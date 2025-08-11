#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复训练脚本：替换为简单的文件配对方案
"""

def fix_training_script():
    """修复训练脚本"""
    print("🔧 开始修复训练脚本...")
    
    # 读取原文件
    with open('lightgbm_stock_train.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"   原文件行数: {len(lines)}")
    
    # 新的_load_direct_data方法
    new_method = '''    def _load_direct_data(self) -> bool:
        """直接加载parquet格式的股票数据 - 简化版"""
        try:
            self.logger.info("📊 使用直接训练模式加载数据...")
            
            data_config = self.config.get('data', {})
            direct_training = data_config.get('direct_training', {})
            
            data_dir = Path(data_config.get('data_dir', './data/professional_parquet'))
            data_format = direct_training.get('data_format', 'parquet')
            target_column = direct_training.get('target_column', '涨跌幅')
            exclude_columns = direct_training.get('exclude_columns', ['name', '涨跌幅'])
            
            # 加载数据文件
            if data_format == 'parquet':
                # 查找parquet文件
                parquet_files = list(data_dir.glob("*.parquet"))
                if not parquet_files:
                    self.logger.error(f"❌ 在{data_dir}中未找到parquet文件")
                    return False
                
                self.logger.info(f"   发现 {len(parquet_files)} 个parquet文件")
                
                # 使用简单高效的文件配对方案
                parquet_files = sorted(parquet_files)  # 按日期排序
                self.logger.info("   📅 使用文件配对方案：今天文件 → 明天目标")
                
                features_list = []
                targets_list = []
                processed_pairs = 0
                
                # 相邻文件配对
                for i in range(len(parquet_files) - 1):
                    today_file = parquet_files[i]      # 今天的特征
                    tomorrow_file = parquet_files[i+1]  # 明天的目标
                    
                    try:
                        # 读取今天的数据作为特征
                        today_data = pd.read_parquet(today_file)
                        # 读取明天的数据提取目标
                        tomorrow_data = pd.read_parquet(tomorrow_file)
                        
                        # 按股票代码匹配（取交集）
                        common_stocks = today_data.index.intersection(tomorrow_data.index)
                        
                        if len(common_stocks) > 0:
                            # 今天的所有信息作为特征
                            features_list.append(today_data.loc[common_stocks])
                            # 明天的涨跌幅作为目标
                            targets_list.append(tomorrow_data.loc[common_stocks, target_column])
                            processed_pairs += 1
                            
                        self.logger.info(f"   ✅ 配对: {today_file.name} → {tomorrow_file.name}, 股票: {len(common_stocks)}")
                        
                    except Exception as e:
                        self.logger.warning(f"   跳过配对 {today_file.name} → {tomorrow_file.name}: {e}")
                        continue
                
                if not features_list:
                    self.logger.error("❌ 没有成功配对任何文件")
                    return False
                
                # 合并所有配对的数据
                self.logger.info(f"   🔄 合并 {processed_pairs} 个文件配对的数据...")
                full_data = pd.concat(features_list, ignore_index=False)
                targets_data = pd.concat(targets_list, ignore_index=False)
                
                # 添加目标列
                full_data['next_day_target'] = targets_data
                
                self.logger.info(f"   ✅ 文件配对完成:")
                self.logger.info(f"   - 处理文件对: {processed_pairs}")
                self.logger.info(f"   - 最终样本数: {len(full_data):,}")
                self.logger.info(f"   - 特征列数: {len(full_data.columns)}")
                
            else:
                self.logger.error(f"❌ 不支持的数据格式: {data_format}")
                return False
            
            # 检查次日预测目标列是否存在
            if 'next_day_target' not in full_data.columns:
                self.logger.error(f"❌ 未找到次日预测目标列 'next_day_target'")
                return False
            
            # 设置目标变量（明天的涨跌幅）
            self.y = full_data['next_day_target']
            actual_target_column = 'next_day_target'
            
            # 排除目标列和辅助列，保留今天的涨跌幅作为特征
            exclude_columns = exclude_columns + ['next_day_target']
            self.logger.info(f"   💡 今天的'{target_column}'用作预测明天涨跌幅的特征")
            
            # 选择特征列（排除指定的列）
            feature_columns = [col for col in full_data.columns if col not in exclude_columns]
            self.X = full_data[feature_columns]
            
            # 只保留数值列作为特征
            numeric_columns = self.X.select_dtypes(include=[np.number]).columns
            self.X = self.X[numeric_columns]
            
            self.logger.info(f"   📋 排除的列: {exclude_columns}")
            self.logger.info(f"   📊 数值特征列数: {len(numeric_columns)}")
            
            # 处理缺失值
            self.X = self.X.fillna(0)
            self.y = self.y.fillna(0)
            
            # 保存特征名称
            self.feature_names = list(self.X.columns)
            
            # 保存股票信息
            stock_name_column = direct_training.get('stock_name_column', 'name')
            if stock_name_column in full_data.columns:
                self.stock_info = full_data[[stock_name_column]].copy()
            else:
                self.stock_info = None
            
            self.logger.info(f"   ✅ 次日预测数据加载完成:")
            self.logger.info(f"     - 特征维度: {self.X.shape}")
            self.logger.info(f"     - 目标维度: {self.y.shape}")
            self.logger.info(f"     - 特征数量: {len(self.feature_names)}")
            self.logger.info(f"     - 目标列: {actual_target_column}")
            self.logger.info(f"     - 预测任务: 今天特征 → 明天涨跌幅")
            self.logger.info(f"     - 目标值范围: [{self.y.min():.4f}, {self.y.max():.4f}]")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 直接数据加载失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

'''
    
    # 构建新文件
    new_lines = []
    
    # 1. 复制307行之前的内容（保留_create_next_day_prediction_data之前的部分）
    new_lines.extend(lines[:306])  # 0-305行
    
    # 2. 跳过_create_next_day_prediction_data方法（307-412行）
    # 直接到_load_direct_data方法的开始
    
    # 3. 插入新的_load_direct_data方法
    new_lines.append(new_method)
    new_lines.append('\n')
    
    # 4. 复制526行之后的内容（split_data方法开始）
    new_lines.extend(lines[525:])  # 从525行开始
    
    # 写入新文件
    with open('lightgbm_stock_train.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"   修复后行数: {len(new_lines)}")
    print("   ✅ 修复完成！")
    print("   - 删除了复杂的_create_next_day_prediction_data方法")
    print("   - 替换为简单的文件配对_load_direct_data方法")
    print("   - 大幅提升处理速度")

if __name__ == "__main__":
    fix_training_script()