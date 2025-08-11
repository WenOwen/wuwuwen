#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修改训练脚本，使用简单的文件配对方案
"""

# 需要替换的新代码
new_loading_code = '''
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
'''

print("新的简单文件配对代码已准备好")
print("需要替换lightgbm_stock_train.py中439-466行的复杂逻辑")