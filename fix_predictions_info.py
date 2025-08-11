#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复predictions.csv缺失股票代码和日期的问题
"""

import re
from pathlib import Path

def fix_direct_training_stock_info():
    """修复直接训练模式下的股票信息收集"""
    
    script_path = Path('./lightgbm_stock_train.py')
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找并替换文件配对部分的代码
    old_pairing_code = '''                # 相邻文件配对
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
                        continue'''
    
    new_pairing_code = '''                # 相邻文件配对
                features_list = []
                targets_list = []
                stock_info_list = []  # 新增：收集股票信息
                
                for i in range(len(parquet_files) - 1):
                    today_file = parquet_files[i]      # 今天的特征
                    tomorrow_file = parquet_files[i+1]  # 明天的目标
                    
                    try:
                        # 从文件名提取日期
                        import re
                        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', today_file.name)
                        if date_match:
                            trade_date = date_match.group(1)
                        else:
                            # 如果文件名不包含标准日期格式，使用文件名
                            trade_date = today_file.stem
                        
                        # 读取今天的数据作为特征
                        today_data = pd.read_parquet(today_file)
                        # 读取明天的数据提取目标
                        tomorrow_data = pd.read_parquet(tomorrow_file)
                        
                        # 按股票代码匹配（取交集）
                        common_stocks = today_data.index.intersection(tomorrow_data.index)
                        
                        if len(common_stocks) > 0:
                            # 今天的所有信息作为特征
                            today_features = today_data.loc[common_stocks]
                            features_list.append(today_features)
                            
                            # 明天的涨跌幅作为目标
                            tomorrow_targets = tomorrow_data.loc[common_stocks, target_column]
                            targets_list.append(tomorrow_targets)
                            
                            # 收集股票信息：股票代码、日期、股票名称
                            stock_info_batch = pd.DataFrame({
                                'stock_code': common_stocks,
                                'date': trade_date,
                                'next_day_return': tomorrow_targets.values
                            })
                            
                            # 如果有股票名称信息
                            if 'name' in today_data.columns:
                                stock_info_batch['stock_name'] = today_data.loc[common_stocks, 'name'].values
                            
                            stock_info_list.append(stock_info_batch)
                            processed_pairs += 1
                            
                        self.logger.info(f"   ✅ 配对: {today_file.name} → {tomorrow_file.name}, 日期: {trade_date}, 股票: {len(common_stocks)}")
                        
                    except Exception as e:
                        self.logger.warning(f"   跳过配对 {today_file.name} → {tomorrow_file.name}: {e}")
                        continue'''
    
    # 替换文件配对代码
    if old_pairing_code in content:
        content = content.replace(old_pairing_code, new_pairing_code)
        print("✅ 已更新文件配对逻辑，现在会收集股票代码和日期信息")
    else:
        print("⚠️ 未找到需要替换的文件配对代码")
    
    # 修改数据合并部分
    old_merge_code = '''                # 合并所有配对的数据
                self.logger.info(f"   🔄 合并 {processed_pairs} 个文件配对的数据...")
                full_data = pd.concat(features_list, ignore_index=False)
                targets_data = pd.concat(targets_list, ignore_index=False)
                
                # 添加目标列
                full_data['next_day_target'] = targets_data'''
    
    new_merge_code = '''                # 合并所有配对的数据
                self.logger.info(f"   🔄 合并 {processed_pairs} 个文件配对的数据...")
                full_data = pd.concat(features_list, ignore_index=False)
                targets_data = pd.concat(targets_list, ignore_index=False)
                
                # 合并股票信息
                all_stock_info = pd.concat(stock_info_list, ignore_index=True)
                
                # 添加目标列
                full_data['next_day_target'] = targets_data'''
    
    if old_merge_code in content:
        content = content.replace(old_merge_code, new_merge_code)
        print("✅ 已更新数据合并逻辑")
    else:
        print("⚠️ 未找到需要替换的数据合并代码")
    
    # 修改股票信息保存部分
    old_stock_info_save = '''            # 为直接训练模式保存股票信息
            # 从索引中提取股票代码（假设索引包含股票代码）
            stock_codes = full_data.index.tolist()
            stock_info_data = {
                'stock_code': stock_codes,
                'next_day_return': raw_targets.tolist()  # 保存次日涨跌幅
            }
            
            # 如果原始数据中有其他信息列（如股票名称）
            if 'name' in full_data.columns:
                stock_info_data['stock_name'] = full_data['name'].tolist()
            
            # 如果有日期信息，可以从文件名中推断
            # 这里简化处理，可以根据实际情况添加日期逻辑
            
            self.stock_info = pd.DataFrame(stock_info_data)
            self.logger.info(f"   ✅ 保存股票信息: {list(self.stock_info.columns)}")'''
    
    new_stock_info_save = '''            # 为直接训练模式保存股票信息
            # 使用之前收集的完整股票信息
            self.stock_info = all_stock_info
            self.logger.info(f"   ✅ 保存完整股票信息: {list(self.stock_info.columns)}")
            self.logger.info(f"   📊 股票信息样本数: {len(self.stock_info):,}")
            self.logger.info(f"   📅 日期范围: {self.stock_info['date'].nunique()} 个交易日")
            self.logger.info(f"   🏢 股票数量: {self.stock_info['stock_code'].nunique()} 只股票")'''
    
    if old_stock_info_save in content:
        content = content.replace(old_stock_info_save, new_stock_info_save)
        print("✅ 已更新股票信息保存逻辑")
    else:
        print("⚠️ 未找到需要替换的股票信息保存代码")
    
    # 保存修改后的文件
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ lightgbm_stock_train.py 修改完成")

def create_example_output():
    """展示修复后的predictions.csv示例"""
    
    example = '''
修复后的predictions.csv将包含以下列：

| split | y_true | y_pred | stock_code | stock_name | date       | next_day_return |
|-------|--------|--------|------------|------------|------------|-----------------|
| train | 1      | 0.85   | 000001.SZ  | 平安银行   | 2024-01-15 | 0.023          |
| train | 0      | 0.15   | 000002.SZ  | 万科A     | 2024-01-15 | -0.015         |
| val   | 1      | 0.92   | 000001.SZ  | 平安银行   | 2024-02-01 | 0.031          |
| test  | 0      | 0.08   | 000002.SZ  | 万科A     | 2024-02-15 | -0.012         |

📋 现在包含的信息：
✅ split - 数据集分割（train/val/test）
✅ y_true - 真实标签（涨跌方向）
✅ y_pred - 预测概率
✅ stock_code - 股票代码（从索引获取）
✅ stock_name - 股票名称（如果原数据有name列）
✅ date - 交易日期（从parquet文件名提取）
✅ next_day_return - 次日实际涨跌幅

💡 日期提取逻辑：
- 从文件名中查找 YYYY-MM-DD 格式的日期
- 如果找不到，使用文件名（去除扩展名）作为日期标识
'''
    
    return example

def main():
    print("🔧 开始修复predictions.csv缺失信息问题...")
    print("="*60)
    
    # 修复代码
    fix_direct_training_stock_info()
    
    print("\n📋 修复内容说明：")
    print("1. ✅ 在文件配对时从文件名提取日期")
    print("2. ✅ 收集每个样本的股票代码、日期、名称")
    print("3. ✅ 确保所有信息都保存到predictions.csv")
    
    print("\n📊 修复后的输出示例：")
    print(create_example_output())
    
    print("\n🚀 使用方法：")
    print("1. 重新运行训练脚本")
    print("2. 检查生成的predictions.csv文件")
    print("3. 现在应该包含完整的股票代码、名称和日期信息")

if __name__ == "__main__":
    main()