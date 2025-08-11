#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据预处理脚本
专门用于处理professional_parquet文件夹中的股票数据
转换为适合lightGBM训练的格式
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')


class StockDataProcessor:
    """股票数据预处理器"""
    
    def __init__(self, data_dir: str = "data/professional_parquet", 
                 output_dir: str = "data/processed_stock_data"):
        """
        初始化数据处理器
        
        Args:
            data_dir: 输入数据目录 (professional_parquet)
            output_dir: 输出数据目录
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.stock_data = {}
        self.feature_data = None
        self.target_data = None
        self.stock_codes = []
        self.feature_names = []
        
        print(f"📁 数据输入路径: {self.data_dir}")
        print(f"📁 数据输出路径: {self.output_dir}")
    
    def load_parquet_data(self) -> bool:
        """加载parquet格式的股票数据"""
        try:
            print("📊 加载parquet股票数据...")
            
            # 检查输入目录是否存在
            if not self.data_dir.exists():
                print(f"❌ 数据目录不存在: {self.data_dir}")
                return False
            
            # 获取所有parquet文件
            parquet_files = list(self.data_dir.glob("*.parquet"))
            if not parquet_files:
                print(f"❌ 在{self.data_dir}中未找到parquet文件")
                return False
            
            print(f"   发现 {len(parquet_files)} 个parquet文件")
            
            # 加载并合并所有parquet文件
            all_data = []
            for file_path in tqdm(parquet_files, desc="加载parquet文件"):
                try:
                    df = pd.read_parquet(file_path)
                    # 添加文件名信息（可能包含日期等信息）
                    df['data_source'] = file_path.stem
                    all_data.append(df)
                except Exception as e:
                    print(f"   ⚠️ 跳过文件 {file_path}: {e}")
                    continue
            
            if not all_data:
                print("❌ 没有成功加载任何数据文件")
                return False
            
            # 合并所有数据
            self.stock_data = pd.concat(all_data, ignore_index=True)
            print(f"   ✅ 成功加载数据: {self.stock_data.shape}")
            print(f"   📊 数据列: {list(self.stock_data.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return False
    
    def analyze_data_structure(self):
        """分析数据结构"""
        print("\n📊 数据结构分析:")
        print(f"   总行数: {len(self.stock_data):,}")
        print(f"   总列数: {len(self.stock_data.columns)}")
        
        # 检查是否有股票代码列
        code_columns = [col for col in self.stock_data.columns 
                       if any(keyword in col.lower() for keyword in ['code', '代码', 'symbol'])]
        print(f"   可能的股票代码列: {code_columns}")
        
        # 检查是否有日期列
        date_columns = [col for col in self.stock_data.columns 
                       if any(keyword in col.lower() for keyword in ['date', '日期', 'time', '时间'])]
        print(f"   可能的日期列: {date_columns}")
        
        # 检查数值列
        numeric_columns = self.stock_data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"   数值列数量: {len(numeric_columns)}")
        
        # 显示前几列样例
        print(f"\n   前5列数据预览:")
        print(self.stock_data.head().iloc[:, :5])
        
        return {
            'total_rows': len(self.stock_data),
            'total_columns': len(self.stock_data.columns),
            'code_columns': code_columns,
            'date_columns': date_columns,
            'numeric_columns': numeric_columns
        }
    
    def create_features_and_targets(self, target_column: str = None, 
                                   code_column: str = None,
                                   date_column: str = None,
                                   lookback_days: int = 5,
                                   target_days: int = 1) -> bool:
        """
        创建特征和目标数据
        
        Args:
            target_column: 目标列名（如股价涨跌幅）
            code_column: 股票代码列名
            date_column: 日期列名
            lookback_days: 回看天数（用于创建滞后特征）
            target_days: 预测天数
        """
        try:
            print(f"\n🔧 创建特征和目标数据...")
            
            # 自动识别关键列
            if code_column is None:
                code_columns = [col for col in self.stock_data.columns 
                               if any(keyword in col.lower() for keyword in ['code', '代码', 'symbol'])]
                code_column = code_columns[0] if code_columns else self.stock_data.columns[0]
                print(f"   自动选择股票代码列: {code_column}")
            
            if date_column is None:
                date_columns = [col for col in self.stock_data.columns 
                               if any(keyword in col.lower() for keyword in ['date', '日期', 'time', '时间'])]
                if date_columns:
                    date_column = date_columns[0]
                    print(f"   自动选择日期列: {date_column}")
            
            if target_column is None:
                # 寻找涨跌幅相关列
                pct_columns = [col for col in self.stock_data.columns 
                              if any(keyword in col.lower() for keyword in ['pct', '涨跌', 'return', '收益'])]
                if pct_columns:
                    target_column = pct_columns[0]
                    print(f"   自动选择目标列: {target_column}")
                else:
                    print("❌ 无法自动识别目标列，请手动指定")
                    return False
            
            # 确保日期列为datetime格式
            if date_column and date_column in self.stock_data.columns:
                self.stock_data[date_column] = pd.to_datetime(self.stock_data[date_column])
                self.stock_data = self.stock_data.sort_values([code_column, date_column])
            
            # 获取所有股票代码
            self.stock_codes = self.stock_data[code_column].unique().tolist()
            print(f"   股票数量: {len(self.stock_codes)}")
            
            # 构建特征和目标数据
            features_list = []
            targets_list = []
            
            for stock_code in tqdm(self.stock_codes, desc="处理股票数据"):
                stock_df = self.stock_data[self.stock_data[code_column] == stock_code].copy()
                
                if len(stock_df) < lookback_days + target_days:
                    continue  # 数据不足，跳过
                
                # 创建滞后特征
                numeric_cols = stock_df.select_dtypes(include=[np.number]).columns.tolist()
                
                for i in range(lookback_days, len(stock_df) - target_days + 1):
                    # 特征：过去lookback_days天的数据
                    feature_row = {}
                    feature_row['stock_code'] = stock_code
                    
                    if date_column:
                        feature_row['date'] = stock_df.iloc[i + target_days - 1][date_column]
                    
                    # 添加历史特征
                    for day in range(lookback_days):
                        day_data = stock_df.iloc[i - lookback_days + day]
                        for col in numeric_cols:
                            if col == target_column:
                                continue
                            feature_row[f'{col}_lag_{lookback_days - day}'] = day_data[col]
                    
                    # 添加统计特征
                    window_data = stock_df.iloc[i - lookback_days:i]
                    for col in numeric_cols:
                        if col == target_column:
                            continue
                        values = window_data[col]
                        feature_row[f'{col}_mean'] = values.mean()
                        feature_row[f'{col}_std'] = values.std()
                        feature_row[f'{col}_max'] = values.max()
                        feature_row[f'{col}_min'] = values.min()
                    
                    features_list.append(feature_row)
                    
                    # 目标：未来target_days天的收益
                    target_value = stock_df.iloc[i + target_days - 1][target_column]
                    targets_list.append({
                        'stock_code': stock_code,
                        'target': target_value
                    })
            
            # 转换为DataFrame
            self.feature_data = pd.DataFrame(features_list)
            self.target_data = pd.DataFrame(targets_list)
            
            print(f"   ✅ 特征数据shape: {self.feature_data.shape}")
            print(f"   ✅ 目标数据shape: {self.target_data.shape}")
            
            # 保存特征名称
            self.feature_names = [col for col in self.feature_data.columns 
                                 if col not in ['stock_code', 'date']]
            
            return True
            
        except Exception as e:
            print(f"❌ 创建特征和目标数据失败: {e}")
            return False
    
    def save_processed_data(self) -> bool:
        """保存处理后的数据"""
        try:
            print(f"\n💾 保存处理后的数据...")
            
            # 创建时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = self.output_dir / f"processed_{timestamp}"
            output_folder.mkdir(exist_ok=True)
            
            # 分离特征和目标
            X_features = self.feature_data[self.feature_names]
            y_targets = self.target_data['target']
            
            # 保存CSV文件
            X_features.to_csv(output_folder / "X_features.csv", index=False, encoding='utf-8')
            y_targets.to_csv(output_folder / "y_targets.csv", index=False, encoding='utf-8')
            
            # 保存完整数据（包含股票代码等信息）
            full_data = self.feature_data.copy()
            full_data['target'] = y_targets
            full_data.to_csv(output_folder / "full_data.csv", index=False, encoding='utf-8')
            
            # 保存股票代码信息
            stock_info = {
                'stock_codes': self.stock_codes,
                'total_stocks': len(self.stock_codes),
                'total_samples': len(self.feature_data),
                'feature_count': len(self.feature_names)
            }
            with open(output_folder / "stock_codes.json", 'w', encoding='utf-8') as f:
                json.dump(stock_info, f, ensure_ascii=False, indent=2)
            
            # 保存数据处理信息
            data_info = {
                'processing_time': timestamp,
                'input_data_dir': str(self.data_dir),
                'output_data_dir': str(output_folder),
                'total_samples': len(self.feature_data),
                'feature_count': len(self.feature_names),
                'feature_names': self.feature_names[:10],  # 只保存前10个特征名作为样例
                'data_shape': {
                    'features': list(X_features.shape),
                    'targets': list(y_targets.shape)
                }
            }
            with open(output_folder / "data_info.json", 'w', encoding='utf-8') as f:
                json.dump(data_info, f, ensure_ascii=False, indent=2)
            
            print(f"   ✅ 数据已保存到: {output_folder}")
            print(f"   📁 X_features.csv: {X_features.shape}")
            print(f"   📁 y_targets.csv: {y_targets.shape}")
            print(f"   📁 full_data.csv: {full_data.shape}")
            print(f"   📁 stock_codes.json: {len(self.stock_codes)} 只股票")
            print(f"   📁 data_info.json: 数据处理信息")
            
            # 返回输出路径供后续使用
            self.processed_data_path = output_folder
            return True
            
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
            return False
    
    def run_full_pipeline(self, target_column: str = None, 
                         code_column: str = None,
                         date_column: str = None,
                         lookback_days: int = 5,
                         target_days: int = 1) -> str:
        """
        运行完整的数据处理流程
        
        Returns:
            str: 处理后数据的路径
        """
        print("🚀 开始股票数据处理流程...")
        
        # 1. 加载数据
        if not self.load_parquet_data():
            return None
        
        # 2. 分析数据结构
        self.analyze_data_structure()
        
        # 3. 创建特征和目标
        if not self.create_features_and_targets(
            target_column=target_column,
            code_column=code_column,
            date_column=date_column,
            lookback_days=lookback_days,
            target_days=target_days
        ):
            return None
        
        # 4. 保存处理后的数据
        if not self.save_processed_data():
            return None
        
        print(f"\n✅ 数据处理完成!")
        print(f"📊 最终数据概况:")
        print(f"   - 样本数量: {len(self.feature_data):,}")
        print(f"   - 特征数量: {len(self.feature_names)}")
        print(f"   - 股票数量: {len(self.stock_codes)}")
        print(f"   - 数据路径: {self.processed_data_path}")
        
        return str(self.processed_data_path)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='股票数据预处理')
    parser.add_argument('--data_dir', type=str, default='data/professional_parquet',
                       help='输入数据目录')
    parser.add_argument('--output_dir', type=str, default='data/processed_stock_data',
                       help='输出数据目录')
    parser.add_argument('--target_column', type=str, default=None,
                       help='目标列名')
    parser.add_argument('--code_column', type=str, default=None,
                       help='股票代码列名')
    parser.add_argument('--date_column', type=str, default=None,
                       help='日期列名')
    parser.add_argument('--lookback_days', type=int, default=5,
                       help='回看天数')
    parser.add_argument('--target_days', type=int, default=1,
                       help='预测天数')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = StockDataProcessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # 运行处理流程
    result_path = processor.run_full_pipeline(
        target_column=args.target_column,
        code_column=args.code_column,
        date_column=args.date_column,
        lookback_days=args.lookback_days,
        target_days=args.target_days
    )
    
    if result_path:
        print(f"\n🎉 处理成功! 数据已保存到: {result_path}")
        print("\n📝 下一步可以运行训练脚本:")
        print(f"python lightgbm_train.py --config config/train/lightGBM_train.yaml")
    else:
        print("\n❌ 数据处理失败")
        sys.exit(1)


if __name__ == "__main__":
    main()