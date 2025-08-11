#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM股票训练演示脚本
用于验证完整的训练流程
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def create_demo_stock_data():
    """创建演示用的股票数据"""
    print("📊 创建演示股票数据...")
    
    # 创建目录
    demo_dir = Path("data/demo_professional_parquet")
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成模拟股票数据
    np.random.seed(42)
    
    # 股票代码列表
    stock_codes = ['000001', '000002', '600036', '600519', '000858']
    
    # 日期范围
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    all_data = []
    
    for stock_code in stock_codes:
        # 生成该股票的数据
        n_days = len(date_range)
        
        # 基础价格走势（随机游走）
        base_price = 100.0
        price_changes = np.random.normal(0, 0.02, n_days)  # 2%的日波动
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # 价格不能为负
        
        for i, date in enumerate(date_range):
            # 跳过周末（简化处理）
            if date.weekday() >= 5:
                continue
                
            price = prices[i]
            
            # 计算各种指标
            volume = np.random.randint(1000000, 10000000)  # 成交量
            
            # 技术指标
            ma5 = np.mean(prices[max(0, i-4):i+1])  # 5日均线
            ma20 = np.mean(prices[max(0, i-19):i+1])  # 20日均线
            
            # 波动率
            if i >= 5:
                volatility = np.std(price_changes[max(0, i-4):i+1])
            else:
                volatility = 0.02
            
            # 涨跌幅
            if i > 0:
                pct_change = (prices[i] - prices[i-1]) / prices[i-1]
            else:
                pct_change = 0.0
            
            # RSI指标（简化计算）
            rsi = 50 + np.random.normal(0, 10)
            rsi = max(0, min(100, rsi))
            
            # MACD指标（简化）
            macd = np.random.normal(0, 0.5)
            macd_signal = macd * 0.8 + np.random.normal(0, 0.1)
            macd_hist = macd - macd_signal
            
            # 布林带
            bollinger_upper = ma20 * 1.02
            bollinger_lower = ma20 * 0.98
            
            row_data = {
                'stock_code': stock_code,
                'date': date.strftime('%Y-%m-%d'),
                'open': price * (1 + np.random.normal(0, 0.005)),
                'high': price * (1 + abs(np.random.normal(0, 0.01))),
                'low': price * (1 - abs(np.random.normal(0, 0.01))),
                'close': price,
                'volume': volume,
                'pct_change': pct_change,
                'ma5': ma5,
                'ma20': ma20,
                'volatility': volatility,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'bollinger_upper': bollinger_upper,
                'bollinger_lower': bollinger_lower,
                'turnover_rate': np.random.uniform(0.5, 5.0),
                'pe_ratio': np.random.uniform(10, 50),
                'pb_ratio': np.random.uniform(1, 10),
            }
            
            all_data.append(row_data)
    
    # 转换为DataFrame并保存
    df = pd.DataFrame(all_data)
    
    # 保存为parquet文件
    output_file = demo_dir / "demo_stock_data.parquet"
    df.to_parquet(output_file, index=False)
    
    print(f"   ✅ 演示数据已创建: {output_file}")
    print(f"   📊 数据规模: {df.shape}")
    print(f"   📈 股票数量: {len(stock_codes)}")
    print(f"   📅 日期范围: {df['date'].min()} 到 {df['date'].max()}")
    
    return str(demo_dir)

def run_demo_training():
    """运行演示训练"""
    print("\n🚀 开始演示训练流程...")
    
    try:
        # 1. 创建演示数据
        demo_data_dir = create_demo_stock_data()
        
        # 2. 运行数据预处理
        print("\n📊 运行数据预处理...")
        from stock_data_processor import StockDataProcessor
        
        processor = StockDataProcessor(
            data_dir=demo_data_dir,
            output_dir="data/demo_processed"
        )
        
        processed_path = processor.run_full_pipeline(
            target_column='pct_change',
            code_column='stock_code',
            date_column='date',
            lookback_days=5,
            target_days=1
        )
        
        if not processed_path:
            print("❌ 数据预处理失败")
            return False
        
        print(f"✅ 数据预处理完成: {processed_path}")
        
        # 3. 验证数据
        print("\n🔍 验证处理后的数据...")
        X_features = pd.read_csv(Path(processed_path) / "X_features.csv")
        y_targets = pd.read_csv(Path(processed_path) / "y_targets.csv")
        
        print(f"   特征数据: {X_features.shape}")
        print(f"   目标数据: {y_targets.shape}")
        print(f"   特征列样例: {X_features.columns[:5].tolist()}")
        
        # 4. 创建简化的训练配置
        demo_config = {
            'data': {
                'data_dir': processed_path,
                'source_data': {
                    'parquet_dir': demo_data_dir,
                    'auto_process': False
                },
                'loading_options': {
                    'prefer_full_data': True,
                    'encoding': 'utf-8',
                    'validate_data': True
                },
                'preprocessing': {
                    'normalization': {
                        'method': 'robust'
                    },
                    'outlier_handling': {
                        'enabled': True,
                        'method': 'winsorize',
                        'winsorize_limits': [0.01, 0.01]
                    }
                }
            },
            'training': {
                'data_split': {
                    'test_size': 0.2,
                    'validation_size': 0.1,
                    'random_state': 42,
                    'time_series_split': True
                }
            },
            'lightgbm': {
                'basic_params': {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'verbose': -1,
                    'random_state': 42
                },
                'fit_params': {
                    'num_boost_round': 100
                }
            },
            'feature_selection': {
                'enabled': False
            },
            'hyperparameter_tuning': {
                'enabled': False
            },
            'output': {
                'file_naming': {
                    'identifier_type': 'unique_id',
                    'folder_name_prefix': 'demo_training',
                    'show_id_in_log': True
                },
                'model_save': {
                    'save_dir': './models/demo_lightgbm',
                    'model_name': 'demo_stock_model',
                    'save_format': ['pkl']
                },
                'results_save': {
                    'save_dir': './results/demo_lightgbm',
                    'save_predictions': True,
                    'save_feature_importance': True,
                    'save_metrics': True
                },
                'logging': {
                    'log_level': 'INFO',
                    'console_output': True
                }
            },
            'evaluation': {
                'metrics': ['rmse', 'mae', 'r2_score']
            },
            'misc': {
                'n_jobs': 1,
                'random_seed': 42
            }
        }
        
        # 5. 保存演示配置
        import yaml
        demo_config_path = "config/train/demo_config.yaml"
        os.makedirs(os.path.dirname(demo_config_path), exist_ok=True)
        
        with open(demo_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(demo_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 演示配置已保存: {demo_config_path}")
        
        print("\n🎉 演示数据准备完成!")
        print("\n📝 接下来可以运行:")
        print(f"python lightgbm_stock_train.py --config {demo_config_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🎯 LightGBM股票训练演示")
    print("=" * 60)
    
    success = run_demo_training()
    
    if success:
        print("\n✅ 演示准备成功!")
    else:
        print("\n❌ 演示准备失败!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())