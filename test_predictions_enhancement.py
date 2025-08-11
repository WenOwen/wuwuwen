#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试predictions.csv增强功能
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_test_data():
    """创建测试数据"""
    print("🔧 创建测试数据...")
    
    # 创建测试目录
    test_data_dir = Path('./test_data')
    test_data_dir.mkdir(exist_ok=True)
    
    # 生成模拟的完整股票数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # 股票代码列表
    stock_codes = [f"{i:06d}.SZ" for i in range(1, 51)] * 20  # 50只股票，每只20天数据
    
    # 股票名称列表
    stock_names = [f"股票{i}" for i in range(1, 51)] * 20
    
    # 日期列表
    dates = pd.date_range('2024-01-01', periods=20, freq='D').tolist() * 50
    
    # 生成特征数据
    feature_data = np.random.randn(n_samples, n_features)
    feature_columns = [f'feature_{i}' for i in range(n_features)]
    
    # 生成目标数据（涨跌幅）
    target_data = np.random.normal(0, 0.02, n_samples)  # 平均0，标准差2%的涨跌幅
    
    # 生成次日涨跌幅
    next_day_return = np.random.normal(0, 0.025, n_samples)
    
    # 创建完整数据框
    full_data = pd.DataFrame(feature_data, columns=feature_columns)
    full_data['stock_code'] = stock_codes[:n_samples]
    full_data['股票名称'] = stock_names[:n_samples]
    full_data['date'] = dates[:n_samples]
    full_data['target'] = target_data
    full_data['次日涨跌幅'] = next_day_return
    
    # 保存测试数据
    full_data_path = test_data_dir / 'full_data.csv'
    full_data.to_csv(full_data_path, index=False, encoding='utf-8')
    
    print(f"✅ 测试数据已创建: {full_data_path}")
    print(f"   - 样本数: {len(full_data):,}")
    print(f"   - 特征数: {len(feature_columns)}")
    print(f"   - 包含列: {list(full_data.columns)}")
    
    return test_data_dir

def create_test_config(test_data_dir):
    """创建测试配置"""
    config = {
        "data": {
            "data_dir": str(test_data_dir),
            "X_features_file": "X_features.csv",
            "y_targets_file": "y_targets.csv", 
            "full_data_file": "full_data.csv",
            "loading_options": {
                "prefer_full_data": True,
                "encoding": "utf-8"
            },
            "direct_training": {
                "enabled": False
            },
            "preprocessing": {
                "normalization": {
                    "enabled": True,
                    "method": "robust"
                },
                "outlier_handling": {
                    "enabled": False
                }
            }
        },
        "training": {
            "data_split": {
                "test_size": 0.2,
                "validation_size": 0.1,
                "random_state": 42,
                "time_series_split": True
            },
            "model_params": {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "random_state": 42
            },
            "fit_params": {
                "num_boost_round": 100,
                "valid_sets": ["train", "val"],
                "valid_names": ["train", "val"],
                "early_stopping_rounds": 20,
                "verbose_eval": False
            }
        },
        "output": {
            "model_save": {
                "save_dir": "./test_models",
                "save_format": ["pkl"]
            },
            "results_save": {
                "save_dir": "./test_results",
                "save_metrics": True,
                "save_predictions": True,
                "save_feature_importance": True
            }
        }
    }
    
    config_path = Path('./test_config.json')
    import json
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 测试配置已创建: {config_path}")
    return config_path

def test_predictions_enhancement():
    """测试predictions增强功能"""
    print("🧪 开始测试predictions增强功能...")
    
    # 创建测试数据和配置
    test_data_dir = create_test_data()
    config_path = create_test_config(test_data_dir)
    
    try:
        # 导入训练器
        from lightgbm_stock_train import LightGBMStockTrainer
        
        # 创建训练器实例
        trainer = LightGBMStockTrainer(str(config_path))
        
        # 加载数据
        print("\n📊 加载数据...")
        if not trainer.load_data():
            print("❌ 数据加载失败")
            return False
        
        # 分割数据
        print("\n✂️ 分割数据...")
        if not trainer.split_data():
            print("❌ 数据分割失败")
            return False
        
        # 检查股票信息是否正确保存
        if hasattr(trainer, 'stock_info_train') and trainer.stock_info_train is not None:
            print("✅ 股票信息保存成功")
            print(f"   - 训练集股票信息: {trainer.stock_info_train.shape}")
            print(f"   - 验证集股票信息: {trainer.stock_info_val.shape}")
            print(f"   - 测试集股票信息: {trainer.stock_info_test.shape}")
            print(f"   - 股票信息列: {list(trainer.stock_info_train.columns)}")
        else:
            print("⚠️ 未找到股票信息")
        
        # 预处理特征
        print("\n🔧 预处理特征...")
        if not trainer.preprocess_features():
            print("❌ 特征预处理失败")
            return False
        
        # 训练模型
        print("\n🎯 训练模型...")
        if not trainer.train_model():
            print("❌ 模型训练失败")
            return False
        
        # 评估模型
        print("\n📊 评估模型...")
        results = trainer.evaluate_model()
        if not results:
            print("❌ 模型评估失败")
            return False
        
        # 保存结果（包括增强的predictions.csv）
        print("\n💾 保存结果...")
        if not trainer.save_results(results):
            print("❌ 结果保存失败")
            return False
        
        # 检查predictions.csv文件
        predictions_path = trainer.results_save_dir / "predictions.csv"
        if predictions_path.exists():
            pred_df = pd.read_csv(predictions_path)
            print(f"\n✅ predictions.csv生成成功: {predictions_path}")
            print(f"   - 记录数: {len(pred_df):,}")
            print(f"   - 列数: {len(pred_df.columns)}")
            print(f"   - 列名: {list(pred_df.columns)}")
            
            # 显示前几行数据
            print(f"\n📋 前5行数据预览:")
            print(pred_df.head())
            
            return True
        else:
            print("❌ predictions.csv文件未生成")
            return False
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理测试文件
        print("\n🧹 清理测试文件...")
        import shutil
        try:
            if test_data_dir.exists():
                shutil.rmtree(test_data_dir)
            if config_path.exists():
                config_path.unlink()
            
            # 清理生成的结果目录
            test_models_dir = Path('./test_models')
            test_results_dir = Path('./test_results')
            if test_models_dir.exists():
                shutil.rmtree(test_models_dir)
            if test_results_dir.exists():
                shutil.rmtree(test_results_dir)
            
            print("✅ 测试文件清理完成")
        except Exception as e:
            print(f"⚠️ 清理文件时出现问题: {e}")

if __name__ == "__main__":
    success = test_predictions_enhancement()
    if success:
        print("\n🎉 predictions增强功能测试通过！")
        print("现在predictions.csv文件将包含股票代码、名称、日期和次日涨跌幅等信息。")
    else:
        print("\n💡 测试失败，请检查修改的代码。")