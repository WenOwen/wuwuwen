#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM 自动参数调优脚本
使用网格搜索和贝叶斯优化来找到最佳参数组合
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class LightGBMAutoTuner:
    """LightGBM自动调优器"""
    
    def __init__(self, data_dir="./data/professional_parquet"):
        self.data_dir = Path(data_dir)
        self.X = None
        self.y = None
        self.best_params = None
        self.best_score = 0
        self.results = []
        
    def load_data(self):
        """加载数据（复用现有的文件配对逻辑）"""
        print("📊 加载数据...")
        
        parquet_files = sorted(list(self.data_dir.glob("*.parquet")))
        if len(parquet_files) < 2:
            raise ValueError("需要至少2个parquet文件")
        
        features_list = []
        targets_list = []
        
        # 文件配对策略
        for i in range(len(parquet_files) - 1):
            today_file = parquet_files[i]
            tomorrow_file = parquet_files[i+1]
            
            try:
                today_data = pd.read_parquet(today_file)
                tomorrow_data = pd.read_parquet(tomorrow_file)
                
                common_stocks = today_data.index.intersection(tomorrow_data.index)
                if len(common_stocks) > 0:
                    features_list.append(today_data.loc[common_stocks])
                    targets_list.append(tomorrow_data.loc[common_stocks, '涨跌幅'])
                    
            except Exception as e:
                print(f"跳过文件对 {today_file.name}: {e}")
                continue
        
        # 合并数据
        full_data = pd.concat(features_list, ignore_index=False)
        targets_data = pd.concat(targets_list, ignore_index=False)
        
        # 创建方向目标
        self.y = (targets_data > 0).astype(int)
        
        # 选择特征
        exclude_columns = ['name', 'symbol']
        feature_columns = [col for col in full_data.columns if col not in exclude_columns]
        self.X = full_data[feature_columns]
        
        # 只保留数值列
        numeric_columns = self.X.select_dtypes(include=[np.number]).columns
        self.X = self.X[numeric_columns].fillna(0)
        self.y = self.y.fillna(0)
        
        print(f"✅ 数据加载完成: {self.X.shape[0]} 样本, {self.X.shape[1]} 特征")
        print(f"目标分布: 看多={sum(self.y)}, 看空={len(self.y)-sum(self.y)}")
        
    def define_search_space(self):
        """定义搜索空间"""
        
        # 🎯 基础网格搜索参数
        base_grid = {
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.01, 0.03, 0.05],
            'feature_fraction': [0.5, 0.7, 0.9],
            'bagging_fraction': [0.5, 0.7, 0.9],
            'lambda_l1': [0.1, 0.5, 1.0],
            'lambda_l2': [0.1, 0.5, 1.0],
            'min_data_in_leaf': [50, 100, 200],
            'max_depth': [3, 5, 7]
        }
        
        # 🔍 精细搜索参数（在最佳区域附近）
        fine_grid = {
            'num_leaves': [20, 25, 30, 35, 40],
            'learning_rate': [0.02, 0.025, 0.03, 0.035, 0.04],
            'feature_fraction': [0.6, 0.65, 0.7, 0.75, 0.8],
            'bagging_fraction': [0.6, 0.65, 0.7, 0.75, 0.8],
            'lambda_l1': [0.2, 0.3, 0.4, 0.5, 0.6],
            'lambda_l2': [0.2, 0.3, 0.4, 0.5, 0.6]
        }
        
        return base_grid, fine_grid
    
    def evaluate_params(self, params, cv_folds=3):
        """评估参数组合"""
        
        # 固定参数
        fixed_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42,
            'bagging_freq': 1
        }
        
        # 合并参数
        all_params = {**fixed_params, **params}
        
        # 时序交叉验证
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # 检查目标分布
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue
                
            try:
                # 训练模型
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    all_params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=200,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
                
                # 预测和评估
                y_pred = model.predict(X_val)
                auc_score = roc_auc_score(y_val, y_pred)
                scores.append(auc_score)
                
            except Exception as e:
                print(f"  ❌ Fold {fold} 失败: {e}")
                continue
        
        if len(scores) == 0:
            return 0.0
            
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        return mean_score, std_score
    
    def grid_search(self, param_grid, max_combinations=50):
        """网格搜索"""
        print(f"🔍 开始网格搜索，最多测试 {max_combinations} 个组合...")
        
        # 生成参数组合
        param_combinations = list(ParameterGrid(param_grid))
        
        # 如果组合太多，随机采样
        if len(param_combinations) > max_combinations:
            import random
            param_combinations = random.sample(param_combinations, max_combinations)
        
        print(f"📊 将测试 {len(param_combinations)} 个参数组合")
        
        best_score = 0
        best_params = None
        
        for i, params in enumerate(param_combinations):
            print(f"\n🧪 测试组合 {i+1}/{len(param_combinations)}: {params}")
            
            try:
                mean_score, std_score = self.evaluate_params(params)
                
                result = {
                    'params': params,
                    'mean_auc': mean_score,
                    'std_auc': std_score,
                    'combination_id': i+1
                }
                self.results.append(result)
                
                print(f"   📈 平均AUC: {mean_score:.4f} ± {std_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    print(f"   🎯 新的最佳结果!")
                    
            except Exception as e:
                print(f"   ❌ 评估失败: {e}")
                continue
        
        self.best_score = best_score
        self.best_params = best_params
        
        print(f"\n🏆 搜索完成!")
        print(f"最佳AUC: {best_score:.4f}")
        print(f"最佳参数: {best_params}")
        
        return best_params, best_score
    
    def save_results(self, output_dir="./tuning_results"):
        """保存调优结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 保存详细结果
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('mean_auc', ascending=False)
        results_df.to_csv(output_dir / "tuning_results.csv", index=False)
        
        # 保存最佳参数
        with open(output_dir / "best_params.json", 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'best_score': self.best_score,
                'total_combinations': len(self.results)
            }, f, indent=2)
        
        # 保存配置文件
        config = self.create_config_from_params(self.best_params)
        with open(output_dir / "lightGBM_optimized.yaml", 'w', encoding='utf-8') as f:
            f.write(config)
        
        print(f"📁 结果已保存到: {output_dir}")
    
    def create_config_from_params(self, params):
        """从最佳参数创建配置文件"""
        config = f"""# LightGBM 自动调优结果配置文件
# 最佳AUC: {self.best_score:.4f}
# 生成时间: {pd.Timestamp.now()}

data:
  data_dir: "./data/professional_parquet"
  direct_training:
    enabled: true
    data_format: "parquet"
    stock_name_column: "name"
    target_column: "涨跌幅"
    prediction_mode: "direction"
    exclude_columns:
      - "name"
      - "symbol"

training:
  data_split:
    test_size: 0.2
    validation_size: 0.15
    random_state: 42
    stratify: true
    time_series_split: true
    
  training_params:
    early_stopping_rounds: 50
    verbose: 50
    eval_metric: ["auc", "binary_logloss"]

lightgbm:
  basic_params:
    objective: "binary"
    metric: "auc"
    boosting_type: "gbdt"
    num_leaves: {params.get('num_leaves', 31)}
    learning_rate: {params.get('learning_rate', 0.03)}
    feature_fraction: {params.get('feature_fraction', 0.7)}
    bagging_fraction: {params.get('bagging_fraction', 0.7)}
    bagging_freq: 1
    verbose: -1
    random_state: 42
    
  advanced_params:
    max_depth: {params.get('max_depth', 5)}
    min_data_in_leaf: {params.get('min_data_in_leaf', 100)}
    lambda_l1: {params.get('lambda_l1', 0.1)}
    lambda_l2: {params.get('lambda_l2', 0.1)}
    
  fit_params:
    num_boost_round: 500

output:
  file_naming:
    folder_name_prefix: "optimized_training"
  model_save:
    save_dir: "./models/lightgbm_optimized"
  results_save:
    save_dir: "./results/lightgbm_optimized"

evaluation:
  metrics:
    - "auc"
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
"""
        return config

def main():
    """主函数"""
    print("🚀 LightGBM 自动参数调优")
    print("=" * 50)
    
    # 创建调优器
    tuner = LightGBMAutoTuner()
    
    # 加载数据
    tuner.load_data()
    
    # 定义搜索空间
    base_grid, fine_grid = tuner.define_search_space()
    
    # 第一阶段：粗搜索
    print("\n🎯 第一阶段：基础网格搜索")
    best_params_base, best_score_base = tuner.grid_search(base_grid, max_combinations=30)
    
    if best_score_base > 0.5:
        print(f"\n✅ 基础搜索成功! AUC: {best_score_base:.4f}")
        
        # 第二阶段：精细搜索（可选）
        print("\n🎯 第二阶段：精细网格搜索")
        # 基于最佳结果调整精细搜索范围
        fine_grid_adjusted = {
            'num_leaves': [max(15, best_params_base['num_leaves']-10), 
                          best_params_base['num_leaves'], 
                          best_params_base['num_leaves']+10],
            'learning_rate': [best_params_base['learning_rate']*0.8,
                             best_params_base['learning_rate'],
                             best_params_base['learning_rate']*1.2],
            # 其他参数类似调整...
        }
        
        tuner.grid_search(fine_grid_adjusted, max_combinations=20)
    
    # 保存结果
    tuner.save_results()
    
    print(f"\n🎉 调优完成!")
    print(f"🏆 最终最佳AUC: {tuner.best_score:.4f}")
    print(f"📋 最佳参数: {tuner.best_params}")
    print(f"📁 配置文件已生成: ./tuning_results/lightGBM_optimized.yaml")

if __name__ == "__main__":
    main()