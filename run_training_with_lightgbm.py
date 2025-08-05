#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI股市预测系统 - LightGBM完整训练脚本
功能：启动包含LightGBM的完整训练流程
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
from datetime import datetime, timedelta
import logging

# 添加项目路径到系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'core'))
sys.path.insert(0, os.path.join(project_root, 'utils'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def check_data_directory():
    """检查数据目录是否存在"""
    data_dirs = ['datas_em', 'datas_index', 'financial_csv']
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if csv_files:
                logger.info(f"发现数据目录 {data_dir}，包含 {len(csv_files)} 个CSV文件")
                return data_dir, csv_files[:5]  # 返回前5个文件作为测试
    
    logger.error("未找到有效的数据目录！")
    return None, []

def fix_lightgbm_setup():
    """修复LightGBM设置"""
    try:
        import lightgbm as lgb
        logger.info(f"✅ LightGBM版本: {lgb.__version__}")
        
        # 测试基本功能
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=10,
            verbose=-1,
            random_state=42
        )
        model.fit(X, y)
        logger.info("✅ LightGBM基本功能测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ LightGBM设置失败: {str(e)}")
        return False

def simplified_training():
    """简化的训练流程，专注于LightGBM"""
    logger.info("🚀 开始简化版LightGBM训练...")
    
    # 检查数据
    data_dir, sample_files = check_data_directory()
    if not data_dir:
        return False
    
    # 选择一个有数据的股票文件进行训练
    for file_name in sample_files:
        try:
            file_path = os.path.join(data_dir, file_name)
            df = pd.read_csv(file_path)
            
            if len(df) < 100:  # 需要足够的数据
                continue
                
            logger.info(f"使用数据文件: {file_name}, 数据量: {len(df)}")
            
            # 简单的特征工程
            if '收盘价' in df.columns or 'close' in df.columns:
                close_col = '收盘价' if '收盘价' in df.columns else 'close'
                
                # 创建基本特征
                df['price_change'] = df[close_col].pct_change()
                df['price_ma5'] = df[close_col].rolling(5).mean()
                df['price_ma10'] = df[close_col].rolling(10).mean()
                
                # 创建标签（第二天涨跌）
                df['target'] = (df[close_col].shift(-1) > df[close_col]).astype(int)
                
                # 准备训练数据
                feature_cols = ['price_change', 'price_ma5', 'price_ma10']
                df_clean = df[feature_cols + ['target']].dropna()
                
                if len(df_clean) < 50:
                    continue
                
                X = df_clean[feature_cols]
                y = df_clean['target']
                
                # 分割数据
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # 训练LightGBM模型
                import lightgbm as lgb
                
                model = lgb.LGBMClassifier(
                    objective='binary',
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    verbose=-1
                )
                
                logger.info("🔄 开始训练LightGBM模型...")
                model.fit(X_train, y_train)
                
                # 预测和评估
                from sklearn.metrics import accuracy_score, classification_report
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                logger.info(f"✅ 训练完成！")
                logger.info(f"📊 测试集准确率: {accuracy:.4f}")
                logger.info(f"📊 分类报告:\n{classification_report(y_test, y_pred)}")
                
                # 保存模型
                os.makedirs('models', exist_ok=True)
                model_path = f'models/lightgbm_model_{file_name.replace(".csv", "")}.pkl'
                joblib.dump(model, model_path)
                logger.info(f"💾 模型已保存: {model_path}")
                
                return True
                
        except Exception as e:
            logger.error(f"处理文件 {file_name} 时出错: {str(e)}")
            continue
    
    logger.error("❌ 没有找到可用的训练数据")
    return False

def full_training_pipeline():
    """完整的训练流水线"""
    logger.info("🚀 开始完整训练流水线...")
    
    try:
        # 导入训练模块
        from core.training_pipeline import ModelTrainingPipeline
        
        # 修复导入路径问题
        import sys
        sys.path.append('core')
        sys.path.append('utils')
        
        # 创建训练管道
        pipeline = ModelTrainingPipeline()
        
        # 检查数据
        data_dir, sample_files = check_data_directory()
        if not data_dir:
            return simplified_training()
        
        # 提取股票代码
        stock_codes = [f.replace('.csv', '') for f in sample_files]
        logger.info(f"准备训练股票: {stock_codes}")
        
        # 开始训练
        model = pipeline.train_model(
            stock_codes=stock_codes,
            prediction_days=1,
            use_hyperparameter_optimization=False,  # 先不用超参数优化，加快训练
            save_model=True
        )
        
        logger.info("✅ 完整训练流程完成！")
        return True
        
    except Exception as e:
        logger.error(f"完整训练流程失败: {str(e)}")
        logger.info("🔄 切换到简化训练模式...")
        return simplified_training()

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("🤖 AI股市预测系统 - LightGBM训练启动")
    logger.info("=" * 60)
    
    # 1. 检查LightGBM
    if not fix_lightgbm_setup():
        logger.error("❌ LightGBM设置失败，请检查安装")
        return
    
    # 2. 尝试完整训练流程
    try:
        success = full_training_pipeline()
    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}")
        success = False
    
    if success:
        logger.info("🎉 训练成功完成！")
    else:
        logger.info("⚠️ 训练过程中遇到问题，请检查日志")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()