# -*- coding: utf-8 -*-
"""
AI股市预测系统 - 核心模块
包含预测服务、模型管理、特征工程等核心功能
"""

__version__ = "1.0.0"
__author__ = "AI股市预测系统"

# 主要模块导入
try:
    from .prediction_service import PredictionService
    from .ai_models import EnsembleModel, create_ensemble_model
    from .feature_engineering import FeatureEngineering
    from .training_pipeline import ModelTrainingPipeline, AutoRetrainingSystem
except ImportError as e:
    # 某些依赖可能还未安装，这是正常的
    pass

__all__ = [
    'PredictionService',
    'EnsembleModel', 
    'create_ensemble_model',
    'FeatureEngineering',
    'ModelTrainingPipeline',
    'AutoRetrainingSystem'
]