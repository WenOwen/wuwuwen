#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM 股票预测脚本
使用训练好的模型进行股票预测
"""

import os
import yaml
import numpy as np
import pandas as pd
import json
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import argparse

import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class LightGBMPredictor:
    """LightGBM 预测器"""
    
    def __init__(self, config_path: str, model_path: str, 
                 scaler_path: Optional[str] = None, 
                 feature_names_path: Optional[str] = None):
        """
        初始化预测器
        
        Args:
            config_path: 配置文件路径
            model_path: 模型文件路径
            scaler_path: 预处理器文件路径（可选）
            feature_names_path: 特征名称文件路径（可选）
        """
        self.config_path = config_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_names_path = feature_names_path
        
        # 加载配置和模型
        self.config = self._load_config()
        self.setup_logging()
        self.load_model()
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """设置日志"""
        log_config = self.config.get('output', {}).get('logging', {})
        log_level = getattr(logging, log_config.get('log_level', 'INFO'))
        
        # 配置日志
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """加载模型和预处理器"""
        self.logger.info("加载模型...")
        
        # 加载模型
        if self.model_path.endswith('.pkl'):
            self.model = joblib.load(self.model_path)
        elif self.model_path.endswith('.txt'):
            self.model = lgb.Booster(model_file=self.model_path)
        else:
            raise ValueError(f"不支持的模型文件格式: {self.model_path}")
        
        # 加载预处理器
        self.scaler = None
        if self.scaler_path and os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            self.logger.info("预处理器加载完成")
        
        # 加载特征名称
        self.feature_names = None
        if self.feature_names_path and os.path.exists(self.feature_names_path):
            with open(self.feature_names_path, 'r', encoding='utf-8') as f:
                self.feature_names = json.load(f)
            self.logger.info(f"特征名称加载完成，共 {len(self.feature_names)} 个特征")
        
        self.logger.info("模型加载完成")
    
    def load_prediction_data(self, data_path: str) -> Tuple[np.ndarray, Optional[List]]:
        """
        加载预测数据（支持NPY和CSV格式）
        
        Args:
            data_path: 数据路径，可以是文件夹或文件
            
        Returns:
            X: 特征数据
            stock_codes: 股票代码（如果有的话）
        """
        self.logger.info(f"加载预测数据: {data_path}")
        
        if os.path.isdir(data_path):
            # 数据文件夹 - 优先使用CSV格式
            X_csv_path = os.path.join(data_path, "X_features.csv")
            X_npy_path = os.path.join(data_path, "X_features.npy")
            stock_codes_path = os.path.join(data_path, "stock_codes.json")
            data_info_path = os.path.join(data_path, "data_info.json")
            
            # 加载数据信息
            if os.path.exists(data_info_path):
                with open(data_info_path, 'r', encoding='utf-8') as f:
                    self.data_info = json.load(f)
            
            if os.path.exists(X_csv_path):
                # 加载CSV格式数据
                self.logger.info("检测到CSV格式数据，正在加载...")
                df = pd.read_csv(X_csv_path)
                
                if 'stock_code' in df.columns:
                    stock_codes = df['stock_code'].tolist()
                    X = df.drop('stock_code', axis=1).values
                else:
                    X = df.values
                    stock_codes = None
                    
                # 如果没有股票代码，尝试从JSON文件加载
                if stock_codes is None and os.path.exists(stock_codes_path):
                    with open(stock_codes_path, 'r', encoding='utf-8') as f:
                        stock_codes = json.load(f)
                        
            elif os.path.exists(X_npy_path):
                # 加载NPY格式数据（兼容旧版本）
                self.logger.info("检测到NPY格式数据，正在加载...")
                X = np.load(X_npy_path)
                
                stock_codes = None
                if os.path.exists(stock_codes_path):
                    with open(stock_codes_path, 'r', encoding='utf-8') as f:
                        stock_codes = json.load(f)
            else:
                raise FileNotFoundError(f"未找到特征数据文件: {data_path}")
                    
        elif data_path.endswith('.npy'):
            # 单个numpy文件
            X = np.load(data_path)
            stock_codes = None
            
        elif data_path.endswith('.csv'):
            # CSV文件
            df = pd.read_csv(data_path)
            
            # 检查是否包含股票代码
            if 'stock_code' in df.columns:
                stock_codes = df['stock_code'].tolist()
                df = df.drop('stock_code', axis=1)
            else:
                stock_codes = None
                
            # 检查是否包含目标变量
            if 'target' in df.columns:
                X = df.drop('target', axis=1).values
            elif 'y' in df.columns:
                X = df.drop('y', axis=1).values
            else:
                X = df.values
            
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
        
        self.logger.info(f"数据加载完成: {X.shape}")
        return X, stock_codes
    
    def preprocess_timeseries_data(self, X: np.ndarray) -> np.ndarray:
        """
        预处理时序数据（与训练时保持一致）
        
        Args:
            X: 原始数据，可能是时序数据 [samples, timesteps, features] 或已展平的数据 [samples, features]
            
        Returns:
            处理后的2D数据 [samples, features]
        """
        self.logger.info("预处理时序数据...")
        
        # 如果数据已经是2D（从CSV加载），则检查是否需要重新构造时序结构
        if len(X.shape) == 2:
            self.logger.info("检测到2D数据格式（可能来自CSV），检查是否需要时序处理...")
            
            # 如果配置要求时序处理且数据信息中有原始shape信息
            if hasattr(self, 'data_info') and 'original_X_shape' in self.data_info:
                original_shape = self.data_info['original_X_shape']
                if len(original_shape) == 3:
                    # 尝试重新构造3D格式
                    samples, timesteps, features = original_shape
                    if X.shape[0] <= samples and X.shape[1] == timesteps * features:
                        self.logger.info(f"重新构造时序数据: {X.shape} -> ({X.shape[0]}, {timesteps}, {features})")
                        X = X.reshape(X.shape[0], timesteps, features)
                    else:
                        self.logger.warning("无法重新构造时序结构，将使用2D数据直接处理")
                        return X
                else:
                    # 原本就是2D数据
                    return X
            else:
                # 没有原始shape信息，直接使用2D数据
                return X
        
        # 如果是3D数据，按配置处理
        if len(X.shape) == 3:
            method = self.config['data']['preprocessing']['timeseries_method']
            
            if method == 'flatten':
                # 直接展平
                X_processed = X.reshape(X.shape[0], -1)
                
            elif method == 'statistical':
                # 提取统计特征
                stat_features = self.config['data']['preprocessing']['feature_engineering']['statistical_features']
                X_processed = self._extract_statistical_features(X, stat_features)
                
            elif method == 'last_step':
                # 只使用最后一个时间步
                X_processed = X[:, -1, :]
                
            else:
                raise ValueError(f"不支持的时序处理方法: {method}")
                
            self.logger.info(f"时序数据预处理完成: {X.shape} -> {X_processed.shape}")
            return X_processed
        else:
            # 如果不是2D也不是3D，报错
            raise ValueError(f"不支持的数据维度: {X.shape}")
        
        return X
    
    def _extract_statistical_features(self, X: np.ndarray, stat_features: List[str]) -> np.ndarray:
        """提取统计特征"""
        features_list = []
        
        for i in range(X.shape[2]):
            feature_data = X[:, :, i]  # [samples, timesteps]
            
            for stat in stat_features:
                if stat == 'mean':
                    features_list.append(np.mean(feature_data, axis=1))
                elif stat == 'std':
                    features_list.append(np.std(feature_data, axis=1))
                elif stat == 'max':
                    features_list.append(np.max(feature_data, axis=1))
                elif stat == 'min':
                    features_list.append(np.min(feature_data, axis=1))
                elif stat == 'median':
                    features_list.append(np.median(feature_data, axis=1))
                elif stat == 'skew':
                    features_list.append(stats.skew(feature_data, axis=1))
                elif stat == 'kurtosis':
                    features_list.append(stats.kurtosis(feature_data, axis=1))
        
        return np.column_stack(features_list)
    
    def add_technical_features(self, X: np.ndarray, raw_X: np.ndarray) -> np.ndarray:
        """添加技术指标特征（与训练时保持一致）"""
        if not self.config['data']['preprocessing']['feature_engineering']['technical_features']['enabled']:
            return X
        
        # 检查raw_X是否为3D数据
        if len(raw_X.shape) != 3:
            self.logger.warning("技术指标特征需要3D时序数据，当前数据已被展平，跳过技术特征计算")
            return X
            
        self.logger.info("添加技术指标特征...")
        
        tech_features = []
        
        try:
            # 计算动量特征
            momentum_features = self._calculate_momentum_features(raw_X)
            tech_features.extend(momentum_features)
            
            # 计算波动率特征
            volatility_features = self._calculate_volatility_features(raw_X)
            tech_features.extend(volatility_features)
            
            # 计算趋势特征
            trend_features = self._calculate_trend_features(raw_X)
            tech_features.extend(trend_features)
            
            # 合并特征
            if tech_features:
                tech_array = np.column_stack(tech_features)
                X_enhanced = np.column_stack([X, tech_array])
                self.logger.info(f"技术指标特征添加完成: {X.shape} -> {X_enhanced.shape}")
                return X_enhanced
        except Exception as e:
            self.logger.warning(f"技术指标特征计算失败: {e}，跳过技术特征")
        
        return X
    
    def _calculate_momentum_features(self, X: np.ndarray) -> List[np.ndarray]:
        """计算动量特征"""
        features = []
        
        # 简单动量 (最后值 - 第一值)
        momentum = X[:, -1, :] - X[:, 0, :]
        for i in range(momentum.shape[1]):
            features.append(momentum[:, i])
            
        return features
    
    def _calculate_volatility_features(self, X: np.ndarray) -> List[np.ndarray]:
        """计算波动率特征"""
        features = []
        
        # 标准差作为波动率
        volatility = np.std(X, axis=1)
        for i in range(volatility.shape[1]):
            features.append(volatility[:, i])
            
        return features
    
    def _calculate_trend_features(self, X: np.ndarray) -> List[np.ndarray]:
        """计算趋势特征"""
        features = []
        
        # 线性趋势斜率
        time_steps = np.arange(X.shape[1])
        for feat_idx in range(X.shape[2]):
            slopes = []
            for sample_idx in range(X.shape[0]):
                slope, _, _, _, _ = stats.linregress(time_steps, X[sample_idx, :, feat_idx])
                slopes.append(slope)
            features.append(np.array(slopes))
            
        return features
    
    def normalize_data(self, X: np.ndarray) -> np.ndarray:
        """数据标准化"""
        if self.scaler is None:
            self.logger.warning("未找到预处理器，跳过数据标准化")
            return X
        
        self.logger.info("应用数据标准化...")
        return self.scaler.transform(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """进行预测"""
        self.logger.info(f"开始预测，样本数量: {X.shape[0]}")
        
        # 检查特征数量
        if hasattr(self.model, 'n_features_'):
            expected_features = self.model.n_features_
            if X.shape[1] != expected_features:
                self.logger.warning(f"特征数量不匹配: 期望 {expected_features}, 实际 {X.shape[1]}")
        
        # 进行预测
        if hasattr(self.model, 'predict'):
            # sklearn风格的模型
            predictions = self.model.predict(X)
        else:
            # LightGBM Booster对象
            predictions = self.model.predict(X)
        
        self.logger.info("预测完成")
        return predictions
    
    def save_predictions(self, predictions: np.ndarray, stock_codes: Optional[List] = None, 
                        output_path: str = None) -> str:
        """
        保存预测结果
        
        Args:
            predictions: 预测结果
            stock_codes: 股票代码列表
            output_path: 输出路径
            
        Returns:
            保存的文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"predictions_{timestamp}.csv"
        
        # 创建结果DataFrame
        if stock_codes is not None:
            # 假设stock_codes包含了每个样本对应的股票代码
            if len(stock_codes) == len(predictions):
                df = pd.DataFrame({
                    'stock_code': stock_codes,
                    'prediction': predictions
                })
            else:
                # 如果stock_codes是唯一股票列表，需要其他逻辑来映射
                df = pd.DataFrame({
                    'sample_id': range(len(predictions)),
                    'prediction': predictions
                })
        else:
            df = pd.DataFrame({
                'sample_id': range(len(predictions)),
                'prediction': predictions
            })
        
        # 添加预测时间
        df['prediction_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存结果
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        self.logger.info(f"预测结果已保存到: {output_path}")
        return output_path
    
    def run_prediction_pipeline(self, data_path: str, output_path: str = None) -> Tuple[np.ndarray, str]:
        """
        运行完整的预测流程
        
        Args:
            data_path: 输入数据路径
            output_path: 输出路径
            
        Returns:
            predictions: 预测结果
            output_file: 输出文件路径
        """
        try:
            self.logger.info("开始LightGBM预测流程...")
            
            # 1. 加载数据
            X_raw, stock_codes = self.load_prediction_data(data_path)
            
            # 2. 预处理时序数据
            X = self.preprocess_timeseries_data(X_raw)
            
            # 3. 添加技术特征
            X = self.add_technical_features(X, X_raw)
            
            # 4. 数据标准化
            X = self.normalize_data(X)
            
            # 5. 进行预测
            predictions = self.predict(X)
            
            # 6. 保存结果
            output_file = self.save_predictions(predictions, stock_codes, output_path)
            
            self.logger.info("预测流程完成！")
            
            return predictions, output_file
            
        except Exception as e:
            self.logger.error(f"预测过程中发生错误: {str(e)}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LightGBM 股票预测')
    parser.add_argument('--config', type=str, default='config/train/lightGBM_train.yaml',
                      help='配置文件路径')
    parser.add_argument('--model', type=str, required=True,
                      help='模型文件路径')
    parser.add_argument('--data', type=str, required=True,
                      help='预测数据路径')
    parser.add_argument('--output', type=str, default=None,
                      help='输出文件路径')
    parser.add_argument('--scaler', type=str, default=None,
                      help='预处理器文件路径')
    parser.add_argument('--features', type=str, default=None,
                      help='特征名称文件路径')
    
    args = parser.parse_args()
    
    # 创建预测器并运行
    predictor = LightGBMPredictor(
        config_path=args.config,
        model_path=args.model,
        scaler_path=args.scaler,
        feature_names_path=args.features
    )
    
    predictions, output_file = predictor.run_prediction_pipeline(
        data_path=args.data,
        output_path=args.output
    )
    
    print(f"预测完成！结果保存在: {output_file}")
    print(f"预测样本数量: {len(predictions)}")
    print(f"预测结果统计:")
    print(f"  均值: {np.mean(predictions):.4f}")
    print(f"  标准差: {np.std(predictions):.4f}")
    print(f"  最小值: {np.min(predictions):.4f}")
    print(f"  最大值: {np.max(predictions):.4f}")


if __name__ == '__main__':
    main()