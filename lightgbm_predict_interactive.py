#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM 股票预测脚本 - 交互式版本
支持自动扫描模型和数据，快捷键选择
"""

import os
import glob
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


class InteractiveLightGBMPredictor:
    """交互式 LightGBM 预测器"""
    
    def __init__(self):
        """初始化预测器"""
        self.models_dir = "models"
        self.data_dir = "data/datas_predict"
        self.config_path = "config/train/lightGBM_train.yaml"
        
        # 扫描可用的模型和数据
        self.available_models = self._scan_models()
        self.available_data = self._scan_prediction_data()
        
        # 设置日志
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
    
    def _scan_models(self) -> List[Dict]:
        """扫描可用的模型"""
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
            
        # 扫描所有模型类型文件夹
        model_types = ['lightgbm', 'lightgbm_extended', 'lightgbm_fine_tuned', 'lightgbm_optimized']
        
        for model_type in model_types:
            model_type_dir = os.path.join(self.models_dir, model_type)
            if not os.path.exists(model_type_dir):
                continue
                
            # 扫描训练文件夹
            training_dirs = glob.glob(os.path.join(model_type_dir, "training*"))
            training_dirs.sort(reverse=True)  # 最新的在前面
            
            for training_dir in training_dirs:
                # 查找模型文件
                pkl_files = glob.glob(os.path.join(training_dir, "*model*.pkl"))
                txt_files = glob.glob(os.path.join(training_dir, "*model*.txt"))
                
                for model_file in pkl_files + txt_files:
                    # 查找配套文件
                    scaler_file = None
                    feature_names_file = None
                    
                    scaler_path = os.path.join(training_dir, "scaler.pkl")
                    if os.path.exists(scaler_path):
                        scaler_file = scaler_path
                        
                    feature_names_path = os.path.join(training_dir, "feature_names.json")
                    if os.path.exists(feature_names_path):
                        feature_names_file = feature_names_path
                    
                    # 获取模型信息
                    model_name = f"{model_type}_{os.path.basename(training_dir)}"
                    
                    models.append({
                        'name': model_name,
                        'type': model_type,
                        'training_dir': os.path.basename(training_dir),
                        'model_file': model_file,
                        'scaler_file': scaler_file,
                        'feature_names_file': feature_names_file,
                        'full_path': training_dir
                    })
        
        return models
    
    def _scan_prediction_data(self) -> List[Dict]:
        """扫描可用的预测数据"""
        data_list = []
        
        if not os.path.exists(self.data_dir):
            return data_list
            
        # 扫描所有处理过的数据文件夹
        processed_dirs = glob.glob(os.path.join(self.data_dir, "processed_*"))
        processed_dirs.sort(reverse=True)  # 最新的在前面
        
        for data_dir in processed_dirs:
            # 查找特征文件
            x_features_file = os.path.join(data_dir, "X_features.csv")
            full_data_file = os.path.join(data_dir, "full_data.csv")
            
            if os.path.exists(x_features_file):
                # 查找配套文件
                stock_codes_file = None
                data_info_file = None
                feature_names_file = None
                
                stock_codes_path = os.path.join(data_dir, "stock_codes.json")
                if os.path.exists(stock_codes_path):
                    stock_codes_file = stock_codes_path
                    
                data_info_path = os.path.join(data_dir, "data_info.json")
                if os.path.exists(data_info_path):
                    data_info_file = data_info_path
                    
                feature_names_path = os.path.join(data_dir, "feature_names.json")
                if os.path.exists(feature_names_path):
                    feature_names_file = feature_names_path
                
                # 获取数据信息
                data_name = os.path.basename(data_dir)
                
                # 读取数据基本信息
                try:
                    df = pd.read_csv(x_features_file)
                    sample_count = len(df)
                    feature_count = len(df.columns)
                except:
                    sample_count = "未知"
                    feature_count = "未知"
                
                data_list.append({
                    'name': data_name,
                    'data_dir': data_dir,
                    'x_features_file': x_features_file,
                    'full_data_file': full_data_file if os.path.exists(full_data_file) else None,
                    'stock_codes_file': stock_codes_file,
                    'data_info_file': data_info_file,
                    'feature_names_file': feature_names_file,
                    'sample_count': sample_count,
                    'feature_count': feature_count
                })
        
        return data_list
    
    def display_models(self):
        """显示可用的模型"""
        print("\n" + "="*80)
        print("🤖 可用的 LightGBM 模型:")
        print("="*80)
        
        if not self.available_models:
            print("❌ 未找到可用的模型文件")
            return
            
        for i, model in enumerate(self.available_models[:9], 1):  # 最多显示9个
            print(f"[{i}] {model['name']}")
            print(f"    📁 类型: {model['type']}")
            print(f"    📂 训练: {model['training_dir']}")
            print(f"    📄 模型: {os.path.basename(model['model_file'])}")
            print(f"    🔧 预处理器: {'✅' if model['scaler_file'] else '❌'}")
            print(f"    📋 特征名称: {'✅' if model['feature_names_file'] else '❌'}")
            print()
    
    def display_data(self):
        """显示可用的预测数据"""
        print("\n" + "="*80)
        print("📊 可用的预测数据:")
        print("="*80)
        
        if not self.available_data:
            print("❌ 未找到可用的预测数据文件")
            return
            
        for i, data in enumerate(self.available_data[:9], 1):  # 最多显示9个
            print(f"[{i}] {data['name']}")
            print(f"    📈 样本数量: {data['sample_count']}")
            print(f"    🔢 特征数量: {data['feature_count']}")
            print(f"    📄 特征文件: X_features.csv")
            print(f"    🏷️  股票代码: {'✅' if data['stock_codes_file'] else '❌'}")
            print(f"    ℹ️  数据信息: {'✅' if data['data_info_file'] else '❌'}")
            print()
    
    def get_user_choice(self, prompt: str, max_choice: int) -> int:
        """获取用户选择"""
        while True:
            try:
                choice = input(f"\n{prompt} (1-{max_choice}, 或 'q' 退出): ").strip()
                
                if choice.lower() == 'q':
                    print("👋 退出程序")
                    exit(0)
                
                choice_num = int(choice)
                if 1 <= choice_num <= max_choice:
                    return choice_num - 1  # 转换为索引
                else:
                    print(f"❌ 请输入 1-{max_choice} 之间的数字")
                    
            except ValueError:
                print("❌ 请输入有效的数字")
    
    def load_model_and_components(self, model_info: Dict):
        """加载模型和相关组件"""
        self.logger.info(f"加载模型: {model_info['name']}")
        
        # 加载配置文件
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            # 默认配置
            self.config = {
                'data': {
                    'preprocessing': {
                        'timeseries_method': 'flatten',
                        'feature_engineering': {
                            'technical_features': {'enabled': False},
                            'statistical_features': ['mean', 'std', 'max', 'min']
                        }
                    }
                }
            }
        
        # 加载模型
        model_file = model_info['model_file']
        if model_file.endswith('.pkl'):
            self.model = joblib.load(model_file)
        elif model_file.endswith('.txt'):
            self.model = lgb.Booster(model_file=model_file)
        else:
            raise ValueError(f"不支持的模型文件格式: {model_file}")
        
        # 加载预处理器
        self.scaler = None
        if model_info['scaler_file'] and os.path.exists(model_info['scaler_file']):
            self.scaler = joblib.load(model_info['scaler_file'])
            self.logger.info("✅ 预处理器加载完成")
        
        # 加载特征名称
        self.feature_names = None
        if model_info['feature_names_file'] and os.path.exists(model_info['feature_names_file']):
            with open(model_info['feature_names_file'], 'r', encoding='utf-8') as f:
                self.feature_names = json.load(f)
            self.logger.info(f"✅ 特征名称加载完成，共 {len(self.feature_names)} 个特征")
        
        self.logger.info("✅ 模型加载完成")
    
    def load_prediction_data(self, data_info: Dict) -> Tuple[np.ndarray, Optional[List], List[str], Optional[np.ndarray]]:
        """加载预测数据（包括真实目标值）"""
        self.logger.info(f"加载预测数据: {data_info['name']}")
        
        # 加载数据信息
        if data_info['data_info_file']:
            with open(data_info['data_info_file'], 'r', encoding='utf-8') as f:
                self.data_info = json.load(f)
        
        # 加载特征数据
        df = pd.read_csv(data_info['x_features_file'])
        
        # 获取特征名称
        feature_names = df.columns.tolist()
        
        # 检查是否包含股票代码
        if 'stock_code' in df.columns:
            stock_codes = df['stock_code'].tolist()
            X = df.drop('stock_code', axis=1).values
            feature_names.remove('stock_code')
        else:
            X = df.values
            stock_codes = None
            
        # 如果没有股票代码，尝试从JSON文件加载
        if stock_codes is None and data_info['stock_codes_file']:
            with open(data_info['stock_codes_file'], 'r', encoding='utf-8') as f:
                stock_codes = json.load(f)
        
        # 尝试加载真实目标值
        y_true = None
        y_targets_file = os.path.join(data_info['data_dir'], 'y_targets.csv')
        if os.path.exists(y_targets_file):
            try:
                y_df = pd.read_csv(y_targets_file)
                if 'target' in y_df.columns:
                    y_true = y_df['target'].values
                elif 'y' in y_df.columns:
                    y_true = y_df['y'].values
                elif len(y_df.columns) == 1:
                    y_true = y_df.iloc[:, 0].values
                else:
                    # 如果有多列，尝试选择数值列
                    numeric_cols = y_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        y_true = y_df[numeric_cols[0]].values
                
                if y_true is not None and len(y_true) == X.shape[0]:
                    self.logger.info(f"✅ 真实目标值加载完成: {y_true.shape}")
                else:
                    self.logger.warning(f"目标值数量不匹配: y={len(y_true) if y_true is not None else 0}, X={X.shape[0]}")
                    y_true = None
            except Exception as e:
                self.logger.warning(f"加载目标值失败: {e}")
                y_true = None
        else:
            self.logger.info("未找到y_targets.csv文件，无法进行预测效果对比")
        
        # 尝试从特征名称文件加载更准确的特征名称
        if data_info['feature_names_file']:
            try:
                with open(data_info['feature_names_file'], 'r', encoding='utf-8') as f:
                    loaded_feature_names = json.load(f)
                if isinstance(loaded_feature_names, list) and len(loaded_feature_names) == len(feature_names):
                    feature_names = loaded_feature_names
                    self.logger.info("使用特征名称文件中的特征名称")
            except Exception as e:
                self.logger.warning(f"加载特征名称文件失败: {e}")
        
        self.logger.info(f"✅ 数据加载完成: {X.shape}, 特征数量: {len(feature_names)}")
        return X, stock_codes, feature_names, y_true
    
    def preprocess_data(self, X: np.ndarray, data_feature_names: List[str]) -> np.ndarray:
        """预处理数据（按照训练时的顺序）"""
        self.logger.info("🔄 预处理数据...")
        
        # 第一步：特征对齐和统计特征提取（生成完整的特征向量）
        if self.feature_names is not None:
            X = self._align_and_expand_features(X, data_feature_names)
        
        # 第二步：数据标准化（在完整特征向量上）
        if self.scaler is not None:
            self.logger.info("📊 应用数据标准化...")
            X = self.scaler.transform(X)
        
        # 第三步：特征选择（模拟训练时的特征选择）
        if self.feature_names is not None and 'feature_names' in self.feature_names:
            X = self._select_final_features(X)
        
        return X
    
    def _align_and_expand_features(self, X: np.ndarray, data_feature_names: List[str]) -> np.ndarray:
        """第一步：特征对齐和扩展，直接生成模型期望的特征数量"""
        self.logger.info("🔧 特征对齐和扩展...")
        
        try:
            # 获取模型期望的特征数量
            expected_features = self.scaler.n_features_in_ if self.scaler is not None else 36
            final_feature_names = self.feature_names.get('feature_names', [])
            
            self.logger.info(f"预测数据特征数量: {len(data_feature_names)}")
            self.logger.info(f"期望特征数量: {expected_features}")
            self.logger.info(f"最终模型特征数量: {len(final_feature_names)}")
            
            # 创建特征映射字典（移除step_00_前缀进行匹配）
            data_feature_dict = {}
            for i, name in enumerate(data_feature_names):
                clean_name = name.replace('step_00_', '')
                data_feature_dict[clean_name] = i
                data_feature_dict[name] = i  # 也保留原名
            
            # 选择最终模型需要的特征
            selected_features = []
            used_indices = []
            
            for final_name in final_feature_names:
                clean_final = final_name.replace('step_00_', '')
                found_idx = None
                
                # 精确匹配
                if final_name in data_feature_dict:
                    found_idx = data_feature_dict[final_name]
                elif clean_final in data_feature_dict:
                    found_idx = data_feature_dict[clean_final]
                else:
                    # 模糊匹配
                    for data_name, idx in data_feature_dict.items():
                        if clean_final in data_name or data_name in clean_final:
                            found_idx = idx
                            break
                
                if found_idx is not None and found_idx not in used_indices:
                    selected_features.append(X[:, found_idx])
                    used_indices.append(found_idx)
                    self.logger.debug(f"匹配特征: {final_name} -> {data_feature_names[found_idx]}")
                else:
                    self.logger.warning(f"未找到特征 {final_name}，使用零填充")
                    selected_features.append(np.zeros(X.shape[0]))
            
            if len(selected_features) != len(final_feature_names):
                raise ValueError(f"特征数量不匹配: 期望{len(final_feature_names)}，实际{len(selected_features)}")
            
            # 构造特征矩阵
            X_selected = np.column_stack(selected_features)
            self.logger.info(f"特征选择完成: {X.shape} -> {X_selected.shape}")
            
            # 如果需要扩展到期望的特征数量（例如mean+std扩展）
            if expected_features > X_selected.shape[1]:
                expansion_factor = expected_features // X_selected.shape[1]
                if expansion_factor == 2:
                    # mean + std 扩展
                    self.logger.info("应用mean+std扩展")
                    features_list = []
                    for i in range(X_selected.shape[1]):
                        features_list.append(X_selected[:, i])  # mean
                        features_list.append(np.zeros(X_selected.shape[0]))  # std (零值)
                    X_expanded = np.column_stack(features_list)
                    self.logger.info(f"特征扩展完成: {X_selected.shape} -> {X_expanded.shape}")
                    return X_expanded
                else:
                    # 其他扩展方式
                    repeat_data = np.tile(X_selected, (1, expansion_factor))
                    remaining = expected_features - repeat_data.shape[1]
                    if remaining > 0:
                        extra_data = X_selected[:, :remaining]
                        X_expanded = np.column_stack([repeat_data, extra_data])
                    else:
                        X_expanded = repeat_data[:, :expected_features]
                    self.logger.info(f"特征重复扩展完成: {X_selected.shape} -> {X_expanded.shape}")
                    return X_expanded
            
            return X_selected
                
        except Exception as e:
            self.logger.error(f"特征对齐和扩展失败: {e}")
            self.logger.info("使用原始特征...")
            return X
    
    def _expand_features_with_stats(self, X: np.ndarray, stat_features: List[str], base_names: List[str]) -> np.ndarray:
        """使用统计特征扩展单时间步数据（简化版本：只使用mean+std）"""
        # 根据训练时的逻辑，似乎只使用了mean和std，而且可能是先选择特征再扩展
        final_feature_names = self.feature_names.get('feature_names', [])
        
        # 如果StandardScaler期望36个特征，而最终特征是18个，说明是18*2=36
        if hasattr(self, 'scaler') and self.scaler is not None:
            expected_features = self.scaler.n_features_in_
            if expected_features == len(final_feature_names) * 2:
                self.logger.info(f"使用简化统计特征扩展: mean+std，目标特征数: {expected_features}")
                
                # 先选择最终需要的特征
                target_features = []
                data_feature_dict = {name.replace('step_00_', ''): i for i, name in enumerate(base_names)}
                
                for final_name in final_feature_names:
                    clean_name = final_name.replace('step_00_', '')
                    # 尝试多种匹配方式
                    found_idx = None
                    if clean_name in data_feature_dict:
                        found_idx = data_feature_dict[clean_name]
                    else:
                        # 模糊匹配
                        for data_name, idx in data_feature_dict.items():
                            if clean_name in data_name or data_name in clean_name:
                                found_idx = idx
                                break
                    
                    if found_idx is not None and found_idx < X.shape[1]:
                        target_features.append(X[:, found_idx])
                    else:
                        self.logger.warning(f"未找到特征 {final_name}，使用零填充")
                        target_features.append(np.zeros(X.shape[0]))
                
                if len(target_features) == len(final_feature_names):
                    # 构造mean+std特征
                    features_list = []
                    for feature_data in target_features:
                        features_list.append(feature_data)  # mean
                        features_list.append(np.zeros_like(feature_data))  # std (零值)
                    
                    return np.column_stack(features_list)
        
        # 备用方案：使用原有逻辑但只用mean+std
        features_list = []
        for i, base_name in enumerate(base_names):
            if i >= X.shape[1]:
                break
                
            feature_data = X[:, i]
            features_list.append(feature_data)  # mean
            features_list.append(np.zeros_like(feature_data))  # std
        
        return np.column_stack(features_list)
    
    def _select_final_features(self, X: np.ndarray) -> np.ndarray:
        """第三步：特征选择，选择最终用于模型的特征"""
        try:
            final_feature_names = self.feature_names.get('feature_names', [])
            final_feature_count = len(final_feature_names)
            
            self.logger.info(f"🎯 选择最终特征: {X.shape[1]} -> {final_feature_count}")
            
            # 简单选择前N个特征（假设特征选择保留了前面的特征）
            if X.shape[1] >= final_feature_count:
                X_final = X[:, :final_feature_count]
                self.logger.info(f"最终特征选择完成: {X.shape} -> {X_final.shape}")
                return X_final
            else:
                self.logger.warning(f"可用特征数量({X.shape[1]})少于需要的特征数量({final_feature_count})")
                return X
                
        except Exception as e:
            self.logger.error(f"最终特征选择失败: {e}")
            return X

    def _align_features(self, X: np.ndarray, data_feature_names: List[str]) -> np.ndarray:
        """根据训练时的特征映射对齐特征"""
        self.logger.info("🔧 对齐特征...")
        
        try:
            # 获取训练时的特征信息
            training_features = self.feature_names.get('feature_names', [])
            feature_mapping = self.feature_names.get('feature_mapping', {})
            timeseries_method = self.feature_names.get('timeseries_method', 'flatten')
            
            self.logger.info(f"训练特征数量: {len(training_features)}")
            self.logger.info(f"预测数据特征数量: {len(data_feature_names)}")
            self.logger.info(f"时序处理方法: {timeseries_method}")
            
            # 创建特征映射字典
            data_feature_dict = {name: i for i, name in enumerate(data_feature_names)}
            
            # 选择训练时使用的特征
            selected_features = []
            feature_indices = []
            
            for feature_name in training_features:
                if feature_name in data_feature_dict:
                    selected_features.append(feature_name)
                    feature_indices.append(data_feature_dict[feature_name])
                else:
                    self.logger.warning(f"预测数据中缺少特征: {feature_name}")
            
            if not feature_indices:
                raise ValueError("没有找到匹配的特征")
            
            # 选择对应的特征列
            X_selected = X[:, feature_indices]
            self.logger.info(f"特征选择完成: {X.shape} -> {X_selected.shape}")
            
            # 检查是否需要进行统计特征提取
            # 对于单时间步数据（如预测数据），不进行统计特征提取
            if timeseries_method == 'statistical':
                # 检查模型期望的特征数量
                if hasattr(self, 'scaler') and self.scaler is not None:
                    expected_features = self.scaler.n_features_in_
                    self.logger.info(f"模型期望特征数量: {expected_features}")
                    
                    # 如果选择的特征数量已经匹配，直接返回
                    if X_selected.shape[1] == expected_features:
                        self.logger.info("特征数量已匹配，跳过统计特征提取")
                        return X_selected
                    
                    # 如果需要扩展特征，进行统计特征提取
                    elif X_selected.shape[1] < expected_features:
                        X_processed = self._apply_statistical_features_for_single_step(X_selected, expected_features)
                        self.logger.info(f"统计特征提取完成: {X_selected.shape} -> {X_processed.shape}")
                        return X_processed
                
                # 默认情况，不进行统计特征提取
                self.logger.info("单时间步数据，跳过统计特征提取")
                return X_selected
            
            return X_selected
            
        except Exception as e:
            self.logger.error(f"特征对齐失败: {e}")
            self.logger.info("使用原始特征...")
            return X
    
    def _apply_statistical_features_for_single_step(self, X: np.ndarray, expected_features: int) -> np.ndarray:
        """为单时间步数据应用统计特征扩展"""
        current_features = X.shape[1]
        
        if expected_features == current_features * 2:
            # 最常见情况：每个特征扩展为2个（mean + std）
            self.logger.info("应用mean+std扩展")
            features_list = []
            
            for i in range(current_features):
                feature_data = X[:, i]
                # Mean: 原始值
                features_list.append(feature_data)
                # Std: 零值（单时间步无变化）
                features_list.append(np.zeros_like(feature_data))
            
            return np.column_stack(features_list)
            
        elif expected_features == current_features:
            # 特征数量已匹配
            return X
            
        else:
            # 其他情况：尝试简单复制或截断
            if expected_features > current_features:
                # 需要扩展特征
                repeat_times = expected_features // current_features
                remaining = expected_features % current_features
                
                features_list = []
                for _ in range(repeat_times):
                    features_list.append(X)
                
                if remaining > 0:
                    features_list.append(X[:, :remaining])
                
                return np.column_stack(features_list)
            else:
                # 需要截断特征
                return X[:, :expected_features]

    def _apply_statistical_features(self, X: np.ndarray) -> np.ndarray:
        """应用统计特征提取（模拟时序统计处理）"""
        # 获取统计特征配置
        stat_features = ['mean', 'std']  # 默认统计特征
        if hasattr(self, 'config') and 'data' in self.config:
            stat_config = self.config['data']['preprocessing']['feature_engineering'].get('statistical_features', ['mean', 'std'])
            if stat_config:
                stat_features = stat_config
        
        # 为每个特征计算统计量
        features_list = []
        
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            
            for stat in stat_features:
                if stat == 'mean':
                    # 对于单时间步数据，均值就是自身
                    features_list.append(feature_data)
                elif stat == 'std':
                    # 对于单时间步数据，标准差设为0（或小值）
                    features_list.append(np.zeros_like(feature_data))
                elif stat == 'max':
                    features_list.append(feature_data)
                elif stat == 'min':
                    features_list.append(feature_data)
                elif stat == 'median':
                    features_list.append(feature_data)
        
        if features_list:
            return np.column_stack(features_list)
        else:
            return X
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """进行预测"""
        self.logger.info(f"🎯 开始预测，样本数量: {X.shape[0]}")
        
        # 检查特征数量
        if hasattr(self.model, 'n_features_'):
            expected_features = self.model.n_features_
            if X.shape[1] != expected_features:
                self.logger.warning(f"⚠️  特征数量不匹配: 期望 {expected_features}, 实际 {X.shape[1]}")
        
        # 进行预测
        if hasattr(self.model, 'predict'):
            # sklearn风格的模型
            predictions = self.model.predict(X)
        else:
            # LightGBM Booster对象
            predictions = self.model.predict(X)
        
        self.logger.info("✅ 预测完成")
        return predictions
    
    def save_predictions(self, predictions: np.ndarray, stock_codes: Optional[List] = None, 
                        model_name: str = "", data_name: str = "", y_true: Optional[np.ndarray] = None) -> str:
        """
        保存预测结果（包含真实值对比）
        
        Args:
            predictions: 预测结果
            stock_codes: 股票代码列表
            model_name: 模型名称
            data_name: 数据名称  
            y_true: 真实目标值
            
        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "results/predictions"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"predictions_{model_name}_{data_name}_{timestamp}.csv")
        
        # 创建结果DataFrame
        if stock_codes is not None and len(stock_codes) == len(predictions):
            df = pd.DataFrame({
                'stock_code': stock_codes,
                'prediction': predictions
            })
        else:
            df = pd.DataFrame({
                'sample_id': range(len(predictions)),
                'prediction': predictions
            })
        
        # 添加真实值和差值对比
        if y_true is not None and len(y_true) == len(predictions):
            df['y_true'] = y_true
            df['difference'] = predictions - y_true
            df['abs_difference'] = np.abs(predictions - y_true)
            df['error_rate'] = np.abs(predictions - y_true) / (np.abs(y_true) + 1e-8) * 100  # 相对误差率(%)
            
            # 计算评估指标
            mae = np.mean(np.abs(predictions - y_true))
            mse = np.mean((predictions - y_true) ** 2)
            rmse = np.sqrt(mse)
            r2 = 1 - (np.sum((y_true - predictions) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            
            # 添加评估指标到每行（方便查看）
            df['mae'] = mae
            df['mse'] = mse
            df['rmse'] = rmse
            df['r2_score'] = r2
            
            self.logger.info(f"📊 预测评估指标:")
            self.logger.info(f"   MAE (平均绝对误差): {mae:.4f}")
            self.logger.info(f"   MSE (均方误差): {mse:.4f}")
            self.logger.info(f"   RMSE (均方根误差): {rmse:.4f}")
            self.logger.info(f"   R² (决定系数): {r2:.4f}")
        
        # 添加元信息
        df['model_name'] = model_name
        df['data_name'] = data_name
        df['prediction_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存结果
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        self.logger.info(f"💾 预测结果已保存到: {output_file}")
        return output_file
    
    def run_interactive_prediction(self):
        """运行交互式预测"""
        print("🚀 欢迎使用 LightGBM 交互式股票预测系统!")
        print("=" * 80)
        
        # 检查是否有可用的模型和数据
        if not self.available_models:
            print("❌ 未找到可用的模型文件，请检查 models 文件夹")
            return
        
        if not self.available_data:
            print("❌ 未找到可用的预测数据，请检查 data/datas_predict 文件夹")
            return
        
        while True:
            try:
                # 显示并选择模型
                self.display_models()
                model_idx = self.get_user_choice("请选择要使用的模型", len(self.available_models))
                selected_model = self.available_models[model_idx]
                
                # 显示并选择数据
                self.display_data()
                data_idx = self.get_user_choice("请选择要预测的数据", len(self.available_data))
                selected_data = self.available_data[data_idx]
                
                print(f"\n🎯 开始预测:")
                print(f"   📊 模型: {selected_model['name']}")
                print(f"   📈 数据: {selected_data['name']}")
                print("-" * 60)
                
                # 加载模型
                self.load_model_and_components(selected_model)
                
                # 加载数据
                X, stock_codes, data_feature_names, y_true = self.load_prediction_data(selected_data)
                
                # 预处理数据
                X = self.preprocess_data(X, data_feature_names)
                
                # 进行预测
                predictions = self.predict(X)
                
                # 保存结果
                output_file = self.save_predictions(
                    predictions, 
                    stock_codes, 
                    selected_model['name'],
                    selected_data['name'],
                    y_true
                )
                
                # 显示预测结果统计
                print(f"\n📈 预测结果统计:")
                print(f"   样本数量: {len(predictions)}")
                print(f"   预测均值: {np.mean(predictions):.4f}")
                print(f"   预测标准差: {np.std(predictions):.4f}")
                print(f"   最小值: {np.min(predictions):.4f}")
                print(f"   最大值: {np.max(predictions):.4f}")
                
                # 如果有真实值，显示对比统计
                if y_true is not None:
                    mae = np.mean(np.abs(predictions - y_true))
                    mse = np.mean((predictions - y_true) ** 2)
                    rmse = np.sqrt(mse)
                    r2 = 1 - (np.sum((y_true - predictions) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
                    
                    print(f"\n🎯 预测效果评估:")
                    print(f"   真实值均值: {np.mean(y_true):.4f}")
                    print(f"   真实值标准差: {np.std(y_true):.4f}")
                    print(f"   平均绝对误差 (MAE): {mae:.4f}")
                    print(f"   均方根误差 (RMSE): {rmse:.4f}")
                    print(f"   决定系数 (R²): {r2:.4f}")
                    print(f"   平均相对误差: {np.mean(np.abs(predictions - y_true) / (np.abs(y_true) + 1e-8) * 100):.2f}%")
                
                print(f"\n💾 输出文件: {output_file}")
                
                # 询问是否继续
                continue_choice = input("\n🔄 是否继续进行其他预测? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes', '是']:
                    print("👋 感谢使用，再见!")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，程序退出")
                break
            except Exception as e:
                print(f"\n❌ 预测过程中发生错误: {str(e)}")
                print("请检查模型和数据文件是否正确")
                
                retry_choice = input("🔄 是否重试? (y/n): ").strip().lower()
                if retry_choice not in ['y', 'yes', '是']:
                    break


def main():
    """主函数"""
    predictor = InteractiveLightGBMPredictor()
    predictor.run_interactive_prediction()


if __name__ == '__main__':
    main()