#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM 股票预测模型训练脚本
基于时序特征数据进行股票预测模型训练
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import json
import joblib
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# 抑制常见的警告信息，保持进度条清晰
warnings.filterwarnings('ignore', category=UserWarning, message='.*X does not have valid feature names.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.feature_selection import SelectFromModel, RFE
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 导入字体配置模块
from font_config import setup_chinese_plot
setup_chinese_plot()  # 设置中文字体

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("警告: Optuna 未安装，超参数优化功能将不可用")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("警告: tqdm 未安装，将使用简单的进度显示")


class SuppressOutput:
    """上下文管理器：完全抑制stdout和stderr输出"""
    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        return self
    
    def __exit__(self, *args):
        sys.stdout = self.stdout
        sys.stderr = self.stderr


class LightGBMTrainer:
    """LightGBM 训练器"""
    
    def __init__(self, config_path: str):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 创建当前训练的唯一文件夹
        self.training_folder = self._create_training_folder()
        
        self.setup_logging()
        
        # 初始化变量
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        self.model = None
        
        # 创建输出目录
        self._create_output_dirs()
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _create_training_folder(self) -> str:
        """创建当前训练的唯一文件夹"""
        file_naming_config = self.config['output'].get('file_naming', {})
        identifier_type = file_naming_config.get('identifier_type', 'unique_id')
        folder_name_prefix = file_naming_config.get('folder_name_prefix', 'training')
        
        # 根据配置生成文件夹标识符
        if identifier_type == 'timestamp':
            folder_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            digits = file_naming_config.get('unique_id_digits', 3)
            # 检查模型和结果目录，选择更大的ID
            model_dir = self.config['output']['model_save']['save_dir']
            results_dir = self.config['output']['results_save']['save_dir']
            
            # 获取现有训练文件夹的最大ID
            max_id = 0
            for base_dir in [model_dir, results_dir]:
                if os.path.exists(base_dir):
                    for folder in os.listdir(base_dir):
                        if folder.startswith(folder_name_prefix + '_'):
                            folder_id_part = folder.replace(folder_name_prefix + '_', '')
                            if folder_id_part.isdigit() and len(folder_id_part) == digits:
                                max_id = max(max_id, int(folder_id_part))
            
            folder_id = f"{max_id + 1:0{digits}d}"
        
        folder_name = f"{folder_name_prefix}_{folder_id}"
        return folder_name
    
    def setup_logging(self):
        """设置日志"""
        log_config = self.config.get('output', {}).get('logging', {})
        log_level = getattr(logging, log_config.get('log_level', 'INFO'))
        
        # 创建日志目录
        log_file = log_config.get('log_file', './logs/lightgbm_training.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 配置日志
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler() if log_config.get('console_output', True) else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _create_output_dirs(self):
        """创建输出目录"""
        # 创建基础目录
        base_dirs = [
            self.config['output']['model_save']['save_dir'],
            self.config['output']['results_save']['save_dir'],
            os.path.dirname(self.config['output']['logging']['log_file'])
        ]
        
        for dir_path in base_dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        # 创建当前训练的专用文件夹
        self.model_save_dir = os.path.join(self.config['output']['model_save']['save_dir'], self.training_folder)
        self.results_save_dir = os.path.join(self.config['output']['results_save']['save_dir'], self.training_folder)
        
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_save_dir, exist_ok=True)
        
        print(f"📁 创建训练文件夹: {self.training_folder}")
        print(f"   模型保存路径: {self.model_save_dir}")
        print(f"   结果保存路径: {self.results_save_dir}")
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载数据
        
        Returns:
            X: 特征数据
            y: 目标数据
        """
        self.logger.info("开始加载数据...")
        
        data_config = self.config['data']
        data_dir = data_config['data_dir']
        
        # 检查数据目录是否存在
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        # 获取加载选项
        loading_options = data_config.get('loading_options', {})
        prefer_full_data = loading_options.get('prefer_full_data', True)
        encoding = loading_options.get('encoding', 'utf-8')
        validate_data = loading_options.get('validate_data', True)
        
        # 优先使用完整数据文件，如果不存在则分别加载特征和目标文件
        full_data_file = data_config.get('full_data_file')
        if prefer_full_data and full_data_file and os.path.exists(os.path.join(data_dir, full_data_file)):
            self.logger.info("使用完整数据文件加载数据...")
            full_data_path = os.path.join(data_dir, full_data_file)
            
            # 添加数据加载进度提示
            if TQDM_AVAILABLE:
                self.logger.info("正在读取CSV文件...")
                with tqdm(desc="读取数据文件", unit="行", leave=False) as pbar:
                    df = pd.read_csv(full_data_path, encoding=encoding)
                    pbar.update(len(df))
            else:
                df = pd.read_csv(full_data_path, encoding=encoding)
            
            # 时序窗口数据验证和整理
            self.logger.info("处理时序窗口数据...")
            
            # 检查数据是否为时序窗口格式
            step_columns = [col for col in df.columns if col.startswith('step_')]
            if step_columns:
                # 分析时序窗口结构
                step_numbers = set()
                feature_types = set()
                
                for col in step_columns:
                    parts = col.split('_')
                    if len(parts) >= 3:
                        step_numbers.add(parts[1])  # step_XX部分
                        feature_types.add('_'.join(parts[2:]))  # 特征名部分
                
                max_step = max([int(s) for s in step_numbers if s.isdigit()])
                self.logger.info(f"检测到时序窗口数据: {max_step + 1} 个时间步, {len(feature_types)} 种特征类型")
                self.logger.info(f"时间窗口范围: step_00 到 step_{max_step:02d}")
                
                # 验证窗口完整性
                expected_steps = [f"{i:02d}" for i in range(max_step + 1)]
                missing_steps = set(expected_steps) - step_numbers
                if missing_steps:
                    self.logger.warning(f"发现缺失的时间步: {missing_steps}")
                
                # 按股票代码分组，检查数据一致性
                if 'stock_code' in df.columns:
                    stock_groups = df.groupby('stock_code').size()
                    self.logger.info(f"数据包含 {len(stock_groups)} 只股票")
                    self.logger.info(f"每只股票的样本数量 - 最少: {stock_groups.min()}, 最多: {stock_groups.max()}, 平均: {stock_groups.mean():.1f}")
                    
                    # 检查是否有不完整的窗口数据
                    incomplete_stocks = stock_groups[stock_groups < 5].index.tolist()  # 少于5个样本可能有问题
                    if incomplete_stocks:
                        self.logger.warning(f"发现样本数量较少的股票 (< 5个样本): {incomplete_stocks[:5]}..." if len(incomplete_stocks) > 5 else incomplete_stocks)
                
                # 保存时序窗口信息
                self.timeseries_info = {
                    'window_size': max_step + 1,
                    'feature_types': list(feature_types),
                    'step_columns': step_columns,
                    'is_timeseries_window': True
                }
                
            else:
                self.logger.info("未检测到时序窗口格式，按传统数据处理")
                self.timeseries_info = {'is_timeseries_window': False}
            
            # 分离特征和目标
            if 'target' in df.columns and 'stock_code' in df.columns:
                # 移除非特征列（包括日期列）
                excluded_cols = ['target', 'stock_code']
                if 'date' in df.columns:
                    excluded_cols.append('date')
                if 'Date' in df.columns:
                    excluded_cols.append('Date')
                    
                feature_cols = [col for col in df.columns if col not in excluded_cols]
                X = df[feature_cols].values
                y = df['target'].values
                self.stock_codes = df['stock_code'].tolist()
                
                # 保存真实的特征名称
                self.original_feature_names = feature_cols
                self.logger.info(f"提取特征列 {len(feature_cols)} 个: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"提取特征列: {feature_cols}")
            else:
                raise ValueError("完整数据文件缺少必要的列: 'target' 或 'stock_code'")
                
        else:
            self.logger.info("分别加载特征和目标文件...")
            # 分别加载特征数据和目标数据
            X_path = os.path.join(data_dir, data_config['X_features_file'])
            y_path = os.path.join(data_dir, data_config['y_targets_file'])
            
            if not os.path.exists(X_path):
                raise FileNotFoundError(f"特征文件不存在: {X_path}")
            if not os.path.exists(y_path):
                raise FileNotFoundError(f"目标文件不存在: {y_path}")
            
            # 加载特征数据
            if TQDM_AVAILABLE:
                with tqdm(desc="读取特征文件", leave=False) as pbar:
                    X_df = pd.read_csv(X_path, encoding=encoding)
                    pbar.update(len(X_df))
            else:
                X_df = pd.read_csv(X_path, encoding=encoding)
            
            # 加载目标数据（需要同时处理两个文件的日期过滤）
            if TQDM_AVAILABLE:
                with tqdm(desc="读取目标文件", leave=False) as pbar:
                    y_df = pd.read_csv(y_path, encoding=encoding)
                    pbar.update(len(y_df))
            else:
                y_df = pd.read_csv(y_path, encoding=encoding)
            
            # 时序窗口数据验证和整理（分文件模式）
            self.logger.info("处理时序窗口数据...")
            
            # 检查特征文件是否为时序窗口格式
            step_columns = [col for col in X_df.columns if col.startswith('step_')]
            if step_columns:
                # 分析时序窗口结构
                step_numbers = set()
                feature_types = set()
                
                for col in step_columns:
                    parts = col.split('_')
                    if len(parts) >= 3:
                        step_numbers.add(parts[1])  # step_XX部分
                        feature_types.add('_'.join(parts[2:]))  # 特征名部分
                
                max_step = max([int(s) for s in step_numbers if s.isdigit()])
                self.logger.info(f"检测到时序窗口数据: {max_step + 1} 个时间步, {len(feature_types)} 种特征类型")
                
                # 验证X和y文件的数据一致性
                if len(X_df) != len(y_df):
                    self.logger.error(f"特征文件和目标文件的行数不匹配: {len(X_df)} vs {len(y_df)}")
                    raise ValueError("特征文件和目标文件的数据量不匹配")
                
                # 如果y文件也有股票代码，验证顺序一致性
                if 'stock_code' in X_df.columns and 'stock_code' in y_df.columns:
                    if not X_df['stock_code'].equals(y_df['stock_code']):
                        self.logger.warning("特征文件和目标文件的股票代码顺序不一致，正在对齐...")
                        # 按股票代码对齐
                        y_df = y_df.set_index('stock_code').loc[X_df['stock_code']].reset_index()
                
                # 按股票代码分组，检查数据一致性
                if 'stock_code' in X_df.columns:
                    stock_groups = X_df.groupby('stock_code').size()
                    self.logger.info(f"数据包含 {len(stock_groups)} 只股票")
                    self.logger.info(f"每只股票的样本数量 - 最少: {stock_groups.min()}, 最多: {stock_groups.max()}")
                
                # 保存时序窗口信息
                self.timeseries_info = {
                    'window_size': max_step + 1,
                    'feature_types': list(feature_types),
                    'step_columns': step_columns,
                    'is_timeseries_window': True
                }
                
            else:
                self.logger.info("未检测到时序窗口格式，按传统数据处理")
                self.timeseries_info = {'is_timeseries_window': False}
            
            # 提取股票代码和特征
            if 'stock_code' in X_df.columns:
                self.stock_codes = X_df['stock_code'].tolist()
                # 移除股票代码列和日期列，只保留特征
                excluded_cols = ['stock_code']
                if date_col:
                    excluded_cols.append(date_col)
                feature_cols = [col for col in X_df.columns if col not in excluded_cols]
                X = X_df[feature_cols].values
            else:
                # 从目标文件中获取股票代码
                if 'stock_code' in y_df.columns:
                    self.stock_codes = y_df['stock_code'].tolist()
                else:
                    self.stock_codes = [f'stock_{i}' for i in range(len(X_df))]
                
                # 获取特征列
                excluded_cols = []
                if date_col:
                    excluded_cols.append(date_col)
                feature_cols = [col for col in X_df.columns if col not in excluded_cols]
                X = X_df[feature_cols].values
            
            # 保存真实的特征名称
            self.original_feature_names = feature_cols
            self.logger.info(f"提取特征列 {len(feature_cols)} 个: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"提取特征列: {feature_cols}")
            
            # 提取目标数据（y_df已经加载过了）
            if 'target' in y_df.columns:
                y = y_df['target'].values
            else:
                # 如果没有target列，假设最后一列是目标
                excluded_cols = ['stock_code']
                if date_col and date_col in y_df.columns:
                    excluded_cols.append(date_col)
                target_cols = [col for col in y_df.columns if col not in excluded_cols]
                if target_cols:
                    y = y_df[target_cols[-1]].values  # 使用最后一个非排除列
                else:
                    y = y_df.iloc[:, -1].values
        
        self.logger.info(f"数据加载完成: X.shape={X.shape}, y.shape={y.shape}")
        
        # 数据验证和清理
        if validate_data:
            X, y = self._validate_loaded_data(X, y)
        
        # 加载股票代码信息（如果单独的json文件存在）
        stock_codes_path = os.path.join(data_dir, data_config['stock_codes_file'])
        if os.path.exists(stock_codes_path):
            with open(stock_codes_path, 'r', encoding='utf-8') as f:
                stock_codes_from_json = json.load(f)
                # 如果CSV中没有股票代码，使用JSON中的
                if not hasattr(self, 'stock_codes') or not self.stock_codes:
                    self.stock_codes = stock_codes_from_json
                    
        # 加载数据信息
        data_info_path = os.path.join(data_dir, data_config['data_info_file'])
        if os.path.exists(data_info_path):
            with open(data_info_path, 'r', encoding='utf-8') as f:
                self.data_info = json.load(f)
                self.logger.info(f"数据信息: {self.data_info.get('output_format', 'Unknown')} 格式")
        else:
            self.data_info = {}
            
        return X, y
    
    def _validate_loaded_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """验证和清理加载的数据"""
        self.logger.info("验证数据完整性...")
        
        # 检查基本形状
        if len(X) != len(y):
            raise ValueError(f"特征数据和目标数据长度不匹配: {len(X)} vs {len(y)}")
        
        # 检查并处理NaN值和无穷值
        X_nan_count = np.isnan(X).sum()
        y_nan_count = np.isnan(y).sum()
        X_inf_count = np.isinf(X).sum()
        y_inf_count = np.isinf(y).sum()
        
        if X_nan_count > 0:
            self.logger.warning(f"特征数据中发现 {X_nan_count} 个 NaN 值，将替换为0")
        if y_nan_count > 0:
            self.logger.warning(f"目标数据中发现 {y_nan_count} 个 NaN 值，将替换为0")
        if X_inf_count > 0:
            self.logger.warning(f"特征数据中发现 {X_inf_count} 个无穷值，将替换为0")
        if y_inf_count > 0:
            self.logger.warning(f"目标数据中发现 {y_inf_count} 个无穷值，将替换为0")
        
        # 如果发现问题数据，进行清理
        total_issues = X_nan_count + y_nan_count + X_inf_count + y_inf_count
        if total_issues > 0:
            self.logger.info(f"开始清理数据中的 {total_issues} 个问题值...")
            X, y = self._clean_raw_data(X, y)
        
        # 检查股票代码数量
        if hasattr(self, 'stock_codes') and self.stock_codes:
            if len(self.stock_codes) != len(X):
                self.logger.warning(f"股票代码数量与样本数量不匹配: {len(self.stock_codes)} vs {len(X)}")
        
        self.logger.info("数据验证完成")
        return X, y
    
    def _clean_raw_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """清理原始数据中的NaN值和无穷值"""
        X_clean = np.copy(X)
        y_clean = np.copy(y)
        
        # 处理特征数据中的无穷值和NaN值
        X_clean[np.isnan(X_clean)] = 0
        X_clean[np.isinf(X_clean)] = 0
        
        # 处理目标数据中的无穷值和NaN值
        y_clean[np.isnan(y_clean)] = 0
        y_clean[np.isinf(y_clean)] = 0
        
        # 验证清理结果
        X_issues_after = np.isnan(X_clean).sum() + np.isinf(X_clean).sum()
        y_issues_after = np.isnan(y_clean).sum() + np.isinf(y_clean).sum()
        
        if X_issues_after == 0 and y_issues_after == 0:
            self.logger.info("原始数据清理完成，所有问题值已处理")
        else:
            self.logger.warning(f"清理后仍有问题：特征数据 {X_issues_after} 个，目标数据 {y_issues_after} 个")
        
        return X_clean, y_clean
    
    def preprocess_timeseries_data(self, X: np.ndarray) -> np.ndarray:
        """
        预处理时序数据
        
        Args:
            X: 原始数据，可能是2D [samples, features] 或 3D [samples, timesteps, features]
            
        Returns:
            处理后的2D数据 [samples, features]
        """
        self.logger.info("开始预处理时序数据...")
        self.logger.info(f"输入数据形状: {X.shape}")
        
        # 检查数据维度和时序窗口类型
        if len(X.shape) == 2:
            # 数据已经是2D，检查是否为时序窗口数据
            if hasattr(self, 'timeseries_info') and self.timeseries_info.get('is_timeseries_window', False):
                self.logger.info("检测到2D时序窗口数据，进行特征名称处理...")
                X_processed = X
                
                # 使用step_格式的特征名称
                if hasattr(self, 'original_feature_names') and self.original_feature_names:
                    # 验证特征名称是否为step格式
                    step_features = [name for name in self.original_feature_names if name.startswith('step_')]
                    if step_features:
                        self.feature_names = self.original_feature_names.copy()
                        self.logger.info(f"使用时序窗口特征名称: {len(step_features)} 个step特征")
                        
                        # 分析时序窗口结构
                        window_size = self.timeseries_info.get('window_size', 30)
                        feature_types = self.timeseries_info.get('feature_types', [])
                        self.logger.info(f"时序窗口结构: {window_size} 个时间步 × {len(feature_types)} 种特征")
                    else:
                        self.feature_names = self.original_feature_names.copy()
                        self.logger.info(f"使用原始特征名称: {len(self.feature_names)} 个特征")
                else:
                    self.feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]
                    self.logger.warning("未找到原始特征名称，使用默认名称")
            else:
                # 传统2D数据处理
                self.logger.info("检测到2D数据，跳过时序预处理")
                X_processed = X
                
                if hasattr(self, 'original_feature_names') and self.original_feature_names:
                    self.feature_names = self.original_feature_names.copy()
                    self.logger.info(f"使用原始特征名称: {len(self.feature_names)} 个特征")
                else:
                    self.feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]
                    self.logger.warning("未找到原始特征名称，使用默认名称")
            
        elif len(X.shape) == 3:
            # 3D时序数据，需要按配置处理
            method = self.config['data']['preprocessing']['timeseries_method']
            
            if method == 'flatten':
                # 直接展平
                X_processed = X.reshape(X.shape[0], -1)
                # 基于原始特征名称生成展平后的特征名称
                if hasattr(self, 'original_feature_names') and self.original_feature_names:
                    base_names = self.original_feature_names
                else:
                    base_names = [f'feature_{f}' for f in range(X.shape[2])]
                    
                self.feature_names = [f'timestep_{t}_{name}' 
                                    for t in range(X.shape[1]) 
                                    for name in base_names]
                
            elif method == 'statistical':
                # 提取统计特征
                stat_features = self.config['data']['preprocessing']['feature_engineering']['statistical_features']
                X_processed = self._extract_statistical_features(X, stat_features)
                
            elif method == 'last_step':
                # 只使用最后一个时间步
                X_processed = X[:, -1, :]
                # 基于原始特征名称生成特征名称
                if hasattr(self, 'original_feature_names') and self.original_feature_names:
                    self.feature_names = [f'last_step_{name}' for name in self.original_feature_names]
                else:
                    self.feature_names = [f'last_step_feature_{i}' for i in range(X_processed.shape[1])]
                
            else:
                raise ValueError(f"不支持的时序处理方法: {method}")
                
        else:
            raise ValueError(f"不支持的数据维度: {len(X.shape)}D，期望2D或3D")
        
        self.logger.info(f"时序数据预处理完成: {X.shape} -> {X_processed.shape}")
        return X_processed
    
    def _extract_statistical_features(self, X: np.ndarray, stat_features: List[str]) -> np.ndarray:
        """提取统计特征"""
        features_list = []
        feature_names = []
        
        # 基于原始特征名称生成基础名称
        if hasattr(self, 'original_feature_names') and self.original_feature_names:
            base_feature_names = self.original_feature_names
        else:
            base_feature_names = [f'feature_{j}' for j in range(X.shape[2])]
        
        for i, base_name in enumerate(base_feature_names):
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
                
                feature_names.append(f'{base_name}_{stat}')
        
        self.feature_names = feature_names
        self.logger.info(f"生成统计特征名称: {len(feature_names)} 个，示例: {feature_names[:5]}...")
        return np.column_stack(features_list)
    
    def add_technical_features(self, X: np.ndarray, raw_X: np.ndarray) -> np.ndarray:
        """添加技术指标特征"""
        if not self.config['data']['preprocessing']['feature_engineering']['technical_features']['enabled']:
            return X
        
        # 对于时序窗口数据，可以计算一些基于窗口的技术指标
        if hasattr(self, 'timeseries_info') and self.timeseries_info.get('is_timeseries_window', False):
            self.logger.info("为时序窗口数据计算技术指标...")
            
            try:
                tech_features = []
                tech_feature_names = []
                
                # 从时序窗口数据中提取技术指标
                window_size = self.timeseries_info.get('window_size', 30)
                feature_types = self.timeseries_info.get('feature_types', [])
                
                # 寻找价格相关的特征
                price_features = [ft for ft in feature_types if any(keyword in ft for keyword in ['开盘价', '收盘价', '最高价', '最低价'])]
                volume_features = [ft for ft in feature_types if '成交量' in ft or '成交额' in ft]
                
                self.logger.info(f"发现 {len(price_features)} 种价格特征, {len(volume_features)} 种成交量特征")
                
                if price_features:
                    # 计算基于时序窗口的技术指标
                    window_tech_features = self._calculate_window_technical_features(X, window_size, feature_types)
                    tech_features.extend(window_tech_features['features'])
                    tech_feature_names.extend(window_tech_features['names'])
                
                if tech_features:
                    tech_array = np.column_stack(tech_features)
                    X_enhanced = np.column_stack([X, tech_array])
                    self.feature_names.extend(tech_feature_names)
                    self.logger.info(f"时序窗口技术指标特征添加完成: {X.shape} -> {X_enhanced.shape}")
                    return X_enhanced
                
            except Exception as e:
                self.logger.warning(f"时序窗口技术指标计算失败: {e}，跳过技术特征")
                
            return X
        
        # 传统的3D数据技术指标计算
        if len(raw_X.shape) == 2:
            self.logger.info("原始数据已为2D格式，跳过传统技术指标计算")
            return X
            
        elif len(raw_X.shape) != 3:
            self.logger.warning(f"原始数据维度异常: {raw_X.shape}，跳过技术指标特征计算")
            return X
            
        self.logger.info("添加传统技术指标特征...")
        
        tech_features = []
        tech_feature_names = []
        
        try:
            # 计算动量特征
            momentum_features = self._calculate_momentum_features(raw_X)
            tech_features.extend(momentum_features)
            tech_feature_names.extend([f'momentum_{i}' for i in range(len(momentum_features))])
            
            # 计算波动率特征
            volatility_features = self._calculate_volatility_features(raw_X)
            tech_features.extend(volatility_features)
            tech_feature_names.extend([f'volatility_{i}' for i in range(len(volatility_features))])
            
            # 计算趋势特征
            trend_features = self._calculate_trend_features(raw_X)
            tech_features.extend(trend_features)
            tech_feature_names.extend([f'trend_{i}' for i in range(len(trend_features))])
            
            # 合并特征
            if tech_features:
                tech_array = np.column_stack(tech_features)
                X_enhanced = np.column_stack([X, tech_array])
                self.feature_names.extend(tech_feature_names)
                self.logger.info(f"技术指标特征添加完成: {X.shape} -> {X_enhanced.shape}")
                return X_enhanced
                
        except Exception as e:
            self.logger.warning(f"技术指标特征计算失败: {e}，跳过技术特征")
            
        return X
    
    def _calculate_window_technical_features(self, X: np.ndarray, window_size: int, feature_types: List[str]) -> Dict:
        """计算基于时序窗口的技术指标"""
        tech_features = []
        tech_feature_names = []
        
        # 寻找价格特征的列索引
        price_cols = {}
        for i, feature_name in enumerate(self.feature_names):
            for price_type in ['开盘价', '收盘价', '最高价', '最低价']:
                if price_type in feature_name:
                    step_match = feature_name.split('_')[1] if feature_name.startswith('step_') else None
                    if step_match:
                        if price_type not in price_cols:
                            price_cols[price_type] = []
                        price_cols[price_type].append((i, int(step_match)))
        
        # 对每种价格类型计算技术指标
        for price_type, col_info in price_cols.items():
            if len(col_info) >= window_size:
                # 按时间步排序
                col_info.sort(key=lambda x: x[1])
                price_indices = [col[0] for col in col_info]
                
                # 提取该价格类型的时序数据
                price_series = X[:, price_indices]  # [samples, time_steps]
                
                # 计算技术指标
                # 1. 移动平均（不同窗口大小）
                for ma_window in [5, 10, 20]:
                    if ma_window <= window_size:
                        ma_values = []
                        for i in range(len(price_series)):
                            series = price_series[i]
                            # 计算最后ma_window个数据的移动平均
                            if len(series) >= ma_window:
                                window_data = series[-ma_window:]
                                # 过滤掉NaN值
                                valid_data = window_data[~np.isnan(window_data)]
                                if len(valid_data) > 0:
                                    ma = np.mean(valid_data)
                                    ma_values.append(ma)
                                else:
                                    ma_values.append(0)  # 如果全是NaN，使用0
                            else:
                                ma_values.append(0)
                        
                        tech_features.append(np.array(ma_values))
                        tech_feature_names.append(f'MA{ma_window}_{price_type}')
                
                # 2. 相对强弱指标RSI的简化版本（基于最近几天的涨跌）
                rsi_values = []
                for i in range(len(price_series)):
                    series = price_series[i]
                    if len(series) >= 14:
                        # 计算价格变化
                        price_changes = np.diff(series[-14:])
                        gains = price_changes[price_changes > 0]
                        losses = -price_changes[price_changes < 0]
                        
                        avg_gain = np.mean(gains) if len(gains) > 0 else 0
                        avg_loss = np.mean(losses) if len(losses) > 0 else 0
                        
                        if avg_loss != 0 and not np.isnan(avg_loss) and not np.isnan(avg_gain):
                            rs = avg_gain / avg_loss
                            rsi = 100 - (100 / (1 + rs))
                            # 确保RSI在0-100范围内
                            rsi = np.clip(rsi, 0, 100)
                        else:
                            rsi = 100 if avg_gain > 0 else 50
                        
                        rsi_values.append(rsi)
                    else:
                        rsi_values.append(50)  # 默认中性值
                
                tech_features.append(np.array(rsi_values))
                tech_feature_names.append(f'RSI_{price_type}')
                
                # 3. 价格波动率（标准差）
                volatility_values = []
                for i in range(len(price_series)):
                    series = price_series[i]
                    vol = np.std(series) if len(series) > 1 else 0
                    volatility_values.append(vol)
                
                tech_features.append(np.array(volatility_values))
                tech_feature_names.append(f'Volatility_{price_type}')
                
                # 4. 动量指标（最新价格相对于N天前的变化率）
                for momentum_period in [5, 10]:
                    if momentum_period < window_size:
                        momentum_values = []
                        for i in range(len(price_series)):
                            series = price_series[i]
                            if len(series) > momentum_period:
                                prev_price = series[-momentum_period-1]
                                curr_price = series[-1]
                                
                                # 防止除零错误
                                if prev_price != 0 and not np.isnan(prev_price) and not np.isnan(curr_price):
                                    momentum = (curr_price - prev_price) / abs(prev_price) * 100
                                    # 限制动量值范围，防止极端值
                                    momentum = np.clip(momentum, -1000, 1000)
                                else:
                                    momentum = 0
                                
                                momentum_values.append(momentum)
                            else:
                                momentum_values.append(0)
                        
                        tech_features.append(np.array(momentum_values))
                        tech_feature_names.append(f'Momentum{momentum_period}D_{price_type}')
        
        # 数据清理：处理无穷值和NaN值
        cleaned_features = []
        for i, feature_array in enumerate(tech_features):
            # 替换无穷值和NaN值
            clean_feature = np.copy(feature_array)
            
            # 处理无穷值
            clean_feature[np.isinf(clean_feature)] = 0
            
            # 处理NaN值
            clean_feature[np.isnan(clean_feature)] = 0
            
            # 处理极端值（超过10个标准差的值）
            if len(clean_feature) > 1:
                std_val = np.std(clean_feature)
                mean_val = np.mean(clean_feature)
                if std_val > 0:
                    # 限制在±10个标准差范围内
                    outlier_mask = np.abs(clean_feature - mean_val) > 10 * std_val
                    clean_feature[outlier_mask] = mean_val
            
            cleaned_features.append(clean_feature)
        
        return {
            'features': cleaned_features,
            'names': tech_feature_names
        }
    
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
    
    def _clean_data_for_scaling(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """清理数据中的无穷值、NaN值和极端值，为标准化做准备"""
        def clean_array(X: np.ndarray, array_name: str) -> np.ndarray:
            X_clean = np.copy(X)
            
            # 统计问题数据
            inf_count = np.isinf(X_clean).sum()
            nan_count = np.isnan(X_clean).sum()
            
            if inf_count > 0 or nan_count > 0:
                self.logger.warning(f"{array_name}: 发现 {inf_count} 个无穷值, {nan_count} 个NaN值")
            
            # 处理无穷值：替换为0
            X_clean[np.isinf(X_clean)] = 0
            
            # 处理NaN值：替换为0  
            X_clean[np.isnan(X_clean)] = 0
            
            # 处理极端值：每个特征维度分别处理
            for feature_idx in range(X_clean.shape[1]):
                feature_values = X_clean[:, feature_idx]
                if len(feature_values) > 1:
                    # 计算该特征的统计信息
                    q75 = np.percentile(feature_values, 75)
                    q25 = np.percentile(feature_values, 25)
                    iqr = q75 - q25
                    
                    if iqr > 0:
                        # 使用IQR方法识别极端值
                        lower_bound = q25 - 3.0 * iqr
                        upper_bound = q75 + 3.0 * iqr
                        
                        # 限制极端值
                        outlier_mask = (feature_values < lower_bound) | (feature_values > upper_bound)
                        if outlier_mask.any():
                            # 用中位数替换极端值
                            median_val = np.median(feature_values[~outlier_mask]) if (~outlier_mask).any() else np.median(feature_values)
                            X_clean[outlier_mask, feature_idx] = median_val
            
            return X_clean
        
        # 清理所有数据集
        X_train_clean = clean_array(X_train, "训练集")
        X_val_clean = clean_array(X_val, "验证集")
        X_test_clean = clean_array(X_test, "测试集")
        
        # 最终检查
        for name, data in [("训练集", X_train_clean), ("验证集", X_val_clean), ("测试集", X_test_clean)]:
            if np.isinf(data).any() or np.isnan(data).any():
                self.logger.error(f"{name}仍包含无穷值或NaN值！")
            else:
                self.logger.info(f"{name}数据清理完成")
        
        return X_train_clean, X_val_clean, X_test_clean
    
    def normalize_data(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """数据标准化"""
        norm_config = self.config['data']['preprocessing']['normalization']
        
        if norm_config['method'] is None:
            return X_train, X_val, X_test
        
        # 数据清理：在标准化前处理无穷值和NaN值
        self.logger.info("数据清理：处理无穷值和NaN值...")
        X_train, X_val, X_test = self._clean_data_for_scaling(X_train, X_val, X_test)
            
        self.logger.info(f"使用 {norm_config['method']} 方法进行数据标准化...")
        
        if norm_config['method'] == 'standard':
            self.scaler = StandardScaler()
        elif norm_config['method'] == 'minmax':
            self.scaler = MinMaxScaler()
        elif norm_config['method'] == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的标准化方法: {norm_config['method']}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def handle_outliers(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """增强的异常值处理"""
        outlier_config = self.config['data']['preprocessing']['outlier_handling']
        
        if not outlier_config['enabled']:
            return X, y
            
        self.logger.info("处理异常值...")
        
        method = outlier_config['method']
        
        if method == 'multi_stage':
            # 多阶段异常值处理（新增）
            return self._handle_outliers_multi_stage(X, y, outlier_config)
        elif method == 'iqr':
            # 使用IQR方法
            threshold = outlier_config['threshold']
            Q1 = np.percentile(y, 25)
            Q3 = np.percentile(y, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (y >= lower_bound) & (y <= upper_bound)
            
        elif method == 'zscore':
            # 使用Z-score方法
            threshold = outlier_config['threshold']
            z_scores = np.abs(stats.zscore(y))
            mask = z_scores < threshold
            
        else:
            self.logger.warning(f"不支持的异常值处理方法: {method}")
            return X, y
        
        X_clean = X[mask]
        y_clean = y[mask]
        
        removed_count = len(y) - len(y_clean)
        self.logger.info(f"异常值处理完成，移除了 {removed_count} 个样本")
        
        return X_clean, y_clean
    
    def _handle_outliers_multi_stage(self, X: np.ndarray, y: np.ndarray, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """多阶段异常值处理"""
        extreme_threshold = config.get('extreme_threshold', 5.0)
        mild_threshold = config.get('mild_threshold', 2.5)
        strategy = config.get('strategy', 'winsorize')
        winsorize_limits = config.get('winsorize_limits', [0.01, 0.01])
        
        original_count = len(y)
        
        # 第一阶段：处理极端异常值
        z_scores = np.abs(stats.zscore(y))
        extreme_mask = z_scores < extreme_threshold
        
        if strategy == 'remove':
            X_clean = X[extreme_mask]
            y_clean = y[extreme_mask]
            extreme_removed = original_count - len(y_clean)
            self.logger.info(f"第一阶段：移除了 {extreme_removed} 个极端异常值")
            
        elif strategy == 'winsorize':
            # 使用Winsorize方法截断极端值
            from scipy.stats import mstats
            y_clean = mstats.winsorize(y, limits=winsorize_limits)
            X_clean = X.copy()
            extreme_modified = np.sum(y != y_clean)
            self.logger.info(f"第一阶段：调整了 {extreme_modified} 个极端异常值")
            
        elif strategy == 'clip':
            # 使用clip方法截断
            lower_percentile = winsorize_limits[0] * 100
            upper_percentile = (1 - winsorize_limits[1]) * 100
            lower_bound = np.percentile(y, lower_percentile)
            upper_bound = np.percentile(y, upper_percentile)
            y_clean = np.clip(y, lower_bound, upper_bound)
            X_clean = X.copy()
            extreme_modified = np.sum(y != y_clean)
            self.logger.info(f"第一阶段：截断了 {extreme_modified} 个极端异常值")
        
        # 第二阶段：处理温和异常值（仅在remove策略下执行）
        if strategy == 'remove':
            z_scores_clean = np.abs(stats.zscore(y_clean))
            mild_mask = z_scores_clean < mild_threshold
            X_final = X_clean[mild_mask]
            y_final = y_clean[mild_mask]
            mild_removed = len(y_clean) - len(y_final)
            self.logger.info(f"第二阶段：移除了 {mild_removed} 个温和异常值")
        else:
            X_final = X_clean
            y_final = y_clean
        
        total_processed = original_count - len(y_final) if strategy == 'remove' else np.sum(y != y_final)
        self.logger.info(f"多阶段异常值处理完成，总共处理了 {total_processed} 个样本")
        
        return X_final, y_final
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """数据分割"""
        self.logger.info("分割数据...")
        
        split_config = self.config['training']['data_split']
        
        # 先分离测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=split_config['test_size'],
            random_state=split_config['random_state'],
            stratify=None if not split_config['stratify'] else y
        )
        
        # 再分离训练集和验证集
        val_size = split_config['validation_size'] / (1 - split_config['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=split_config['random_state'],
            stratify=None if not split_config['stratify'] else y_temp
        )
        
        self.logger.info(f"数据分割完成: 训练集={len(X_train)}, 验证集={len(X_val)}, 测试集={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def feature_selection(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, ...]:
        """增强的特征选择"""
        fs_config = self.config['feature_selection']
        
        if not fs_config['enabled']:
            return X_train, X_val, X_test
            
        self.logger.info("进行特征选择...")
        original_features = X_train.shape[1]
        
        # 检查是否启用渐进式特征选择
        progressive_config = self.config.get('progressive_feature_selection', {})
        if progressive_config.get('enabled', False):
            return self._progressive_feature_selection(
                X_train, X_val, X_test, y_train, progressive_config
            )
        
        # 传统特征选择
        # 基于重要性的选择
        if fs_config['methods']['importance_based']['enabled']:
            X_train, X_val, X_test = self._importance_based_selection(
                X_train, X_val, X_test, y_train, fs_config['methods']['importance_based']
            )
            
        # 基于相关性的选择
        if fs_config['methods']['correlation_based']['enabled']:
            X_train, X_val, X_test = self._correlation_based_selection(
                X_train, X_val, X_test, fs_config['methods']['correlation_based']
            )
            
        # 递归特征消除
        if fs_config['methods']['rfe']['enabled']:
            X_train, X_val, X_test = self._rfe_selection(
                X_train, X_val, X_test, y_train, fs_config['methods']['rfe']
            )
        
        final_features = X_train.shape[1]
        self.logger.info(f"特征选择完成：{original_features} -> {final_features} 个特征")
        
        return X_train, X_val, X_test
    
    def _progressive_feature_selection(self, X_train, X_val, X_test, y_train, config):
        """渐进式特征选择"""
        self.logger.info("开始渐进式特征选择...")
        original_features = X_train.shape[1]
        
        stages = config.get('stages', [])
        current_features = X_train.shape[1]
        
        for stage in stages:
            stage_name = stage['name']
            method = stage['method']
            threshold = stage['threshold']
            
            self.logger.info(f"执行阶段: {stage_name}")
            
            if method == 'variance_threshold':
                # 移除低方差特征
                X_train, X_val, X_test = self._variance_threshold_selection(
                    X_train, X_val, X_test, threshold
                )
                
            elif method == 'correlation':
                # 移除高相关特征
                X_train, X_val, X_test = self._correlation_based_selection(
                    X_train, X_val, X_test, {'threshold': threshold}
                )
                
            elif method == 'importance':
                # 基于重要性选择
                X_train, X_val, X_test = self._importance_based_selection(
                    X_train, X_val, X_test, y_train, 
                    {'threshold': threshold, 'method': 'gain'}
                )
            
            new_features = X_train.shape[1]
            removed = current_features - new_features
            self.logger.info(f"{stage_name}完成：移除了{removed}个特征，剩余{new_features}个")
            current_features = new_features
        
        final_features = X_train.shape[1]
        self.logger.info(f"渐进式特征选择完成：{original_features} -> {final_features} 个特征")
        
        return X_train, X_val, X_test
    
    def _variance_threshold_selection(self, X_train, X_val, X_test, threshold):
        """基于方差阈值的特征选择"""
        from sklearn.feature_selection import VarianceThreshold
        
        selector = VarianceThreshold(threshold=threshold)
        X_train_selected = selector.fit_transform(X_train)
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        # 更新特征名称
        selected_features = selector.get_support()
        self.feature_names = [name for name, selected in zip(self.feature_names, selected_features) if selected]
        
        return X_train_selected, X_val_selected, X_test_selected
    
    def _importance_based_selection(self, X_train, X_val, X_test, y_train, config):
        """基于重要性的特征选择"""
        threshold = config['threshold']
        method = config.get('method', 'split')
        
        # 训练临时模型获取特征重要性
        temp_params = {
            'random_state': 42,
            'verbose': -1,
            'n_estimators': 100,  # 减少估计器数量，加快速度
            'importance_type': method
        }
        temp_model = lgb.LGBMRegressor(**temp_params)
        
        with SuppressOutput():  # 静默训练
            temp_model.fit(X_train, y_train)
        
        selector = SelectFromModel(temp_model, threshold=threshold)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        selected_features = selector.get_support()
        self.feature_names = [name for name, selected in zip(self.feature_names, selected_features) if selected]
        
        selected_count = X_train_selected.shape[1]
        self.logger.info(f"基于重要性选择了 {selected_count} 个特征")
        
        return X_train_selected, X_val_selected, X_test_selected
    
    def _correlation_based_selection(self, X_train, X_val, X_test, config):
        """基于相关性的特征选择"""
        threshold = config['threshold']
        
        # 计算特征间相关性矩阵
        corr_matrix = np.corrcoef(X_train.T)
        corr_matrix = np.abs(corr_matrix)
        
        # 找出高相关性的特征对
        upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        high_corr_pairs = np.where((corr_matrix > threshold) & upper_triangle)
        
        # 选择要移除的特征（保留方差较大的）
        features_to_remove = set()
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            var_i = np.var(X_train[:, i])
            var_j = np.var(X_train[:, j])
            # 移除方差较小的特征
            if var_i < var_j:
                features_to_remove.add(i)
            else:
                features_to_remove.add(j)
        
        # 创建保留特征的掩码
        features_to_keep = [i for i in range(X_train.shape[1]) if i not in features_to_remove]
        
        X_train_selected = X_train[:, features_to_keep]
        X_val_selected = X_val[:, features_to_keep]
        X_test_selected = X_test[:, features_to_keep]
        
        # 更新特征名称
        self.feature_names = [self.feature_names[i] for i in features_to_keep]
        
        removed_count = len(features_to_remove)
        self.logger.info(f"基于相关性移除了 {removed_count} 个高相关特征")
        
        return X_train_selected, X_val_selected, X_test_selected
    
    def _rfe_selection(self, X_train, X_val, X_test, y_train, config):
        """递归特征消除"""
        n_features_to_select = config['n_features_to_select']
        step = config.get('step', 0.1)
        
        # 创建基础估计器
        estimator = lgb.LGBMRegressor(
            random_state=42,
            verbose=-1,
            n_estimators=50  # 减少估计器数量，加快RFE速度
        )
        
        # 创建RFE选择器
        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=step,
            verbose=0
        )
        
        self.logger.info(f"开始递归特征消除，目标特征数: {n_features_to_select}")
        
        # 执行特征选择（可能需要一些时间）
        with SuppressOutput():
            X_train_selected = selector.fit_transform(X_train, y_train)
        
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        # 更新特征名称
        selected_features = selector.get_support()
        self.feature_names = [name for name, selected in zip(self.feature_names, selected_features) if selected]
        
        self.logger.info(f"RFE选择了 {X_train_selected.shape[1]} 个特征")
        
        return X_train_selected, X_val_selected, X_test_selected
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray) -> lgb.LGBMRegressor:
        """训练模型"""
        # 获取模型参数
        lgb_config = self.config['lightgbm']
        model_params = {**lgb_config['basic_params'], **lgb_config['advanced_params']}
        fit_params = lgb_config['fit_params']
        train_params = self.config['training']['training_params']
        
        # 创建模型（不包含early_stopping_rounds，这个参数在fit中使用）
        self.model = lgb.LGBMRegressor(**model_params)
        
        # 设置回调函数
        callbacks = [
            lgb.early_stopping(train_params['early_stopping_rounds'])
        ]
        
        # 根据是否有tqdm设置不同的回调
        if TQDM_AVAILABLE:
            # 有tqdm时，直接静默训练，用简单的进度指示
            self.logger.info("🚀 开始模型训练...")
            callbacks.append(lgb.log_evaluation(0))  # 完全关闭日志
        else:
            # 没有tqdm时，使用较少的日志输出
            callbacks.append(lgb.log_evaluation(train_params.get('verbose', 100)))
        
        # 训练模型（静默训练）
        print("🚀 正在训练最终模型...")
        if TQDM_AVAILABLE:
            # 静默训练，避免冗长日志
            with SuppressOutput():
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=callbacks
                )
            print("✅ 模型训练完成")
        else:
            # 没有tqdm时，正常显示LightGBM输出
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )
        
        self.logger.info("模型训练完成")
        return self.model
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """超参数优化"""
        tuning_config = self.config['hyperparameter_tuning']
        
        if not tuning_config['enabled'] or not OPTUNA_AVAILABLE:
            return self.config['lightgbm']['basic_params']
            
        n_trials = tuning_config['optuna_config']['n_trials']
        
        # 设置Optuna日志级别，抑制详细输出
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # 创建研究对象，使用TPE采样器
        sampler = TPESampler(seed=42) if OPTUNA_AVAILABLE else None
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        # 创建清晰的进度条
        print(f"\n🔍 开始超参数优化 (共{n_trials}次尝试)...")
        if TQDM_AVAILABLE:
            pbar = tqdm(
                total=n_trials, 
                desc="🔍 优化进度", 
                unit="次",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                leave=True
            )
            
        # 用于存储最佳值的变量
        best_score = float('inf')
        
        def objective(trial):
            nonlocal best_score
            
            # 定义参数搜索空间
            param_ranges = tuning_config['optuna_config']['param_ranges']
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'random_state': 42,
                'num_leaves': trial.suggest_int('num_leaves', *param_ranges['num_leaves']),
                'learning_rate': trial.suggest_float('learning_rate', *param_ranges['learning_rate']),
                'feature_fraction': trial.suggest_float('feature_fraction', *param_ranges['feature_fraction']),
                'bagging_fraction': trial.suggest_float('bagging_fraction', *param_ranges['bagging_fraction']),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', *param_ranges['min_data_in_leaf']),
                'lambda_l1': trial.suggest_float('lambda_l1', *param_ranges['lambda_l1']),
                'lambda_l2': trial.suggest_float('lambda_l2', *param_ranges['lambda_l2']),
            }
            
            # 使用交叉验证评估
            cv_config = self.config['training']['cross_validation']
            if cv_config['enabled']:
                scores = []
                
                if cv_config['cv_strategy'] == 'time_series':
                    tscv = TimeSeriesSplit(n_splits=cv_config['cv_folds'])
                    cv_splits = list(tscv.split(X_train))
                else:
                    from sklearn.model_selection import KFold
                    kf = KFold(n_splits=cv_config['cv_folds'], shuffle=True, random_state=42)
                    cv_splits = list(kf.split(X_train))
                
                # 完全静默执行交叉验证
                for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    
                    # 使用上下文管理器完全抑制输出
                    with SuppressOutput():
                        model = lgb.LGBMRegressor(**params)
                        model.fit(
                            X_tr, y_tr, 
                            eval_set=[(X_val, y_val)],
                            callbacks=[
                                lgb.early_stopping(50, verbose=False),
                                lgb.log_evaluation(0)  # 完全关闭日志输出
                            ]
                        )
                        
                        y_pred = model.predict(X_val)
                    
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    scores.append(rmse)
                
                current_score = np.mean(scores)
                
                # 更新最佳分数
                if current_score < best_score:
                    best_score = current_score
                
                # 更新主进度条
                if TQDM_AVAILABLE:
                    pbar.update(1)
                    pbar.set_postfix({
                        '最佳': f"{best_score:.4f}",
                        '当前': f"{current_score:.4f}"
                    })
                
                return current_score
        
        try:
            # 执行优化
            study.optimize(
                objective, 
                n_trials=n_trials,
                timeout=tuning_config['optuna_config']['timeout'],
                show_progress_bar=False  # 我们使用自己的进度条
            )
        finally:
            if TQDM_AVAILABLE:
                pbar.close()
        
        best_params = {**self.config['lightgbm']['basic_params'], **study.best_params}
        
        # 美化最佳参数输出
        print(f"\n{'='*60}")
        print(f"🎯 超参数优化完成!")
        print(f"{'='*60}")
        print(f"最佳RMSE: {study.best_value:.4f}")
        print(f"最佳参数:")
        print(f"{'-'*40}")
        for param, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {param:<20}: {value:.6f}")
            else:
                print(f"  {param:<20}: {value}")
        print(f"{'='*60}\n")
        
        return best_params
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估模型"""
        self.logger.info("评估模型性能...")
        
        y_pred = self.model.predict(X_test)
        
        # 计算评估指标
        metrics = {}
        eval_config = self.config['evaluation']['metrics']
        
        if 'rmse' in eval_config:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        if 'mae' in eval_config:
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
        if 'r2_score' in eval_config:
            metrics['r2_score'] = r2_score(y_test, y_pred)
        if 'explained_variance' in eval_config:
            metrics['explained_variance'] = explained_variance_score(y_test, y_pred)
        if 'mape' in eval_config:
            # 避免除零错误：过滤掉y_test中的零值
            non_zero_mask = np.abs(y_test) > 1e-8  # 使用小阈值而不是严格的零
            if non_zero_mask.any():
                y_test_filtered = y_test[non_zero_mask]
                y_pred_filtered = y_pred[non_zero_mask]
                metrics['mape'] = np.mean(np.abs((y_test_filtered - y_pred_filtered) / y_test_filtered)) * 100
                
                # 如果有数据被过滤，记录信息
                if not non_zero_mask.all():
                    filtered_count = len(y_test) - non_zero_mask.sum()
                    self.logger.info(f"MAPE计算时过滤了{filtered_count}个接近零值的样本")
            else:
                # 如果所有真实值都接近零，MAPE无意义
                metrics['mape'] = np.inf
                self.logger.warning("所有目标值都接近零，MAPE设为无穷大")
        
        # 美化评估结果输出
        print(f"\n{'='*50}")
        print(f"📊 模型评估结果")
        print(f"{'='*50}")
        for metric, value in metrics.items():
            if np.isinf(value):
                print(f"  {metric.upper():<20}: ∞ (无穷大)")
            elif np.isnan(value):
                print(f"  {metric.upper():<20}: NaN")
            else:
                print(f"  {metric.upper():<20}: {value:.4f}")
        print(f"{'='*50}\n")
        
        return metrics, y_pred
    

    
    def save_model(self):
        """保存模型"""
        self.logger.info("保存模型...")
        
        save_config = self.config['output']['model_save']
        model_name = save_config['model_name']
        
        for format_type in save_config['save_format']:
            if format_type == 'pkl':
                model_path = os.path.join(self.model_save_dir, f"{model_name}.pkl")
                joblib.dump(self.model, model_path)
                
                # 保存预处理器
                if self.scaler is not None:
                    scaler_path = os.path.join(self.model_save_dir, "scaler.pkl")
                    joblib.dump(self.scaler, scaler_path)
                
                # 保存特征名称（详细信息）
                feature_info = {
                    'feature_names': self.feature_names,
                    'original_feature_names': getattr(self, 'original_feature_names', []),
                    'feature_count': len(self.feature_names),
                    'timeseries_method': self.config['data']['preprocessing']['timeseries_method'],
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'feature_mapping': {}
                }
                
                # 创建特征映射（原始特征 -> 处理后特征）
                if hasattr(self, 'original_feature_names') and self.original_feature_names:
                    for i, original_name in enumerate(self.original_feature_names):
                        related_features = [fname for fname in self.feature_names 
                                          if original_name in fname or fname.endswith(f'_{i}')]
                        feature_info['feature_mapping'][original_name] = related_features
                
                feature_names_path = os.path.join(self.model_save_dir, "feature_names.json")
                with open(feature_names_path, 'w', encoding='utf-8') as f:
                    json.dump(feature_info, f, ensure_ascii=False, indent=2)
                
                # 也保存简单的特征名称列表（便于快速查看）
                simple_names_path = os.path.join(self.model_save_dir, "feature_names_simple.txt")
                with open(simple_names_path, 'w', encoding='utf-8') as f:
                    f.write("特征名称列表:\n")
                    f.write("=" * 50 + "\n")
                    for i, name in enumerate(self.feature_names):
                        f.write(f"{i+1:4d}. {name}\n")
                    
            elif format_type == 'txt':
                model_path = os.path.join(self.model_save_dir, f"{model_name}.txt")
                self.model.booster_.save_model(model_path)
        
        self.logger.info(f"模型已保存到: {self.model_save_dir}")
    
    def save_results(self, metrics: Dict, y_pred: np.ndarray, y_test: np.ndarray):
        """保存结果"""
        save_config = self.config['output']['results_save']
        
        # 保存指标
        if save_config['save_metrics']:
            metrics_path = os.path.join(self.results_save_dir, "metrics.json")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        # 保存预测结果
        if save_config['save_predictions']:
            predictions_path = os.path.join(self.results_save_dir, "predictions.csv")
            results_df = pd.DataFrame({
                'y_true': y_test,
                'y_pred': y_pred,
                'residual': y_test - y_pred
            })
            results_df.to_csv(predictions_path, index=False)
        
        # 保存特征重要性
        if save_config['save_feature_importance'] and hasattr(self.model, 'feature_importances_'):
            # 创建详细的特征重要性信息
            feature_names_used = self.feature_names[:len(self.model.feature_importances_)]
            importance_df = pd.DataFrame({
                'feature_name': feature_names_used,
                'importance': self.model.feature_importances_,
                'feature_index': range(len(self.model.feature_importances_))
            }).sort_values('importance', ascending=False)
            
            # 添加特征排名
            importance_df['importance_rank'] = range(1, len(importance_df) + 1)
            
            # 添加相对重要性（百分比）
            total_importance = importance_df['importance'].sum()
            importance_df['importance_percent'] = (importance_df['importance'] / total_importance * 100).round(4)
            
            # 添加累积重要性百分比
            importance_df['cumulative_percent'] = importance_df['importance_percent'].cumsum().round(4)
            
            importance_path = os.path.join(self.results_save_dir, "feature_importance.csv")
            importance_df.to_csv(importance_path, index=False)
            
            # 保存前20个最重要特征的可视化文本
            top_features_path = os.path.join(self.results_save_dir, "top_features.txt")
            with open(top_features_path, 'w', encoding='utf-8') as f:
                f.write("前20个最重要特征:\n")
                f.write("=" * 80 + "\n")
                f.write(f"{'排名':<4} {'特征名称':<40} {'重要性':<12} {'占比(%)':<10} {'累积(%)':<10}\n")
                f.write("-" * 80 + "\n")
                for idx, row in importance_df.head(20).iterrows():
                    f.write(f"{row['importance_rank']:<4} {row['feature_name']:<40} "
                           f"{row['importance']:<12.6f} {row['importance_percent']:<10.2f} "
                           f"{row['cumulative_percent']:<10.2f}\n")
            
            self.logger.info(f"特征重要性已保存，前5个重要特征: {list(importance_df.head(5)['feature_name'])}")
        
        self.logger.info(f"结果已保存到: {self.results_save_dir}")
    
    def run_training_pipeline(self):
        """运行完整的训练流程"""
        try:
            self.logger.info("开始LightGBM训练流程...")
            
            # 定义训练步骤
            steps = [
                ("加载数据", self._step_load_data),
                ("处理异常值", self._step_handle_outliers),
                ("预处理数据", self._step_preprocess_data),
                ("添加技术特征", self._step_add_features),
                ("分割数据", self._step_split_data),
                ("标准化数据", self._step_normalize_data),
                ("特征选择", self._step_feature_selection),
                ("超参数优化", self._step_hyperparameter_tuning),
                ("训练模型", self._step_train_model),
                ("评估模型", self._step_evaluate_model),
                ("保存结果", self._step_save_results)
            ]
            
            # 使用简化的进度显示
            print(f"\n{'='*60}")
            print(f"🚀 开始LightGBM训练流程")
            print(f"{'='*60}")
            
            # 存储中间结果的变量
            results = {}
            
            for i, (step_name, step_func) in enumerate(steps):
                # 清晰的步骤指示
                print(f"\n📍 [{i+1}/{len(steps)}] {step_name}...")
                
                # 执行步骤
                step_result = step_func(results)
                if step_result is not None:
                    results.update(step_result)
                
                print(f"✅ {step_name} 完成")
            
            # 完成提示
            print(f"\n{'='*60}")
            print(f"🎉 训练流程全部完成！")
            print(f"{'='*60}")
            print(f"📂 训练结果已保存在以下位置:")
            print(f"   🔧 模型文件: {self.model_save_dir}")
            print(f"   📊 结果文件: {self.results_save_dir}")
            print(f"   📁 训练文件夹: {self.training_folder}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {str(e)}")
            raise
    
    # 将原来的步骤拆分为独立的方法（优化日志输出）
    def _step_load_data(self, results):
        """静默加载数据"""
        # 临时调整日志级别
        original_level = self.logger.level
        self.logger.setLevel(logging.WARNING)
        
        X_raw, y = self.load_data()
        
        # 恢复日志级别
        self.logger.setLevel(original_level)
        return {"X_raw": X_raw, "y": y}
    
    def _step_handle_outliers(self, results):
        """静默处理异常值"""
        original_level = self.logger.level
        self.logger.setLevel(logging.WARNING)
        
        X_raw, y = self.handle_outliers(results["X_raw"], results["y"])
        
        self.logger.setLevel(original_level)
        return {"X_raw": X_raw, "y": y}
    
    def _step_preprocess_data(self, results):
        """静默预处理数据"""
        original_level = self.logger.level
        self.logger.setLevel(logging.WARNING)
        
        X = self.preprocess_timeseries_data(results["X_raw"])
        
        self.logger.setLevel(original_level)
        return {"X": X}
    
    def _step_add_features(self, results):
        """静默添加技术特征"""
        original_level = self.logger.level
        self.logger.setLevel(logging.WARNING)
        
        X = self.add_technical_features(results["X"], results["X_raw"])
        
        self.logger.setLevel(original_level)
        return {"X": X}
    
    def _step_split_data(self, results):
        """静默分割数据"""
        original_level = self.logger.level
        self.logger.setLevel(logging.WARNING)
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(results["X"], results["y"])
        
        self.logger.setLevel(original_level)
        return {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test
        }
    
    def _step_normalize_data(self, results):
        """静默标准化数据"""
        original_level = self.logger.level
        self.logger.setLevel(logging.WARNING)
        
        X_train, X_val, X_test = self.normalize_data(
            results["X_train"], results["X_val"], results["X_test"]
        )
        
        self.logger.setLevel(original_level)
        return {"X_train": X_train, "X_val": X_val, "X_test": X_test}
    
    def _step_feature_selection(self, results):
        """静默特征选择"""
        original_level = self.logger.level
        self.logger.setLevel(logging.WARNING)
        
        X_train, X_val, X_test = self.feature_selection(
            results["X_train"], results["y_train"], results["X_val"], results["X_test"]
        )
        
        self.logger.setLevel(original_level)
        return {"X_train": X_train, "X_val": X_val, "X_test": X_test}
    
    def _step_hyperparameter_tuning(self, results):
        """执行超参数优化"""
        if self.config['hyperparameter_tuning']['enabled']:
            best_params = self.hyperparameter_tuning(results["X_train"], results["y_train"])
            self.config['lightgbm']['basic_params'].update(best_params)
        else:
            print("⚠️  超参数优化已禁用，使用默认参数")
        return None
    
    def _step_train_model(self, results):
        """训练模型"""
        self.model = self.train_model(
            results["X_train"], results["y_train"], results["X_val"], results["y_val"]
        )
        return None
    
    def _step_evaluate_model(self, results):
        """静默评估模型"""
        original_level = self.logger.level
        self.logger.setLevel(logging.WARNING)
        
        metrics, y_pred = self.evaluate_model(results["X_test"], results["y_test"])
        
        self.logger.setLevel(original_level)
        return {"metrics": metrics, "y_pred": y_pred}
    
    def _step_save_results(self, results):
        """静默保存结果"""
        original_level = self.logger.level
        self.logger.setLevel(logging.WARNING)
        
        self.save_model()
        self.save_results(results["metrics"], results["y_pred"], results["y_test"])
        
        self.logger.setLevel(original_level)
        return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LightGBM 股票预测模型训练')
    parser.add_argument('--config', type=str, default='config/train/lightGBM_train.yaml',
                      help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建训练器并运行
    trainer = LightGBMTrainer(args.config)
    trainer.run_training_pipeline()


if __name__ == '__main__':
    main()