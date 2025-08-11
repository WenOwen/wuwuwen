#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM 股票预测模型训练脚本
专门针对parquet格式股票数据的训练脚本
集成数据预处理和模型训练功能
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

# 导入自定义模块
from stock_data_processor import StockDataProcessor

# 抑制常见的警告信息
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


class StockLightGBMTrainer:
    """股票数据专用的LightGBM训练器"""
    
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
        self.stock_data_processor = None
        
        # 创建输出目录
        self._create_output_dirs()
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            raise
    
    def _create_training_folder(self) -> str:
        """创建当前训练的唯一文件夹"""
        file_naming = self.config.get('output', {}).get('file_naming', {})
        identifier_type = file_naming.get('identifier_type', 'unique_id')
        folder_prefix = file_naming.get('folder_name_prefix', 'stock_training')
        
        if identifier_type == 'timestamp':
            identifier = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:  # unique_id
            digits = file_naming.get('unique_id_digits', 3)
            # 查找现有文件夹，确定下一个ID
            model_dir = Path(self.config.get('output', {}).get('model_save', {}).get('save_dir', './models/lightgbm_stock'))
            existing_folders = []
            if model_dir.exists():
                existing_folders = [d.name for d in model_dir.iterdir() 
                                  if d.is_dir() and d.name.startswith(folder_prefix)]
            
            # 提取已存在的ID号
            max_id = 0
            for folder in existing_folders:
                try:
                    id_part = folder.replace(folder_prefix + '_', '')
                    if id_part.isdigit():
                        max_id = max(max_id, int(id_part))
                except:
                    continue
            
            identifier = str(max_id + 1).zfill(digits)
        
        training_folder = f"{folder_prefix}_{identifier}"
        return training_folder
    
    def setup_logging(self):
        """设置日志"""
        log_config = self.config.get('output', {}).get('logging', {})
        log_level = log_config.get('log_level', 'INFO')
        log_file = log_config.get('log_file', './logs/lightgbm_stock_training.log')
        
        # 创建日志目录
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置日志
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler() if log_config.get('console_output', True) else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 开始股票LightGBM训练: {self.training_folder}")
    
    def _create_output_dirs(self):
        """创建输出目录"""
        # 模型保存目录
        model_save_config = self.config.get('output', {}).get('model_save', {})
        self.model_save_dir = Path(model_save_config.get('save_dir', './models/lightgbm_stock')) / self.training_folder
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 结果保存目录
        results_save_config = self.config.get('output', {}).get('results_save', {})
        self.results_save_dir = Path(results_save_config.get('save_dir', './results/lightgbm_stock')) / self.training_folder
        self.results_save_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_stock_data(self) -> bool:
        """预处理股票数据"""
        try:
            # 检查是否启用直接训练模式
            data_config = self.config.get('data', {})
            direct_training = data_config.get('direct_training', {})
            
            if direct_training.get('enabled', False):
                self.logger.info("🎯 使用直接训练模式，跳过数据预处理")
                return True
            
            self.logger.info("📊 开始股票数据预处理...")
            
            # 获取数据配置
            source_data_config = data_config.get('source_data', {})
            
            # 检查是否需要自动处理数据
            if source_data_config.get('auto_process', True):
                parquet_dir = source_data_config.get('parquet_dir', './data/professional_parquet')
                output_dir = data_config.get('data_dir', './data/processed_stock_data')
                
                # 检查是否已有处理好的数据
                processed_dir = Path(output_dir)
                if processed_dir.exists() and any(processed_dir.glob('processed_*')):
                    self.logger.info("✅ 发现已处理的数据，跳过预处理步骤")
                    return True
                
                # 创建股票数据处理器
                self.stock_data_processor = StockDataProcessor(
                    data_dir=parquet_dir,
                    output_dir=output_dir
                )
                
                # 获取股票特定配置
                stock_config = data_config.get('stock_specific', {})
                time_series_config = stock_config.get('time_series', {})
                
                # 运行数据处理流程
                processed_path = self.stock_data_processor.run_full_pipeline(
                    lookback_days=time_series_config.get('lookback_days', 5),
                    target_days=time_series_config.get('target_days', 1)
                )
                
                if processed_path:
                    # 更新配置中的数据路径
                    self.config['data']['data_dir'] = processed_path
                    self.logger.info(f"✅ 数据预处理完成: {processed_path}")
                    return True
                else:
                    self.logger.error("❌ 数据预处理失败")
                    return False
            else:
                self.logger.info("⏭️ 跳过自动数据预处理")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 数据预处理失败: {e}")
            return False
    
    def load_data(self) -> bool:
        """加载数据"""
        try:
            self.logger.info("📂 加载股票训练数据...")
            
            data_config = self.config.get('data', {})
            direct_training = data_config.get('direct_training', {})
            
            # 检查是否使用直接训练模式
            if direct_training.get('enabled', False):
                return self._load_direct_data()
            
            # 原有的加载逻辑
            data_dir = Path(data_config.get('data_dir', './data/processed_stock_data'))
            
            # 如果data_dir是目录，查找最新的processed_*文件夹
            if data_dir.is_dir() and not (data_dir / data_config.get('X_features_file', 'X_features.csv')).exists():
                processed_folders = list(data_dir.glob('processed_*'))
                if processed_folders:
                    # 选择最新的文件夹
                    data_dir = max(processed_folders, key=lambda x: x.stat().st_mtime)
                    self.logger.info(f"   使用最新的数据文件夹: {data_dir}")
            
            # 文件路径
            X_features_file = data_dir / data_config.get('X_features_file', 'X_features.csv')
            y_targets_file = data_dir / data_config.get('y_targets_file', 'y_targets.csv')
            full_data_file = data_dir / data_config.get('full_data_file', 'full_data.csv')
            
            loading_options = data_config.get('loading_options', {})
            prefer_full_data = loading_options.get('prefer_full_data', True)
            encoding = loading_options.get('encoding', 'utf-8')
            
            # 加载数据
            if prefer_full_data and full_data_file.exists():
                self.logger.info("   加载完整数据文件...")
                full_data = pd.read_csv(full_data_file, encoding=encoding)
                
                # 分离特征和目标
                exclude_cols = ['stock_code', 'date', 'target']
                feature_cols = [col for col in full_data.columns if col not in exclude_cols]
                
                self.X = full_data[feature_cols]
                self.y = full_data['target']
                self.stock_info = full_data[['stock_code']].copy() if 'stock_code' in full_data.columns else None
                
            else:
                self.logger.info("   分别加载特征和目标文件...")
                self.X = pd.read_csv(X_features_file, encoding=encoding)
                self.y = pd.read_csv(y_targets_file, encoding=encoding)
                if hasattr(self.y, 'iloc'):
                    self.y = self.y.iloc[:, 0]  # 取第一列作为目标
                self.stock_info = None
            
            # 保存特征名称
            self.feature_names = list(self.X.columns)
            
            self.logger.info(f"   ✅ 数据加载完成:")
            self.logger.info(f"     - 特征维度: {self.X.shape}")
            self.logger.info(f"     - 目标维度: {self.y.shape}")
            self.logger.info(f"     - 特征数量: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 数据加载失败: {e}")
            return False
    
    def _load_direct_data(self) -> bool:
        """直接加载parquet格式的股票数据 - 简化版"""
        try:
            self.logger.info("📊 使用直接训练模式加载数据...")
            
            data_config = self.config.get('data', {})
            direct_training = data_config.get('direct_training', {})
            
            data_dir = Path(data_config.get('data_dir', './data/professional_parquet'))
            data_format = direct_training.get('data_format', 'parquet')
            target_column = direct_training.get('target_column', '涨跌幅')
            exclude_columns = direct_training.get('exclude_columns', ['name', '涨跌幅'])
            
            # 加载数据文件
            if data_format == 'parquet':
                # 查找parquet文件
                parquet_files = list(data_dir.glob("*.parquet"))
                if not parquet_files:
                    self.logger.error(f"❌ 在{data_dir}中未找到parquet文件")
                    return False
                
                self.logger.info(f"   发现 {len(parquet_files)} 个parquet文件")
                
                # 使用简单高效的文件配对方案
                parquet_files = sorted(parquet_files)  # 按日期排序
                self.logger.info("   📅 使用文件配对方案：今天文件 → 明天目标")
                
                features_list = []
                targets_list = []
                processed_pairs = 0
                
                # 相邻文件配对
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
                        continue
                
                if not features_list:
                    self.logger.error("❌ 没有成功配对任何文件")
                    return False
                
                # 合并所有配对的数据
                self.logger.info(f"   🔄 合并 {processed_pairs} 个文件配对的数据...")
                full_data = pd.concat(features_list, ignore_index=False)
                targets_data = pd.concat(targets_list, ignore_index=False)
                
                # 添加目标列
                full_data['next_day_target'] = targets_data
                
                self.logger.info(f"   ✅ 文件配对完成:")
                self.logger.info(f"   - 处理文件对: {processed_pairs}")
                self.logger.info(f"   - 最终样本数: {len(full_data):,}")
                self.logger.info(f"   - 特征列数: {len(full_data.columns)}")
                
            else:
                self.logger.error(f"❌ 不支持的数据格式: {data_format}")
                return False
            
            # 检查次日预测目标列是否存在
            if 'next_day_target' not in full_data.columns:
                self.logger.error(f"❌ 未找到次日预测目标列 'next_day_target'")
                return False
            
            # 设置目标变量（明天的涨跌幅）
            self.y = full_data['next_day_target']
            actual_target_column = 'next_day_target'
            
            # 排除目标列和辅助列，保留今天的涨跌幅作为特征
            exclude_columns = exclude_columns + ['next_day_target']
            self.logger.info(f"   💡 今天的'{target_column}'用作预测明天涨跌幅的特征")
            
            # 选择特征列（排除指定的列）
            feature_columns = [col for col in full_data.columns if col not in exclude_columns]
            self.X = full_data[feature_columns]
            
            # 只保留数值列作为特征
            numeric_columns = self.X.select_dtypes(include=[np.number]).columns
            self.X = self.X[numeric_columns]
            
            self.logger.info(f"   📋 排除的列: {exclude_columns}")
            self.logger.info(f"   📊 数值特征列数: {len(numeric_columns)}")
            
            # 处理缺失值
            self.X = self.X.fillna(0)
            self.y = self.y.fillna(0)
            
            # 保存特征名称
            self.feature_names = list(self.X.columns)
            
            # 保存股票信息
            stock_name_column = direct_training.get('stock_name_column', 'name')
            if stock_name_column in full_data.columns:
                self.stock_info = full_data[[stock_name_column]].copy()
            else:
                self.stock_info = None
            
            self.logger.info(f"   ✅ 次日预测数据加载完成:")
            self.logger.info(f"     - 特征维度: {self.X.shape}")
            self.logger.info(f"     - 目标维度: {self.y.shape}")
            self.logger.info(f"     - 特征数量: {len(self.feature_names)}")
            self.logger.info(f"     - 目标列: {actual_target_column}")
            self.logger.info(f"     - 预测任务: 今天特征 → 明天涨跌幅")
            self.logger.info(f"     - 目标值范围: [{self.y.min():.4f}, {self.y.max():.4f}]")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 直接数据加载失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False


    def split_data(self) -> bool:
        """分割数据"""
        try:
            self.logger.info("✂️ 分割训练/验证/测试数据...")
            
            split_config = self.config.get('training', {}).get('data_split', {})
            test_size = split_config.get('test_size', 0.2)
            validation_size = split_config.get('validation_size', 0.1)
            random_state = split_config.get('random_state', 42)
            time_series_split = split_config.get('time_series_split', True)
            
            if time_series_split:
                # 时序分割（股票数据的推荐方式）
                n_samples = len(self.X)
                test_start = int(n_samples * (1 - test_size))
                val_start = int(n_samples * (1 - test_size - validation_size))
                
                # 分割数据
                self.X_train = self.X.iloc[:val_start]
                self.X_val = self.X.iloc[val_start:test_start]
                self.X_test = self.X.iloc[test_start:]
                
                self.y_train = self.y.iloc[:val_start]
                self.y_val = self.y.iloc[val_start:test_start]
                self.y_test = self.y.iloc[test_start:]
                
                self.logger.info("   使用时序分割方式")
                
            else:
                # 随机分割
                X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                    self.X, self.y, test_size=test_size, random_state=random_state
                )
                
                val_size_adjusted = validation_size / (1 - test_size)
                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
                )
                
                self.logger.info("   使用随机分割方式")
            
            self.logger.info(f"   ✅ 数据分割完成:")
            self.logger.info(f"     - 训练集: {self.X_train.shape[0]} 样本")
            self.logger.info(f"     - 验证集: {self.X_val.shape[0]} 样本")
            self.logger.info(f"     - 测试集: {self.X_test.shape[0]} 样本")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 数据分割失败: {e}")
            return False
    
    def preprocess_features(self) -> bool:
        """特征预处理"""
        try:
            self.logger.info("🔧 特征预处理...")
            
            preprocessing_config = self.config.get('data', {}).get('preprocessing', {})
            normalization_config = preprocessing_config.get('normalization', {})
            outlier_config = preprocessing_config.get('outlier_handling', {})
            
            # 数据标准化
            method = normalization_config.get('method', 'robust')
            if method:
                if method == 'standard':
                    self.scaler = StandardScaler()
                elif method == 'minmax':
                    self.scaler = MinMaxScaler()
                elif method == 'robust':
                    self.scaler = RobustScaler()
                else:
                    self.logger.warning(f"未知的标准化方法: {method}，跳过标准化")
                    self.scaler = None
                
                if self.scaler:
                    self.X_train = pd.DataFrame(
                        self.scaler.fit_transform(self.X_train),
                        columns=self.X_train.columns,
                        index=self.X_train.index
                    )
                    self.X_val = pd.DataFrame(
                        self.scaler.transform(self.X_val),
                        columns=self.X_val.columns,
                        index=self.X_val.index
                    )
                    self.X_test = pd.DataFrame(
                        self.scaler.transform(self.X_test),
                        columns=self.X_test.columns,
                        index=self.X_test.index
                    )
                    
                    self.logger.info(f"   ✅ 使用 {method} 标准化")
            
            # 异常值处理
            if outlier_config.get('enabled', False):
                method = outlier_config.get('method', 'winsorize')
                if method == 'winsorize':
                    limits = outlier_config.get('winsorize_limits', [0.01, 0.01])
                    from scipy.stats.mstats import winsorize
                    
                    # 只对训练集进行winsorize，验证集和测试集使用相同的限制
                    for col in self.X_train.columns:
                        self.X_train[col] = winsorize(self.X_train[col], limits=limits)
                    
                    self.logger.info(f"   ✅ 使用 winsorize 处理异常值，限制: {limits}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 特征预处理失败: {e}")
            return False
    
    def train_model(self) -> bool:
        """训练模型"""
        try:
            self.logger.info("🎯 开始训练LightGBM模型...")
            
            # 获取模型参数
            lgb_config = self.config.get('lightgbm', {})
            basic_params = lgb_config.get('basic_params', {})
            advanced_params = lgb_config.get('advanced_params', {})
            fit_params = lgb_config.get('fit_params', {})
            
            # 合并参数
            model_params = {**basic_params, **advanced_params}
            
            # 训练参数
            training_config = self.config.get('training', {}).get('training_params', {})
            early_stopping_rounds = training_config.get('early_stopping_rounds', 100)
            verbose = training_config.get('verbose', 100)
            eval_metric = training_config.get('eval_metric', ['rmse'])
            
            # 创建数据集
            train_data = lgb.Dataset(self.X_train, label=self.y_train)
            val_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)
            
            # 训练模型
            self.logger.info("   开始训练...")
            
            callbacks = [
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(verbose)
            ]
            
            self.model = lgb.train(
                model_params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                num_boost_round=fit_params.get('num_boost_round', 1000),
                callbacks=callbacks
            )
            
            self.logger.info("   ✅ 模型训练完成")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 模型训练失败: {e}")
            return False
    
    def evaluate_model(self) -> Dict:
        """评估模型"""
        try:
            self.logger.info("📊 评估模型性能...")
            
            # 获取预测结果
            y_train_pred = self.model.predict(self.X_train)
            y_val_pred = self.model.predict(self.X_val)
            y_test_pred = self.model.predict(self.X_test)
            
            # 计算评估指标
            eval_config = self.config.get('evaluation', {})
            metrics_list = eval_config.get('metrics', ['rmse', 'mae', 'r2_score'])
            
            results = {}
            
            for split, y_true, y_pred in [
                ('train', self.y_train, y_train_pred),
                ('val', self.y_val, y_val_pred),
                ('test', self.y_test, y_test_pred)
            ]:
                split_metrics = {}
                
                for metric in metrics_list:
                    if metric == 'rmse':
                        value = np.sqrt(mean_squared_error(y_true, y_pred))
                    elif metric == 'mae':
                        value = mean_absolute_error(y_true, y_pred)
                    elif metric == 'mape':
                        value = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
                    elif metric == 'r2_score':
                        value = r2_score(y_true, y_pred)
                    elif metric == 'explained_variance':
                        value = explained_variance_score(y_true, y_pred)
                    elif metric == 'directional_accuracy':
                        # 方向准确率（股票预测特有指标）
                        direction_true = np.sign(y_true)
                        direction_pred = np.sign(y_pred)
                        value = np.mean(direction_true == direction_pred) * 100
                    else:
                        continue
                    
                    split_metrics[metric] = float(value)
                
                results[split] = split_metrics
            
            # 输出结果
            self.logger.info("   📈 评估结果:")
            for split, metrics in results.items():
                self.logger.info(f"     {split.upper()}:")
                for metric, value in metrics.items():
                    if metric in ['mape', 'directional_accuracy']:
                        self.logger.info(f"       {metric}: {value:.2f}%")
                    else:
                        self.logger.info(f"       {metric}: {value:.6f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 模型评估失败: {e}")
            return {}
    
    def save_model(self) -> bool:
        """保存模型"""
        try:
            self.logger.info("💾 保存模型和相关文件...")
            
            model_config = self.config.get('output', {}).get('model_save', {})
            model_name = model_config.get('model_name', 'lightgbm_stock_model')
            save_formats = model_config.get('save_format', ['pkl'])
            
            # 保存模型
            for fmt in save_formats:
                if fmt == 'pkl':
                    model_path = self.model_save_dir / f"{model_name}.pkl"
                    joblib.dump(self.model, model_path)
                    self.logger.info(f"   ✅ 模型已保存: {model_path}")
                elif fmt == 'txt':
                    model_path = self.model_save_dir / f"{model_name}.txt"
                    self.model.save_model(str(model_path))
                    self.logger.info(f"   ✅ 模型已保存: {model_path}")
            
            # 保存标准化器
            if self.scaler:
                scaler_path = self.model_save_dir / "scaler.pkl"
                joblib.dump(self.scaler, scaler_path)
                self.logger.info(f"   ✅ 标准化器已保存: {scaler_path}")
            
            # 保存特征名称
            feature_info = {
                'feature_names': self.feature_names,
                'feature_count': len(self.feature_names),
                'training_config': self.config_path
            }
            
            with open(self.model_save_dir / "feature_names.json", 'w', encoding='utf-8') as f:
                json.dump(feature_info, f, ensure_ascii=False, indent=2)
            
            with open(self.model_save_dir / "feature_names_simple.txt", 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.feature_names))
            
            self.logger.info(f"   ✅ 特征信息已保存")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 保存模型失败: {e}")
            return False
    
    def save_results(self, evaluation_results: Dict) -> bool:
        """保存结果"""
        try:
            self.logger.info("📊 保存训练结果...")
            
            results_config = self.config.get('output', {}).get('results_save', {})
            
            # 保存评估指标
            if results_config.get('save_metrics', True):
                metrics_path = self.results_save_dir / "metrics.json"
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
                self.logger.info(f"   ✅ 评估指标已保存: {metrics_path}")
            
            # 保存预测结果
            if results_config.get('save_predictions', True):
                predictions = {
                    'y_train_true': self.y_train.tolist(),
                    'y_train_pred': self.model.predict(self.X_train).tolist(),
                    'y_val_true': self.y_val.tolist(),
                    'y_val_pred': self.model.predict(self.X_val).tolist(),
                    'y_test_true': self.y_test.tolist(),
                    'y_test_pred': self.model.predict(self.X_test).tolist()
                }
                
                pred_df = pd.DataFrame({
                    'split': (['train'] * len(self.y_train) + 
                             ['val'] * len(self.y_val) + 
                             ['test'] * len(self.y_test)),
                    'y_true': (self.y_train.tolist() + 
                              self.y_val.tolist() + 
                              self.y_test.tolist()),
                    'y_pred': (predictions['y_train_pred'] + 
                              predictions['y_val_pred'] + 
                              predictions['y_test_pred'])
                })
                
                pred_path = self.results_save_dir / "predictions.csv"
                pred_df.to_csv(pred_path, index=False, encoding='utf-8')
                self.logger.info(f"   ✅ 预测结果已保存: {pred_path}")
            
            # 保存特征重要性
            if results_config.get('save_feature_importance', True):
                importance = self.model.feature_importance(importance_type='gain')
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                importance_path = self.results_save_dir / "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False, encoding='utf-8')
                
                # 保存前20个重要特征
                top_features_path = self.results_save_dir / "top_features.txt"
                with open(top_features_path, 'w', encoding='utf-8') as f:
                    f.write("前20个最重要特征:\n")
                    f.write("=" * 50 + "\n")
                    for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
                        f.write(f"{i+1:2d}. {row['feature']}: {row['importance']:.6f}\n")
                
                self.logger.info(f"   ✅ 特征重要性已保存: {importance_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 保存结果失败: {e}")
            return False
    
    def run_training_pipeline(self) -> bool:
        """运行完整的训练流程"""
        try:
            self.logger.info("🚀 开始股票LightGBM训练流程...")
            
            # 1. 数据预处理
            if not self.preprocess_stock_data():
                return False
            
            # 2. 加载数据
            if not self.load_data():
                return False
            
            # 3. 分割数据
            if not self.split_data():
                return False
            
            # 4. 特征预处理
            if not self.preprocess_features():
                return False
            
            # 5. 训练模型
            if not self.train_model():
                return False
            
            # 6. 评估模型
            evaluation_results = self.evaluate_model()
            if not evaluation_results:
                return False
            
            # 7. 保存模型
            if not self.save_model():
                return False
            
            # 8. 保存结果
            if not self.save_results(evaluation_results):
                return False
            
            self.logger.info("🎉 训练流程完成!")
            self.logger.info(f"📁 模型保存路径: {self.model_save_dir}")
            self.logger.info(f"📁 结果保存路径: {self.results_save_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 训练流程失败: {e}")
            return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LightGBM股票预测模型训练')
    parser.add_argument('--config', type=str, 
                       default='config/train/lightGBM_stock_train.yaml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 LightGBM股票预测模型训练脚本")
    print("=" * 60)
    
    # 检查配置文件是否存在
    if not Path(args.config).exists():
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 创建训练器
    try:
        trainer = StockLightGBMTrainer(args.config)
    except Exception as e:
        print(f"❌ 创建训练器失败: {e}")
        sys.exit(1)
    
    # 运行训练流程
    success = trainer.run_training_pipeline()
    
    if success:
        print("\n🎉 训练成功完成!")
        print(f"📁 模型文件: {trainer.model_save_dir}")
        print(f"📁 结果文件: {trainer.results_save_dir}")
        sys.exit(0)
    else:
        print("\n❌ 训练失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()