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
import itertools
import copy
import random
from collections import defaultdict

# 导入字体配置模块
try:
    from font_config import setup_chinese_plot
    setup_chinese_plot()  # 设置中文字体
    print("✅ 中文字体配置已加载")
except ImportError as e:
    print(f"⚠️ 字体配置模块导入失败: {e}")
    # 使用备用字体配置
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 使用备用中文字体配置")
except Exception as e:
    print(f"⚠️ 字体配置失败: {e}")

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


class BuiltinVisualizer:
    """内置可视化器，替代外部可视化模块"""
    
    def __init__(self, output_dir, logger=None):
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.training_history = {
            'iteration': [],
            'train_loss': [],
            'val_loss': []
        }
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def record_callback(self):
        """返回训练记录回调函数"""
        def callback(env):
            if env.evaluation_result_list:
                iteration = env.iteration
                self.training_history['iteration'].append(iteration)
                
                for eval_result in env.evaluation_result_list:
                    dataset_name, eval_name, result, is_higher_better = eval_result
                    if dataset_name == 'train':
                        self.training_history['train_loss'].append(result)
                    elif dataset_name == 'val':
                        self.training_history['val_loss'].append(result)
        
        return callback
    
    def plot_learning_curves(self, model=None):
        """绘制学习曲线"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            
            # 确保中文字体正确设置
            if 'Microsoft YaHei' not in plt.rcParams['font.sans-serif']:
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei'] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if self.training_history['iteration']:
                ax.plot(self.training_history['iteration'], 
                       self.training_history['train_loss'], 
                       label='训练损失', linewidth=2, color='blue')
                ax.plot(self.training_history['iteration'], 
                       self.training_history['val_loss'], 
                       label='验证损失', linewidth=2, color='red')
                
                ax.set_xlabel('训练轮数')
                ax.set_ylabel('损失值')
                ax.set_title('LightGBM 学习曲线')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 保存图表
                save_path = self.output_dir / 'learning_curves.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                if self.logger:
                    self.logger.info(f"   📈 学习曲线已保存: {save_path}")
                
                return str(save_path)
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"   ⚠️ 学习曲线绘制失败: {e}")
            return None
    
    def plot_feature_importance(self, model, feature_names, top_n=20):
        """绘制特征重要性图"""
        try:
            import matplotlib.pyplot as plt
            
            # 确保中文字体正确设置
            if 'Microsoft YaHei' not in plt.rcParams['font.sans-serif']:
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei'] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
            
            # 获取特征重要性
            importance = model.feature_importance(importance_type='gain')
            feature_importance = list(zip(feature_names, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # 取前N个重要特征
            top_features = feature_importance[:top_n]
            features, scores = zip(*top_features)
            
            # 创建水平条形图
            fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
            
            y_pos = range(len(features))
            bars = ax.barh(y_pos, scores, color='skyblue', alpha=0.8)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()  # 重要性高的在上面
            ax.set_xlabel('重要性得分')
            ax.set_title(f'前 {top_n} 个最重要特征')
            ax.grid(True, alpha=0.3, axis='x')
            
            # 在条形图上添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.0f}', ha='left', va='center', fontsize=8)
            
            # 保存图表
            save_path = self.output_dir / 'feature_importance.png'
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            if self.logger:
                self.logger.info(f"   📊 特征重要性图已保存: {save_path}")
            
            return str(save_path)
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"   ⚠️ 特征重要性图绘制失败: {e}")
            return None
    
    def plot_predictions_scatter(self, y_true, y_pred, split_name='测试集'):
        """绘制预测值vs真实值散点图"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 确保中文字体正确设置
            if 'Microsoft YaHei' not in plt.rcParams['font.sans-serif']:
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei'] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # 绘制散点图
            ax.scatter(y_true, y_pred, alpha=0.6, s=10, color='blue')
            
            # 绘制理想预测线 (y=x)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')
            
            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            ax.set_title(f'{split_name} - 预测值 vs 真实值')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加相关系数
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            ax.text(0.05, 0.95, f'相关系数: {correlation:.4f}', 
                   transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # 保存图表
            save_path = self.output_dir / f'predictions_scatter_{split_name}.png'
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            if self.logger:
                self.logger.info(f"   📊 预测散点图已保存: {save_path}")
            
            return str(save_path)
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"   ⚠️ 预测散点图绘制失败: {e}")
            return None
    
    def plot_residuals(self, y_true, y_pred, split_name='测试集'):
        """绘制残差图"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 确保中文字体正确设置
            if 'Microsoft YaHei' not in plt.rcParams['font.sans-serif']:
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei'] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
            
            residuals = y_true - y_pred
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 残差vs预测值图
            ax1.scatter(y_pred, residuals, alpha=0.6, s=10, color='green')
            ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax1.set_xlabel('预测值')
            ax1.set_ylabel('残差 (真实值 - 预测值)')
            ax1.set_title(f'{split_name} - 残差分析')
            ax1.grid(True, alpha=0.3)
            
            # 残差分布直方图
            ax2.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('残差值')
            ax2.set_ylabel('频次')
            ax2.set_title(f'{split_name} - 残差分布')
            ax2.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            ax2.text(0.05, 0.95, f'均值: {mean_residual:.4f}\n标准差: {std_residual:.4f}', 
                    transform=ax2.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # 保存图表
            save_path = self.output_dir / f'residuals_analysis_{split_name}.png'
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            if self.logger:
                self.logger.info(f"   📊 残差分析图已保存: {save_path}")
            
            return str(save_path)
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"   ⚠️ 残差分析图绘制失败: {e}")
            return None
    
    def generate_all_visualizations(self, model, y_train, y_val, 
                                   y_train_pred, y_val_pred):
        """生成所有可视化图表 - 只使用训练集和验证集"""
        results = {}
        
        # 学习曲线
        learning_curve_path = self.plot_learning_curves(model)
        if learning_curve_path:
            results['learning_curves'] = learning_curve_path
        
        # 特征重要性
        if hasattr(model, 'feature_importance'):
            # 尝试获取特征名称，如果没有则生成默认名称
            if hasattr(self, 'feature_names') and self.feature_names:
                feature_names = self.feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(model.num_feature())]
            
            feature_importance_path = self.plot_feature_importance(model, feature_names)
            if feature_importance_path:
                results['feature_importance'] = feature_importance_path
        
        # 预测散点图 - 只生成训练集和验证集
        for split_name, y_true, y_pred in [
            ('训练集', y_train, y_train_pred),
            ('验证集', y_val, y_val_pred)
        ]:
            scatter_path = self.plot_predictions_scatter(y_true, y_pred, split_name)
            if scatter_path:
                results[f'predictions_scatter_{split_name}'] = scatter_path
            
            residuals_path = self.plot_residuals(y_true, y_pred, split_name)
            if residuals_path:
                results[f'residuals_{split_name}'] = residuals_path
        
        return results


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
        self.visualizer = None  # 可视化器
        
        # 创建输出目录
        self._create_output_dirs()
        
        # 训练历史记录
        self.training_history = {}
        
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
    
    def _create_builtin_visualizer(self):
        """创建内置可视化器"""
        return BuiltinVisualizer(self.results_save_dir, self.logger)
    
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
                
                # 保存股票相关信息用于预测结果
                info_cols = []
                if 'stock_code' in full_data.columns:
                    info_cols.append('stock_code')
                if 'date' in full_data.columns:
                    info_cols.append('date')
                
                # 尝试查找更多可能的列
                for col in full_data.columns:
                    if '股票名称' in col or 'stock_name' in col or '名称' in col:
                        info_cols.append(col)
                    elif '次日涨跌幅' in col or 'next_day_return' in col:
                        info_cols.append(col)
                
                self.stock_info = full_data[info_cols].copy() if info_cols else None
                if self.stock_info is not None:
                    self.logger.info(f"   ✅ 保存股票信息列: {list(self.stock_info.columns)}")
                
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
                file_dates_list = []  # 存储每个样本对应的日期信息
                processed_pairs = 0
                
                # 相邻文件配对
                for i in range(len(parquet_files) - 1):
                    today_file = parquet_files[i]      # 今天的特征
                    tomorrow_file = parquet_files[i+1]  # 明天的目标
                    
                    try:
                        # 从文件名中提取日期信息
                        today_date = today_file.stem  # 移除.parquet扩展名
                        tomorrow_date = tomorrow_file.stem
                        
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
                            
                            # 记录每个股票样本的日期信息
                            date_info = pd.DataFrame({
                                'feature_date': [today_date] * len(common_stocks),
                                'target_date': [tomorrow_date] * len(common_stocks),
                                'stock_code': common_stocks.tolist()
                            }, index=common_stocks)  # 使用股票代码作为索引
                            file_dates_list.append(date_info)
                            
                            processed_pairs += 1
                            
                        self.logger.info(f"   ✅ 配对: {today_file.name} → {tomorrow_file.name}, 股票: {len(common_stocks)}")
                        
                    except Exception as e:
                        self.logger.warning(f"   跳过配对 {today_file.name} → {tomorrow_file.name}: {e}")
                        continue
                
                if not features_list:
                    self.logger.error("❌ 没有成功配对任何文件")
                    return False
                
                # 合并所有配对的数据（内存优化）
                self.logger.info(f"   🔄 合并 {processed_pairs} 个文件配对的数据（内存优化）...")
                
                # 内存优化：分批合并以减少峰值内存使用
                import gc
                self.logger.info("   🧹 清理内存...")
                gc.collect()
                
                # 合并特征数据
                self.logger.info("   📊 合并特征数据...")
                full_data = pd.concat(features_list, ignore_index=False, copy=False)
                del features_list  # 立即释放内存
                gc.collect()
                
                # 合并目标数据
                self.logger.info("   🎯 合并目标数据...")
                targets_data = pd.concat(targets_list, ignore_index=False, copy=False)
                del targets_list  # 立即释放内存
                gc.collect()
                
                # 合并日期信息
                if file_dates_list:
                    self.logger.info("   📅 合并日期信息...")
                    all_dates_info = pd.concat(file_dates_list, ignore_index=False, copy=False)
                    del file_dates_list  # 立即释放内存
                    gc.collect()
                    self.logger.info(f"   ✅ 合并日期信息: {len(all_dates_info)} 条记录")
                    self.logger.info(f"   📋 日期信息索引示例: {list(all_dates_info.index[:5])}")
                else:
                    all_dates_info = None
                
                # 添加目标列
                self.logger.info("   🎯 添加目标列...")
                full_data['next_day_target'] = targets_data
                del targets_data  # 立即释放内存
                gc.collect()
                
                self.logger.info(f"   ✅ 文件配对完成:")
                self.logger.info(f"   - 处理文件对: {processed_pairs}")
                self.logger.info(f"   - 最终样本数: {len(full_data):,}")
                self.logger.info(f"   - 特征列数: {len(full_data.columns)}")
                self.logger.info(f"   📋 full_data索引示例: {list(full_data.index[:5])}")
                
            else:
                self.logger.error(f"❌ 不支持的数据格式: {data_format}")
                return False
            
            # 检查次日预测目标列是否存在
            if 'next_day_target' not in full_data.columns:
                self.logger.error(f"❌ 未找到次日预测目标列 'next_day_target'")
                return False
            
            # 检查预测模式
            prediction_mode = direct_training.get('prediction_mode', 'regression')
            self.prediction_mode = prediction_mode  # 保存预测模式
            
            # 设置目标变量
            raw_targets = full_data['next_day_target']
            
            if prediction_mode == 'direction':
                # 🎯 方向预测模式：涨跌幅 > 0 为看多(1)，<= 0 为看空(0)
                self.y = (raw_targets > 0).astype(int)
                actual_target_column = 'next_day_direction'
                self.logger.info(f"   🎯 预测模式: 涨跌方向预测（二分类）")
                self.logger.info(f"   📊 看多样本: {(self.y == 1).sum():,} ({(self.y == 1).mean()*100:.1f}%)")
                self.logger.info(f"   📊 看空样本: {(self.y == 0).sum():,} ({(self.y == 0).mean()*100:.1f}%)")
            else:
                # 📈 回归预测模式：预测具体涨跌幅
                self.y = raw_targets
                actual_target_column = 'next_day_target'
                self.logger.info(f"   📈 预测模式: 涨跌幅预测（回归）")
                self.logger.info(f"   📊 目标值范围: [{self.y.min():.4f}, {self.y.max():.4f}]")            
            # 排除目标列和辅助列，保留今天的涨跌幅作为特征
            exclude_columns = exclude_columns + ['next_day_target']
            self.logger.info(f"   💡 今天的'{target_column}'用作预测明天涨跌幅的特征")
            
            # 选择特征列（排除指定的列）
            feature_columns = [col for col in full_data.columns if col not in exclude_columns]
            self.X = full_data[feature_columns]
            
            # 只保留数值列作为特征（内存优化版本）
            self.logger.info("   🔧 筛选数值列（内存优化处理）...")
            numeric_columns = []
            for col in self.X.columns:
                if pd.api.types.is_numeric_dtype(self.X[col]):
                    numeric_columns.append(col)
            
            # 如果需要筛选列，使用内存友好的方式
            if len(numeric_columns) < len(self.X.columns):
                # 逐列筛选，避免大内存分配
                self.X = self.X[numeric_columns].copy()
            
            numeric_columns = list(self.X.columns)  # 更新列列表
            
            self.logger.info(f"   📋 排除的列: {exclude_columns}")
            self.logger.info(f"   📊 数值特征列数: {len(numeric_columns)}")
            
            # 处理缺失值（内存优化）
            self.logger.info("   🔧 处理缺失值（内存优化）...")
            self.X.fillna(0, inplace=True)  # 使用inplace避免创建副本
            self.y.fillna(0, inplace=True)
            
            # 内存优化：转换为float32降低内存使用
            self.logger.info("   🔧 优化数据类型（float64 → float32）...")
            for col in self.X.columns:
                if self.X[col].dtype == 'float64':
                    self.X[col] = self.X[col].astype('float32')
            
            if self.y.dtype == 'float64':
                self.y = self.y.astype('float32')
            
            # 保存特征名称
            self.feature_names = list(self.X.columns)
            
            # 为直接训练模式保存股票信息
            # 构建包含股票代码、日期和收益率的完整信息
            stock_info_data = {
                'stock_code': full_data.index.tolist(),
                'next_day_return': raw_targets.tolist()  # 保存次日涨跌幅
            }
            
            # 添加日期信息（从文件名中提取）
            if all_dates_info is not None:
                # 直接使用位置索引匹配，因为all_dates_info和full_data按相同顺序合并
                try:
                    self.logger.info(f"   🔍 full_data行数: {len(full_data)}")
                    self.logger.info(f"   🔍 all_dates_info行数: {len(all_dates_info)}")
                    
                    if len(full_data) == len(all_dates_info):
                        # 行数匹配，直接按位置对应
                        all_dates_reset = all_dates_info.reset_index(drop=True)
                        stock_info_data['feature_date'] = all_dates_reset['feature_date'].astype(str).tolist()
                        stock_info_data['target_date'] = all_dates_reset['target_date'].astype(str).tolist()
                        
                        # 统计日期分布
                        unique_feature_dates = set(stock_info_data['feature_date'])
                        unique_target_dates = set(stock_info_data['target_date'])
                        
                        self.logger.info(f"   ✅ 添加日期信息: feature_date, target_date")
                        self.logger.info(f"   📅 唯一特征日期数: {len(unique_feature_dates)}")
                        self.logger.info(f"   📅 唯一目标日期数: {len(unique_target_dates)}")
                        self.logger.info(f"   📅 特征日期示例: {list(unique_feature_dates)[:5]}")
                        self.logger.info(f"   📅 目标日期示例: {list(unique_target_dates)[:5]}")
                        
                    else:
                        self.logger.warning(f"   ⚠️  数据行数不匹配: full_data={len(full_data)}, all_dates_info={len(all_dates_info)}")
                        # 添加默认值
                        stock_info_data['feature_date'] = ['unknown'] * len(full_data)
                        stock_info_data['target_date'] = ['unknown'] * len(full_data)
                    
                except Exception as e:
                    self.logger.warning(f"   ⚠️  日期信息合并失败: {e}")
                    import traceback
                    self.logger.warning(f"   详细错误: {traceback.format_exc()}")
                    # 添加默认值
                    stock_info_data['feature_date'] = ['unknown'] * len(full_data)
                    stock_info_data['target_date'] = ['unknown'] * len(full_data)
            else:
                self.logger.warning("   ⚠️  无法从文件名提取日期信息")
                # 添加默认值
                stock_info_data['feature_date'] = ['unknown'] * len(full_data)
                stock_info_data['target_date'] = ['unknown'] * len(full_data)
            
            # 如果原始数据中有其他信息列（如股票名称）
            if 'name' in full_data.columns:
                stock_info_data['stock_name'] = full_data['name'].tolist()
                self.logger.info(f"   ✅ 添加股票名称信息")
            
            # 构建最终的stock_info
            self.stock_info = pd.DataFrame(stock_info_data)
            self.logger.info(f"   ✅ 完整股票信息已保存: {list(self.stock_info.columns)}")
            self.logger.info(f"   📊 股票信息维度: {self.stock_info.shape}")
            
            # 最终内存清理
            import gc
            gc.collect()
            
            # 计算内存使用情况
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.logger.info(f"   💾 当前内存使用: {memory_mb:.1f} MB")
            except ImportError:
                self.logger.info("   💾 内存优化已完成")
            
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
            self.logger.info("✂️ 分割训练/验证数据...")
            
            split_config = self.config.get('training', {}).get('data_split', {})
            validation_size = split_config.get('validation_size', 0.2)  # 增加验证集比例
            random_state = split_config.get('random_state', 42)
            time_series_split = split_config.get('time_series_split', True)
            
            if time_series_split:
                # 时序分割（股票数据的推荐方式）
                n_samples = len(self.X)
                val_start = int(n_samples * (1 - validation_size))
                
                # 分割数据 - 只分训练集和验证集
                self.X_train = self.X.iloc[:val_start]
                self.X_val = self.X.iloc[val_start:]
                
                self.y_train = self.y.iloc[:val_start]
                self.y_val = self.y.iloc[val_start:]
                
                # 保存对应的股票信息索引
                if self.stock_info is not None:
                    self.stock_info_train = self.stock_info.iloc[:val_start]
                    self.stock_info_val = self.stock_info.iloc[val_start:]
                else:
                    self.stock_info_train = self.stock_info_val = None
                
                self.logger.info("   使用时序分割方式")
                
            else:
                # 随机分割
                if self.stock_info is not None:
                    self.X_train, self.X_val, self.y_train, self.y_val, self.stock_info_train, self.stock_info_val = train_test_split(
                        self.X, self.y, self.stock_info, test_size=validation_size, random_state=random_state
                    )
                else:
                    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                        self.X, self.y, test_size=validation_size, random_state=random_state
                    )
                    
                    self.stock_info_train = self.stock_info_val = None
                
                self.logger.info("   使用随机分割方式")
            
            # 设置测试集为None（不使用）
            self.X_test = None
            self.y_test = None
            self.stock_info_test = None
            
            self.logger.info(f"   ✅ 数据分割完成:")
            self.logger.info(f"     - 训练集: {self.X_train.shape[0]} 样本")
            self.logger.info(f"     - 验证集: {self.X_val.shape[0]} 样本")
            self.logger.info(f"     - 测试集: 已禁用")
            
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
    
    def hyperparameter_tuning(self) -> bool:
        """超参数调优"""
        try:
            # 检查是否启用超参数调优
            tuning_config = self.config.get('hyperparameter_tuning', {})
            if not tuning_config.get('enabled', False):
                self.logger.info("⚠️ 超参数调优未启用，使用默认参数")
                return True
            
            self.logger.info("🔍 开始超参数调优...")
            
            # 获取调优配置
            strategy = tuning_config.get('strategy', 'grid_search')
            max_trials = tuning_config.get('max_trials', 20)
            optimization_metric = tuning_config.get('optimization_metric', 'auc')
            optimization_direction = tuning_config.get('optimization_direction', 'maximize')
            param_space = tuning_config.get('param_space', {})
            
            # 早停设置
            early_stopping_config = tuning_config.get('early_stopping', {})
            patience = early_stopping_config.get('patience', 50)
            min_improvement = early_stopping_config.get('min_improvement', 0.001)
            
            # 生成参数组合
            if strategy == 'grid_search':
                param_combinations = self._generate_grid_search_params(param_space)
                self.logger.info(f"   📊 网格搜索：总共 {len(param_combinations)} 种参数组合")
            elif strategy == 'random_search':
                param_combinations = self._generate_random_search_params(param_space, max_trials)
                self.logger.info(f"   🎲 随机搜索：总共 {len(param_combinations)} 种参数组合")
            else:
                self.logger.error(f"❌ 不支持的搜索策略: {strategy}")
                return False
            
            # 存储调优结果
            tuning_results = []
            best_score = float('-inf') if optimization_direction == 'maximize' else float('inf')
            best_params = None
            best_model = None
            
            # 获取基础模型参数
            lgb_config = self.config.get('lightgbm', {})
            base_params = {**lgb_config.get('basic_params', {}), **lgb_config.get('advanced_params', {})}
            fit_params = lgb_config.get('fit_params', {})
            
            # 训练参数
            training_config = self.config.get('training', {}).get('training_params', {})
            verbose_eval = training_config.get('verbose', 100)
            
            # 创建数据集
            train_data = lgb.Dataset(self.X_train, label=self.y_train)
            val_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)
            
            # 开始调参循环
            self.logger.info(f"   🚀 开始调参，目标指标: {optimization_metric} ({optimization_direction})")
            
            for trial_idx, trial_params in enumerate(param_combinations):
                try:
                    self.logger.info(f"   🔄 第 {trial_idx + 1}/{len(param_combinations)} 次尝试")
                    self.logger.info(f"     参数: {trial_params}")
                    
                    # 合并参数
                    model_params = base_params.copy()
                    model_params.update(trial_params)
                    
                    # 训练模型
                    with SuppressOutput():  # 抑制训练输出
                        # 确保模型参数中有评估指标
                        if 'metric' not in model_params:
                            if hasattr(self, 'prediction_mode') and self.prediction_mode == 'direction':
                                model_params['metric'] = 'binary_logloss'
                            else:
                                model_params['metric'] = 'rmse'
                        
                        trial_model = lgb.train(
                            model_params,
                            train_data,
                            valid_sets=[val_data],  # 只使用验证集进行早停
                            valid_names=['val'],
                            num_boost_round=fit_params.get('num_boost_round', 1000),
                            callbacks=[
                                lgb.early_stopping(patience),
                                lgb.log_evaluation(0)  # 不输出训练日志
                            ]
                        )
                    
                    # 评估模型
                    trial_score = self._evaluate_single_trial(trial_model, optimization_metric)
                    
                    # 记录结果
                    trial_result = {
                        'trial': trial_idx + 1,
                        'params': trial_params.copy(),
                        'score': trial_score,
                        'metric': optimization_metric
                    }
                    tuning_results.append(trial_result)
                    
                    self.logger.info(f"     📊 {optimization_metric}: {trial_score:.6f}")
                    
                    # 更新最佳结果
                    if self._is_better_score(trial_score, best_score, optimization_direction):
                        best_score = trial_score
                        best_params = trial_params.copy()
                        best_model = trial_model
                        self.logger.info(f"     🏆 发现更好的参数！{optimization_metric}: {best_score:.6f}")
                    
                except Exception as e:
                    self.logger.warning(f"     ❌ 第 {trial_idx + 1} 次尝试失败: {e}")
                    continue
            
            # 保存调优结果
            self._save_tuning_results(tuning_results, best_params, best_score, tuning_config)
            
            # 使用最佳参数更新模型配置
            if best_params is not None:
                self.logger.info(f"   🎯 调优完成！最佳 {optimization_metric}: {best_score:.6f}")
                self.logger.info(f"   🏆 最佳参数: {best_params}")
                
                # 更新配置中的参数
                self.config['lightgbm']['basic_params'].update(best_params)
                self.model = best_model  # 保存最佳模型
                
                return True
            else:
                self.logger.error("❌ 调优失败，未找到有效的参数组合")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 超参数调优失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _generate_grid_search_params(self, param_space: Dict) -> List[Dict]:
        """生成网格搜索的参数组合"""
        if not param_space:
            return [{}]
        
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _generate_random_search_params(self, param_space: Dict, max_trials: int) -> List[Dict]:
        """生成随机搜索的参数组合"""
        if not param_space:
            return [{}]
        
        combinations = []
        for _ in range(max_trials):
            param_dict = {}
            for param_name, param_values in param_space.items():
                param_dict[param_name] = random.choice(param_values)
            combinations.append(param_dict)
        
        return combinations
    
    def _evaluate_single_trial(self, model, metric: str) -> float:
        """评估单次试验的模型"""
        # 获取验证集预测
        if hasattr(self, 'prediction_mode') and self.prediction_mode == 'direction':
            # 二分类
            y_pred_proba = model.predict(self.X_val)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            # 回归
            y_pred = model.predict(self.X_val)
            y_pred_proba = None
        
        # 计算指标
        try:
            if metric == 'auc' and y_pred_proba is not None:
                from sklearn.metrics import roc_auc_score
                return roc_auc_score(self.y_val, y_pred_proba)
            elif metric == 'accuracy':
                return np.mean(self.y_val == y_pred)
            elif metric == 'precision':
                from sklearn.metrics import precision_score
                return precision_score(self.y_val, y_pred, zero_division=0)
            elif metric == 'recall':
                from sklearn.metrics import recall_score
                return recall_score(self.y_val, y_pred, zero_division=0)
            elif metric == 'f1_score':
                from sklearn.metrics import f1_score
                return f1_score(self.y_val, y_pred, zero_division=0)
            elif metric == 'rmse':
                return np.sqrt(mean_squared_error(self.y_val, y_pred))
            elif metric == 'mae':
                return mean_absolute_error(self.y_val, y_pred)
            elif metric == 'r2_score':
                return r2_score(self.y_val, y_pred)
            else:
                self.logger.warning(f"未知的评估指标: {metric}，使用默认AUC")
                if y_pred_proba is not None:
                    from sklearn.metrics import roc_auc_score
                    return roc_auc_score(self.y_val, y_pred_proba)
                else:
                    return r2_score(self.y_val, y_pred)
        except Exception as e:
            self.logger.warning(f"计算指标 {metric} 失败: {e}")
            return 0.0
    
    def _is_better_score(self, current_score: float, best_score: float, direction: str) -> bool:
        """判断当前分数是否更好"""
        if direction == 'maximize':
            return current_score > best_score
        else:
            return current_score < best_score
    
    def _save_tuning_results(self, results: List[Dict], best_params: Dict, best_score: float, config: Dict):
        """保存调优结果"""
        try:
            results_save_config = config.get('results_save', {})
            
            if results_save_config.get('save_all_trials', True):
                # 保存所有试验结果
                results_df = pd.DataFrame(results)
                results_path = self.results_save_dir / "hyperparameter_tuning_results.csv"
                results_df.to_csv(results_path, index=False, encoding='utf-8')
                self.logger.info(f"   ✅ 调优结果已保存: {results_path}")
            
            if results_save_config.get('save_best_model', True):
                # 保存最佳参数
                best_result = {
                    'best_score': best_score,
                    'best_params': best_params,
                    'total_trials': len(results),
                    'optimization_metric': config.get('optimization_metric', 'auc'),
                    'strategy': config.get('strategy', 'grid_search')
                }
                
                best_params_path = self.results_save_dir / "best_hyperparameters.json"
                with open(best_params_path, 'w', encoding='utf-8') as f:
                    json.dump(best_result, f, ensure_ascii=False, indent=2)
                self.logger.info(f"   ✅ 最佳参数已保存: {best_params_path}")
            
            if results_save_config.get('detailed_log', True):
                # 保存详细日志
                log_content = []
                log_content.append("超参数调优详细结果")
                log_content.append("=" * 50)
                log_content.append(f"策略: {config.get('strategy', 'grid_search')}")
                log_content.append(f"优化指标: {config.get('optimization_metric', 'auc')}")
                log_content.append(f"优化方向: {config.get('optimization_direction', 'maximize')}")
                log_content.append(f"总试验次数: {len(results)}")
                log_content.append(f"最佳分数: {best_score:.6f}")
                log_content.append(f"最佳参数: {best_params}")
                log_content.append("")
                log_content.append("所有试验结果:")
                log_content.append("-" * 30)
                
                for result in results:
                    log_content.append(f"试验 {result['trial']}: {result['metric']}={result['score']:.6f}, 参数={result['params']}")
                
                log_path = self.results_save_dir / "hyperparameter_tuning_log.txt"
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(log_content))
                self.logger.info(f"   ✅ 详细日志已保存: {log_path}")
                
        except Exception as e:
            self.logger.warning(f"保存调优结果失败: {e}")

    def train_model(self) -> bool:
        """训练模型"""
        try:
            self.logger.info("🎯 开始训练LightGBM模型...")
            
            # 检查是否已经通过调优训练了模型
            if hasattr(self, 'model') and self.model is not None:
                self.logger.info("   ✅ 使用调优后的最佳模型")
                
                # 创建内置可视化器
                self.visualizer = self._create_builtin_visualizer()
                self.logger.info("   🎨 内置可视化器已启用")
                
                return True
            
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
            
            # 创建内置可视化器（用于记录训练历史）
            self.visualizer = self._create_builtin_visualizer()
            self.logger.info("   🎨 内置可视化器已启用")
            
            callbacks = [
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(verbose)
            ]
            
            # 添加可视化记录回调
            if self.visualizer and hasattr(self.visualizer, 'record_callback'):
                callbacks.append(self.visualizer.record_callback())
            
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
            if hasattr(self, 'prediction_mode') and self.prediction_mode == 'direction':
                # 二分类：获取概率预测
                y_train_pred_proba = self.model.predict(self.X_train)
                y_val_pred_proba = self.model.predict(self.X_val)
                
                # 转换为类别预测（概率 > 0.5 为看多）
                y_train_pred = (y_train_pred_proba > 0.5).astype(int)
                y_val_pred = (y_val_pred_proba > 0.5).astype(int)
            else:
                # 回归：直接预测数值
                y_train_pred = self.model.predict(self.X_train)
                y_val_pred = self.model.predict(self.X_val)
                y_train_pred_proba = y_val_pred_proba = None
            
            # 计算评估指标
            eval_config = self.config.get('evaluation', {})
            metrics_list = eval_config.get('metrics', ['rmse', 'mae', 'r2_score'])
            
            results = {}
            
            for split, y_true, y_pred, y_pred_proba in [
                ('train', self.y_train, y_train_pred, y_train_pred_proba),
                ('val', self.y_val, y_val_pred, y_val_pred_proba)
            ]:
                split_metrics = {}
                
                for metric in metrics_list:
                    try:
                        if hasattr(self, 'prediction_mode') and self.prediction_mode == 'direction':
                            # 🎯 分类指标
                            if metric == 'accuracy':
                                value = np.mean(y_true == y_pred) * 100
                            elif metric == 'auc' and y_pred_proba is not None:
                                from sklearn.metrics import roc_auc_score
                                value = roc_auc_score(y_true, y_pred_proba)
                            elif metric == 'precision':
                                from sklearn.metrics import precision_score
                                value = precision_score(y_true, y_pred, zero_division=0)
                            elif metric == 'recall':
                                from sklearn.metrics import recall_score
                                value = recall_score(y_true, y_pred, zero_division=0)
                            elif metric == 'f1_score':
                                from sklearn.metrics import f1_score
                                value = f1_score(y_true, y_pred, zero_division=0)
                            elif metric == 'log_loss' and y_pred_proba is not None:
                                from sklearn.metrics import log_loss
                                # 处理概率边界问题
                                y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1-1e-15)
                                value = log_loss(y_true, y_pred_proba_clipped)
                            else:
                                continue
                        else:
                            # 📈 回归指标
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
                        
                    except Exception as e:
                        self.logger.warning(f"   计算指标 {metric} 失败: {e}")
                        continue
                
                results[split] = split_metrics
            
            # 输出结果
            prediction_type = "方向预测" if hasattr(self, 'prediction_mode') and self.prediction_mode == 'direction' else "回归预测"
            self.logger.info(f"   📈 评估结果 ({prediction_type}):")
            for split, metrics in results.items():
                self.logger.info(f"     {split.upper()}:")
                for metric, value in metrics.items():
                    if metric in ['mape', 'directional_accuracy', 'accuracy']:
                        self.logger.info(f"       {metric}: {value:.2f}%")
                    else:
                        self.logger.info(f"       {metric}: {value:.6f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 模型评估失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
            
            # 保存预测结果 - 只保存验证集
            if results_config.get('save_predictions', True):
                # 只获取验证集预测
                y_val_pred = self.model.predict(self.X_val)
                
                # 基础预测数据框 - 只包含验证集
                pred_data = {
                    'split': ['val'] * len(self.y_val),
                    'y_true': self.y_val.tolist(),
                    'y_pred': y_val_pred.tolist()
                }
                
                # 如果有股票信息，添加验证集的股票信息
                if self.stock_info_val is not None:
                    # 添加验证集股票信息到预测数据中
                    for col in self.stock_info_val.columns:
                        pred_data[col] = self.stock_info_val[col].tolist()
                    
                    self.logger.info(f"   ✅ 预测结果包含股票信息: {list(self.stock_info_val.columns)}")
                
                pred_df = pd.DataFrame(pred_data)
                
                pred_path = self.results_save_dir / "predictions.csv"
                pred_df.to_csv(pred_path, index=False, encoding='utf-8')
                self.logger.info(f"   ✅ 验证集预测结果已保存: {pred_path}")
                self.logger.info(f"   📊 预测结果包含 {len(pred_df)} 条记录，{len(pred_df.columns)} 列信息")
            
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
            
            # 🎨 生成可视化图表
            if hasattr(self, 'visualizer') and self.visualizer:
                try:
                    self.logger.info("   🎨 开始生成可视化图表...")
                    
                    # 获取预测结果
                    y_train_pred = self.model.predict(self.X_train)
                    y_val_pred = self.model.predict(self.X_val)
                    y_test_pred = self.model.predict(self.X_test)
                    
                    # 设置特征名称给可视化器
                    if hasattr(self.visualizer, '__dict__'):
                        self.visualizer.feature_names = self.feature_names
                    
                    # 检查是否为分类任务
                    if hasattr(self, 'prediction_mode') and self.prediction_mode == 'direction':
                        # 对于方向预测，只生成部分图表
                        viz_results = {}
                        
                        # 学习曲线
                        learning_curve_path = self.visualizer.plot_learning_curves(self.model)
                        if learning_curve_path:
                            viz_results['learning_curves'] = learning_curve_path
                        
                        # 特征重要性
                        feature_importance_path = self.visualizer.plot_feature_importance(
                            self.model, self.feature_names, top_n=20)
                        if feature_importance_path:
                            viz_results['feature_importance'] = feature_importance_path
                        
                        self.logger.info(f"   🎯 方向预测模式，生成了 {len(viz_results)} 个图表")
                        
                    else:
                        # 回归任务生成所有图表 - 只使用训练集和验证集
                        viz_results = {}
                        
                        # 学习曲线
                        learning_curve_path = self.visualizer.plot_learning_curves(self.model)
                        if learning_curve_path:
                            viz_results['learning_curves'] = learning_curve_path
                        
                        # 特征重要性
                        feature_importance_path = self.visualizer.plot_feature_importance(
                            self.model, self.feature_names, top_n=20)
                        if feature_importance_path:
                            viz_results['feature_importance'] = feature_importance_path
                        
                        # 预测散点图和残差图 - 只生成训练集和验证集
                        for split_name, y_true, y_pred in [
                            ('训练集', self.y_train, y_train_pred),
                            ('验证集', self.y_val, y_val_pred)
                        ]:
                            scatter_path = self.visualizer.plot_predictions_scatter(y_true, y_pred, split_name)
                            if scatter_path:
                                viz_results[f'predictions_scatter_{split_name}'] = scatter_path
                            
                            residuals_path = self.visualizer.plot_residuals(y_true, y_pred, split_name)
                            if residuals_path:
                                viz_results[f'residuals_{split_name}'] = residuals_path
                        
                        self.logger.info(f"   📈 回归模式，生成了 {len(viz_results)} 个图表")
                    
                    # 显示图表保存位置
                    for chart_type, chart_path in viz_results.items():
                        self.logger.info(f"     📊 {chart_type}: {chart_path}")
                        
                except Exception as e:
                    self.logger.warning(f"   ⚠️ 可视化图表生成失败: {e}")
                    import traceback
                    self.logger.warning(f"   详细错误: {traceback.format_exc()}")
            else:
                self.logger.info("   ℹ️ 未启用可视化功能")
            
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
            
            # 5. 超参数调优（可选）
            if not self.hyperparameter_tuning():
                return False
            
            # 6. 训练模型（如果没有通过调优得到模型）
            if not self.train_model():
                return False
            
            # 7. 评估模型
            evaluation_results = self.evaluate_model()
            if not evaluation_results:
                return False
            
            # 8. 保存模型
            if not self.save_model():
                return False
            
            # 9. 保存结果
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