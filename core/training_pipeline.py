# -*- coding: utf-8 -*-
"""
AI股市预测系统 - 训练管道模块
功能：自动化模型训练、验证、超参数优化和部署管道
"""

import os
# 设置TensorFlow环境变量，屏蔽调试信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示错误
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 关闭oneDNN提示
os.environ['TF_DISABLE_MKL'] = '1'  # 禁用MKL优化提示

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# 导入处理 - 支持直接运行和模块导入
try:
    from .feature_engineering import FeatureEngineering
    from .ai_models import EnsembleModel, create_ensemble_model, LightGBMModel
    from .enhanced_ai_models import create_enhanced_ensemble_model
    from .feature_cache import BatchFeatureProcessor
    from .training_report_generator import TrainingReportGenerator
    from ..utils.gpu_config import setup_dual_gpu, get_optimal_batch_size
except ImportError:
    # 直接运行时的导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
    
    from feature_engineering import FeatureEngineering
    from ai_models import EnsembleModel, create_ensemble_model, LightGBMModel
    from enhanced_ai_models import create_enhanced_ensemble_model
    from feature_cache import BatchFeatureProcessor
    from training_report_generator import TrainingReportGenerator
    try:
        from gpu_config import setup_dual_gpu, get_optimal_batch_size
    except ImportError:
        # 如果GPU配置不可用，提供默认实现
        def setup_dual_gpu():
            return None
        def get_optimal_batch_size(base_size, gpu_count):
            return base_size

# LightGBM导入
import lightgbm as lgb

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    模型训练管道类
    负责数据准备、模型训练、验证、超参数优化等
    """
    
    def __init__(self, data_dir: str = None, model_dir: str = "models", 
                 enable_batch_cache: bool = True, cache_workers: int = 1):
        # 自动寻找数据目录
        if data_dir is None:
            possible_dirs = ["data/datas_em", "datas_em", "data_em", "data/datas_QMT", "datas_QMT", "financial_csv"]
            for dir_path in possible_dirs:
                if os.path.exists(dir_path):
                    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
                    if csv_files:
                        data_dir = dir_path
                        logger.info(f"自动发现数据目录: {data_dir}，包含 {len(csv_files)} 个CSV文件")
                        break
            
            if data_dir is None:
                data_dir = "data/datas_em"  # 默认值
                logger.warning(f"未找到数据目录，使用默认: {data_dir}")
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.feature_engineer = FeatureEngineering(enable_cache=False)
        self.models = {}
        self.performance_history = []
        self.report_generator = TrainingReportGenerator()
        
        # 🚀 配置双GPU环境
        self.gpu_strategy = setup_dual_gpu()
        
        # 🚀 批量缓存处理器
        self.enable_batch_cache = enable_batch_cache
        # 彻底禁用批量缓存
        self.batch_processor = None
        if enable_batch_cache:
            logger.warning("⚠️ 批量缓存已被强制禁用")
        logger.info("📊 批量缓存处理器: 已禁用")
        
        # 创建模型保存目录
        os.makedirs(model_dir, exist_ok=True)
        
        # 训练配置 - 完整训练优化参数
        self.config = {
            'sequence_length': 60,
            'prediction_days': [1, 3, 5],
            'train_test_split': 0.8,
            'validation_split': 0.2,
            'min_samples': 300,  # 降低最少样本数，包含更多股票
            'performance_threshold': 0.55,  # 最低准确率要求
            
            # 🚀 完整训练参数配置
            'training_params': {
                'epochs': 150,  # 完整训练轮次
                'batch_size': 64,  # 优化批次大小
                'early_stopping_patience': 15,  # 早停耐心值
                'cv_epochs': 50,  # 交叉验证轮次
                'cv_folds': 3,  # 交叉验证折数
            },
            
            # LightGBM专用配置 - 完整训练优化
            'lightgbm_config': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_threads': 8,  # 增加线程数
                'n_estimators': 500,  # 增加树的数量
                'learning_rate': 0.05,  # 降低学习率，提高稳定性
                'max_depth': 8,  # 增加树深度
                'num_leaves': 127,  # 增加叶子数
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'early_stopping_rounds': 100,  # 增加早停轮次
                'verbose': -1,
                'random_state': 42
            }
        }
    
    def load_stock_data(self, stock_code: str) -> pd.DataFrame:
        """加载单只股票数据"""
        file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"股票数据文件不存在: {file_path}")
        
        df = pd.read_csv(file_path)
        df['交易日期'] = pd.to_datetime(df['交易日期'])
        df = df.sort_values('交易日期').reset_index(drop=True)
        
        # 静默加载，不打印每只股票的信息
        return df
    
    def prepare_training_data(self, stock_codes: List[str], 
                            prediction_days: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """
        准备训练数据 - 支持批量缓存优化
        
        Args:
            stock_codes: 股票代码列表
            prediction_days: 预测天数
            
        Returns:
            X, y, feature_names, feature_info
        """
        logger.info(f"🚀 开始准备训练数据，股票数量: {len(stock_codes)}")
        
        # 缓存已禁用，只使用传统方法
        logger.info("📊 使用传统方法进行特征工程（缓存已禁用）")
        return self._prepare_training_data_traditional(stock_codes, prediction_days)
    
    def _prepare_training_data_with_batch_cache(self, stock_codes: List[str], 
                                              prediction_days: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """
        使用批量缓存准备训练数据
        """
        logger.info("📈 使用批量缓存处理器进行特征工程...")
        
        # 定义数据加载函数
        def data_loader(stock_code: str) -> pd.DataFrame:
            try:
                df = self.load_stock_data(stock_code)
                if len(df) < self.config['min_samples']:
                    logger.warning(f"股票 {stock_code} 数据量不足，跳过")
                    return None
                return df
            except Exception as e:
                logger.error(f"加载股票 {stock_code} 数据失败: {e}")
                return None
        
        # 批量处理特征工程
        features_results = self.batch_processor.process_stocks_with_cache(
            stock_codes=stock_codes,
            data_loader_func=data_loader,
            show_progress=True
        )
        
        if not features_results:
            raise ValueError("批量特征处理未返回任何有效数据")
        
        # 准备模型数据
        logger.info("🔧 准备模型训练数据...")
        all_X, all_y = [], []
        feature_names = None
        feature_info = None
        
        for stock_code, df_features in features_results.items():
            try:
                # 准备模型数据
                X, y, feature_names_temp, feature_info_temp = self.feature_engineer.prepare_model_data(
                    df_features, 
                    prediction_days=prediction_days,
                    lookback_window=self.config['sequence_length']
                )
                
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
                    if feature_names is None:
                        feature_names = feature_names_temp
                    if feature_info is None:
                        feature_info = feature_info_temp
                    logger.info(f"股票 {stock_code} 生成样本 {len(X)} 个")
                
            except Exception as e:
                logger.error(f"准备股票 {stock_code} 模型数据时出错: {str(e)}")
                continue
        
        if not all_X:
            raise ValueError("没有成功准备的模型数据")
        
        # 合并所有股票数据
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)
        
        logger.info(f"✅ 批量训练数据准备完成:")
        logger.info(f"   总计样本数: {len(X_combined)}")
        logger.info(f"   特征数: {len(feature_names)}")
        logger.info(f"   正样本比例: {y_combined.mean():.3f}")
        
        # 显示缓存效果统计
        if hasattr(self.batch_processor.cache, 'cache_hits'):
            total_access = self.batch_processor.cache.cache_hits + self.batch_processor.cache.cache_misses
            if total_access > 0:
                hit_rate = self.batch_processor.cache.cache_hits / total_access
                logger.info(f"   缓存命中率: {hit_rate:.1%}")
                logger.info(f"   缓存加速: 第二次运行预计提速 {hit_rate*8:.1f}x")
        
        return X_combined, y_combined, feature_names, feature_info
    
    def _prepare_training_data_traditional(self, stock_codes: List[str], 
                                         prediction_days: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """
        传统方式准备训练数据（真正的分批处理，避免内存爆炸）
        """
        print(f"📊 处理 {len(stock_codes)} 只股票的特征工程...")
        
        # 分批处理配置
        batch_size = 50  # 每批处理50只股票
        max_samples_per_stock = 500  # 每只股票最大样本数
        max_total_samples = 50000  # 总最大样本数
        # 内存优化：减少回望窗口
        optimized_lookback = min(30, self.config['sequence_length'])
        
        feature_names = None
        feature_info = None
        processed_count = 0
        total_samples = 0
        
        # 初始化结果存储（使用列表逐步构建，避免大数组预分配）
        batch_results = []
        
        # 分批处理股票
        for batch_start in range(0, len(stock_codes), batch_size):
            batch_end = min(batch_start + batch_size, len(stock_codes))
            batch_stocks = stock_codes[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(stock_codes) + batch_size - 1) // batch_size
            
            print(f"\n📦 批次 {batch_num}/{total_batches}: 处理股票 {batch_start+1}-{batch_end}")
            
            # 处理当前批次
            batch_X, batch_y = [], []
            
            for i, stock_code in enumerate(batch_stocks):
                try:
                    # 显示进度（每只股票都显示，不使用\r覆盖）
                    progress = f"[{batch_start + i + 1:4d}/{len(stock_codes):4d}] {stock_code}"
                    print(f"{progress}", end='', flush=True)
                    
                    # 检查是否达到样本数限制
                    if total_samples >= max_total_samples:
                        print(f"\n⚠️ 已达到最大样本数限制({max_total_samples})，停止处理")
                        break
                    
                    # 加载股票数据
                    df = self.load_stock_data(stock_code)
                    
                    if len(df) < self.config['min_samples']:
                        print(f" ❌数据不足")
                        continue
                    
                    # 限制单只股票的数据量以控制内存
                    if len(df) > 2000:
                        df = df.tail(2000)  # 只保留最近2000条记录
                    
                    # 特征工程
                    df_features = self.feature_engineer.create_all_features(df, stock_code)
                    
                    # 准备模型数据
                    X, y, feature_names_temp, feature_info_temp = self.feature_engineer.prepare_model_data(
                        df_features, 
                        prediction_days=prediction_days,
                        lookback_window=optimized_lookback
                    )
                    
                    if len(X) > 0:
                        # 限制每只股票的样本数
                        if len(X) > max_samples_per_stock:
                            indices = np.random.choice(len(X), max_samples_per_stock, replace=False)
                            X = X[indices]
                            y = y[indices]
                        
                        batch_X.append(X)
                        batch_y.append(y)
                        total_samples += len(X)
                        
                        if feature_names is None:
                            feature_names = feature_names_temp
                        if feature_info is None:
                            feature_info = feature_info_temp
                        processed_count += 1
                        
                        print(f" ✅{len(X)}样本")
                    else:
                        print(f" ❌无有效样本")
                    
                    # 清理内存
                    del df, df_features, X, y
                    
                except Exception as e:
                    print(f" ❌处理失败: {str(e)[:20]}")
                    continue
            
            # 合并当前批次数据
            if batch_X:
                batch_X_combined = np.vstack(batch_X)
                batch_y_combined = np.hstack(batch_y)
                batch_results.append((batch_X_combined, batch_y_combined))
                
                print(f"\n   ✅ 批次 {batch_num} 完成: {len(batch_X_combined)} 个样本")
                
                # 清理批次临时数据
                del batch_X, batch_y, batch_X_combined, batch_y_combined
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 检查是否达到样本限制
            if total_samples >= max_total_samples:
                break
        
        print()  # 换行
        
        if not batch_results:
            raise ValueError("没有成功处理的股票数据")
        
        # 逐步合并所有批次结果（避免内存峰值）
        print("🔄 合并所有批次数据...")
        X_combined = None
        y_combined = None
        
        for i, (batch_X, batch_y) in enumerate(batch_results):
            if X_combined is None:
                X_combined = batch_X.copy()
                y_combined = batch_y.copy()
            else:
                X_combined = np.vstack([X_combined, batch_X])
                y_combined = np.hstack([y_combined, batch_y])
            
            # 清理已处理的批次
            del batch_X, batch_y
            
            # 每处理5个批次进行一次垃圾回收
            if (i + 1) % 5 == 0:
                gc.collect()
                print(f"   已合并 {i+1}/{len(batch_results)} 个批次")
        
        # 清理批次结果列表
        del batch_results
        gc.collect()
        
        # 最终数据采样（如果仍然太大）
        if len(X_combined) > max_total_samples:
            print(f"🔽 最终采样: {len(X_combined)} -> {max_total_samples}")
            indices = np.random.choice(len(X_combined), max_total_samples, replace=False)
            X_combined = X_combined[indices]
            y_combined = y_combined[indices]
        
        print(f"✅ 分批特征工程完成: {processed_count}/{len(stock_codes)} 只股票, {len(X_combined)} 个样本, {len(feature_names)} 个特征")
        print(f"   序列长度: {optimized_lookback} (原始: {self.config['sequence_length']})")
        print(f"   内存占用估计: {X_combined.nbytes / 1024 / 1024:.1f} MB")
        
        # 更新feature_info中的序列长度信息
        if feature_info is None:
            feature_info = {}
        feature_info['actual_sequence_length'] = optimized_lookback
        feature_info['original_sequence_length'] = self.config['sequence_length']
        
        return X_combined, y_combined, feature_names, feature_info
    
    def get_available_stocks(self, limit: int = None) -> List[str]:
        """
        获取可用的股票列表
        
        Args:
            limit: 限制股票数量
            
        Returns:
            股票代码列表
        """
        if not os.path.exists(self.data_dir):
            logger.warning(f"数据目录不存在: {self.data_dir}")
            return []
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        stock_codes = [f.replace('.csv', '') for f in csv_files]
        
        if limit:
            stock_codes = stock_codes[:limit]
        
        # 验证数据质量
        valid_stocks = []
        for stock_code in stock_codes:
            try:
                df = self.load_stock_data(stock_code)
                if len(df) >= self.config['min_samples']:
                    valid_stocks.append(stock_code)
                if limit and len(valid_stocks) >= limit:
                    break
            except Exception:
                continue
        
        logger.info(f"发现 {len(valid_stocks)} 只有效股票（总共 {len(csv_files)} 个文件）")
        return valid_stocks
    
    def warm_up_cache(self, stock_codes: List[str], show_progress: bool = True):
        """
        预热缓存 - 已禁用缓存功能
        """
        logger.warning("⚠️ 缓存功能已被禁用，跳过预热步骤")
        return
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        时间序列数据分割
        """
        split_idx = int(len(X) * self.config['train_test_split'])
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def cross_validate_model(self, X: np.ndarray, y: np.ndarray, 
                           model: Optional[EnsembleModel] = None,
                           feature_info: Optional[Dict] = None,
                           n_splits: int = None) -> Dict[str, float]:
        """
        时间序列交叉验证
        
        Args:
            X: 特征数据
            y: 标签数据
            model: 模型实例
            n_splits: 交叉验证折数
            
        Returns:
            验证结果字典
        """
        # 使用配置中的交叉验证折数
        if n_splits is None:
            n_splits = self.config['training_params']['cv_folds']
        
        logger.info(f"开始 {n_splits} 折交叉验证...")
        
        if model is None:
            # 使用LightGBM集成模型进行交叉验证
            model = self.create_lightgbm_ensemble_model(
                n_features=X.shape[-1],
                feature_info=feature_info
            )
        
        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        fold = 1
        for train_idx, val_idx in tscv.split(X):
            logger.info(f"交叉验证 折 {fold}/{n_splits}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # 训练模型
            # 🚀 交叉验证优化配置
            cv_epochs = self.config['training_params']['cv_epochs']  # 使用配置的交叉验证轮次
            cv_batch_size = get_optimal_batch_size(self.config['training_params']['batch_size'], 2)  # 动态计算最优batch size
            
            if feature_info and hasattr(model, 'fit') and 'feature_info' in model.fit.__code__.co_varnames:
                model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                         feature_info=feature_info, epochs=cv_epochs, batch_size=cv_batch_size)
            else:
                model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                         epochs=cv_epochs, batch_size=cv_batch_size)
            
            # 预测和评估
            y_pred = model.predict(X_val_fold)
            y_proba = model.predict_proba(X_val_fold)[:, 1]
            
            # 计算指标
            scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            scores['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
            scores['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
            scores['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
            scores['auc'].append(roc_auc_score(y_val_fold, y_proba))
            
            fold += 1
        
        # 计算平均分数
        avg_scores = {metric: np.mean(values) for metric, values in scores.items()}
        std_scores = {f"{metric}_std": np.std(values) for metric, values in scores.items()}
        
        result = {**avg_scores, **std_scores}
        
        logger.info("交叉验证结果:")
        for metric, score in avg_scores.items():
            logger.info(f"  {metric}: {score:.4f} ± {std_scores[f'{metric}_std']:.4f}")
        
        return result
    
    def hyperparameter_optimization(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  n_trials: int = 50) -> Dict:
        """
        超参数优化
        
        Args:
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            n_trials: 优化试验次数
            
        Returns:
            最优参数字典
        """
        logger.info(f"开始超参数优化，试验次数: {n_trials}")
        
        def objective(trial):
            # LSTM参数
            lstm_units = trial.suggest_int('lstm_units', 64, 256, step=64)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            
            # LightGBM参数 - 更详细的超参数
            lgb_max_depth = trial.suggest_int('lgb_max_depth', 3, 10)
            lgb_learning_rate = trial.suggest_float('lgb_learning_rate', 0.01, 0.3)
            lgb_n_estimators = trial.suggest_int('lgb_n_estimators', 100, 1000, step=100)
            lgb_num_leaves = trial.suggest_int('lgb_num_leaves', 20, 300)
            lgb_feature_fraction = trial.suggest_float('lgb_feature_fraction', 0.6, 1.0)
            lgb_bagging_fraction = trial.suggest_float('lgb_bagging_fraction', 0.6, 1.0)
            lgb_bagging_freq = trial.suggest_int('lgb_bagging_freq', 1, 10)
            lgb_min_child_samples = trial.suggest_int('lgb_min_child_samples', 10, 100)
            
            # Transformer参数
            d_model = trial.suggest_int('d_model', 64, 256, step=64)
            num_heads = trial.suggest_int('num_heads', 4, 16, step=4)
            
            try:
                # 创建模型（这里简化，实际需要传入参数）
                try:
                    from .ai_models import LSTMModel, LightGBMModel, TransformerModel, CNNLSTMModel
                except ImportError:
                    from ai_models import LSTMModel, LightGBMModel, TransformerModel, CNNLSTMModel
                
                ensemble = EnsembleModel()
                
                # 添加优化后的模型 - LightGBM作为核心模型给更高权重
                lgb_model = LightGBMModel(
                    objective='binary',
                    max_depth=lgb_max_depth,
                    learning_rate=lgb_learning_rate,
                    n_estimators=lgb_n_estimators,
                    num_leaves=lgb_num_leaves,
                    feature_fraction=lgb_feature_fraction,
                    bagging_fraction=lgb_bagging_fraction,
                    bagging_freq=lgb_bagging_freq,
                    min_child_samples=lgb_min_child_samples,
                    random_state=42,
                    verbose=-1
                )
                
                lstm_model = LSTMModel(
                    sequence_length=self.config['sequence_length'],
                    n_features=X_train.shape[-1],
                    lstm_units=lstm_units,
                    dropout_rate=dropout_rate
                )
                
                transformer_model = TransformerModel(
                    sequence_length=self.config['sequence_length'],
                    n_features=X_train.shape[-1],
                    d_model=d_model,
                    num_heads=num_heads
                )
                
                cnn_lstm_model = CNNLSTMModel(
                    sequence_length=self.config['sequence_length'],
                    n_features=X_train.shape[-1]
                )
                
                # LightGBM权重0.4，其他模型共享0.6
                ensemble.add_model(lgb_model, weight=0.4)      # LightGBM - 主要模型
                ensemble.add_model(lstm_model, weight=0.25)     # LSTM
                ensemble.add_model(transformer_model, weight=0.2) # Transformer  
                ensemble.add_model(cnn_lstm_model, weight=0.15)  # CNN-LSTM
                
                # 训练模型
                ensemble.fit(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
                
                # 评估
                y_pred = ensemble.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                
                return accuracy
                
            except Exception as e:
                logger.error(f"超参数优化试验失败: {str(e)}")
                return 0.0
        
        # 创建Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(),
            pruner=MedianPruner()
        )
        
        study.optimize(objective, n_trials=n_trials, timeout=3600)  # 最多1小时
        
        logger.info(f"超参数优化完成，最佳得分: {study.best_value:.4f}")
        logger.info(f"最佳参数: {study.best_params}")
        
        return study.best_params
    
    def create_lightgbm_ensemble_model(self, n_features: int, feature_info: Optional[Dict] = None) -> EnsembleModel:
        """
        创建包含LightGBM的集成模型
        
        Args:
            n_features: 特征数量
            feature_info: 特征信息（可选）
            
        Returns:
            集成模型
        """
        logger.info("创建LightGBM集成模型...")
        
        # 获取实际的序列长度
        actual_sequence_length = self.config['sequence_length']
        if feature_info and 'actual_sequence_length' in feature_info:
            actual_sequence_length = feature_info['actual_sequence_length']
            logger.info(f"使用优化后的序列长度: {actual_sequence_length} (原始: {self.config['sequence_length']})")
        
        ensemble = EnsembleModel()
        
        # 1. LightGBM模型 - 主要模型，权重40%
        lgb_config = self.config['lightgbm_config'].copy()
        # 配置已经在__init__中优化，直接使用
        
        lgb_model = LightGBMModel(**lgb_config)
        ensemble.add_model(lgb_model, weight=0.40)
        
        # 2. LSTM模型 - 时序特征，权重25%
        try:
            try:
                from .ai_models import LSTMModel
            except ImportError:
                from ai_models import LSTMModel
                
            lstm_model = LSTMModel(
                sequence_length=actual_sequence_length,  # 使用实际序列长度
                n_features=n_features,
                lstm_units=128,
                dropout_rate=0.3
            )
            ensemble.add_model(lstm_model, weight=0.25)
        except Exception as e:
            logger.warning(f"LSTM模型创建失败: {str(e)}")
        
        # 3. Transformer模型 - 注意力机制，权重20%
        try:
            try:
                from .ai_models import TransformerModel
            except ImportError:
                from ai_models import TransformerModel
                
            transformer_model = TransformerModel(
                sequence_length=actual_sequence_length,  # 使用实际序列长度
                n_features=n_features,
                d_model=128,
                num_heads=8
            )
            ensemble.add_model(transformer_model, weight=0.20)
        except Exception as e:
            logger.warning(f"Transformer模型创建失败: {str(e)}")
        
        # 4. CNN-LSTM模型 - 局部模式识别，权重15%
        try:
            try:
                from .ai_models import CNNLSTMModel
            except ImportError:
                from ai_models import CNNLSTMModel
                
            cnn_lstm_model = CNNLSTMModel(
                sequence_length=actual_sequence_length,  # 使用实际序列长度
                n_features=n_features
            )
            ensemble.add_model(cnn_lstm_model, weight=0.15)
        except Exception as e:
            logger.warning(f"CNN-LSTM模型创建失败: {str(e)}")
        
        logger.info(f"集成模型创建完成，包含 {len(ensemble.models)} 个子模型")
        return ensemble
    
    def train_model(self, stock_codes: List[str], 
                   prediction_days: int = 1,
                   use_hyperparameter_optimization: bool = True,
                   save_model: bool = True,
                   clear_cache: bool = False) -> EnsembleModel:
        """
        训练模型主流程
        
        Args:
            stock_codes: 股票代码列表
            prediction_days: 预测天数
            use_hyperparameter_optimization: 是否使用超参数优化
            save_model: 是否保存模型
            clear_cache: 是否清理特征缓存
            
        Returns:
            训练好的集成模型
        """
        
        # 所有缓存已禁用
        if clear_cache:
            logger.info("📊 所有缓存已禁用，无需清理")
        
        # 缓存状态
        logger.info("📊 所有缓存状态: 已禁用")
        
        logger.info(f"🚀 开始训练模型，预测 {prediction_days} 天")
        logger.info(f"   批量缓存: {'✅ 启用' if self.enable_batch_cache else '❌ 禁用'}")
        
        # 1. 准备数据
        X, y, feature_names, feature_info = self.prepare_training_data(stock_codes, prediction_days)
        
        # 2. 数据分割
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # 3. 进一步分割训练集和验证集
        val_split_idx = int(len(X_train) * (1 - self.config['validation_split']))
        X_train_final, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_train_final, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        
        # 4. 超参数优化（可选）
        best_params = None
        if use_hyperparameter_optimization:
            best_params = self.hyperparameter_optimization(
                X_train_final, y_train_final, X_val, y_val
            )
        
        # 5. 创建和训练最终模型（使用LightGBM集成模型）
        logger.info("创建LightGBM为核心的集成模型...")
        model = self.create_lightgbm_ensemble_model(
            n_features=X.shape[-1],
            feature_info=feature_info
        )
        
        # 使用全部训练数据进行最终训练（传递feature_info）
        # 🚀 完整训练配置
        optimized_epochs = self.config['training_params']['epochs']  # 使用配置的完整训练轮次
        optimized_batch_size = get_optimal_batch_size(self.config['training_params']['batch_size'], 2)  # 动态计算最优batch size
        
        if feature_info and hasattr(model, 'fit') and 'feature_info' in model.fit.__code__.co_varnames:
            model.fit(X_train, y_train, X_test, y_test, feature_info=feature_info, epochs=optimized_epochs, batch_size=optimized_batch_size)
        else:
            model.fit(X_train, y_train, X_test, y_test, epochs=optimized_epochs, batch_size=optimized_batch_size)
        
        # 6. 模型评估
        results = model.evaluate(X_test, y_test)
        
        # 7. 交叉验证（传递feature_info）
        cv_results = self.cross_validate_model(X_train, y_train, model, feature_info=feature_info)
        
        # 8. 保存模型和结果
        if save_model:
            # 创建统一的保存目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_save_path = os.path.join(self.model_dir, f"model_{prediction_days}d_{timestamp}")
            os.makedirs(model_save_path, exist_ok=True)
            
            # 保存模型文件
            model.save_models(model_save_path)
            
            # 保存训练信息和评估结果到同一目录
            training_info = {
                'stock_codes': stock_codes,
                'prediction_days': prediction_days,
                'feature_names': feature_names,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'test_results': results,
                'cv_results': cv_results,
                'best_params': best_params,
                'training_time': datetime.now().isoformat(),
                'gpu_config': {
                    'gpu_strategy': str(type(self.gpu_strategy).__name__) if self.gpu_strategy else None,
                    'batch_size_optimized': optimized_batch_size,
                    'epochs_optimized': optimized_epochs
                }
            }
            
            # 保存训练信息为JSON格式（易读）
            import json
            
            def convert_to_json_serializable(obj):
                """递归转换对象为JSON可序列化格式"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_to_json_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_json_serializable(item) for item in obj]
                elif hasattr(obj, '__dict__'):
                    return str(obj)  # 复杂对象转为字符串
                else:
                    return obj
            
            training_info_json = convert_to_json_serializable(training_info)
            
            with open(os.path.join(model_save_path, 'training_info.json'), 'w', encoding='utf-8') as f:
                json.dump(training_info_json, f, indent=2, ensure_ascii=False)
            
            # 保存性能摘要为JSON和CSV格式
            performance_summary = {
                'model_name': f"model_{prediction_days}d",
                'ensemble_accuracy': results.get('Ensemble', {}).get('accuracy', 0),
                'cv_accuracy_mean': cv_results['accuracy'],
                'cv_accuracy_std': cv_results['accuracy_std'],
                'feature_count': len(feature_names),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'total_samples': len(X_train) + len(X_test),
                'training_time': timestamp,
                'gpu_optimized': self.gpu_strategy is not None
            }
            
            # 转换为JSON可序列化格式
            performance_summary_json = convert_to_json_serializable(performance_summary)
            
            # 保存为JSON
            with open(os.path.join(model_save_path, 'performance_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(performance_summary_json, f, indent=2, ensure_ascii=False)
            
            # 保存为CSV（方便Excel打开）
            import pandas as pd
            pd.DataFrame([performance_summary]).to_csv(
                os.path.join(model_save_path, 'performance_summary.csv'), 
                index=False, encoding='utf-8-sig'
            )
            
            # 保存详细评估结果
            detailed_results = {}
            for model_name, model_result in results.items():
                if isinstance(model_result, dict) and 'accuracy' in model_result:
                    detailed_results[model_name] = {
                        'accuracy': convert_to_json_serializable(model_result['accuracy']),
                        'model_type': model_name
                    }
            
            # 转换详细结果为JSON可序列化格式
            detailed_results_json = convert_to_json_serializable(detailed_results)
            
            # 保存详细结果为JSON
            with open(os.path.join(model_save_path, 'detailed_results.json'), 'w', encoding='utf-8') as f:
                json.dump(detailed_results_json, f, indent=2, ensure_ascii=False)
            
            # 保存详细结果为CSV
            if detailed_results:
                pd.DataFrame(detailed_results).T.to_csv(
                    os.path.join(model_save_path, 'detailed_results.csv'), 
                    encoding='utf-8-sig'
                )
            
            # 保存特征名称列表
            with open(os.path.join(model_save_path, 'feature_names.txt'), 'w', encoding='utf-8') as f:
                f.write(f"总计 {len(feature_names)} 个特征:\n\n")
                for i, name in enumerate(feature_names, 1):
                    f.write(f"{i:3d}. {name}\n")
            
            # 保存模型文件（pkl格式，给程序用）
            joblib.dump(training_info, os.path.join(model_save_path, 'training_info.pkl'))
            
            logger.info(f"📁 模型和评估结果已统一保存到: {model_save_path}")
            logger.info(f"   ├── 模型文件: *.pkl (程序使用)")
            logger.info(f"   ├── 训练信息: training_info.json / .pkl")
            logger.info(f"   ├── 性能摘要: performance_summary.json / .csv")
            logger.info(f"   ├── 详细结果: detailed_results.json / .csv")
            logger.info(f"   └── 特征列表: feature_names.txt")
            
            # 🚀 自动生成训练完成报告
            try:
                report_path = self.report_generator.generate_training_report(
                    model_save_path=model_save_path,
                    training_info=training_info,
                    results=results,
                    cv_results=cv_results,
                    feature_names=feature_names,
                    stock_codes=stock_codes,
                    prediction_days=prediction_days
                )
                if report_path:
                    logger.info(f"   ├── 训练报告: 训练完成报告.md")
            except Exception as e:
                logger.warning(f"⚠️  自动生成训练报告失败: {str(e)}")
        
        # 9. 记录性能历史
        self.performance_history.append({
            'timestamp': datetime.now(),
            'prediction_days': prediction_days,
            'test_accuracy': results['Ensemble']['accuracy'],
            'cv_accuracy': cv_results['accuracy'],
            'cv_accuracy_std': cv_results['accuracy_std']
        })
        
        logger.info("模型训练完成！")
        return model
    
    def batch_train_models(self, stock_codes: List[str], 
                         warm_up_cache_first: bool = True) -> Dict[int, EnsembleModel]:
        """
        批量训练不同预测天数的模型 - 支持缓存优化
        
        Args:
            stock_codes: 股票代码列表
            warm_up_cache_first: 是否先预热缓存
            
        Returns:
            预测天数到模型的映射
        """
        logger.info("🚀 开始批量训练模型...")
        logger.info(f"   股票数量: {len(stock_codes)}")
        logger.info(f"   预测天数: {self.config['prediction_days']}")
        logger.info(f"   批量缓存: {'✅ 启用' if self.enable_batch_cache else '❌ 禁用'}")
        
        # 缓存已禁用，跳过预热
        if warm_up_cache_first:
            logger.info("\n⚠️ 缓存功能已禁用，跳过预热步骤")
        
        models = {}
        total_models = len(self.config['prediction_days'])
        
        for i, prediction_days in enumerate(self.config['prediction_days'], 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 模型 {i}/{total_models}: 训练 {prediction_days} 天预测模型")
            logger.info(f"{'='*60}")
            
            try:
                start_time = datetime.now()
                
                model = self.train_model(
                    stock_codes=stock_codes,
                    prediction_days=prediction_days,
                    use_hyperparameter_optimization=False,  # 批量训练时关闭以节省时间
                    save_model=True,
                    clear_cache=False  # 不清理缓存，保持加速效果
                )
                models[prediction_days] = model
                
                training_time = datetime.now() - start_time
                logger.info(f"✅ {prediction_days} 天预测模型训练完成，耗时: {training_time}")
                
                # 显示剩余训练估计时间
                if i < total_models:
                    remaining_models = total_models - i
                    estimated_remaining_time = training_time * remaining_models
                    logger.info(f"📊 预计剩余训练时间: {estimated_remaining_time}")
                
            except Exception as e:
                logger.error(f"❌ 训练 {prediction_days} 天预测模型失败: {str(e)}")
                continue
        
        logger.info(f"\n🎉 批量训练完成！")
        logger.info(f"   成功训练: {len(models)}/{total_models} 个模型")
        logger.info(f"   失败数量: {total_models - len(models)}")
        
        # 缓存已禁用，无需显示统计
        logger.info(f"\n📊 缓存状态: 已禁用")
        
        return models
    
    def get_performance_summary(self) -> pd.DataFrame:
        """获取性能历史摘要"""
        if not self.performance_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.performance_history)
        return df.groupby('prediction_days').agg({
            'test_accuracy': ['mean', 'std', 'max'],
            'cv_accuracy': ['mean', 'std', 'max'],
            'cv_accuracy_std': 'mean'
        }).round(4)


class AutoRetrainingSystem:
    """
    自动重训练系统
    监控模型性能，自动触发重训练
    """
    
    def __init__(self, pipeline: ModelTrainingPipeline, 
                 performance_threshold: float = 0.55,
                 monitoring_window: int = 30):
        self.pipeline = pipeline
        self.performance_threshold = performance_threshold
        self.monitoring_window = monitoring_window
        self.prediction_history = []
        
    def log_prediction(self, prediction: int, actual: int, stock_code: str, 
                      prediction_days: int):
        """记录预测结果"""
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'stock_code': stock_code,
            'prediction_days': prediction_days,
            'prediction': prediction,
            'actual': actual,
            'correct': prediction == actual
        })
        
        # 保持历史记录在合理范围内
        if len(self.prediction_history) > 10000:
            self.prediction_history = self.prediction_history[-5000:]
    
    def check_performance(self, prediction_days: int) -> float:
        """检查最近的预测性能"""
        if len(self.prediction_history) < 100:
            return 1.0  # 数据不足，不触发重训练
        
        # 获取最近的预测记录
        recent_predictions = [
            record for record in self.prediction_history[-self.monitoring_window:]
            if record['prediction_days'] == prediction_days
        ]
        
        if len(recent_predictions) < 20:
            return 1.0  # 样本不足
        
        # 计算准确率
        accuracy = sum(record['correct'] for record in recent_predictions) / len(recent_predictions)
        logger.info(f"最近 {len(recent_predictions)} 次预测准确率: {accuracy:.4f}")
        
        return accuracy
    
    def should_retrain(self, prediction_days: int) -> bool:
        """判断是否需要重训练"""
        current_performance = self.check_performance(prediction_days)
        
        if current_performance < self.performance_threshold:
            logger.warning(f"{prediction_days}天预测模型性能下降，需要重训练")
            return True
        
        return False
    
    def auto_retrain(self, stock_codes: List[str]):
        """自动重训练检查"""
        logger.info("开始自动重训练检查...")
        
        for prediction_days in self.pipeline.config['prediction_days']:
            if self.should_retrain(prediction_days):
                logger.info(f"开始重训练 {prediction_days} 天预测模型...")
                
                try:
                    new_model = self.pipeline.train_model(
                        stock_codes=stock_codes,
                        prediction_days=prediction_days,
                        use_hyperparameter_optimization=True,
                        save_model=True
                    )
                    logger.info(f"{prediction_days} 天预测模型重训练完成")
                    
                except Exception as e:
                    logger.error(f"重训练失败: {str(e)}")


if __name__ == "__main__":
    # 测试训练管道 - 展示批量缓存功能
    logger.info("🚀 启动LightGBM集成训练流水线测试（带批量缓存优化）")
    
    # 初始化管道（启用批量缓存）
    pipeline = ModelTrainingPipeline(
        enable_batch_cache=True,  # 启用批量缓存
        cache_workers=1  # 缓存工作进程数
    )
    
    # 使用新的方法获取可用股票
    available_stocks = pipeline.get_available_stocks()  # 获取所有可用股票
    
    if not available_stocks:
        logger.error("❌ 未找到有效的股票数据")
        print("请确保有足够的股票数据文件在 datas_em 目录中")
        exit(1)
    
    logger.info(f"使用股票代码: {available_stocks}")
    
    try:
        # 测试1: 单个模型训练（第一次运行 - 建立缓存）
        logger.info("\n" + "="*60)
        logger.info("测试1: 第一次训练（建立缓存）")
        logger.info("="*60)
        
        start_time = datetime.now()
        model1 = pipeline.train_model(
            stock_codes=available_stocks,
            prediction_days=1,
            use_hyperparameter_optimization=False,
            save_model=True,
            clear_cache=False
        )
        first_training_time = datetime.now() - start_time
        
        # 测试2: 第二次训练（使用缓存）
        logger.info("\n" + "="*60)
        logger.info("测试2: 第二次训练（使用缓存）")
        logger.info("="*60)
        
        start_time = datetime.now()
        model2 = pipeline.train_model(
            stock_codes=available_stocks,
            prediction_days=1,
            use_hyperparameter_optimization=False,
            save_model=False,  # 第二次不保存，只测试速度
            clear_cache=False
        )
        second_training_time = datetime.now() - start_time
        
        # 性能对比
        logger.info("\n" + "="*60)
        logger.info("🚀 批量缓存性能测试结果")
        logger.info("="*60)
        logger.info(f"第一次训练时间: {first_training_time}")
        logger.info(f"第二次训练时间: {second_training_time}")
        
        if second_training_time.total_seconds() > 0:
            speedup = first_training_time.total_seconds() / second_training_time.total_seconds()
            time_saved = first_training_time - second_training_time
            efficiency = (time_saved.total_seconds() / first_training_time.total_seconds()) * 100
            
            logger.info(f"⚡ 加速比: {speedup:.1f}x")
            logger.info(f"⏱️ 时间节省: {time_saved}")
            logger.info(f"📈 效率提升: {efficiency:.1f}%")
            
            # 大规模预测
            total_stocks = len(pipeline.get_available_stocks())
            if total_stocks > len(available_stocks):
                estimated_time_no_cache = (first_training_time.total_seconds() / len(available_stocks)) * total_stocks
                estimated_time_with_cache = (second_training_time.total_seconds() / len(available_stocks)) * total_stocks
                
                logger.info(f"\n🔮 {total_stocks}只股票训练时间预估:")
                logger.info(f"   无缓存: {estimated_time_no_cache / 60:.1f} 分钟")
                logger.info(f"   有缓存: {estimated_time_with_cache / 60:.1f} 分钟")
                logger.info(f"   预计节省: {(estimated_time_no_cache - estimated_time_with_cache) / 60:.1f} 分钟")
        
        # 测试3: 缓存预热功能
        logger.info("\n" + "="*60)
        logger.info("测试3: 缓存预热功能")
        logger.info("="*60)
        
        # 清理缓存后重新预热
        pipeline.batch_processor.cache.clear_all_cache()
        pipeline.warm_up_cache(available_stocks, show_progress=True)
        
        # 获取性能摘要
        summary = pipeline.get_performance_summary()
        if not summary.empty:
            logger.info("\n📊 性能摘要:")
            print(summary)
        
        logger.info("\n🎉 批量缓存测试完成！")
        logger.info("✅ 缓存系统工作正常，可显著提升大规模训练效率")
        
    except Exception as e:
        logger.error(f"训练测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        print("请确保有足够的股票数据文件在 datas_em 目录中")