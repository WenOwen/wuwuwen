# -*- coding: utf-8 -*-
"""
AI股市预测系统 - 训练管道模块
功能：自动化模型训练、验证、超参数优化和部署管道
"""

import os
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

from .feature_engineering import FeatureEngineering
from .ai_models import EnsembleModel, create_ensemble_model
from .enhanced_ai_models import create_enhanced_ensemble_model
from ..utils.gpu_config import setup_dual_gpu, get_optimal_batch_size

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    模型训练管道类
    负责数据准备、模型训练、验证、超参数优化等
    """
    
    def __init__(self, data_dir: str = "datas_em", model_dir: str = "models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.feature_engineer = FeatureEngineering()
        self.models = {}
        self.performance_history = []
        
        # 🚀 配置双GPU环境
        self.gpu_strategy = setup_dual_gpu()
        
        # 创建模型保存目录
        os.makedirs(model_dir, exist_ok=True)
        
        # 训练配置
        self.config = {
            'sequence_length': 60,
            'prediction_days': [1, 3, 5],
            'train_test_split': 0.8,
            'validation_split': 0.2,
            'min_samples': 500,  # 最少样本数
            'performance_threshold': 0.55,  # 最低准确率要求
        }
    
    def load_stock_data(self, stock_code: str) -> pd.DataFrame:
        """加载单只股票数据"""
        file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"股票数据文件不存在: {file_path}")
        
        df = pd.read_csv(file_path)
        df['交易日期'] = pd.to_datetime(df['交易日期'])
        df = df.sort_values('交易日期').reset_index(drop=True)
        
        logger.info(f"加载股票 {stock_code} 数据，共 {len(df)} 条记录")
        return df
    
    def prepare_training_data(self, stock_codes: List[str], 
                            prediction_days: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据
        
        Args:
            stock_codes: 股票代码列表
            prediction_days: 预测天数
            
        Returns:
            X, y, feature_names
        """
        logger.info(f"开始准备训练数据，股票数量: {len(stock_codes)}")
        
        all_X, all_y = [], []
        feature_names = None
        feature_info = None
        
        for stock_code in stock_codes:
            try:
                # 加载股票数据
                df = self.load_stock_data(stock_code)
                
                if len(df) < self.config['min_samples']:
                    logger.warning(f"股票 {stock_code} 数据量不足，跳过")
                    continue
                
                # 特征工程（传递股票代码）
                df_features = self.feature_engineer.create_all_features(df, stock_code)
                
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
                logger.error(f"处理股票 {stock_code} 时出错: {str(e)}")
                continue
        
        if not all_X:
            raise ValueError("没有成功处理的股票数据")
        
        # 合并所有股票数据
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)
        
        logger.info(f"总计样本数: {len(X_combined)}, 特征数: {len(feature_names)}")
        logger.info(f"正样本比例: {y_combined.mean():.3f}")
        
        return X_combined, y_combined, feature_names, feature_info
    
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
                           n_splits: int = 3) -> Dict[str, float]:  # 从5折减少到3折
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
        logger.info("开始交叉验证...")
        
        if model is None:
            if feature_info:
                model = create_enhanced_ensemble_model(
                    sequence_length=self.config['sequence_length'],
                    n_features=X.shape[-1],
                    n_stocks=feature_info.get('n_stocks', 100),
                    n_sectors=feature_info.get('n_sectors', 20)
                )
            else:
                model = create_ensemble_model(
                    sequence_length=self.config['sequence_length'],
                    n_features=X.shape[-1]
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
            cv_epochs = 10  # 交叉验证减少epochs加快速度
            cv_batch_size = get_optimal_batch_size(32, 2)  # 动态计算最优batch size
            
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
            
            # LightGBM参数
            lgb_max_depth = trial.suggest_int('lgb_max_depth', 3, 10)
            lgb_learning_rate = trial.suggest_float('lgb_learning_rate', 0.01, 0.3)
            lgb_n_estimators = trial.suggest_int('lgb_n_estimators', 100, 1000, step=100)
            
            # Transformer参数
            d_model = trial.suggest_int('d_model', 64, 256, step=64)
            num_heads = trial.suggest_int('num_heads', 4, 16, step=4)
            
            try:
                # 创建模型（这里简化，实际需要传入参数）
                from ai_models import LSTMModel, LightGBMModel, TransformerModel, CNNLSTMModel
                
                ensemble = EnsembleModel()
                
                # 添加优化后的模型
                lstm_model = LSTMModel(
                    sequence_length=self.config['sequence_length'],
                    n_features=X_train.shape[-1],
                    lstm_units=lstm_units,
                    dropout_rate=dropout_rate
                )
                
                lgb_model = LightGBMModel(
                    max_depth=lgb_max_depth,
                    learning_rate=lgb_learning_rate,
                    n_estimators=lgb_n_estimators
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
                
                ensemble.add_model(lstm_model)
                ensemble.add_model(lgb_model)
                ensemble.add_model(transformer_model)
                ensemble.add_model(cnn_lstm_model)
                
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
    
    def train_model(self, stock_codes: List[str], 
                   prediction_days: int = 1,
                   use_hyperparameter_optimization: bool = True,
                   save_model: bool = True) -> EnsembleModel:
        """
        训练模型主流程
        
        Args:
            stock_codes: 股票代码列表
            prediction_days: 预测天数
            use_hyperparameter_optimization: 是否使用超参数优化
            save_model: 是否保存模型
            
        Returns:
            训练好的集成模型
        """
        logger.info(f"开始训练模型，预测 {prediction_days} 天")
        
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
        
        # 5. 创建和训练最终模型（使用增强模型）
        if feature_info:
            model = create_enhanced_ensemble_model(
                sequence_length=self.config['sequence_length'],
                n_features=X.shape[-1],
                n_stocks=feature_info.get('n_stocks', 100),
                n_sectors=feature_info.get('n_sectors', 20)
            )
        else:
            # 回退到原始模型
            model = create_ensemble_model(
                sequence_length=self.config['sequence_length'],
                n_features=X.shape[-1]
            )
        
        # 使用全部训练数据进行最终训练（传递feature_info）
        # 🚀 双3090优化配置
        optimized_epochs = 30  # 从100减少到30，平衡速度和效果
        optimized_batch_size = get_optimal_batch_size(48, 2)  # 动态计算最优batch size
        
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
            model_save_path = os.path.join(
                self.model_dir, 
                f"model_{prediction_days}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            model.save_models(model_save_path)
            
            # 保存训练信息
            training_info = {
                'stock_codes': stock_codes,
                'prediction_days': prediction_days,
                'feature_names': feature_names,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'test_results': results,
                'cv_results': cv_results,
                'best_params': best_params,
                'training_time': datetime.now().isoformat()
            }
            
            joblib.dump(training_info, os.path.join(model_save_path, 'training_info.pkl'))
            logger.info(f"模型已保存到: {model_save_path}")
        
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
    
    def batch_train_models(self, stock_codes: List[str]) -> Dict[int, EnsembleModel]:
        """
        批量训练不同预测天数的模型
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            预测天数到模型的映射
        """
        logger.info("开始批量训练模型...")
        
        models = {}
        for prediction_days in self.config['prediction_days']:
            logger.info(f"\n{'='*50}")
            logger.info(f"训练 {prediction_days} 天预测模型")
            logger.info(f"{'='*50}")
            
            try:
                model = self.train_model(
                    stock_codes=stock_codes,
                    prediction_days=prediction_days,
                    use_hyperparameter_optimization=False,  # 批量训练时关闭以节省时间
                    save_model=True
                )
                models[prediction_days] = model
                
            except Exception as e:
                logger.error(f"训练 {prediction_days} 天预测模型失败: {str(e)}")
                continue
        
        logger.info(f"批量训练完成，成功训练 {len(models)} 个模型")
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
    # 测试训练管道
    pipeline = ModelTrainingPipeline()
    
    # 🚀 快速调试：只用一只股票
    stock_codes = ['sz301636','sh600000']  # 先用一只有数据的股票快速验证
    
    try:
        # 训练单个模型
        model = pipeline.train_model(
            stock_codes=stock_codes,
            prediction_days=1,
            use_hyperparameter_optimization=False,
            save_model=False  # 快速调试时不保存模型
        )
        
        # 获取性能摘要
        summary = pipeline.get_performance_summary()
        print("\n性能摘要:")
        print(summary)
        
    except Exception as e:
        logger.error(f"训练测试失败: {str(e)}")
        print("请确保有足够的股票数据文件在 datas_em 目录中")