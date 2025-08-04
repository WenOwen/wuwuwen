# -*- coding: utf-8 -*-
"""
AI股市预测系统 - 模型架构模块
功能：实现多模型融合的股市预测系统
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

# 深度学习框架
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Attention, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, RobustScaler

# 机器学习模型
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 模型评估
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error


class StockPredictionModel:
    """
    股票预测模型基类
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.is_fitted = False
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        raise NotImplementedError
        
    def predict(self, X):
        """预测"""
        raise NotImplementedError
        
    def predict_proba(self, X):
        """预测概率"""
        raise NotImplementedError


class LSTMModel(StockPredictionModel):
    """
    LSTM时序预测模型
    """
    
    def __init__(self, sequence_length=60, n_features=50, 
                 lstm_units=128, dropout_rate=0.2):
        super().__init__("LSTM")
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.scaler = StandardScaler()
        
    def _build_model(self):
        """构建LSTM模型"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(self.dropout_rate),
            
            LSTM(self.lstm_units // 2, return_sequences=True),
            Dropout(self.dropout_rate),
            
            LSTM(self.lstm_units // 4, return_sequences=False),
            Dropout(self.dropout_rate),
            
            Dense(64, activation='relu'),
            Dropout(self.dropout_rate),
            
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')  # 二分类输出
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """训练LSTM模型"""
        print(f"🔥 开始训练{self.model_name}模型...")
        
        # 数据预处理
        X_train_scaled = self._preprocess_data(X_train, fit_scaler=True)
        
        if X_val is not None:
            X_val_scaled = self._preprocess_data(X_val, fit_scaler=False)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # 构建模型
        self.model = self._build_model()
        
        # 训练回调
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # 训练模型
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_fitted = True
        print(f"✅ {self.model_name}模型训练完成")
        return history
    
    def _preprocess_data(self, X, fit_scaler=False):
        """数据预处理"""
        # 重塑数据为2D用于标准化
        n_samples, n_timesteps, n_features = X.shape
        X_2d = X.reshape(-1, n_features)
        
        if fit_scaler:
            X_scaled_2d = self.scaler.fit_transform(X_2d)
        else:
            X_scaled_2d = self.scaler.transform(X_2d)
        
        # 重塑回3D
        X_scaled = X_scaled_2d.reshape(n_samples, n_timesteps, n_features)
        return X_scaled
    
    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        X_scaled = self._preprocess_data(X, fit_scaler=False)
        predictions = self.model.predict(X_scaled)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        X_scaled = self._preprocess_data(X, fit_scaler=False)
        probas = self.model.predict(X_scaled)
        # 返回[neg_proba, pos_proba]格式
        return np.column_stack([1 - probas.flatten(), probas.flatten()])


class TransformerModel(StockPredictionModel):
    """
    Transformer注意力机制模型
    """
    
    def __init__(self, sequence_length=60, n_features=50, 
                 d_model=128, num_heads=8, num_layers=4):
        super().__init__("Transformer")
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.scaler = StandardScaler()
        
    def _build_model(self):
        """构建Transformer模型"""
        inputs = tf.keras.Input(shape=(self.sequence_length, self.n_features))
        
        # 输入投影层
        x = Dense(self.d_model)(inputs)
        
        # 多层Transformer块
        for _ in range(self.num_layers):
            # 多头注意力
            attn_output = MultiHeadAttention(
                num_heads=self.num_heads, 
                key_dim=self.d_model // self.num_heads
            )(x, x)
            
            # 残差连接和层归一化
            x = LayerNormalization()(x + attn_output)
            
            # 前馈网络
            ffn_output = Dense(self.d_model * 2, activation='relu')(x)
            ffn_output = Dense(self.d_model)(ffn_output)
            
            # 残差连接和层归一化
            x = LayerNormalization()(x + ffn_output)
        
        # 全局平均池化
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # 分类头
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """训练Transformer模型"""
        print(f"🔥 开始训练{self.model_name}模型...")
        
        # 数据预处理（与LSTM相同）
        X_train_scaled = self._preprocess_data(X_train, fit_scaler=True)
        
        if X_val is not None:
            X_val_scaled = self._preprocess_data(X_val, fit_scaler=False)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # 构建模型
        self.model = self._build_model()
        
        # 训练回调
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # 训练模型
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_fitted = True
        print(f"✅ {self.model_name}模型训练完成")
        return history
    
    def _preprocess_data(self, X, fit_scaler=False):
        """数据预处理（与LSTM相同）"""
        n_samples, n_timesteps, n_features = X.shape
        X_2d = X.reshape(-1, n_features)
        
        if fit_scaler:
            X_scaled_2d = self.scaler.fit_transform(X_2d)
        else:
            X_scaled_2d = self.scaler.transform(X_2d)
        
        X_scaled = X_scaled_2d.reshape(n_samples, n_timesteps, n_features)
        return X_scaled
    
    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        X_scaled = self._preprocess_data(X, fit_scaler=False)
        predictions = self.model.predict(X_scaled)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        X_scaled = self._preprocess_data(X, fit_scaler=False)
        probas = self.model.predict(X_scaled)
        return np.column_stack([1 - probas.flatten(), probas.flatten()])


class XGBoostModel(StockPredictionModel):
    """
    XGBoost梯度提升模型
    """
    
    def __init__(self, **params):
        super().__init__("XGBoost")
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            **params
        }
        self.scaler = RobustScaler()
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """训练XGBoost模型"""
        print(f"🔥 开始训练{self.model_name}模型...")
        
        # 如果输入是3D（时序数据），需要flatten
        if len(X_train.shape) == 3:
            X_train_2d = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_2d = X_train
        
        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train_2d)
        
        # 验证集处理
        eval_set = None
        if X_val is not None and y_val is not None:
            if len(X_val.shape) == 3:
                X_val_2d = X_val.reshape(X_val.shape[0], -1)
            else:
                X_val_2d = X_val
            X_val_scaled = self.scaler.transform(X_val_2d)
            eval_set = [(X_val_scaled, y_val)]
        
        # 训练模型
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )
        
        self.is_fitted = True
        print(f"✅ {self.model_name}模型训练完成")
        
    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        if len(X.shape) == 3:
            X_2d = X.reshape(X.shape[0], -1)
        else:
            X_2d = X
            
        X_scaled = self.scaler.transform(X_2d)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        if len(X.shape) == 3:
            X_2d = X.reshape(X.shape[0], -1)
        else:
            X_2d = X
            
        X_scaled = self.scaler.transform(X_2d)
        return self.model.predict_proba(X_scaled)


class CNNLSTMModel(StockPredictionModel):
    """
    CNN-LSTM混合模型，用于捕捉局部模式和长期依赖
    """
    
    def __init__(self, sequence_length=60, n_features=50):
        super().__init__("CNN-LSTM")
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.scaler = StandardScaler()
        
    def _build_model(self):
        """构建CNN-LSTM模型"""
        model = Sequential([
            # CNN层提取局部特征
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                   input_shape=(self.sequence_length, self.n_features)),
            MaxPooling1D(pool_size=2),
            
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            
            # LSTM层捕捉时序依赖
            LSTM(100, return_sequences=True),
            Dropout(0.3),
            
            LSTM(50, return_sequences=False),
            Dropout(0.3),
            
            # 全连接层
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """训练模型"""
        print(f"🔥 开始训练{self.model_name}模型...")
        
        # 数据预处理
        X_train_scaled = self._preprocess_data(X_train, fit_scaler=True)
        
        if X_val is not None:
            X_val_scaled = self._preprocess_data(X_val, fit_scaler=False)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # 构建模型
        self.model = self._build_model()
        
        # 训练回调
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # 训练模型
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_fitted = True
        print(f"✅ {self.model_name}模型训练完成")
        return history
    
    def _preprocess_data(self, X, fit_scaler=False):
        """数据预处理"""
        n_samples, n_timesteps, n_features = X.shape
        X_2d = X.reshape(-1, n_features)
        
        if fit_scaler:
            X_scaled_2d = self.scaler.fit_transform(X_2d)
        else:
            X_scaled_2d = self.scaler.transform(X_2d)
        
        X_scaled = X_scaled_2d.reshape(n_samples, n_timesteps, n_features)
        return X_scaled
    
    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        X_scaled = self._preprocess_data(X, fit_scaler=False)
        predictions = self.model.predict(X_scaled)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        X_scaled = self._preprocess_data(X, fit_scaler=False)
        probas = self.model.predict(X_scaled)
        return np.column_stack([1 - probas.flatten(), probas.flatten()])


class EnsembleModel:
    """
    集成模型 - 多模型融合预测系统
    """
    
    def __init__(self, model_weights: Dict[str, float] = None):
        """
        初始化集成模型
        
        Args:
            model_weights: 模型权重字典，如 {'LSTM': 0.4, 'XGBoost': 0.3, 'Transformer': 0.2, 'CNN-LSTM': 0.1}
        """
        self.models = {}
        self.model_weights = model_weights or {
            'LSTM': 0.4,
            'XGBoost': 0.3, 
            'Transformer': 0.2,
            'CNN-LSTM': 0.1
        }
        self.performance_history = []
        
    def add_model(self, model: StockPredictionModel):
        """添加模型到集成"""
        self.models[model.model_name] = model
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """训练所有模型"""
        print("🚀 开始训练集成模型...")
        
        # 训练各个子模型
        for model_name, model in self.models.items():
            print(f"\n--- 训练 {model_name} ---")
            model.fit(X_train, y_train, X_val, y_val, **kwargs)
        
        # 动态调整权重（基于验证集性能）
        if X_val is not None and y_val is not None:
            self._adjust_weights(X_val, y_val)
        
        print("✅ 集成模型训练完成")
        
    def _adjust_weights(self, X_val, y_val):
        """基于验证集性能动态调整模型权重"""
        print("\n🔧 基于验证集性能调整模型权重...")
        
        model_scores = {}
        for model_name, model in self.models.items():
            if model.is_fitted:
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                model_scores[model_name] = accuracy
                print(f"{model_name} 验证准确率: {accuracy:.4f}")
        
        # 基于性能重新分配权重
        total_score = sum(model_scores.values())
        if total_score > 0:
            for model_name in self.model_weights.keys():
                if model_name in model_scores:
                    self.model_weights[model_name] = model_scores[model_name] / total_score
        
        print("调整后的模型权重:")
        for model_name, weight in self.model_weights.items():
            print(f"  {model_name}: {weight:.3f}")
    
    def predict(self, X):
        """集成预测"""
        if not self.models:
            raise ValueError("没有可用的模型")
        
        # 收集各模型预测
        predictions = {}
        for model_name, model in self.models.items():
            if model.is_fitted:
                predictions[model_name] = model.predict(X)
        
        if not predictions:
            raise ValueError("没有已训练的模型")
        
        # 加权平均（硬投票）
        ensemble_pred = np.zeros(len(predictions[list(predictions.keys())[0]]))
        
        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 0)
            ensemble_pred += weight * pred
        
        return (ensemble_pred > 0.5).astype(int)
    
    def predict_proba(self, X):
        """集成概率预测"""
        if not self.models:
            raise ValueError("没有可用的模型")
        
        # 收集各模型概率预测
        prob_predictions = {}
        for model_name, model in self.models.items():
            if model.is_fitted:
                prob_predictions[model_name] = model.predict_proba(X)
        
        if not prob_predictions:
            raise ValueError("没有已训练的模型")
        
        # 加权平均概率
        first_pred = list(prob_predictions.values())[0]
        ensemble_proba = np.zeros_like(first_pred)
        
        for model_name, proba in prob_predictions.items():
            weight = self.model_weights.get(model_name, 0)
            ensemble_proba += weight * proba
        
        return ensemble_proba
    
    def evaluate(self, X_test, y_test):
        """评估集成模型性能"""
        print("\n📊 评估模型性能...")
        
        # 各子模型性能
        results = {}
        for model_name, model in self.models.items():
            if model.is_fitted:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                results[model_name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'probabilities': y_proba
                }
                
                print(f"{model_name} 测试准确率: {accuracy:.4f}")
        
        # 集成模型性能
        ensemble_pred = self.predict(X_test)
        ensemble_proba = self.predict_proba(X_test)[:, 1]
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        results['Ensemble'] = {
            'accuracy': ensemble_accuracy,
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba
        }
        
        print(f"集成模型 测试准确率: {ensemble_accuracy:.4f}")
        
        # 详细报告
        print("\n集成模型分类报告:")
        print(classification_report(y_test, ensemble_pred))
        
        return results
    
    def save_models(self, save_dir: str):
        """保存所有模型"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model.is_fitted:
                if hasattr(model.model, 'save'):  # Keras模型
                    model.model.save(f"{save_dir}/{model_name}_model.h5")
                    joblib.dump(model.scaler, f"{save_dir}/{model_name}_scaler.pkl")
                else:  # sklearn模型
                    joblib.dump(model.model, f"{save_dir}/{model_name}_model.pkl")
                    joblib.dump(model.scaler, f"{save_dir}/{model_name}_scaler.pkl")
        
        # 保存权重
        joblib.dump(self.model_weights, f"{save_dir}/model_weights.pkl")
        print(f"✅ 模型已保存到 {save_dir}")


def create_ensemble_model(sequence_length=60, n_features=50) -> EnsembleModel:
    """
    创建完整的集成模型
    
    Args:
        sequence_length: 时序长度
        n_features: 特征数量
        
    Returns:
        EnsembleModel实例
    """
    
    # 创建集成模型
    ensemble = EnsembleModel()
    
    # 添加各个子模型
    lstm_model = LSTMModel(sequence_length, n_features)
    xgb_model = XGBoostModel()
    transformer_model = TransformerModel(sequence_length, n_features)
    cnn_lstm_model = CNNLSTMModel(sequence_length, n_features)
    
    ensemble.add_model(lstm_model)
    ensemble.add_model(xgb_model)
    ensemble.add_model(transformer_model)
    ensemble.add_model(cnn_lstm_model)
    
    return ensemble


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试AI模型架构...")
    
    # 模拟数据
    n_samples, sequence_length, n_features = 1000, 60, 50
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # 划分训练集和测试集
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 创建和训练集成模型
    ensemble = create_ensemble_model(sequence_length, n_features)
    ensemble.fit(X_train, y_train, X_test, y_test, epochs=5, batch_size=32)
    
    # 评估模型
    results = ensemble.evaluate(X_test, y_test)
    
    print("🎉 测试完成！")