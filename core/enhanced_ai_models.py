# -*- coding: utf-8 -*-
"""
增强AI股市预测系统 - 支持股票标识和板块效应的模型架构
功能：实现包含Embedding层的多模型融合预测系统
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

# 深度学习框架
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, 
    Embedding, Concatenate, MultiHeadAttention, LayerNormalization,
    Add, GlobalAveragePooling1D, Lambda, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, RobustScaler

# 机器学习模型
import lightgbm as lgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 导入原有基类
from ai_models import StockPredictionModel, EnsembleModel


class EnhancedLSTMModel(StockPredictionModel):
    """
    增强LSTM模型 - 支持股票和板块Embedding
    """
    
    def __init__(self, sequence_length=60, n_features=50, n_stocks=100, n_sectors=20,
                 lstm_units=128, dropout_rate=0.3, embedding_dim=32):
        super().__init__("Enhanced_LSTM")
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_stocks = n_stocks
        self.n_sectors = n_sectors
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.scaler = StandardScaler()
        
        # 记录分类特征的索引
        self.stock_id_idx = None
        self.sector_id_idx = None
        self.categorical_indices = []
        self.numerical_indices = []
        
    def _build_model(self, feature_info: Dict):
        """构建增强LSTM模型"""
        
        # 更新特征映射
        self.n_stocks = feature_info.get('n_stocks', self.n_stocks)
        self.n_sectors = feature_info.get('n_sectors', self.n_sectors)
        
        # 找到分类特征的索引
        all_cols = feature_info['all_cols']
        categorical_cols = feature_info['categorical_cols']
        numerical_cols = feature_info['numerical_cols']
        
        self.categorical_indices = [all_cols.index(col) for col in categorical_cols if col in all_cols]
        self.numerical_indices = [all_cols.index(col) for col in numerical_cols if col in all_cols]
        
        if 'stock_id' in all_cols:
            self.stock_id_idx = all_cols.index('stock_id')
        if 'sector_id' in all_cols:
            self.sector_id_idx = all_cols.index('sector_id')
        
        # 输入层
        main_input = Input(shape=(self.sequence_length, self.n_features), name='main_input')
        
        # 分离数值特征和分类特征 - 使用Keras层而非TF函数
        if self.categorical_indices:
            # 提取分类特征（取最后一个时间步的值）
            # 使用Lambda层来安全地提取特征
            def extract_categorical_features(x):
                # 取最后一个时间步
                last_step = x[:, -1, :]
                # 提取分类特征
                cat_features = tf.gather(last_step, self.categorical_indices, axis=1)
                return cat_features
            
            categorical_features = Lambda(extract_categorical_features)(main_input)
            
            # 股票ID Embedding
            if self.stock_id_idx is not None and 'stock_id' in categorical_cols:
                stock_id_idx = categorical_cols.index('stock_id')
                
                def extract_stock_id(x):
                    return tf.cast(x[:, stock_id_idx:stock_id_idx+1], tf.int32)
                
                stock_id = Lambda(extract_stock_id)(categorical_features)
                stock_id_flat = Reshape((1,))(stock_id)
                stock_embedding = Embedding(self.n_stocks, self.embedding_dim, name='stock_embedding')(stock_id_flat)
                stock_embedding = Reshape((self.embedding_dim,))(stock_embedding)
            else:
                # 创建零向量
                def create_zero_embedding(x):
                    batch_size = tf.shape(x)[0]
                    return tf.zeros((batch_size, self.embedding_dim))
                
                stock_embedding = Lambda(create_zero_embedding)(main_input)
            
            # 板块ID Embedding
            if self.sector_id_idx is not None and 'sector_id' in categorical_cols:
                sector_id_idx = categorical_cols.index('sector_id')
                
                def extract_sector_id(x):
                    return tf.cast(x[:, sector_id_idx:sector_id_idx+1], tf.int32)
                
                sector_id = Lambda(extract_sector_id)(categorical_features)
                sector_id_flat = Reshape((1,))(sector_id)
                sector_embedding = Embedding(self.n_sectors, self.embedding_dim, name='sector_embedding')(sector_id_flat)
                sector_embedding = Reshape((self.embedding_dim,))(sector_embedding)
            else:
                # 创建零向量
                def create_zero_embedding(x):
                    batch_size = tf.shape(x)[0]
                    return tf.zeros((batch_size, self.embedding_dim))
                
                sector_embedding = Lambda(create_zero_embedding)(main_input)
            
            # 其他分类特征
            other_categorical = []
            for i, col in enumerate(categorical_cols):
                if col not in ['stock_id', 'sector_id']:
                    def extract_feature(x, idx=i):
                        return x[:, idx:idx+1]
                    
                    feat = Lambda(lambda x: extract_feature(x, i))(categorical_features)
                    other_categorical.append(feat)
            
            if other_categorical:
                other_cat_concat = Concatenate()(other_categorical)
            else:
                # 创建单维零向量
                def create_zero_feature(x):
                    batch_size = tf.shape(x)[0]
                    return tf.zeros((batch_size, 1))
                
                other_cat_concat = Lambda(create_zero_feature)(main_input)
        else:
            # 如果没有分类特征，创建零向量
            def create_zero_embedding(x):
                batch_size = tf.shape(x)[0]
                return tf.zeros((batch_size, self.embedding_dim))
            
            def create_zero_feature(x):
                batch_size = tf.shape(x)[0]
                return tf.zeros((batch_size, 1))
            
            stock_embedding = Lambda(create_zero_embedding)(main_input)
            sector_embedding = Lambda(create_zero_embedding)(main_input)
            other_cat_concat = Lambda(create_zero_feature)(main_input)
        
        # 数值特征的LSTM处理 - 使用Lambda层
        if self.numerical_indices:
            def extract_numerical_features(x):
                return tf.gather(x, self.numerical_indices, axis=2)
            
            numerical_features = Lambda(extract_numerical_features)(main_input)
        else:
            numerical_features = main_input
        
        # LSTM层
        lstm_out = LSTM(self.lstm_units, return_sequences=True, name='lstm1')(numerical_features)
        lstm_out = Dropout(self.dropout_rate)(lstm_out)
        
        lstm_out = LSTM(self.lstm_units // 2, return_sequences=True, name='lstm2')(lstm_out)
        lstm_out = Dropout(self.dropout_rate)(lstm_out)
        
        lstm_out = LSTM(self.lstm_units // 4, return_sequences=False, name='lstm3')(lstm_out)
        lstm_out = Dropout(self.dropout_rate)(lstm_out)
        
        # 合并所有特征
        combined_features = Concatenate(name='feature_concat')([
            lstm_out, 
            stock_embedding, 
            sector_embedding, 
            other_cat_concat
        ])
        
        # 全连接层
        dense_out = Dense(64, activation='relu', name='dense1')(combined_features)
        dense_out = Dropout(self.dropout_rate)(dense_out)
        
        dense_out = Dense(32, activation='relu', name='dense2')(dense_out)
        dense_out = Dropout(self.dropout_rate)(dense_out)
        
        # 输出层
        output = Dense(1, activation='sigmoid', name='output')(dense_out)
        
        model = Model(inputs=main_input, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_info=None, epochs=100, batch_size=32):
        """训练增强LSTM模型"""
        print(f"🔥 开始训练{self.model_name}模型...")
        
        if feature_info is None:
            raise ValueError("需要feature_info来构建模型")
        
        # 数据预处理
        X_train_scaled = self._preprocess_data(X_train, fit_scaler=True)
        
        if X_val is not None:
            X_val_scaled = self._preprocess_data(X_val, fit_scaler=False)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # 构建模型
        self.model = self._build_model(feature_info)
        
        # 训练回调
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # 训练模型
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_fitted = True
        print(f"✅ {self.model_name}模型训练完成")
        return history
    
    def _preprocess_data(self, X, fit_scaler=False):
        """数据预处理（只处理数值特征）"""
        n_samples, n_timesteps, n_features = X.shape
        
        if self.numerical_indices:
            # 只标准化数值特征
            X_numerical = X[:, :, self.numerical_indices]
            X_categorical = X[:, :, self.categorical_indices] if self.categorical_indices else None
            
            # 重塑为2D进行标准化
            X_num_2d = X_numerical.reshape(-1, len(self.numerical_indices))
            
            if fit_scaler:
                X_num_scaled_2d = self.scaler.fit_transform(X_num_2d)
            else:
                X_num_scaled_2d = self.scaler.transform(X_num_2d)
            
            # 重塑回3D
            X_num_scaled = X_num_scaled_2d.reshape(n_samples, n_timesteps, len(self.numerical_indices))
            
            # 重新组合特征
            if X_categorical is not None:
                # 保持原始顺序
                X_scaled = np.zeros_like(X)
                X_scaled[:, :, self.numerical_indices] = X_num_scaled
                X_scaled[:, :, self.categorical_indices] = X_categorical
            else:
                X_scaled = X_num_scaled
        else:
            # 如果没有数值特征索引，按原方式处理
            X_2d = X.reshape(-1, n_features)
            if fit_scaler:
                X_scaled_2d = self.scaler.fit_transform(X_2d)
            else:
                X_scaled_2d = self.scaler.transform(X_2d)
            X_scaled = X_scaled_2d.reshape(n_samples, n_timesteps, n_features)
        
        return X_scaled


class EnhancedTransformerModel(StockPredictionModel):
    """
    增强Transformer模型 - 支持股票和板块Embedding
    """
    
    def __init__(self, sequence_length=60, n_features=50, n_stocks=100, n_sectors=20,
                 d_model=128, num_heads=8, num_layers=4, embedding_dim=32):
        super().__init__("Enhanced_Transformer")
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_stocks = n_stocks
        self.n_sectors = n_sectors
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.scaler = StandardScaler()
        
        # 记录分类特征的索引
        self.categorical_indices = []
        self.numerical_indices = []
        
    def _build_model(self, feature_info: Dict):
        """构建增强Transformer模型"""
        
        # 更新特征映射
        self.n_stocks = feature_info.get('n_stocks', self.n_stocks)
        self.n_sectors = feature_info.get('n_sectors', self.n_sectors)
        
        # 找到分类特征的索引
        all_cols = feature_info['all_cols']
        categorical_cols = feature_info['categorical_cols']
        numerical_cols = feature_info['numerical_cols']
        
        self.categorical_indices = [all_cols.index(col) for col in categorical_cols if col in all_cols]
        self.numerical_indices = [all_cols.index(col) for col in numerical_cols if col in all_cols]
        
        # 输入层
        main_input = Input(shape=(self.sequence_length, self.n_features), name='main_input')
        
        # 分离数值特征和分类特征 - 使用Lambda层
        if self.numerical_indices:
            def extract_numerical_features(x):
                return tf.gather(x, self.numerical_indices, axis=2)
            
            numerical_features = Lambda(extract_numerical_features)(main_input)
        else:
            numerical_features = main_input
            
        # 数值特征的Transformer处理
        # 输入投影层
        x = Dense(self.d_model)(numerical_features)
        
        # 多层Transformer块
        for _ in range(self.num_layers):
            # 多头注意力
            attn_output = MultiHeadAttention(
                num_heads=self.num_heads, 
                key_dim=self.d_model // self.num_heads
            )(x, x)
            
            # 残差连接和层归一化
            x = Add()([x, attn_output])
            x = LayerNormalization()(x)
            
            # 前馈网络
            ff_output = Dense(self.d_model * 2, activation='relu')(x)
            ff_output = Dense(self.d_model)(ff_output)
            
            # 残差连接和层归一化
            x = Add()([x, ff_output])
            x = LayerNormalization()(x)
        
        # 全局平均池化
        transformer_out = GlobalAveragePooling1D()(x)
        
        # 处理分类特征（Embedding）- 使用Lambda层
        if self.categorical_indices:
            # 提取分类特征
            def extract_categorical_features(x):
                last_step = x[:, -1, :]
                return tf.gather(last_step, self.categorical_indices, axis=1)
            
            categorical_features = Lambda(extract_categorical_features)(main_input)
            
            embeddings = []
            for i, col in enumerate(categorical_cols):
                if col == 'stock_id':
                    def extract_stock_id(x, idx=i):
                        return tf.cast(x[:, idx:idx+1], tf.int32)
                    
                    stock_id = Lambda(lambda x: extract_stock_id(x, i))(categorical_features)
                    stock_id_flat = Reshape((1,))(stock_id)
                    stock_emb = Embedding(self.n_stocks, self.embedding_dim)(stock_id_flat)
                    stock_emb = Reshape((self.embedding_dim,))(stock_emb)
                    embeddings.append(stock_emb)
                elif col == 'sector_id':
                    def extract_sector_id(x, idx=i):
                        return tf.cast(x[:, idx:idx+1], tf.int32)
                    
                    sector_id = Lambda(lambda x: extract_sector_id(x, i))(categorical_features)
                    sector_id_flat = Reshape((1,))(sector_id)
                    sector_emb = Embedding(self.n_sectors, self.embedding_dim)(sector_id_flat)
                    sector_emb = Reshape((self.embedding_dim,))(sector_emb)
                    embeddings.append(sector_emb)
                else:
                    # 其他分类特征直接使用
                    def extract_feature(x, idx=i):
                        return x[:, idx:idx+1]
                    
                    feat = Lambda(lambda x: extract_feature(x, i))(categorical_features)
                    embeddings.append(feat)
            
            if embeddings:
                categorical_concat = Concatenate()(embeddings)
            else:
                def create_zero_feature(x):
                    batch_size = tf.shape(x)[0]
                    return tf.zeros((batch_size, 1))
                
                categorical_concat = Lambda(create_zero_feature)(main_input)
            
            # 合并特征
            combined_features = Concatenate()([transformer_out, categorical_concat])
        else:
            combined_features = transformer_out
        
        # 分类头
        x = Dense(64, activation='relu')(combined_features)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(main_input, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_info=None, epochs=100, batch_size=32):
        """训练增强Transformer模型"""
        print(f"🔥 开始训练{self.model_name}模型...")
        
        if feature_info is None:
            raise ValueError("需要feature_info来构建模型")
        
        # 数据预处理（与EnhancedLSTMModel相同）
        X_train_scaled = self._preprocess_data(X_train, fit_scaler=True)
        
        if X_val is not None:
            X_val_scaled = self._preprocess_data(X_val, fit_scaler=False)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # 构建模型
        self.model = self._build_model(feature_info)
        
        # 训练回调
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # 训练模型
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_fitted = True
        print(f"✅ {self.model_name}模型训练完成")
        return history
    
    def _preprocess_data(self, X, fit_scaler=False):
        """数据预处理（与EnhancedLSTMModel相同）"""
        n_samples, n_timesteps, n_features = X.shape
        
        if self.numerical_indices:
            # 只标准化数值特征
            X_numerical = X[:, :, self.numerical_indices]
            X_categorical = X[:, :, self.categorical_indices] if self.categorical_indices else None
            
            # 重塑为2D进行标准化
            X_num_2d = X_numerical.reshape(-1, len(self.numerical_indices))
            
            if fit_scaler:
                X_num_scaled_2d = self.scaler.fit_transform(X_num_2d)
            else:
                X_num_scaled_2d = self.scaler.transform(X_num_2d)
            
            # 重塑回3D
            X_num_scaled = X_num_scaled_2d.reshape(n_samples, n_timesteps, len(self.numerical_indices))
            
            # 重新组合特征
            if X_categorical is not None:
                X_scaled = np.zeros_like(X)
                X_scaled[:, :, self.numerical_indices] = X_num_scaled
                X_scaled[:, :, self.categorical_indices] = X_categorical
            else:
                X_scaled = X_num_scaled
        else:
            # 如果没有数值特征索引，按原方式处理
            X_2d = X.reshape(-1, n_features)
            if fit_scaler:
                X_scaled_2d = self.scaler.fit_transform(X_2d)
            else:
                X_scaled_2d = self.scaler.transform(X_2d)
            X_scaled = X_scaled_2d.reshape(n_samples, n_timesteps, n_features)
        
        return X_scaled


class EnhancedLightGBMModel(StockPredictionModel):
    """
    增强LightGBM模型 - 直接使用分类特征
    """
    
    def __init__(self, **params):
        super().__init__("Enhanced_LightGBM")
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
            **params
        }
        self.scaler = RobustScaler()
        self.categorical_indices = []
        self.numerical_indices = []
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_info=None, **kwargs):
        """训练增强LightGBM模型"""
        print(f"🔥 开始训练{self.model_name}模型...")
        
        if feature_info is not None:
            all_cols = feature_info['all_cols']
            categorical_cols = feature_info['categorical_cols']
            numerical_cols = feature_info['numerical_cols']
            
            self.categorical_indices = [all_cols.index(col) for col in categorical_cols if col in all_cols]
            self.numerical_indices = [all_cols.index(col) for col in numerical_cols if col in all_cols]
        
        # Flatten 3D to 2D
        if len(X_train.shape) == 3:
            X_train_2d = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_2d = X_train
        
        # 分离数值特征进行标准化
        if self.numerical_indices:
            # 获取最后一个时间步的特征
            last_step_features = X_train[:, -1, :] if len(X_train.shape) == 3 else X_train
            
            X_numerical = last_step_features[:, self.numerical_indices]
            X_categorical = last_step_features[:, self.categorical_indices] if self.categorical_indices else np.array([]).reshape(len(X_train), 0)
            
            # 标准化数值特征
            X_num_scaled = self.scaler.fit_transform(X_numerical)
            
            # 合并特征
            if X_categorical.shape[1] > 0:
                X_train_processed = np.hstack([X_num_scaled, X_categorical])
            else:
                X_train_processed = X_num_scaled
        else:
            # 如果没有特征信息，按原方式处理
            X_train_processed = self.scaler.fit_transform(X_train_2d)
        
        # 验证集处理
        eval_set = None
        if X_val is not None and y_val is not None:
            if len(X_val.shape) == 3:
                last_step_val = X_val[:, -1, :]
            else:
                last_step_val = X_val
                
            if self.numerical_indices:
                X_val_numerical = last_step_val[:, self.numerical_indices]
                X_val_categorical = last_step_val[:, self.categorical_indices] if self.categorical_indices else np.array([]).reshape(len(X_val), 0)
                
                X_val_num_scaled = self.scaler.transform(X_val_numerical)
                
                if X_val_categorical.shape[1] > 0:
                    X_val_processed = np.hstack([X_val_num_scaled, X_val_categorical])
                else:
                    X_val_processed = X_val_num_scaled
            else:
                X_val_2d = X_val.reshape(X_val.shape[0], -1) if len(X_val.shape) == 3 else X_val
                X_val_processed = self.scaler.transform(X_val_2d)
                
            eval_set = [(X_val_processed, y_val)]
        
        # 训练模型
        self.model = lgb.LGBMClassifier(**self.params)
        
        # LightGBM训练
        try:
            self.model.fit(
                X_train_processed, y_train,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(50, verbose=False)] if eval_set else None
            )
        except Exception as e:
            print(f"  ⚠️ 早停训练失败: {e}, 尝试基本训练")
            # 简单训练方式（无早停）
            try:
                self.model = lgb.LGBMClassifier(**self.params)
                self.model.fit(X_train_processed, y_train)
            except Exception as e2:
                print(f"  ❌ 基本训练也失败: {e2}")
                raise e2
        
        self.is_fitted = True
        print(f"✅ {self.model_name}模型训练完成")


class EnhancedEnsembleModel(EnsembleModel):
    """
    增强集成模型 - 支持特征信息传递
    """
    
    def __init__(self, model_weights: Dict[str, float] = None):
        super().__init__(model_weights)
        self.feature_info = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_info=None, **kwargs):
        """训练所有模型（传递feature_info）"""
        print("🚀 开始训练增强集成模型...")
        
        self.feature_info = feature_info
        
        # 训练各个子模型
        for model_name, model in self.models.items():
            print(f"\n--- 训练 {model_name} ---")
            
            # 检查模型是否支持feature_info
            if hasattr(model, 'fit') and 'feature_info' in model.fit.__code__.co_varnames:
                model.fit(X_train, y_train, X_val, y_val, feature_info=feature_info, **kwargs)
            else:
                model.fit(X_train, y_train, X_val, y_val, **kwargs)
        
        # 动态调整权重（基于验证集性能）
        if X_val is not None and y_val is not None:
            self._adjust_weights(X_val, y_val)
        
        print("✅ 增强集成模型训练完成")


def create_enhanced_ensemble_model(sequence_length=60, n_features=50, 
                                 n_stocks=100, n_sectors=20) -> EnhancedEnsembleModel:
    """
    创建增强的集成模型
    
    Args:
        sequence_length: 时序长度
        n_features: 特征数量
        n_stocks: 股票数量
        n_sectors: 板块数量
        
    Returns:
        EnhancedEnsembleModel实例
    """
    
    # 创建集成模型
    ensemble = EnhancedEnsembleModel()
    
    # 添加增强的子模型
    enhanced_lstm = EnhancedLSTMModel(sequence_length, n_features, n_stocks, n_sectors)
    enhanced_transformer = EnhancedTransformerModel(sequence_length, n_features, n_stocks, n_sectors)
    enhanced_lgb = EnhancedLightGBMModel()
    
    # 还可以使用原有的CNN-LSTM模型
    from ai_models import CNNLSTMModel
    cnn_lstm = CNNLSTMModel(sequence_length, n_features)
    
    ensemble.add_model(enhanced_lstm)
    ensemble.add_model(enhanced_transformer)
    ensemble.add_model(enhanced_lgb)
    ensemble.add_model(cnn_lstm)
    
    return ensemble


if __name__ == "__main__":
    # 测试增强模型
    print("🧪 测试增强模型...")
    
    # 模拟数据
    n_samples, sequence_length, n_features = 1000, 60, 45
    n_stocks, n_sectors = 50, 10
    
    X_train = np.random.randn(n_samples, sequence_length, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    # 模拟feature_info
    feature_info = {
        'numerical_cols': [f'feature_{i}' for i in range(40)],
        'categorical_cols': ['stock_id', 'sector_id', 'sector_type', 'is_market_sensitive', 'is_growth_stock'],
        'all_cols': [f'feature_{i}' for i in range(40)] + ['stock_id', 'sector_id', 'sector_type', 'is_market_sensitive', 'is_growth_stock'],
        'n_stocks': n_stocks,
        'n_sectors': n_sectors
    }
    
    # 测试增强LSTM模型
    lstm_model = EnhancedLSTMModel(sequence_length, n_features, n_stocks, n_sectors)
    print("✅ 增强模型创建成功")
    
    print("🎉 增强模型测试完成！")