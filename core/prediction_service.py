# -*- coding: utf-8 -*-
"""
AI股市预测系统 - 预测服务模块
功能：提供实时预测服务、预测结果管理和API接口
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import joblib
import json
import redis
from datetime import datetime, timedelta
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import schedule
import time
from threading import Thread

# 导入处理 - 支持直接运行和模块导入
try:
    from .feature_engineering import FeatureEngineering
    from .ai_models import EnsembleModel, create_ensemble_model
    from .training_pipeline import ModelTrainingPipeline, AutoRetrainingSystem
except ImportError:
    # 直接运行时的导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from feature_engineering import FeatureEngineering
    from ai_models import EnsembleModel, create_ensemble_model
    from training_pipeline import ModelTrainingPipeline, AutoRetrainingSystem

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """预测请求模型"""
    stock_code: str
    prediction_days: int = 1
    include_probability: bool = True
    include_analysis: bool = True


class PredictionResponse(BaseModel):
    """预测响应模型"""
    stock_code: str
    prediction_days: int
    prediction: int  # 0: 下跌, 1: 上涨
    probability: float
    confidence: str  # 'low', 'medium', 'high'
    current_price: float
    predicted_direction: str
    analysis: Dict
    timestamp: str


class RiskAssessment(BaseModel):
    """风险评估模型"""
    stock_code: str
    risk_level: str  # 'low', 'medium', 'high'
    volatility_forecast: float
    max_drawdown_forecast: float
    stop_loss_suggestion: float
    position_size_suggestion: float
    risk_factors: List[str]


class PredictionService:
    """
    预测服务类
    提供股票预测、风险评估、批量预测等功能
    """
    
    def __init__(self, model_dir: str = "models", 
                 data_dir: str = "data/datas_em",
                 redis_host: str = "localhost",
                 redis_port: int = 6379):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.feature_engineer = FeatureEngineering()
        self.models = {}  # prediction_days -> model
        self.model_metadata = {}
        
        # Redis缓存
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
            self.use_cache = True
            logger.info("Redis缓存连接成功")
        except:
            self.redis_client = None
            self.use_cache = False
            logger.warning("Redis缓存连接失败，将不使用缓存")
        
        # 加载模型
        self.load_models()
        
        # 预测历史记录
        self.prediction_history = []
        
        # 风险评估配置
        self.risk_config = {
            'volatility_threshold_low': 0.02,
            'volatility_threshold_high': 0.05,
            'confidence_threshold_low': 0.6,
            'confidence_threshold_high': 0.8,
        }
    
    def load_models(self):
        """加载所有已训练的模型"""
        if not os.path.exists(self.model_dir):
            logger.warning(f"模型目录不存在: {self.model_dir}")
            return
        
        model_folders = [f for f in os.listdir(self.model_dir) 
                        if os.path.isdir(os.path.join(self.model_dir, f))]
        
        for folder in model_folders:
            try:
                folder_path = os.path.join(self.model_dir, folder)
                
                # 加载训练信息
                info_path = os.path.join(folder_path, 'training_info.pkl')
                if os.path.exists(info_path):
                    training_info = joblib.load(info_path)
                    prediction_days = training_info['prediction_days']
                    
                    # 从训练信息中获取正确的参数
                    # 根据错误信息，模型期望的是(None, 30, 170)的输入形状
                    sequence_length = training_info.get('sequence_length', 30)  # 改为默认30
                    n_features = len(training_info['feature_names'])
                    
                    logger.info(f"创建模型: sequence_length={sequence_length}, n_features={n_features}")
                    
                    # 创建模型实例
                    model = create_ensemble_model(
                        sequence_length=sequence_length,
                        n_features=n_features
                    )
                    
                    # 加载各子模型
                    self._load_individual_models(model, folder_path)
                    
                    # 确保training_info中包含sequence_length
                    training_info['sequence_length'] = sequence_length
                    
                    self.models[prediction_days] = model
                    self.model_metadata[prediction_days] = training_info
                    
                    logger.info(f"加载 {prediction_days} 天预测模型成功")
                
            except Exception as e:
                logger.error(f"加载模型 {folder} 失败: {str(e)}")
        
        logger.info(f"成功加载 {len(self.models)} 个模型")
    
    def _load_individual_models(self, ensemble_model: EnsembleModel, model_path: str):
        """加载集成模型中的各个子模型"""
        try:
            import tensorflow as tf
            import lightgbm as lgb
            from sklearn.preprocessing import StandardScaler
            
            # 加载模型权重
            weights_path = os.path.join(model_path, 'model_weights.pkl')
            if os.path.exists(weights_path):
                ensemble_model.model_weights = joblib.load(weights_path)
            
            # 加载各个子模型
            model_files = {
                'LSTM': ('LSTM_model.h5', 'LSTM_scaler.pkl'),
                'CNN-LSTM': ('CNN-LSTM_model.h5', 'CNN-LSTM_scaler.pkl'),
                'Transformer': ('Transformer_model.h5', 'Transformer_scaler.pkl'),
                'LightGBM': ('LightGBM_model.pkl', 'LightGBM_scaler.pkl')
            }
            
            for model_name, (model_file, scaler_file) in model_files.items():
                model_file_path = os.path.join(model_path, model_file)
                scaler_file_path = os.path.join(model_path, scaler_file)
                
                if os.path.exists(model_file_path) and model_name in ensemble_model.models:
                    try:
                        # 加载模型
                        if model_name == 'LightGBM':
                            loaded_model = joblib.load(model_file_path)
                        else:
                            loaded_model = tf.keras.models.load_model(model_file_path)
                        
                        # 加载对应的scaler
                        if os.path.exists(scaler_file_path):
                            scaler = joblib.load(scaler_file_path)
                            ensemble_model.models[model_name].scaler = scaler
                        
                        # 设置模型为已训练状态
                        ensemble_model.models[model_name].model = loaded_model
                        ensemble_model.models[model_name].is_fitted = True
                        
                        logger.info(f"成功加载子模型: {model_name}")
                        
                    except Exception as e:
                        logger.warning(f"加载子模型 {model_name} 失败: {str(e)}")
                        continue
            
            # 检查是否至少有一个模型加载成功
            loaded_models = sum(1 for model in ensemble_model.models.values() if model.is_fitted)
            if loaded_models == 0:
                logger.warning("没有任何子模型加载成功，将使用规则预测")
                # 设置一个简单的规则模型作为后备
                self._setup_fallback_model(ensemble_model)
            else:
                logger.info(f"成功加载 {loaded_models} 个子模型")
                
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            # 设置后备模型
            self._setup_fallback_model(ensemble_model)
    
    def _setup_fallback_model(self, ensemble_model: EnsembleModel):
        """设置后备模型（简单规则模型）"""
        try:
            # 创建一个简单的规则预测模型
            class SimpleFallbackModel:
                def __init__(self):
                    self.is_fitted = True
                
                def predict(self, X):
                    # 简单规则：基于最后几天的价格趋势
                    if len(X) == 0:
                        return np.array([1])  # 默认预测上涨
                    
                    # 计算简单的趋势
                    recent_prices = X[0, -5:, 0] if X.shape[1] >= 5 else X[0, :, 0]
                    if len(recent_prices) >= 2:
                        trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                        return np.array([1 if trend > 0 else 0])
                    else:
                        return np.array([1])
                
                def predict_proba(self, X):
                    pred = self.predict(X)
                    prob = 0.6 if pred[0] == 1 else 0.4
                    return np.array([[1-prob, prob]])
            
            # 清空现有模型，只使用后备模型
            ensemble_model.models.clear()
            ensemble_model.model_weights.clear()
            
            # 设置后备模型
            ensemble_model.models['Fallback'] = SimpleFallbackModel()
            ensemble_model.model_weights['Fallback'] = 1.0
            
            logger.info("已设置简单规则后备模型")
            
        except Exception as e:
            logger.error(f"设置后备模型失败: {str(e)}")
            raise ValueError("无法初始化任何预测模型")
    
    def get_latest_stock_data(self, stock_code: str, days: int = 100) -> pd.DataFrame:
        """获取最新股票数据"""
        file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"股票数据文件不存在: {stock_code}")
        
        df = pd.read_csv(file_path)
        df['交易日期'] = pd.to_datetime(df['交易日期'])
        df = df.sort_values('交易日期').reset_index(drop=True)
        
        # 返回最近的数据
        return df.tail(days).reset_index(drop=True)
    
    def predict_single_stock(self, stock_code: str, prediction_days: int = 1,
                           include_analysis: bool = True, 
                           prediction_threshold: float = 0.6) -> PredictionResponse:
        """
        单只股票预测
        
        Args:
            stock_code: 股票代码
            prediction_days: 预测天数
            include_analysis: 是否包含分析信息
            prediction_threshold: 预测阈值（默认0.6，大于此值预测上涨）
            
        Returns:
            预测结果
        """
        # 检查缓存
        cache_key = f"prediction:{stock_code}:{prediction_days}"
        if self.use_cache:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                logger.info(f"使用缓存结果: {stock_code}")
                return PredictionResponse(**json.loads(cached_result))
        
        # 检查模型是否存在
        if prediction_days not in self.models:
            raise ValueError(f"不支持 {prediction_days} 天预测")
        
        try:
            # 获取股票数据
            df = self.get_latest_stock_data(stock_code)
            current_price = df['收盘价'].iloc[-1]
            
            # 特征工程
            df_features = self.feature_engineer.create_all_features(df, stock_code)
            
            # 获取模型的sequence_length
            model = self.models[prediction_days]
            model_metadata = self.model_metadata.get(prediction_days, {})
            sequence_length = model_metadata.get('sequence_length', 30)  # 改为默认30
            
            # 准备预测数据
            X, _, feature_names, _ = self.feature_engineer.prepare_model_data(
                df_features, 
                prediction_days=prediction_days,
                lookback_window=sequence_length
            )
            
            if len(X) == 0:
                raise ValueError("数据不足，无法进行预测")
            
            # 使用最后一个样本进行预测
            X_pred = X[-1].reshape(1, *X[-1].shape)
            
            # 调试信息：检查实际的数据形状
            logger.info(f"调试信息 - sequence_length: {sequence_length}")
            logger.info(f"调试信息 - X.shape: {X.shape}")
            logger.info(f"调试信息 - X_pred.shape: {X_pred.shape}")
            logger.info(f"调试信息 - 期望形状: (1, {sequence_length}, {len(feature_names)})")
            
            # 获取模型
            model = self.models[prediction_days]
            
            # 预测
            prediction = model.predict(X_pred)[0]
            raw_probability = model.predict_proba(X_pred)[0, 1]  # 上涨概率
            
            # 调试信息：显示各个子模型的预测
            if hasattr(model, 'models') and model.models:
                logger.info("🔍 各子模型预测详情:")
                for sub_model_name, sub_model in model.models.items():
                    if hasattr(sub_model, 'is_fitted') and sub_model.is_fitted:
                        try:
                            sub_pred = sub_model.predict(X_pred)[0]
                            sub_prob = sub_model.predict_proba(X_pred)[0, 1]
                            weight = model.model_weights.get(sub_model_name, 0)
                            logger.info(f"  {sub_model_name}: 预测={sub_pred}, 概率={sub_prob:.3f}, 权重={weight:.3f}")
                        except Exception as e:
                            logger.warning(f"  {sub_model_name}: 预测失败 - {str(e)}")
                logger.info(f"🎯 集成结果: 预测={prediction}, 原始概率={raw_probability:.3f}")
            
            # 基于用户设定的阈值判断预测方向
            if raw_probability > prediction_threshold:
                prediction = 1
                predicted_direction = "上涨"
            else:
                prediction = 0
                predicted_direction = "下跌"
            
            # 直接使用原始概率，不做转换
            probability = raw_probability
            
            # 置信度评估（基于概率距离阈值的远近）
            confidence = self._assess_confidence(probability, prediction_threshold)
            
            # 分析信息
            analysis = {}
            if include_analysis:
                analysis = self._generate_analysis(df_features, stock_code, prediction_days)
            
            # 构建响应
            response = PredictionResponse(
                stock_code=stock_code,
                prediction_days=prediction_days,
                prediction=prediction,
                probability=float(probability),
                confidence=confidence,
                current_price=float(current_price),
                predicted_direction=predicted_direction,
                analysis=analysis,
                timestamp=datetime.now().isoformat()
            )
            
            # 缓存结果（5分钟）
            if self.use_cache:
                self.redis_client.setex(
                    cache_key, 
                    300,  # 5分钟过期
                    json.dumps(response.dict())
                )
            
            # 记录预测历史
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'stock_code': stock_code,
                'prediction_days': prediction_days,
                'prediction': prediction,
                'probability': probability,
                'current_price': current_price
            })
            
            return response
            
        except Exception as e:
            logger.error(f"预测股票 {stock_code} 失败: {str(e)}")
            # 对于streamlit使用，抛出普通异常而不是HTTPException
            raise ValueError(f"预测失败: {str(e)}")
    
    def _assess_confidence(self, probability: float, prediction_threshold: float = 0.6) -> str:
        """评估预测置信度
        
        Args:
            probability: 预测概率
            prediction_threshold: 预测阈值（用于判断上涨/下跌的分界点）
        """
        # 计算概率距离预测阈值的远近
        distance_from_threshold = abs(probability - prediction_threshold)
        
        # 基于距离阈值的远近评估置信度
        if distance_from_threshold >= 0.2:  # 距离阈值20%以上
            return "high"
        elif distance_from_threshold >= 0.1:  # 距离阈值10%-20%
            return "medium"
        else:  # 距离阈值10%以内
            return "low"
    
    def _generate_analysis(self, df_features: pd.DataFrame, 
                          stock_code: str, prediction_days: int) -> Dict:
        """生成预测分析信息"""
        latest_data = df_features.iloc[-1]
        
        analysis = {
            'technical_indicators': {},
            'market_sentiment': {},
            'risk_factors': [],
            'support_resistance': {},
            'trend_analysis': {}
        }
        
        # 技术指标分析
        if 'RSI_14' in df_features.columns:
            rsi = latest_data['RSI_14']
            analysis['technical_indicators']['RSI'] = {
                'value': float(rsi),
                'signal': 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'
            }
        
        if 'MACD' in df_features.columns:
            macd = latest_data['MACD']
            macd_signal = latest_data['MACD_signal']
            analysis['technical_indicators']['MACD'] = {
                'value': float(macd),
                'signal': 'bullish' if macd > macd_signal else 'bearish'
            }
        
        # 布林带分析
        if 'BB_position' in df_features.columns:
            bb_pos = latest_data['BB_position']
            analysis['technical_indicators']['Bollinger'] = {
                'position': float(bb_pos),
                'signal': 'upper' if bb_pos > 0.8 else 'lower' if bb_pos < 0.2 else 'middle'
            }
        
        # 趋势分析
        if 'trend_up' in df_features.columns:
            trend_up = latest_data['trend_up']
            trend_down = latest_data['trend_down']
            analysis['trend_analysis'] = {
                'direction': 'up' if trend_up else 'down' if trend_down else 'sideways',
                'strength': 'strong' if (trend_up or trend_down) else 'weak'
            }
        
        # 成交量分析
        if 'volume_ratio_20' in df_features.columns:
            vol_ratio = latest_data['volume_ratio_20']
            analysis['market_sentiment']['volume'] = {
                'ratio': float(vol_ratio),
                'signal': 'high' if vol_ratio > 1.5 else 'low' if vol_ratio < 0.5 else 'normal'
            }
        
        # 风险因素
        risk_factors = []
        if 'volatility_20d' in df_features.columns:
            volatility = latest_data['volatility_20d']
            if volatility > 0.05:
                risk_factors.append("高波动率")
        
        if 'limit_down' in df_features.columns and latest_data['limit_down']:
            risk_factors.append("跌停风险")
        
        analysis['risk_factors'] = risk_factors
        
        return analysis
    
    def assess_risk(self, stock_code: str) -> RiskAssessment:
        """风险评估"""
        try:
            # 获取股票数据
            df = self.get_latest_stock_data(stock_code)
            df_features = self.feature_engineer.create_all_features(df, stock_code)
            
            latest_data = df_features.iloc[-1]
            current_price = df['收盘价'].iloc[-1]
            
            # 波动率预测
            volatility_forecast = self._forecast_volatility(df_features)
            
            # 最大回撤预测
            max_drawdown_forecast = self._forecast_max_drawdown(df_features)
            
            # 风险等级评估
            risk_level = self._assess_risk_level(volatility_forecast, max_drawdown_forecast)
            
            # 止损建议
            stop_loss_suggestion = self._calculate_stop_loss(current_price, volatility_forecast)
            
            # 仓位建议
            position_size_suggestion = self._calculate_position_size(risk_level, volatility_forecast)
            
            # 风险因素
            risk_factors = self._identify_risk_factors(df_features)
            
            return RiskAssessment(
                stock_code=stock_code,
                risk_level=risk_level,
                volatility_forecast=volatility_forecast,
                max_drawdown_forecast=max_drawdown_forecast,
                stop_loss_suggestion=stop_loss_suggestion,
                position_size_suggestion=position_size_suggestion,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.error(f"风险评估失败 {stock_code}: {str(e)}")
            raise ValueError(f"风险评估失败: {str(e)}")
    
    def _forecast_volatility(self, df_features: pd.DataFrame) -> float:
        """预测未来波动率"""
        if 'volatility_20d' in df_features.columns:
            recent_vol = df_features['volatility_20d'].tail(10).mean()
            return float(recent_vol * 1.2)  # 适当放大作为预测
        return 0.03  # 默认值
    
    def _forecast_max_drawdown(self, df_features: pd.DataFrame) -> float:
        """预测最大回撤"""
        if 'price_change' in df_features.columns:
            returns = df_features['price_change'].dropna()
            # 计算历史最大回撤
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown_hist = drawdown.min()
            return float(abs(max_drawdown_hist) * 1.3)  # 适当放大
        return 0.15  # 默认值
    
    def _assess_risk_level(self, volatility: float, max_drawdown: float) -> str:
        """评估风险等级"""
        if volatility > 0.05 or max_drawdown > 0.2:
            return "high"
        elif volatility > 0.03 or max_drawdown > 0.1:
            return "medium"
        else:
            return "low"
    
    def _calculate_stop_loss(self, current_price: float, volatility: float) -> float:
        """计算止损价格"""
        # 基于波动率的动态止损
        stop_loss_pct = max(0.05, volatility * 2)  # 最少5%止损
        return float(current_price * (1 - stop_loss_pct))
    
    def _calculate_position_size(self, risk_level: str, volatility: float) -> float:
        """计算建议仓位大小"""
        base_position = 0.1  # 基础仓位10%
        
        if risk_level == "low":
            return min(0.2, base_position / volatility * 0.02)
        elif risk_level == "medium":
            return min(0.15, base_position / volatility * 0.015)
        else:  # high risk
            return min(0.1, base_position / volatility * 0.01)
    
    def _identify_risk_factors(self, df_features: pd.DataFrame) -> List[str]:
        """识别风险因素"""
        risk_factors = []
        latest_data = df_features.iloc[-1]
        
        # 技术风险
        if 'RSI_14' in df_features.columns:
            rsi = latest_data['RSI_14']
            if rsi > 80:
                risk_factors.append("技术指标超买")
            elif rsi < 20:
                risk_factors.append("技术指标超卖")
        
        # 成交量异常
        if 'volume_ratio_20' in df_features.columns:
            vol_ratio = latest_data['volume_ratio_20']
            if vol_ratio > 3:
                risk_factors.append("成交量异常放大")
        
        # 波动率风险
        if 'volatility_20d' in df_features.columns:
            volatility = latest_data['volatility_20d']
            if volatility > 0.05:
                risk_factors.append("高波动率")
        
        # 趋势风险
        if 'momentum_20d' in df_features.columns:
            momentum = latest_data['momentum_20d']
            if momentum < -0.2:
                risk_factors.append("下跌趋势强烈")
        
        return risk_factors
    
    def batch_predict(self, stock_codes: List[str], prediction_days: int = 1) -> List[PredictionResponse]:
        """批量预测"""
        logger.info(f"开始批量预测，股票数量: {len(stock_codes)}")
        
        results = []
        for stock_code in stock_codes:
            try:
                result = self.predict_single_stock(
                    stock_code=stock_code,
                    prediction_days=prediction_days,
                    include_analysis=False,  # 批量预测时不包含详细分析
                    prediction_threshold=0.6  # 使用默认阈值
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"批量预测失败 {stock_code}: {str(e)}")
                continue
        
        logger.info(f"批量预测完成，成功 {len(results)} 个")
        return results
    
    def get_prediction_history(self, stock_code: str = None, 
                             days: int = 30) -> List[Dict]:
        """获取预测历史"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        filtered_history = [
            record for record in self.prediction_history
            if record['timestamp'] >= cutoff_date
        ]
        
        if stock_code:
            filtered_history = [
                record for record in filtered_history
                if record['stock_code'] == stock_code
            ]
        
        return filtered_history


# FastAPI 应用
app = FastAPI(title="AI股市预测系统", version="1.0.0")
prediction_service = PredictionService()


@app.get("/")
async def root():
    return {"message": "AI股市预测系统API", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    """单只股票预测接口"""
    return prediction_service.predict_single_stock(
        stock_code=request.stock_code,
        prediction_days=request.prediction_days,
        include_analysis=request.include_analysis,
        prediction_threshold=0.6  # 使用默认阈值
    )


@app.post("/predict/batch")
async def batch_predict_stocks(stock_codes: List[str], prediction_days: int = 1):
    """批量预测接口"""
    return prediction_service.batch_predict(stock_codes, prediction_days)


@app.get("/risk/{stock_code}", response_model=RiskAssessment)
async def assess_stock_risk(stock_code: str):
    """风险评估接口"""
    return prediction_service.assess_risk(stock_code)


@app.get("/history/{stock_code}")
async def get_prediction_history(stock_code: str, days: int = 30):
    """获取预测历史接口"""
    return prediction_service.get_prediction_history(stock_code, days)


@app.get("/models")
async def get_available_models():
    """获取可用模型信息"""
    return {
        "available_models": list(prediction_service.models.keys()),
        "model_metadata": prediction_service.model_metadata
    }


class RealtimeDataUpdater:
    """
    实时数据更新器
    定期更新股票数据和重训练模型
    """
    
    def __init__(self, prediction_service: PredictionService):
        self.prediction_service = prediction_service
        self.pipeline = ModelTrainingPipeline()
        self.auto_retrainer = AutoRetrainingSystem(self.pipeline)
        
    def update_stock_data(self):
        """更新股票数据"""
        logger.info("开始更新股票数据...")
        
        # 这里应该调用您现有的数据获取脚本
        # 例如：samequant_functions.Spider_func()
        try:
            # 假设使用现有的数据获取功能
            from samequant_functions import Spider_func
            spider = Spider_func()
            
            # 更新主要指数和热门股票
            # spider.update_market_data()
            
            logger.info("股票数据更新完成")
            
        except Exception as e:
            logger.error(f"更新股票数据失败: {str(e)}")
    
    def check_model_performance(self):
        """检查模型性能并触发重训练"""
        logger.info("检查模型性能...")
        
        # 获取热门股票列表进行检查
        popular_stocks = ['sh600519', 'sz000001', 'sz000002']  # 示例
        
        try:
            self.auto_retrainer.auto_retrain(popular_stocks)
            
        except Exception as e:
            logger.error(f"性能检查失败: {str(e)}")
    
    def start_scheduled_tasks(self):
        """启动定时任务"""
        logger.info("启动定时任务...")
        
        # 每日收盘后更新数据
        schedule.every().day.at("15:30").do(self.update_stock_data)
        
        # 每周检查模型性能
        schedule.every().monday.at("18:00").do(self.check_model_performance)
        
        # 运行定时任务
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次


def start_realtime_updater():
    """启动实时更新器"""
    updater = RealtimeDataUpdater(prediction_service)
    
    # 在后台线程中运行
    thread = Thread(target=updater.start_scheduled_tasks, daemon=True)
    thread.start()
    
    logger.info("实时更新器已启动")


if __name__ == "__main__":
    import uvicorn
    
    # 启动实时更新器
    start_realtime_updater()
    
    # 启动API服务
    uvicorn.run(app, host="0.0.0.0", port=8000)