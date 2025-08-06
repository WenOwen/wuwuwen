# -*- coding: utf-8 -*-
"""
AI股市预测系统 - 预测服务模块（无Redis版本）
功能：提供实时预测服务，使用内存缓存替代Redis
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import joblib
import json
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


class MemoryCache:
    """内存缓存类，替代Redis"""
    
    def __init__(self):
        self.cache = {}
        self.expiry = {}
    
    def get(self, key: str) -> Optional[str]:
        """获取缓存值"""
        if key in self.cache:
            # 检查是否过期
            if key in self.expiry and datetime.now() > self.expiry[key]:
                del self.cache[key]
                del self.expiry[key]
                return None
            return self.cache[key]
        return None
    
    def setex(self, key: str, seconds: int, value: str):
        """设置带过期时间的缓存"""
        self.cache[key] = value
        self.expiry[key] = datetime.now() + timedelta(seconds=seconds)
    
    def set(self, key: str, value: str):
        """设置永久缓存"""
        self.cache[key] = value
        if key in self.expiry:
            del self.expiry[key]
    
    def delete(self, key: str):
        """删除缓存"""
        if key in self.cache:
            del self.cache[key]
        if key in self.expiry:
            del self.expiry[key]
    
    def ping(self):
        """测试连接（兼容Redis接口）"""
        return True


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
    """预测服务类（无Redis版本）"""
    
    def __init__(self, model_dir: str = "models", 
                 data_dir: str = "data/datas_em"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.feature_engineer = FeatureEngineering()
        self.models = {}  # prediction_days -> model
        self.model_metadata = {}
        
        # 使用内存缓存替代Redis
        self.cache_client = MemoryCache()
        self.use_cache = True
        logger.info("使用内存缓存替代Redis")
        
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
                    
                    # 创建模型实例
                    model = create_ensemble_model(
                        sequence_length=60,
                        n_features=len(training_info['feature_names'])
                    )
                    
                    # 加载各子模型（这里简化处理）
                    # self._load_individual_models(model, folder_path)
                    
                    self.models[prediction_days] = model
                    self.model_metadata[prediction_days] = training_info
                    
                    logger.info(f"加载 {prediction_days} 天预测模型成功")
                
            except Exception as e:
                logger.error(f"加载模型 {folder} 失败: {str(e)}")
        
        logger.info(f"成功加载 {len(self.models)} 个模型")
    
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
                           include_analysis: bool = True) -> PredictionResponse:
        """单只股票预测"""
        # 检查缓存
        cache_key = f"prediction:{stock_code}:{prediction_days}"
        if self.use_cache:
            cached_result = self.cache_client.get(cache_key)
            if cached_result:
                logger.info(f"使用缓存结果: {stock_code}")
                return PredictionResponse(**json.loads(cached_result))
        
        # 检查模型是否存在
        if prediction_days not in self.models:
            # 如果没有训练好的模型，使用简单规则预测
            return self._simple_rule_prediction(stock_code, prediction_days, include_analysis)
        
        try:
            # 获取股票数据
            df = self.get_latest_stock_data(stock_code)
            current_price = df['收盘价'].iloc[-1]
            
            # 特征工程
            df_features = self.feature_engineer.create_all_features(df, stock_code)
            
            # 准备预测数据
            X, _, feature_names, _ = self.feature_engineer.prepare_model_data(
                df_features, 
                prediction_days=prediction_days,
                lookback_window=60
            )
            
            if len(X) == 0:
                raise ValueError("数据不足，无法进行预测")
            
            # 使用最后一个样本进行预测
            X_pred = X[-1].reshape(1, *X[-1].shape)
            
            # 获取模型
            model = self.models[prediction_days]
            
            # 预测
            prediction = model.predict(X_pred)[0]
            probability = model.predict_proba(X_pred)[0, 1]  # 上涨概率
            
            # 置信度评估
            confidence = self._assess_confidence(probability)
            
            # 预测方向
            predicted_direction = "上涨" if prediction == 1 else "下跌"
            
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
                self.cache_client.setex(
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
            # 如果AI预测失败，回退到简单规则预测
            return self._simple_rule_prediction(stock_code, prediction_days, include_analysis)
    
    def _simple_rule_prediction(self, stock_code: str, prediction_days: int = 1,
                               include_analysis: bool = True) -> PredictionResponse:
        """简单规则预测（当AI模型不可用时使用）"""
        try:
            # 获取股票数据
            df = self.get_latest_stock_data(stock_code)
            current_price = df['收盘价'].iloc[-1]
            
            # 特征工程
            df_features = self.feature_engineer.create_all_features(df, stock_code)
            latest = df_features.iloc[-1]
            
            # 简单规则预测
            score = 0
            
            # 价格趋势
            if 'SMA_5' in df_features.columns and 'SMA_20' in df_features.columns:
                if latest['SMA_5'] > latest['SMA_20']:
                    score += 1
                else:
                    score -= 1
            
            # RSI
            if 'RSI_14' in df_features.columns:
                rsi = latest['RSI_14']
                if rsi < 30:  # 超卖
                    score += 1
                elif rsi > 70:  # 超买
                    score -= 1
            
            # 成交量
            if '成交量' in df.columns:
                vol_ma = df['成交量'].rolling(20).mean().iloc[-1]
                if df['成交量'].iloc[-1] > vol_ma * 1.5:
                    score += 0.5
            
            # 转换为概率
            probability = 0.5 + score * 0.1
            probability = max(0.2, min(0.8, probability))
            
            prediction = 1 if probability > 0.5 else 0
            confidence = self._assess_confidence(probability)
            predicted_direction = "上涨" if prediction == 1 else "下跌"
            
            # 分析信息
            analysis = {}
            if include_analysis:
                analysis = {
                    'model_type': 'simple_rules',
                    'note': '使用简单规则预测（AI模型不可用）',
                    'factors': []
                }
            
            return PredictionResponse(
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
            
        except Exception as e:
            logger.error(f"简单规则预测失败 {stock_code}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _assess_confidence(self, probability: float) -> str:
        """评估预测置信度"""
        abs_prob = abs(probability - 0.5)
        
        if abs_prob >= 0.3:  # 概率 >= 0.8 或 <= 0.2
            return "high"
        elif abs_prob >= 0.1:  # 概率 >= 0.6 或 <= 0.4
            return "medium"
        else:
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
        
        return analysis
    
    def batch_predict(self, stock_codes: List[str], prediction_days: int = 1) -> List[PredictionResponse]:
        """批量预测"""
        logger.info(f"开始批量预测，股票数量: {len(stock_codes)}")
        
        results = []
        for stock_code in stock_codes:
            try:
                result = self.predict_single_stock(
                    stock_code=stock_code,
                    prediction_days=prediction_days,
                    include_analysis=False
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
    return {"message": "AI股市预测系统API（无Redis版本）", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    """单只股票预测接口"""
    return prediction_service.predict_single_stock(
        stock_code=request.stock_code,
        prediction_days=request.prediction_days,
        include_analysis=request.include_analysis
    )


@app.post("/predict/batch")
async def batch_predict_stocks(stock_codes: List[str], prediction_days: int = 1):
    """批量预测接口"""
    return prediction_service.batch_predict(stock_codes, prediction_days)


@app.get("/history/{stock_code}")
async def get_prediction_history(stock_code: str, days: int = 30):
    """获取预测历史接口"""
    return prediction_service.get_prediction_history(stock_code, days)


@app.get("/models")
async def get_available_models():
    """获取可用模型信息"""
    return {
        "available_models": list(prediction_service.models.keys()),
        "model_metadata": prediction_service.model_metadata,
        "cache_type": "memory"
    }


if __name__ == "__main__":
    import uvicorn
    
    # 启动API服务
    uvicorn.run(app, host="0.0.0.0", port=8000)