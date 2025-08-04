# -*- coding: utf-8 -*-
"""
AI股市预测系统 - 特征工程模块
功能：基于现有数据构建多维度预测特征
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """
    特征工程类 - 将原始股票数据转换为机器学习特征
    """
    
    def __init__(self):
        self.feature_names = []
        self.technical_indicators = [
            'SMA', 'EMA', 'RSI', 'MACD', 'BOLL', 'KDJ', 
            'CCI', 'WILLR', 'OBV', 'ATR', 'ADXR'
        ]
    
    def create_all_features(self, df: pd.DataFrame, 
                           money_flow_df: pd.DataFrame = None,
                           industry_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        创建所有特征
        
        Args:
            df: 股票K线数据
            money_flow_df: 资金流向数据（可选）
            industry_df: 行业数据（可选）
            
        Returns:
            包含所有特征的DataFrame
        """
        print("🔧 开始特征工程...")
        
        # 确保数据按日期排序
        df = df.sort_values('交易日期').reset_index(drop=True)
        
        # 1. 基础价格特征
        df = self._add_price_features(df)
        print("✅ 价格特征完成")
        
        # 2. 技术指标特征
        df = self._add_technical_indicators(df)
        print("✅ 技术指标完成")
        
        # 3. 成交量特征
        df = self._add_volume_features(df)
        print("✅ 成交量特征完成")
        
        # 4. 波动率特征
        df = self._add_volatility_features(df)
        print("✅ 波动率特征完成")
        
        # 5. 时间序列特征
        df = self._add_time_series_features(df)
        print("✅ 时间序列特征完成")
        
        # 6. 市场微观结构特征
        df = self._add_microstructure_features(df)
        print("✅ 市场微观结构特征完成")
        
        # 7. 资金流向特征（如果有数据）
        if money_flow_df is not None:
            df = self._add_money_flow_features(df, money_flow_df)
            print("✅ 资金流向特征完成")
        
        # 8. 相对强度特征
        df = self._add_relative_strength_features(df)
        print("✅ 相对强度特征完成")
        
        # 9. 形态识别特征
        df = self._add_pattern_features(df)
        print("✅ 形态识别特征完成")
        
        # 10. 创建预测标签
        df = self._create_prediction_labels(df)
        print("✅ 预测标签完成")
        
        print(f"🎉 特征工程完成！总计 {len(df.columns)} 个特征")
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加价格相关特征"""
        
        # 基础价格变化
        df['price_change'] = df['收盘价'].pct_change()
        df['price_change_abs'] = abs(df['price_change'])
        
        # 价格位置特征
        df['close_to_high_ratio'] = df['收盘价'] / df['最高价']
        df['close_to_low_ratio'] = df['收盘价'] / df['最低价']
        df['close_to_open_ratio'] = df['收盘价'] / df['开盘价']
        
        # 价格范围特征
        df['hl_ratio'] = df['最高价'] / df['最低价']  # 高低比
        df['body_ratio'] = abs(df['收盘价'] - df['开盘价']) / (df['最高价'] - df['最低价'] + 1e-8)  # 实体比例
        
        # 上下影线
        df['upper_shadow'] = (df['最高价'] - np.maximum(df['开盘价'], df['收盘价'])) / (df['最高价'] - df['最低价'] + 1e-8)
        df['lower_shadow'] = (np.minimum(df['开盘价'], df['收盘价']) - df['最低价']) / (df['最高价'] - df['最低价'] + 1e-8)
        
        # 多期价格特征
        for period in [3, 5, 10, 20]:
            df[f'close_max_{period}d'] = df['收盘价'].rolling(period).max()
            df[f'close_min_{period}d'] = df['收盘价'].rolling(period).min()
            df[f'close_position_{period}d'] = (df['收盘价'] - df[f'close_min_{period}d']) / (df[f'close_max_{period}d'] - df[f'close_min_{period}d'] + 1e-8)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标特征"""
        
        close = df['收盘价'].values
        high = df['最高价'].values
        low = df['最低价'].values
        open_price = df['开盘价'].values
        volume = df['成交量'].values
        
        # 移动平均线
        for period in [5, 10, 20, 30, 60]:
            df[f'SMA_{period}'] = talib.SMA(close, timeperiod=period)
            df[f'EMA_{period}'] = talib.EMA(close, timeperiod=period)
            df[f'close_sma_{period}_ratio'] = df['收盘价'] / df[f'SMA_{period}']
            df[f'close_ema_{period}_ratio'] = df['收盘价'] / df[f'EMA_{period}']
        
        # RSI
        for period in [6, 14, 21]:
            df[f'RSI_{period}'] = talib.RSI(close, timeperiod=period)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macd_signal
        df['MACD_hist'] = macd_hist
        df['MACD_cross'] = np.where(macd > macd_signal, 1, 0)
        
        # 布林带
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BB_upper'] = bb_upper
        df['BB_middle'] = bb_middle  
        df['BB_lower'] = bb_lower
        df['BB_width'] = (bb_upper - bb_lower) / bb_middle
        df['BB_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        # KDJ指标
        k, d = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
        df['KDJ_K'] = k
        df['KDJ_D'] = d
        df['KDJ_J'] = 3 * k - 2 * d
        
        # CCI
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        
        # Williams %R
        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # 平均真实波幅
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        df['ATR_ratio'] = df['ATR'] / df['收盘价']
        
        # ADX
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量特征"""
        
        volume = df['成交量'].values
        close = df['收盘价'].values
        
        # 成交量移动平均
        for period in [5, 10, 20]:
            df[f'volume_ma_{period}'] = df['成交量'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['成交量'] / df[f'volume_ma_{period}']
        
        # OBV能量潮
        df['OBV'] = talib.OBV(close, volume)
        
        # 成交量相对强度
        df['volume_rsi'] = talib.RSI(volume, timeperiod=14)
        
        # 价量关系
        df['price_volume_trend'] = df['price_change'] * df['成交量']
        df['volume_price_trend'] = talib.OBV(close, volume)
        
        # 成交额特征
        df['amount_ratio'] = df['成交额'] / df['成交额'].rolling(20).mean()
        
        # 换手率特征
        df['turnover_ma_5'] = df['换手率'].rolling(5).mean()
        df['turnover_ma_20'] = df['换手率'].rolling(20).mean()
        df['turnover_relative'] = df['换手率'] / df['turnover_ma_20']
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波动率特征"""
        
        # 历史波动率
        for period in [5, 10, 20]:
            df[f'volatility_{period}d'] = df['price_change'].rolling(period).std()
            df[f'volatility_{period}d_norm'] = df[f'volatility_{period}d'] / df[f'volatility_{period}d'].rolling(60).mean()
        
        # 振幅相关
        df['amplitude_ma_5'] = df['振幅'].rolling(5).mean()
        df['amplitude_ma_20'] = df['振幅'].rolling(20).mean()
        df['amplitude_relative'] = df['振幅'] / df['amplitude_ma_20']
        
        # 真实波动率
        df['true_range'] = np.maximum(
            df['最高价'] - df['最低价'],
            np.maximum(
                abs(df['最高价'] - df['收盘价'].shift(1)),
                abs(df['最低价'] - df['收盘价'].shift(1))
            )
        )
        
        return df
    
    def _add_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间序列特征"""
        
        # 滞后特征
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['收盘价'].shift(lag)
            df[f'return_lag_{lag}'] = df['price_change'].shift(lag)
            df[f'volume_lag_{lag}'] = df['成交量'].shift(lag)
        
        # 滚动统计特征
        for window in [5, 10, 20]:
            df[f'return_mean_{window}d'] = df['price_change'].rolling(window).mean()
            df[f'return_std_{window}d'] = df['price_change'].rolling(window).std()
            df[f'return_skew_{window}d'] = df['price_change'].rolling(window).skew()
            df[f'return_kurt_{window}d'] = df['price_change'].rolling(window).kurt()
        
        # 动量特征
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}d'] = df['收盘价'] / df['收盘价'].shift(period) - 1
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加市场微观结构特征"""
        
        # 跳空特征
        df['gap'] = (df['开盘价'] - df['收盘价'].shift(1)) / df['收盘价'].shift(1)
        df['gap_up'] = np.where(df['gap'] > 0.02, 1, 0)  # 向上跳空
        df['gap_down'] = np.where(df['gap'] < -0.02, 1, 0)  # 向下跳空
        
        # 涨跌停特征
        df['limit_up'] = np.where(df['涨跌幅'] >= 9.5, 1, 0)
        df['limit_down'] = np.where(df['涨跌幅'] <= -9.5, 1, 0)
        
        # 连续涨跌
        df['continuous_up'] = (df['涨跌幅'] > 0).astype(int)
        df['continuous_down'] = (df['涨跌幅'] < 0).astype(int)
        
        for period in [3, 5]:
            df[f'up_days_{period}'] = df['continuous_up'].rolling(period).sum()
            df[f'down_days_{period}'] = df['continuous_down'].rolling(period).sum()
        
        return df
    
    def _add_money_flow_features(self, df: pd.DataFrame, money_flow_df: pd.DataFrame) -> pd.DataFrame:
        """添加资金流向特征"""
        
        # 这里假设money_flow_df包含资金流向数据
        # 实际实现时需要根据具体的资金流向数据结构调整
        
        # 主力资金净流入
        if '主力净流入' in money_flow_df.columns:
            # 合并资金流向数据
            money_flow_df = money_flow_df.rename(columns={'交易日期': '交易日期'})
            df = df.merge(money_flow_df[['交易日期', '主力净流入']], on='交易日期', how='left')
            
            # 资金流向指标
            df['money_flow_ratio'] = df['主力净流入'] / df['成交额']
            df['money_flow_ma_5'] = df['主力净流入'].rolling(5).mean()
            df['money_flow_ma_20'] = df['主力净流入'].rolling(20).mean()
            df['money_flow_trend'] = np.where(df['money_flow_ma_5'] > df['money_flow_ma_20'], 1, 0)
        
        return df
    
    def _add_relative_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加相对强度特征"""
        
        # 相对于自身历史的强度
        for period in [20, 60, 120]:
            df[f'rs_self_{period}d'] = df['收盘价'] / df['收盘价'].rolling(period).mean() - 1
        
        # 价格分位数
        for period in [20, 60]:
            df[f'price_percentile_{period}d'] = df['收盘价'].rolling(period).rank(pct=True)
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术形态识别特征"""
        
        open_price = df['开盘价'].values
        high = df['最高价'].values
        low = df['最低价'].values
        close = df['收盘价'].values
        
        # 常见K线形态
        df['HAMMER'] = talib.CDLHAMMER(open_price, high, low, close)
        df['DOJI'] = talib.CDLDOJI(open_price, high, low, close)
        df['ENGULFING'] = talib.CDLENGULFING(open_price, high, low, close)
        df['HARAMI'] = talib.CDLHARAMI(open_price, high, low, close)
        df['MARUBOZU'] = talib.CDLMARUBOZU(open_price, high, low, close)
        
        # 趋势形态
        df['trend_up'] = np.where(
            (df['收盘价'] > df['SMA_5']) & 
            (df['SMA_5'] > df['SMA_20']) & 
            (df['SMA_20'] > df['SMA_60']), 1, 0
        )
        
        df['trend_down'] = np.where(
            (df['收盘价'] < df['SMA_5']) & 
            (df['SMA_5'] < df['SMA_20']) & 
            (df['SMA_20'] < df['SMA_60']), 1, 0
        )
        
        return df
    
    def _create_prediction_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建预测标签"""
        
        # 未来收益率
        for days in [1, 3, 5]:
            df[f'future_return_{days}d'] = df['收盘价'].shift(-days) / df['收盘价'] - 1
            df[f'future_direction_{days}d'] = np.where(df[f'future_return_{days}d'] > 0, 1, 0)
        
        # 未来最高价和最低价
        for days in [3, 5]:
            df[f'future_max_{days}d'] = df['最高价'].rolling(days).max().shift(-days)
            df[f'future_min_{days}d'] = df['最低价'].rolling(days).min().shift(-days)
            df[f'future_max_return_{days}d'] = df[f'future_max_{days}d'] / df['收盘价'] - 1
            df[f'future_min_return_{days}d'] = df[f'future_min_{days}d'] / df['收盘价'] - 1
        
        # 未来波动率
        for days in [5, 10]:
            future_returns = df['price_change'].shift(-days).rolling(days).apply(lambda x: x.std())
            df[f'future_volatility_{days}d'] = future_returns
        
        return df
    
    def prepare_model_data(self, df: pd.DataFrame, 
                          prediction_days: int = 1,
                          lookback_window: int = 60) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备模型训练数据
        
        Args:
            df: 特征数据
            prediction_days: 预测天数
            lookback_window: 回望窗口
            
        Returns:
            X, y, feature_names
        """
        
        # 选择特征列（排除日期、标签等）
        exclude_cols = ['交易日期'] + [col for col in df.columns if 'future_' in col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 填充缺失值
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)
        
        # 准备目标变量
        target_col = f'future_direction_{prediction_days}d'
        
        # 构建序列数据
        X, y = [], []
        
        for i in range(lookback_window, len(df) - prediction_days):
            # 输入序列
            X.append(df[feature_cols].iloc[i-lookback_window:i].values)
            # 目标值
            y.append(df[target_col].iloc[i])
        
        return np.array(X), np.array(y), feature_cols


if __name__ == "__main__":
    # 测试代码
    fe = FeatureEngineering()
    
    # 读取示例数据
    df = pd.read_csv('datas_em/sz301636.csv')
    print(f"原始数据形状: {df.shape}")
    
    # 创建特征
    df_features = fe.create_all_features(df)
    print(f"特征数据形状: {df_features.shape}")
    
    # 准备模型数据
    X, y, feature_names = fe.prepare_model_data(df_features)
    print(f"模型输入形状: X={X.shape}, y={y.shape}")
    print(f"特征数量: {len(feature_names)}")