# -*- coding: utf-8 -*-
"""
AI股市预测系统 - 特征工程模块
功能：基于现有数据构建多维度预测特征
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 可选导入talib
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️  警告: TA-Lib未安装，将使用简化的技术指标计算")

# 导入处理 - 支持直接运行和模块导入
try:
    from .stock_sector_mapping import StockSectorMapping
    from .sector_features import SectorFeatureEngineer
except ImportError:
    # 直接运行时的导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from stock_sector_mapping import StockSectorMapping
    from sector_features import SectorFeatureEngineer

class FeatureEngineering:
    """
    特征工程类 - 将原始股票数据转换为机器学习特征
    """
    
    def __init__(self, enable_cache: bool = True, cache_dir: str = "cache/features"):
        self.feature_names = []
        self.technical_indicators = [
            'SMA', 'EMA', 'RSI', 'MACD', 'BOLL', 'KDJ', 
            'CCI', 'WILLR', 'OBV', 'ATR', 'ADXR'
        ]
        self.sector_mapping = StockSectorMapping()
        self.sector_feature_engineer = SectorFeatureEngineer()
        self.all_sector_features = {}
        
        # 缓存系统
        self.enable_cache = enable_cache
        if enable_cache:
            try:
                from .feature_cache import FeatureCache
                self.cache = FeatureCache(cache_dir)
            except ImportError:
                # 直接运行时的导入
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from feature_cache import FeatureCache
                self.cache = FeatureCache(cache_dir)
        else:
            self.cache = None
    
    def create_all_features(self, df: pd.DataFrame, 
                           stock_code: str,
                           money_flow_df: pd.DataFrame = None,
                           industry_df: pd.DataFrame = None,
                           sector_data: Optional[pd.DataFrame] = None,
                           force_recalculate: bool = False) -> pd.DataFrame:
        """
        创建所有特征
        
        Args:
            df: 股票K线数据
            stock_code: 股票代码
            money_flow_df: 资金流向数据（可选）
            industry_df: 行业数据（可选）
            sector_data: 板块数据（可选）
            force_recalculate: 是否强制重新计算（忽略缓存）
            
        Returns:
            包含所有特征的DataFrame
        """
        # 保存原始数据用于缓存
        original_df_for_cache = df.copy()
        
        # 尝试从缓存获取（如果启用且不强制重计算）
        if self.enable_cache and self.cache and not force_recalculate:
            cached_features = self.cache.get_cached_features(stock_code, original_df_for_cache)
            if cached_features is not None:
                return cached_features
        
        # 静默特征工程处理
        
        # 确保数据按日期排序
        df = df.sort_values('交易日期').reset_index(drop=True)
        
        # 1. 基础价格特征
        df = self._add_price_features(df)
        
        # 2. 技术指标特征
        df = self._add_technical_indicators(df)
        
        # 3. 成交量特征
        df = self._add_volume_features(df)
        
        # 4. 波动率特征
        df = self._add_volatility_features(df)
        
        # 5. 时间序列特征
        df = self._add_time_series_features(df)
        
        # 6. 市场微观结构特征
        df = self._add_microstructure_features(df)
        
        # 7. 资金流向特征（如果有数据）
        if money_flow_df is not None:
            df = self._add_money_flow_features(df, money_flow_df)
        
        # 8. 相对强度特征
        df = self._add_relative_strength_features(df)
        
        # 9. 形态识别特征
        df = self._add_pattern_features(df)
        
        # 10. 股票标识和板块特征
        df = self._add_stock_and_sector_features(df, stock_code, sector_data)
        
        # 11. 创建预测标签
        df = self._create_prediction_labels(df)
        
        # 特征工程完成
        
        # 保存到缓存（如果启用）
        if self.enable_cache and self.cache:
            try:
                self.cache.save_features_to_cache(stock_code, original_df_for_cache, df)
            except Exception as e:
                print(f"⚠️ 缓存保存失败 {stock_code}: {e}")
        
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
        
        # 确保数据类型为float64，talib要求输入必须是double类型
        close = df['收盘价'].astype(np.float64).values
        high = df['最高价'].astype(np.float64).values
        low = df['最低价'].astype(np.float64).values
        open_price = df['开盘价'].astype(np.float64).values
        volume = df['成交量'].astype(np.float64).values
        
        # 移动平均线
        for period in [5, 10, 20, 30, 60]:
            if TALIB_AVAILABLE:
                df[f'SMA_{period}'] = talib.SMA(close, timeperiod=period)
                df[f'EMA_{period}'] = talib.EMA(close, timeperiod=period)
            else:
                df[f'SMA_{period}'] = close.rolling(window=period).mean()
                df[f'EMA_{period}'] = close.ewm(span=period).mean()
            df[f'close_sma_{period}_ratio'] = df['收盘价'] / df[f'SMA_{period}']
            df[f'close_ema_{period}_ratio'] = df['收盘价'] / df[f'EMA_{period}']
        
        # RSI
        for period in [6, 14, 21]:
            if TALIB_AVAILABLE:
                df[f'RSI_{period}'] = talib.RSI(close, timeperiod=period)
            else:
                # 简化的RSI计算
                delta = pd.Series(close).diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        if TALIB_AVAILABLE:
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df['MACD'] = macd
            df['MACD_signal'] = macd_signal
            df['MACD_hist'] = macd_hist
        else:
            # 简化的MACD计算
            ema12 = pd.Series(close).ewm(span=12).mean()
            ema26 = pd.Series(close).ewm(span=26).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9).mean()
            df['MACD'] = macd
            df['MACD_signal'] = macd_signal
            df['MACD_hist'] = macd - macd_signal
        df['MACD_cross'] = np.where(df['MACD'] > df['MACD_signal'], 1, 0)
        
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
        
        # 确保数据类型为float64，talib要求输入必须是double类型
        volume = df['成交量'].astype(np.float64).values
        close = df['收盘价'].astype(np.float64).values
        
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
        
        # 计算振幅（如果不存在的话）
        if '振幅' not in df.columns:
            df['振幅'] = (df['最高价'] - df['最低价']) / df['收盘价'] * 100
        
        # 振幅相关特征
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
        
        # 确保数据类型为float64，talib要求输入必须是double类型
        open_price = df['开盘价'].astype(np.float64).values
        high = df['最高价'].astype(np.float64).values
        low = df['最低价'].astype(np.float64).values
        close = df['收盘价'].astype(np.float64).values
        
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
    
    def _add_stock_and_sector_features(self, df: pd.DataFrame, stock_code: str, 
                                     sector_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        添加股票标识和板块特征（使用新的板块数据）
        
        Args:
            df: 股票数据
            stock_code: 股票代码
            sector_data: 板块聚合数据（已弃用，使用新的板块特征数据）
            
        Returns:
            添加了股票和板块特征的DataFrame
        """
        
        # 获取股票信息
        stock_info = self.sector_mapping.get_stock_info(stock_code)
        
        # 1. 基础股票标识特征
        df['stock_id'] = stock_info['stock_id']
        df['sector_id'] = stock_info['sector_id']
        df['stock_code'] = stock_code
        df['sector_name'] = stock_info['sector']
        
        # 2. 使用新的板块特征工程器添加板块特征
        df = self.sector_feature_engineer.add_sector_features(df, stock_code, stock_info)
        
        # 3. 传统板块特征（保持兼容性）
        if 'price_change' in df.columns:
            # 板块内股票数量（编码特征）
            sector_stocks = self.sector_mapping.get_sector_stocks(stock_info['sector'])
            df['sector_stock_count'] = len(sector_stocks)
            
            # 板块类型编码（基于新的行业分类）
            industry = stock_info.get('sector', '')
            if industry in ['银行', '保险', '证券', '多元金融']:
                df['sector_type'] = 0  # 金融股
            elif industry in ['专用设备', '通用设备', '电网设备', '汽车零部件']:
                df['sector_type'] = 1  # 制造业
            elif industry in ['软件开发', '半导体', '电子元件', '通信设备']:
                df['sector_type'] = 2  # 科技股
            elif industry in ['化学制药', '生物制品', '医疗器械']:
                df['sector_type'] = 3  # 医药股
            elif industry in ['食品加工', '饮料制造', '纺织服装']:
                df['sector_type'] = 4  # 消费股
            else:
                df['sector_type'] = 5  # 其他
        
        # 4. 跨板块相关性特征（简化版）
        market_sensitive_sectors = ['银行', '保险', '钢铁', '煤炭']
        growth_sectors = ['科技', '新能源', '医药', '创业板']
        
        if stock_info['sector'] in market_sensitive_sectors:
            df['is_market_sensitive'] = 1
            df['is_growth_stock'] = 0
        elif stock_info['sector'] in growth_sectors:
            df['is_market_sensitive'] = 0
            df['is_growth_stock'] = 1
        else:
            df['is_market_sensitive'] = 0
            df['is_growth_stock'] = 0
        
        # 存储板块特征用于后续分析
        sector_features = {
            'sector': stock_info['sector'],
            'sector_id': stock_info['sector_id'],
            'stock_count': len(sector_stocks) if 'sector_stocks' in locals() else 1
        }
        self.all_sector_features[stock_code] = sector_features
        
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
        
        # 选择特征列（排除日期、标签、文本等）
        exclude_cols = ['交易日期', 'stock_code', 'sector_name'] + [col for col in df.columns if 'future_' in col]
        
        # 获取所有可能的特征列
        potential_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 确保只选择数值类型的列，排除所有字符串类型
        feature_cols = []
        for col in potential_cols:
            try:
                # 尝试转换为数值类型
                pd.to_numeric(df[col], errors='raise')
                feature_cols.append(col)
            except (ValueError, TypeError):
                # 如果转换失败，排除该列
                continue
        
        # 分离数值特征和分类特征
        categorical_cols = ['stock_id', 'sector_id', 'sector_type', 'is_market_sensitive', 'is_growth_stock']
        categorical_cols = [col for col in categorical_cols if col in feature_cols]  # 只保留存在的分类列
        numerical_cols = [col for col in feature_cols if col not in categorical_cols]
        
        # 确保所有特征列都是数值类型
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
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
        
        # 返回额外的分类特征信息
        feature_info = {
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols,
            'all_cols': feature_cols,
            'n_stocks': len(self.sector_mapping.get_all_stocks()),
            'n_sectors': len(self.sector_mapping.get_all_sectors())
        }
        
        return np.array(X), np.array(y), feature_cols, feature_info


if __name__ == "__main__":
    # 测试代码
    fe = FeatureEngineering()
    
    # 读取示例数据
    df = pd.read_csv('datas_em/sz301636.csv')
    print(f"原始数据形状: {df.shape}")
    
    # 创建特征
    stock_code = 'sz301636'  # 示例股票代码
    df_features = fe.create_all_features(df, stock_code)
    print(f"特征数据形状: {df_features.shape}")
    
    # 准备模型数据
    X, y, feature_names, feature_info = fe.prepare_model_data(df_features)
    print(f"模型输入形状: X={X.shape}, y={y.shape}")
    print(f"特征数量: {len(feature_names)}")
    print(f"数值特征: {len(feature_info['numerical_cols'])}")
    print(f"分类特征: {len(feature_info['categorical_cols'])}")
    print(f"股票数量: {feature_info['n_stocks']}")
    print(f"板块数量: {feature_info['n_sectors']}")