# 无TA-Lib依赖的特征工程
import pandas as pd
import numpy as np

def calculate_sma(data, window):
    """简单移动平均"""
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    """指数移动平均"""
    return data.ewm(span=window).mean()

def calculate_rsi(data, window=14):
    """相对强弱指标"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window=20, num_std=2):
    """布林带"""
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

def calculate_macd(data, fast=12, slow=26, signal=9):
    """MACD指标"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    histogram = macd - signal_line
    return macd, signal_line, histogram

def add_technical_indicators(df):
    """添加技术指标"""
    df = df.copy()
    df = df.sort_values('交易日期').reset_index(drop=True)
    
    close = df['收盘价']
    high = df['最高价']
    low = df['最低价']
    volume = df['成交量']
    
    # 基础特征
    df['价格变化'] = close.pct_change()
    df['振幅'] = (high - low) / close
    
    # 移动平均线
    for period in [5, 10, 20, 60]:
        df[f'SMA_{period}'] = calculate_sma(close, period)
        df[f'EMA_{period}'] = calculate_ema(close, period)
        df[f'价格_SMA{period}_比率'] = close / df[f'SMA_{period}']
    
    # RSI
    df['RSI_14'] = calculate_rsi(close, 14)
    
    # 布林带
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20)
    df['BB_upper'] = bb_upper
    df['BB_middle'] = bb_middle
    df['BB_lower'] = bb_lower
    df['BB_width'] = (bb_upper - bb_lower) / bb_middle
    df['BB_position'] = (close - bb_lower) / (bb_upper - bb_lower)
    
    # MACD
    macd, signal, histogram = calculate_macd(close)
    df['MACD'] = macd
    df['MACD_signal'] = signal
    df['MACD_histogram'] = histogram
    
    # 成交量指标
    df['成交量_SMA_5'] = calculate_sma(volume, 5)
    df['成交量_SMA_20'] = calculate_sma(volume, 20)
    df['成交量比率'] = volume / df['成交量_SMA_20']
    
    # 价格位置
    df['最高_20'] = high.rolling(20).max()
    df['最低_20'] = low.rolling(20).min()
    df['价格位置_20'] = (close - df['最低_20']) / (df['最高_20'] - df['最低_20'])
    
    # 波动率
    df['波动率_5'] = df['价格变化'].rolling(5).std()
    df['波动率_20'] = df['价格变化'].rolling(20).std()
    
    # 动量
    for period in [5, 10, 20]:
        df[f'动量_{period}'] = close / close.shift(period) - 1
    
    # 预测标签
    df['未来_1日_收益'] = close.shift(-1) / close - 1
    df['未来_1日_方向'] = (df['未来_1日_收益'] > 0).astype(int)
    
    return df

if __name__ == "__main__":
    print("🧪 测试无TA-Lib特征工程...")
    
    # 测试代码
    import os
    csv_files = [f for f in os.listdir('datas_em') if f.endswith('.csv')]
    
    if csv_files:
        test_file = csv_files[0]
        df = pd.read_csv(f'datas_em/{test_file}')
        print(f"原始数据: {df.shape}")
        
        df_with_features = add_technical_indicators(df)
        print(f"添加特征后: {df_with_features.shape}")
        
        feature_cols = [col for col in df_with_features.columns 
                       if any(x in col for x in ['SMA_', 'EMA_', 'RSI', 'MACD', 'BB_', '成交量', '动量'])]
        print(f"技术指标数量: {len(feature_cols)}")
        print("✅ 特征工程测试成功")
    else:
        print("⚠️ 没有数据文件")
