# 简化版特征工程 - 只使用pandas和numpy
import pandas as pd
import numpy as np

def add_basic_features(df):
    """添加基础技术指标"""
    df = df.copy()
    df = df.sort_values('交易日期').reset_index(drop=True)
    
    # 基础价格特征
    df['价格变化'] = df['收盘价'].pct_change()
    df['价格变化绝对值'] = abs(df['价格变化'])
    
    # 移动平均
    for period in [5, 10, 20]:
        df[f'MA_{period}'] = df['收盘价'].rolling(period).mean()
        df[f'价格_MA{period}_比率'] = df['收盘价'] / df[f'MA_{period}']
    
    # 波动率
    df['波动率_5日'] = df['价格变化'].rolling(5).std()
    df['波动率_20日'] = df['价格变化'].rolling(20).std()
    
    # 成交量特征
    df['成交量_MA_5'] = df['成交量'].rolling(5).mean()
    df['成交量比率'] = df['成交量'] / df['成交量_MA_5']
    
    # 价格位置
    df['最高_20日'] = df['最高价'].rolling(20).max()
    df['最低_20日'] = df['最低价'].rolling(20).min()
    df['价格位置'] = (df['收盘价'] - df['最低_20日']) / (df['最高_20日'] - df['最低_20日'])
    
    # 动量指标
    for period in [5, 10]:
        df[f'动量_{period}日'] = df['收盘价'] / df['收盘价'].shift(period) - 1
    
    # 创建预测标签
    df['未来1日收益'] = df['收盘价'].shift(-1) / df['收盘价'] - 1
    df['未来1日方向'] = (df['未来1日收益'] > 0).astype(int)
    
    return df

def prepare_training_data(df, lookback=30):
    """准备训练数据"""
    # 选择特征列
    feature_cols = [col for col in df.columns if 
                   'MA_' in col or '波动率_' in col or '成交量比率' in col or 
                   '价格位置' in col or '动量_' in col or '价格变化' in col]
    
    # 填充缺失值
    df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)
    
    # 构建训练样本
    X, y = [], []
    for i in range(lookback, len(df) - 1):
        X.append(df[feature_cols].iloc[i-lookback:i].values.flatten())
        y.append(df['未来1日方向'].iloc[i])
    
    return np.array(X), np.array(y), feature_cols

if __name__ == "__main__":
    print("🧪 测试简化特征工程...")
    
    # 这里需要有实际的股票数据文件进行测试
    import os
    csv_files = [f for f in os.listdir('datas_em') if f.endswith('.csv')]
    
    if csv_files:
        test_file = csv_files[0]
        df = pd.read_csv(f'datas_em/{test_file}')
        print(f"原始数据: {df.shape}")
        
        df_features = add_basic_features(df)
        print(f"添加特征后: {df_features.shape}")
        
        X, y, feature_names = prepare_training_data(df_features)
        print(f"训练数据: X={X.shape}, y={y.shape}")
        print(f"特征数量: {len(feature_names)}")
        print("✅ 简化特征工程测试通过")
    else:
        print("⚠️ 没有数据文件，无法测试")
