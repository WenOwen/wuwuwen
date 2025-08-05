#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI股市预测系统 - 简化启动脚本（兼容Python 3.8，无TA-Lib依赖）
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 7):
        logger.error("Python版本必须≥3.7，当前版本: %s", sys.version)
        return False
    logger.info("✅ Python版本: %s", sys.version.split()[0])
    return True

def check_basic_packages():
    """检查基础包"""
    basic_packages = ['pandas', 'numpy']
    
    for package in basic_packages:
        try:
            __import__(package)
            logger.info("✅ %s 已安装", package)
        except ImportError:
            logger.warning("❌ %s 未安装，尝试安装...", package)
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                             check=True, capture_output=True)
                logger.info("✅ %s 安装成功", package)
            except:
                logger.error("❌ %s 安装失败", package)
                return False
    
    return True

def create_directories():
    """创建目录"""
    directories = ['datas_em', 'logs', 'models', 'backup', 'config']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info("✅ 目录: %s", directory)

def check_data_files():
    """检查数据文件"""
    data_dir = Path('datas_em')
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        logger.warning("⚠️ 没有股票数据文件")
        return False, 0
    
    valid_files = 0
    for file in csv_files[:10]:
        try:
            df = pd.read_csv(file)
            if len(df) > 50:
                valid_files += 1
        except:
            continue
    
    logger.info("📊 发现 %d 个数据文件，%d 个有效", len(csv_files), valid_files)
    return len(csv_files) > 0, len(csv_files)

def create_no_talib_features():
    """创建不依赖TA-Lib的特征工程"""
    logger.info("🔧 创建无TA-Lib特征工程...")
    
    code = '''# 无TA-Lib依赖的特征工程
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
'''
    
    with open('features_no_talib.py', 'w', encoding='utf-8') as f:
        f.write(code)
    
    logger.info("✅ 无TA-Lib特征工程已创建: features_no_talib.py")

def create_simple_predictor():
    """创建简单预测器"""
    logger.info("🤖 创建简单预测器...")
    
    code = '''# 简单股票预测器
import pandas as pd
import numpy as np
from features_no_talib import add_technical_indicators

class SimpleStockPredictor:
    """简单股票预测器"""
    
    def __init__(self):
        self.trained = False
        self.thresholds = {}
    
    def train(self, df):
        """训练（设置阈值）"""
        df_features = add_technical_indicators(df)
        
        # 计算各指标的阈值
        self.thresholds = {
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'bb_upper_threshold': 0.8,
            'bb_lower_threshold': 0.2,
            'volume_high': df_features['成交量比率'].quantile(0.8),
            'momentum_positive': 0.02,
            'momentum_negative': -0.02
        }
        
        self.trained = True
        print("✅ 简单预测器训练完成")
    
    def predict(self, df):
        """预测股票走势"""
        if not self.trained:
            raise ValueError("预测器未训练")
        
        df_features = add_technical_indicators(df)
        latest = df_features.iloc[-1]
        
        # 预测得分
        score = 0
        signals = []
        
        # RSI信号
        if latest['RSI_14'] > self.thresholds['rsi_overbought']:
            score -= 1
            signals.append("RSI超买")
        elif latest['RSI_14'] < self.thresholds['rsi_oversold']:
            score += 1
            signals.append("RSI超卖")
        
        # 价格相对均线位置
        if latest['价格_SMA5_比率'] > 1.02:
            score += 1
            signals.append("价格高于5日均线")
        elif latest['价格_SMA5_比率'] < 0.98:
            score -= 1
            signals.append("价格低于5日均线")
        
        # 布林带位置
        if latest['BB_position'] > self.thresholds['bb_upper_threshold']:
            score -= 0.5
            signals.append("接近布林带上轨")
        elif latest['BB_position'] < self.thresholds['bb_lower_threshold']:
            score += 0.5
            signals.append("接近布林带下轨")
        
        # MACD信号
        if latest['MACD'] > latest['MACD_signal']:
            score += 0.5
            signals.append("MACD金叉")
        else:
            score -= 0.5
            signals.append("MACD死叉")
        
        # 成交量
        if latest['成交量比率'] > self.thresholds['volume_high']:
            score += 0.5
            signals.append("成交量放大")
        
        # 动量
        if latest['动量_5'] > self.thresholds['momentum_positive']:
            score += 1
            signals.append("短期动量向上")
        elif latest['动量_5'] < self.thresholds['momentum_negative']:
            score -= 1
            signals.append("短期动量向下")
        
        # 转换为概率
        probability = 0.5 + score * 0.08  # 每个信号影响8%
        probability = max(0.1, min(0.9, probability))
        
        prediction = 1 if probability > 0.5 else 0
        confidence = 'high' if abs(probability - 0.5) > 0.2 else 'medium' if abs(probability - 0.5) > 0.1 else 'low'
        
        return {
            'prediction': prediction,
            'direction': '上涨' if prediction == 1 else '下跌',
            'probability': probability,
            'confidence': confidence,
            'signals': signals,
            'score': score,
            'indicators': {
                'RSI': latest['RSI_14'],
                'price_sma5_ratio': latest['价格_SMA5_比率'],
                'bb_position': latest['BB_position'],
                'volume_ratio': latest['成交量比率'],
                'momentum_5d': latest['动量_5']
            }
        }

if __name__ == "__main__":
    print("🧪 测试简单预测器...")
    
    import os
    csv_files = [f for f in os.listdir('datas_em') if f.endswith('.csv')]
    
    if csv_files:
        predictor = SimpleStockPredictor()
        
        for i, file in enumerate(csv_files[:3]):
            try:
                df = pd.read_csv(f'datas_em/{file}')
                if len(df) > 100:
                    # 用前80%数据训练
                    train_size = int(len(df) * 0.8)
                    train_data = df[:train_size]
                    test_data = df[:train_size + 50]  # 用于预测的数据
                    
                    predictor.train(train_data)
                    result = predictor.predict(test_data)
                    
                    stock_code = file.replace('.csv', '')
                    print(f"\\n📊 {stock_code} 预测结果:")
                    print(f"  预测方向: {result['direction']}")
                    print(f"  预测概率: {result['probability']:.1%}")
                    print(f"  置信度: {result['confidence']}")
                    print(f"  信号: {', '.join(result['signals'][:3])}")
                    
                    if i == 0:  # 详细显示第一个
                        print(f"  详细指标:")
                        for key, value in result['indicators'].items():
                            print(f"    {key}: {value:.4f}")
                
            except Exception as e:
                print(f"❌ {file} 预测失败: {e}")
        
        print("\\nβ✅ 预测器测试完成")
    else:
        print("⚠️ 没有数据文件，请先运行数据收集")
'''
    
    with open('simple_predictor_no_talib.py', 'w', encoding='utf-8') as f:
        f.write(code)
    
    logger.info("✅ 简单预测器已创建: simple_predictor_no_talib.py")

def main():
    """主函数"""
    print("🚀 AI股市预测系统 - 简化启动（Python 3.8兼容，无TA-Lib）")
    print("=" * 60)
    
    # 1. 检查Python版本
    if not check_python_version():
        return
    
    # 2. 检查基础包
    if not check_basic_packages():
        logger.error("❌ 基础包检查失败")
        return
    
    # 3. 创建目录
    create_directories()
    
    # 4. 检查数据
    has_data, file_count = check_data_files()
    
    # 5. 创建无TA-Lib工具
    create_no_talib_features()
    create_simple_predictor()
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 简化启动完成:")
    print(f"   🐍 Python版本: {sys.version.split()[0]} ✅")
    print(f"   📦 基础包: pandas + numpy ✅")
    print(f"   📁 数据文件: {file_count} 个")
    print(f"   🔧 无TA-Lib工具: 已创建 ✅")
    
    if has_data:
        print(f"\n🎉 系统就绪！立即可用功能:")
        print(f"   1. 测试特征工程: python features_no_talib.py")
        print(f"   2. 测试预测功能: python simple_predictor_no_talib.py")
        print(f"   3. 收集更多数据: python 2.1获取全数据（东财）.py")
    else:
        print(f"\n📥 请先收集股票数据:")
        print(f"   运行: python 2.1获取全数据（东财）.py")
    
    print(f"\n💡 可选安装TA-Lib:")
    print(f"   运行: python install_talib.py")

if __name__ == "__main__":
    main()