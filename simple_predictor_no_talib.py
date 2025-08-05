# 简单股票预测器
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
                    print(f"\n📊 {stock_code} 预测结果:")
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
        
        print("\nβ✅ 预测器测试完成")
    else:
        print("⚠️ 没有数据文件，请先运行数据收集")
