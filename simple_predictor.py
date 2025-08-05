# 简单预测器 - 基于统计方法
import pandas as pd
import numpy as np
from simple_features import add_basic_features, prepare_training_data

class SimplePredictor:
    """简单预测器 - 基于移动平均和动量"""
    
    def __init__(self):
        self.trained = False
        self.feature_stats = {}
    
    def fit(self, df):
        """训练（计算统计特征）"""
        df_features = add_basic_features(df)
        
        # 计算特征统计
        feature_cols = [col for col in df_features.columns if 
                       'MA_' in col or '波动率_' in col or '成交量比率' in col or 
                       '价格位置' in col or '动量_' in col]
        
        for col in feature_cols:
            self.feature_stats[col] = {
                'mean': df_features[col].mean(),
                'std': df_features[col].std(),
                'up_threshold': df_features[col].quantile(0.7),
                'down_threshold': df_features[col].quantile(0.3)
            }
        
        self.trained = True
        print(f"✅ 训练完成，使用了 {len(feature_cols)} 个特征")
    
    def predict(self, df):
        """预测"""
        if not self.trained:
            raise ValueError("模型未训练")
        
        df_features = add_basic_features(df)
        latest = df_features.iloc[-1]
        
        # 简单规则预测
        score = 0
        
        # 价格趋势
        if latest['价格_MA5_比率'] > 1.02:  # 价格明显高于5日均线
            score += 1
        elif latest['价格_MA5_比率'] < 0.98:
            score -= 1
        
        # 动量
        if latest['动量_5日'] > 0.02:  # 5日动量为正
            score += 1
        elif latest['动量_5日'] < -0.02:
            score -= 1
        
        # 价格位置
        if latest['价格位置'] > 0.7:  # 价格位于高位
            score += 1
        elif latest['价格位置'] < 0.3:
            score -= 1
        
        # 成交量
        if latest['成交量比率'] > 1.5:  # 成交量放大
            score += 0.5
        
        # 转换为概率
        probability = 0.5 + score * 0.1  # 基础概率0.5，每个信号调整0.1
        probability = max(0.1, min(0.9, probability))  # 限制在0.1-0.9之间
        
        prediction = 1 if probability > 0.5 else 0
        
        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': 'high' if abs(probability - 0.5) > 0.2 else 'medium' if abs(probability - 0.5) > 0.1 else 'low',
            'score_details': {
                'price_trend': latest['价格_MA5_比率'],
                'momentum_5d': latest['动量_5日'],
                'price_position': latest['价格位置'],
                'volume_ratio': latest['成交量比率']
            }
        }

if __name__ == "__main__":
    print("🧪 测试简单预测器...")
    
    import os
    csv_files = [f for f in os.listdir('datas_em') if f.endswith('.csv')]
    
    if csv_files:
        # 测试预测器
        predictor = SimplePredictor()
        
        for i, file in enumerate(csv_files[:3]):  # 测试前3个股票
            try:
                df = pd.read_csv(f'datas_em/{file}')
                if len(df) > 100:
                    # 训练
                    train_data = df[:-10]  # 留出最后10天作为测试
                    predictor.fit(train_data)
                    
                    # 预测
                    test_data = df[:-5]  # 用倒数第5天的数据预测
                    result = predictor.predict(test_data)
                    
                    stock_code = file.replace('.csv', '')
                    print(f"\n📊 {stock_code} 预测结果:")
                    print(f"  预测方向: {'上涨' if result['prediction'] == 1 else '下跌'}")
                    print(f"  预测概率: {result['probability']:.1%}")
                    print(f"  置信度: {result['confidence']}")
                    
                    if i == 0:  # 只显示第一个的详细信息
                        print(f"  详细指标:")
                        for key, value in result['score_details'].items():
                            print(f"    {key}: {value:.4f}")
                
            except Exception as e:
                print(f"❌ {file} 预测失败: {e}")
        
        print("\n✅ 简单预测器测试完成")
    else:
        print("⚠️ 没有数据文件，无法测试")
