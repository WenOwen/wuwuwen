#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI股市预测系统 - 最小化启动脚本
只依赖pandas和numpy，让您立即开始使用
"""

import os
import sys
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

def create_directories():
    """创建必要目录"""
    directories = ['datas_em', 'logs', 'models', 'backup', 'config']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ 目录: {directory}")

def check_data_files():
    """检查数据文件"""
    logger.info("📊 检查股票数据文件...")
    
    data_dir = Path('datas_em')
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        logger.warning("⚠️ 没有数据文件，建议运行数据收集脚本")
        return [], 0
    
    # 分析数据质量
    valid_files = []
    total_records = 0
    
    logger.info("🔍 分析数据质量...")
    for file in csv_files[:20]:  # 检查前20个文件
        try:
            df = pd.read_csv(file)
            if len(df) > 50 and '收盘价' in df.columns:
                valid_files.append(file.stem)
                total_records += len(df)
                logger.info(f"  ✅ {file.stem}: {len(df)} 条记录")
            else:
                logger.warning(f"  ⚠️ {file.stem}: 数据不完整")
        except Exception as e:
            logger.warning(f"  ❌ {file.stem}: 读取失败 ({e})")
    
    logger.info(f"📊 数据统计: {len(valid_files)} 个有效文件, {total_records} 条记录")
    return valid_files, len(csv_files)

def create_simple_feature_engineering():
    """创建简化的特征工程"""
    logger.info("🔧 创建简化特征工程...")
    
    code = '''# 简化版特征工程 - 只使用pandas和numpy
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
'''
    
    with open('simple_features.py', 'w', encoding='utf-8') as f:
        f.write(code)
    
    logger.info("✅ 简化特征工程创建完成: simple_features.py")

def create_simple_predictor():
    """创建简单预测器"""
    logger.info("🤖 创建简单预测器...")
    
    code = '''# 简单预测器 - 基于统计方法
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
                    print(f"\\n📊 {stock_code} 预测结果:")
                    print(f"  预测方向: {'上涨' if result['prediction'] == 1 else '下跌'}")
                    print(f"  预测概率: {result['probability']:.1%}")
                    print(f"  置信度: {result['confidence']}")
                    
                    if i == 0:  # 只显示第一个的详细信息
                        print(f"  详细指标:")
                        for key, value in result['score_details'].items():
                            print(f"    {key}: {value:.4f}")
                
            except Exception as e:
                print(f"❌ {file} 预测失败: {e}")
        
        print("\\n✅ 简单预测器测试完成")
    else:
        print("⚠️ 没有数据文件，无法测试")
'''
    
    with open('simple_predictor.py', 'w', encoding='utf-8') as f:
        f.write(code)
    
    logger.info("✅ 简单预测器创建完成: simple_predictor.py")

def main():
    """主函数"""
    print("🚀 AI股市预测系统 - 最小化启动")
    print("=" * 50)
    
    logger.info("💻 检测环境: pandas + numpy")
    
    # 1. 创建目录
    create_directories()
    
    # 2. 检查数据
    valid_files, total_files = check_data_files()
    
    # 3. 创建简化工具
    create_simple_feature_engineering()
    create_simple_predictor()
    
    # 总结
    print("\n" + "=" * 50)
    print("📋 最小化启动完成报告:")
    print(f"   📁 数据文件: {len(valid_files)}/{total_files} 有效")
    print(f"   🔧 简化工具: 已创建")
    print(f"   📊 依赖包: pandas + numpy")
    
    if len(valid_files) > 0:
        print(f"\n🎉 基础系统就绪！")
        print(f"\n📝 立即可用功能:")
        print(f"   1. 测试特征工程: python simple_features.py")
        print(f"   2. 测试预测器: python simple_predictor.py")
        print(f"   3. 收集更多数据: python 2.1获取全数据（东财）.py")
        print(f"\n🔄 安装更多依赖: python install_with_mirror.py")
    else:
        print(f"\n📥 需要先收集数据:")
        print(f"   运行: python 2.1获取全数据（东财）.py")

if __name__ == "__main__":
    main()