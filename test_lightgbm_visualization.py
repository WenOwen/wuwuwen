#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试lightgbm_stock_train.py的可视化功能
模拟实际运行时的图片生成过程
"""

# 模拟lightgbm_stock_train.py的导入顺序
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.feature_selection import SelectFromModel, RFE
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 导入字体配置模块
from font_config import setup_chinese_plot
setup_chinese_plot()  # 设置中文字体

print("开始测试lightgbm可视化功能...")
print(f"当前matplotlib字体设置: {plt.rcParams['font.sans-serif']}")

# 创建模拟数据
np.random.seed(42)
n_samples = 1000
n_features = 10

# 模拟特征数据
X = np.random.randn(n_samples, n_features)
y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练简单的LightGBM模型
train_data = lgb.Dataset(X_train, label=y_train)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'verbose': -1
}

print("训练LightGBM模型...")
model = lgb.train(params, train_data, num_boost_round=100)

# 预测
y_pred = model.predict(X_test)

print("生成可视化图表...")

# 1. 创建特征重要性图 (模拟lightgbm_stock_train.py的风格)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 特征重要性
feature_names = [f'特征_{i+1}' for i in range(n_features)]
importance = model.feature_importance()
feature_importance_df = pd.DataFrame({
    '特征名称': feature_names,
    '重要性': importance
}).sort_values('重要性', ascending=False)

axes[0, 0].barh(feature_importance_df['特征名称'], feature_importance_df['重要性'])
axes[0, 0].set_title('特征重要性分析')
axes[0, 0].set_xlabel('重要性得分')
axes[0, 0].grid(True, alpha=0.3)

# 预测vs实际
axes[0, 1].scatter(y_test, y_pred, alpha=0.6, s=20)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
axes[0, 1].set_xlabel('实际值')
axes[0, 1].set_ylabel('预测值')
axes[0, 1].set_title('预测vs实际值对比')
axes[0, 1].grid(True, alpha=0.3)

# 残差分析
residuals = y_pred - y_test
axes[1, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
axes[1, 0].set_xlabel('预测值')
axes[1, 0].set_ylabel('残差')
axes[1, 0].set_title('残差分析图')
axes[1, 0].grid(True, alpha=0.3)

# 预测分布对比
axes[1, 1].hist(y_test, bins=30, alpha=0.7, label='实际值分布', color='blue')
axes[1, 1].hist(y_pred, bins=30, alpha=0.7, label='预测值分布', color='red')
axes[1, 1].set_xlabel('数值')
axes[1, 1].set_ylabel('频次')
axes[1, 1].set_title('数值分布对比')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lightgbm_test_visualization.png', dpi=300, bbox_inches='tight')
print("测试图片已保存为: lightgbm_test_visualization.png")

# 2. 创建训练历史图（模拟训练过程）
plt.figure(figsize=(12, 8))

# 模拟训练历史
epochs = np.arange(1, 101)
train_rmse = np.exp(-epochs/50) + 0.1 + 0.02*np.random.randn(100)
val_rmse = np.exp(-epochs/45) + 0.12 + 0.03*np.random.randn(100)

plt.subplot(2, 2, 1)
plt.plot(epochs, train_rmse, label='训练RMSE', color='blue')
plt.plot(epochs, val_rmse, label='验证RMSE', color='red')
plt.title('模型训练损失曲线')
plt.xlabel('训练轮次')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
r2_scores = 1 - val_rmse
plt.plot(epochs, r2_scores, label='验证R²得分', color='green')
plt.title('模型性能曲线')
plt.xlabel('训练轮次')
plt.ylabel('R²得分')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(epochs[-50:], train_rmse[-50:], label='训练损失', linewidth=2)
plt.plot(epochs[-50:], val_rmse[-50:], label='验证损失', linewidth=2)
plt.title('最后50轮训练详情')
plt.xlabel('训练轮次')
plt.ylabel('损失值')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
metrics_data = {
    '指标名称': ['训练RMSE', '验证RMSE', '测试RMSE', 'R²得分'],
    '数值': [train_rmse[-1], val_rmse[-1], np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)]
}
plt.bar(metrics_data['指标名称'], metrics_data['数值'], color=['blue', 'red', 'orange', 'green'])
plt.title('最终模型性能指标')
plt.ylabel('指标值')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lightgbm_training_history.png', dpi=300, bbox_inches='tight')
print("训练历史图已保存为: lightgbm_training_history.png")

print("测试完成！检查生成的图片中的中文是否正确显示。")
print(f"最终字体设置: {plt.rcParams['font.sans-serif']}")