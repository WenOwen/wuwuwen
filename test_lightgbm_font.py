#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试lightgbm_stock_train.py的字体配置
"""

print("测试lightgbm_stock_train.py的字体配置...")

# 按照lightgbm_stock_train.py的导入顺序
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

print("导入matplotlib完成，当前字体设置（导入font_config前）:")
print(f"font.sans-serif: {plt.rcParams['font.sans-serif']}")

# 导入字体配置模块
from font_config import setup_chinese_plot
setup_chinese_plot()  # 设置中文字体

print("导入font_config完成，当前字体设置（导入font_config后）:")
print(f"font.sans-serif: {plt.rcParams['font.sans-serif']}")

# 测试生成一个简单的图片
import matplotlib
print(f"matplotlib backend: {matplotlib.get_backend()}")

plt.figure(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label='正弦曲线')
plt.title('lightgbm_stock_train.py 字体测试')
plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')
plt.legend()
plt.grid(True, alpha=0.3)

# 添加一些模拟的中文标签（类似实际训练脚本中的标签）
plt.text(2, 0.5, '训练损失', fontsize=12, bbox=dict(boxstyle="round", facecolor='wheat'))
plt.text(5, -0.5, '验证损失', fontsize=12, bbox=dict(boxstyle="round", facecolor='lightblue'))
plt.text(8, 0.5, '特征重要性', fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgreen'))

plt.savefig('lightgbm_font_test.png', dpi=300, bbox_inches='tight')
print("测试图片已保存为: lightgbm_font_test.png")

print("测试完成！请检查图片中的中文是否正确显示。")