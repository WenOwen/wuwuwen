#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理matplotlib字体缓存并重新配置中文字体
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import shutil

print("🔧 清理matplotlib字体缓存...")

# 1. 清理matplotlib缓存目录
cache_dir = matplotlib.get_cachedir()
print(f"缓存目录: {cache_dir}")

if os.path.exists(cache_dir):
    try:
        shutil.rmtree(cache_dir)
        print("✅ 成功清理缓存目录")
    except Exception as e:
        print(f"⚠️  清理缓存目录失败: {e}")

# 2. 重建字体管理器
try:
    fm.fontManager.__init__()
    print("✅ 重建字体管理器成功")
except Exception as e:
    print(f"⚠️  重建字体管理器失败: {e}")

# 3. 重新配置中文字体
from font_config import setup_chinese_plot
setup_chinese_plot()

print(f"✅ 字体配置完成，当前使用字体: {plt.rcParams['font.sans-serif'][0]}")

# 4. 测试中文显示
import numpy as np

plt.figure(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label='正弦曲线')
plt.title('清理缓存后的中文字体测试')
plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')
plt.legend()
plt.grid(True, alpha=0.3)

# 添加各种中文标签测试
plt.text(2, 0.5, '训练损失', fontsize=12, bbox=dict(boxstyle="round", facecolor='wheat'))
plt.text(5, -0.5, '验证损失', fontsize=12, bbox=dict(boxstyle="round", facecolor='lightblue'))
plt.text(8, 0.5, '特征重要性', fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgreen'))

plt.savefig('cache_cleared_test.png', dpi=300, bbox_inches='tight')
print("🎯 测试图片已保存为: cache_cleared_test.png")
print("请检查图片中的中文是否正确显示！")