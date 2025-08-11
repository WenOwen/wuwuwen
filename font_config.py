#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的中文字体配置模块
确保所有可视化脚本都能正确显示中文
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import warnings

warnings.filterwarnings('ignore')

class ChineseFontConfig:
    """中文字体配置管理器"""
    
    def __init__(self):
        self.font_initialized = False
        
    def setup_chinese_font(self):
        """设置中文字体，适用于Linux系统"""
        if self.font_initialized:
            return True
            
        try:
            # 强制使用非交互式后端
            matplotlib.use('Agg')
            
            # Linux系统中常见的中文字体路径
            font_paths = [
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
            ]
            
            # 尝试添加系统中的中文字体
            added_fonts = []
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        fm.fontManager.addfont(font_path)
                        added_fonts.append(font_path)
                    except Exception:
                        continue
            
            # 重建字体缓存（兼容不同版本的matplotlib）
            if added_fonts:
                try:
                    fm._rebuild()
                except AttributeError:
                    # 新版本matplotlib使用不同的方法
                    try:
                        fm.fontManager.__init__()
                    except:
                        pass
            
            # 获取可用的中文字体
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            # 优先选择的中文字体列表
            preferred_fonts = [
                'WenQuanYi Micro Hei',
                'WenQuanYi Zen Hei', 
                'Noto Sans CJK SC',
                'AR PL UMing CN',
                'AR PL UKai CN',
                'DejaVu Sans',
                'Liberation Sans'
            ]
            
            # 选择第一个可用的字体
            selected_font = 'DejaVu Sans'  # 默认字体
            for font in preferred_fonts:
                if font in available_fonts:
                    selected_font = font
                    break
            
            # 设置matplotlib参数
            plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.family'] = 'sans-serif'
            
            # 设置图表样式
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['figure.dpi'] = 100
            plt.rcParams['savefig.dpi'] = 300
            plt.rcParams['savefig.bbox'] = 'tight'
            plt.rcParams['savefig.facecolor'] = 'white'
            plt.rcParams['savefig.edgecolor'] = 'none'
            
            self.font_initialized = True
            print(f"中文字体配置成功，使用字体: {selected_font}")
            return True
            
        except Exception as e:
            print(f"字体配置失败: {e}")
            # 使用备用配置
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.family'] = 'sans-serif'
            return False
    
    def get_font_info(self):
        """获取当前字体信息"""
        current_font = plt.rcParams['font.sans-serif'][0]
        return {
            'current_font': current_font,
            'font_family': plt.rcParams['font.family'],
            'unicode_minus': plt.rcParams['axes.unicode_minus']
        }

# 创建全局字体配置实例
font_config = ChineseFontConfig()

def setup_chinese_plot():
    """快捷函数：设置中文绘图环境"""
    return font_config.setup_chinese_font()

def get_plot_font_info():
    """快捷函数：获取绘图字体信息"""
    return font_config.get_font_info()

# 自动初始化（导入时自动设置字体）
if __name__ != "__main__":
    setup_chinese_plot()

if __name__ == "__main__":
    # 测试字体配置
    setup_chinese_plot()
    print("字体配置信息:", get_plot_font_info())
    
    # 测试中文显示
    import numpy as np
    
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.plot(x, y, label='正弦曲线')
    plt.title('中文字体测试 - 正弦函数图')
    plt.xlabel('X轴 (横坐标)')
    plt.ylabel('Y轴 (纵坐标)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('font_test.png')
    print("字体测试图片已保存为 font_test.png")