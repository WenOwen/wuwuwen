#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文字体配置模块
解决matplotlib中文显示乱码问题
"""

import os
import sys
import platform
import warnings
from pathlib import Path

def setup_chinese_plot():
    """
    设置matplotlib支持中文显示
    自动检测系统并配置合适的中文字体
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties, fontManager
        import matplotlib.font_manager as fm
        
        # 禁用字体相关警告
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
        
        # 设置matplotlib后端（避免GUI问题）
        matplotlib.use('Agg')
        
        # 清理字体缓存
        try:
            fontManager.findfont(fontManager.fontManager.defaultFont[0], rebuild_if_missing=True)
        except:
            pass
        
        # 检测操作系统并设置对应的中文字体
        system = platform.system()
        
        font_candidates = []
        font_paths = []
        
        if system == "Windows":
            # Windows系统常见中文字体及其路径
            windows_fonts_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
            font_candidates = [
                ('Microsoft YaHei', 'msyh.ttc'),      # 微软雅黑
                ('SimHei', 'simhei.ttf'),             # 黑体
                ('SimSun', 'simsun.ttc'),             # 宋体
                ('KaiTi', 'kaiti.ttf'),               # 楷体
                ('FangSong', 'simfang.ttf'),          # 仿宋
                ('Microsoft YaHei UI', 'msyhbd.ttc'), # 微软雅黑UI
            ]
            
            # 检查字体文件是否存在
            for font_name, font_file in font_candidates:
                font_path = os.path.join(windows_fonts_dir, font_file)
                if os.path.exists(font_path):
                    font_paths.append((font_name, font_path))
        
        elif system == "Darwin":  # macOS
            # macOS系统中文字体
            font_candidates = [
                'PingFang SC',          # 苹方
                'Hiragino Sans GB',     # 冬青黑体
                'STHeiti',              # 华文黑体
                'Arial Unicode MS'
            ]
        else:  # Linux
            # Linux系统中文字体
            font_candidates = [
                'WenQuanYi Micro Hei',  # 文泉驿微米黑
                'WenQuanYi Zen Hei',    # 文泉驿正黑
                'Noto Sans CJK SC',     # 思源黑体
                'Source Han Sans CN',   # 思源黑体
                'DejaVu Sans'
            ]
        
        # 方法1：尝试直接使用字体文件路径（Windows）
        chinese_font = None
        if font_paths:
            for font_name, font_path in font_paths:
                try:
                    # 将字体添加到matplotlib
                    fm.fontManager.addfont(font_path)
                    # 获取字体属性
                    prop = fm.FontProperties(fname=font_path)
                    font_family = prop.get_name()
                    
                    chinese_font = font_family
                    print(f"✅ 通过文件路径设置中文字体成功: {font_name} ({font_family})")
                    break
                except Exception as e:
                    print(f"   字体 {font_name} 设置失败: {e}")
                    continue
        
        # 方法2：使用字体名称查找
        if not chinese_font:
            # 获取所有可用字体
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            print(f"   系统字体总数: {len(available_fonts)}")
            
            # 查找中文字体
            for font in font_candidates if not font_paths else [name for name, _ in font_candidates]:
                if font in available_fonts:
                    chinese_font = font
                    print(f"✅ 通过字体名称找到中文字体: {font}")
                    break
            
            # 如果没找到预设字体，尝试查找任何包含中文的字体
            if not chinese_font:
                chinese_keywords = ['微软雅黑', 'yahei', 'chinese', 'cjk', 'han', 'hei', 'song', 'kai', 'fang', 'simhei', 'simsun']
                for font in available_fonts:
                    if any(keyword in font.lower() for keyword in chinese_keywords):
                        chinese_font = font
                        print(f"✅ 找到包含中文关键词的字体: {font}")
                        break
        
        # 方法3：强制设置常用字体组合
        if chinese_font:
            plt.rcParams['font.sans-serif'] = [chinese_font] + ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
        else:
            # 使用多个备选字体
            plt.rcParams['font.sans-serif'] = [
                'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong',  # Windows
                'PingFang SC', 'Hiragino Sans GB', 'STHeiti',  # macOS
                'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',  # Linux
                'Arial Unicode MS', 'Arial', 'DejaVu Sans', 'Liberation Sans'  # 通用
            ]
            print("⚠️ 使用字体列表备选方案")
        
        # 设置负号正常显示
        plt.rcParams['axes.unicode_minus'] = False
        
        # 强制刷新字体缓存
        try:
            fm.fontManager._rebuild()
            # 清除matplotlib内部缓存
            if hasattr(plt, '_original_font_manager'):
                plt._original_font_manager = None
        except:
            pass
        
        # 其他图表样式设置
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.1
        
        # 设置图表样式
        plt.style.use('default')
        
        # 重新应用字体设置
        if chinese_font:
            plt.rcParams['font.sans-serif'] = [chinese_font] + ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
        
        # 输出当前字体配置
        print(f"   当前字体设置: {plt.rcParams['font.sans-serif'][:3]}")
        print(f"   实际使用的字体: {fm.findfont(fm.FontProperties())}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ 导入matplotlib失败: {e}")
        return False
    except Exception as e:
        print(f"⚠️ 字体配置失败: {e}")
        import traceback
        print(f"   详细错误: {traceback.format_exc()}")
        return False


def get_chinese_font():
    """
    获取中文字体对象
    """
    try:
        import matplotlib.font_manager as fm
        
        # 获取当前设置的字体
        current_font = fm.FontProperties()
        return current_font
    except:
        return None


def test_chinese_display():
    """
    测试中文显示是否正常
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 创建测试图表
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        ax.plot(x, y, label='正弦函数')
        ax.set_title('中文显示测试图表')
        ax.set_xlabel('横轴标签')
        ax.set_ylabel('纵轴标签')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存测试图片
        test_path = Path('./test_chinese_font.png')
        plt.savefig(test_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 中文显示测试完成，图片保存至: {test_path}")
        return True
        
    except Exception as e:
        print(f"❌ 中文显示测试失败: {e}")
        return False


if __name__ == "__main__":
    print("🔧 正在配置中文字体...")
    success = setup_chinese_plot()
    
    if success:
        print("\n🧪 运行中文显示测试...")
        test_chinese_display()
    else:
        print("❌ 字体配置失败")