#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用国内镜像安装依赖 - 解决SSL问题
"""

import subprocess
import sys

def install_with_mirror():
    """使用国内镜像安装"""
    print("🚀 使用国内镜像安装依赖包...")
    
    # 国内镜像源
    mirrors = [
        "https://pypi.tuna.tsinghua.edu.cn/simple",
        "https://mirrors.aliyun.com/pypi/simple", 
        "https://pypi.douban.com/simple"
    ]
    
    # 需要安装的包（按重要性排序）
    packages = [
        ("scikit-learn", "机器学习核心库"),
        ("matplotlib", "绘图库"),
        ("xgboost", "梯度提升库"),
        ("joblib", "模型持久化"),
        ("tqdm", "进度条"),
        ("streamlit", "Web界面"),
        ("fastapi", "API框架"),
        ("uvicorn", "Web服务器"),
        ("plotly", "交互图表")
    ]
    
    success_count = 0
    
    for package, description in packages:
        print(f"\n🔄 安装 {description} ({package})...")
        
        # 尝试不同的镜像源
        installed = False
        for mirror in mirrors:
            try:
                cmd = f"pip install {package} -i {mirror} --timeout 60"
                print(f"  尝试镜像: {mirror}")
                
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    print(f"✅ {package} 安装成功")
                    success_count += 1
                    installed = True
                    break
                else:
                    print(f"  ❌ 该镜像失败，尝试下一个...")
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏰ 超时，尝试下一个镜像...")
                continue
            except Exception as e:
                print(f"  ❌ 异常: {e}")
                continue
        
        if not installed:
            print(f"⚠️ {package} 所有镜像都失败，跳过")
    
    print(f"\n📊 安装结果: {success_count}/{len(packages)} 个包安装成功")
    
    # 测试导入
    print("\n🧪 测试包导入...")
    test_packages = [
        ("pandas", "import pandas as pd"),
        ("numpy", "import numpy as np"),
        ("sklearn", "import sklearn"),
        ("matplotlib", "import matplotlib.pyplot as plt"),
        ("xgboost", "import xgboost as xgb")
    ]
    
    available_packages = []
    for package_name, import_cmd in test_packages:
        try:
            exec(import_cmd)
            print(f"✅ {package_name} 可用")
            available_packages.append(package_name)
        except ImportError:
            print(f"❌ {package_name} 不可用")
    
    return available_packages

if __name__ == "__main__":
    available = install_with_mirror()
    
    print(f"\n🎉 当前可用的包: {', '.join(available)}")
    
    if len(available) >= 2:
        print("\n✅ 基础环境已就绪！可以开始使用系统")
        print("\n📝 下一步:")
        print("1. 运行: python quick_start_minimal.py")
        print("2. 测试数据收集: python 2.1获取全数据（东财）.py")
    else:
        print("\n⚠️ 环境仍不完整，建议联系网络管理员解决SSL问题")