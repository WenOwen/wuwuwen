#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分步安装脚本 - 解决Windows环境的安装问题
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """运行命令并处理错误"""
    print(f"\n🔄 {description}...")
    print(f"执行命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {description} 成功")
            return True
        else:
            print(f"⚠️ {description} 有警告:")
            print(result.stderr[:500])
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 超时")
        return False
    except Exception as e:
        print(f"❌ {description} 失败: {e}")
        return False

def install_core_packages():
    """安装核心包"""
    print("🚀 开始分步安装AI股市预测系统依赖...")
    
    # 核心数据处理包
    core_packages = [
        ("pandas", "数据处理核心库"),
        ("numpy", "数值计算库"),
        ("matplotlib", "基础绘图库"),
        ("requests", "HTTP请求库"),
        ("tqdm", "进度条库")
    ]
    
    success_count = 0
    
    for package, description in core_packages:
        cmd = f"pip install {package} --trusted-host pypi.org --trusted-host files.pythonhosted.org"
        if run_command(cmd, f"安装{description}({package})"):
            success_count += 1
    
    print(f"\n📊 核心包安装结果: {success_count}/{len(core_packages)} 成功")
    
    # 机器学习包
    if success_count >= 3:  # 至少成功安装3个核心包
        ml_packages = [
            ("scikit-learn", "机器学习库"),
            ("lightgbm", "梯度提升库"),
            ("joblib", "模型持久化库")
        ]
        
        print("\n🤖 安装机器学习包...")
        ml_success = 0
        
        for package, description in ml_packages:
            cmd = f"pip install {package} --trusted-host pypi.org --trusted-host files.pythonhosted.org"
            if run_command(cmd, f"安装{description}({package})"):
                ml_success += 1
        
        print(f"📊 机器学习包安装结果: {ml_success}/{len(ml_packages)} 成功")
    
    # 可选包（失败也不影响基本功能）
    optional_packages = [
        ("streamlit", "Web界面库"),
        ("fastapi", "API框架"),
        ("uvicorn", "Web服务器"),
        ("plotly", "交互式图表库")
    ]
    
    print("\n🌐 安装可选包...")
    optional_success = 0
    
    for package, description in optional_packages:
        cmd = f"pip install {package} --trusted-host pypi.org --trusted-host files.pythonhosted.org"
        if run_command(cmd, f"安装{description}({package})"):
            optional_success += 1
    
    print(f"📊 可选包安装结果: {optional_success}/{len(optional_packages)} 成功")
    
    # 测试导入
    print("\n🧪 测试包导入...")
    test_imports = [
        ("pandas", "import pandas as pd"),
        ("numpy", "import numpy as np"), 
        ("sklearn", "import sklearn"),
        ("matplotlib", "import matplotlib.pyplot as plt")
    ]
    
    import_success = 0
    for package_name, import_cmd in test_imports:
        try:
            exec(import_cmd)
            print(f"✅ {package_name} 导入成功")
            import_success += 1
        except ImportError:
            print(f"❌ {package_name} 导入失败")
    
    print(f"\n🎉 安装完成！导入测试: {import_success}/{len(test_imports)} 成功")
    
    if import_success >= 3:
        print("✅ 基本环境已准备就绪，可以开始使用系统")
        return True
    else:
        print("⚠️ 环境不完整，可能影响部分功能")
        return False

def create_minimal_test():
    """创建最小化测试"""
    test_code = '''
import pandas as pd
import numpy as np

print("🧪 最小化功能测试...")

# 测试pandas
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(f"✅ Pandas测试通过，数据形状: {df.shape}")

# 测试numpy
arr = np.array([1, 2, 3, 4, 5])
print(f"✅ Numpy测试通过，数组大小: {arr.size}")

print("🎉 基础功能测试完成！")
'''
    
    with open('test_minimal.py', 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print("📝 创建了最小化测试文件: test_minimal.py")
    print("运行测试: python test_minimal.py")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AI股市预测系统 - Windows环境依赖安装")
    print("=" * 60)
    
    # 检查conda环境
    if 'CONDA_DEFAULT_ENV' in os.environ:
        print(f"✅ 检测到Conda环境: {os.environ['CONDA_DEFAULT_ENV']}")
    
    # 开始安装
    success = install_core_packages()
    
    # 创建测试文件
    create_minimal_test()
    
    if success:
        print("\n🎯 下一步建议:")
        print("1. 运行测试: python test_minimal.py")
        print("2. 快速启动: python quick_start.py")
        print("3. 开始数据收集: python 2.1获取全数据（东财）.py")
    else:
        print("\n🆘 如果仍有问题，请尝试:")
        print("1. 更新conda: conda update conda")
        print("2. 更新pip: python -m pip install --upgrade pip")
        print("3. 使用国内镜像: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name")