#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TA-Lib安装助手 - Windows环境专用
"""

import subprocess
import sys
import platform
import requests
import os

def get_python_info():
    """获取Python版本信息"""
    version = sys.version_info
    arch = platform.architecture()[0]
    
    # 确定Python版本
    if version.major == 3 and version.minor == 9:
        py_version = "cp39"
    elif version.major == 3 and version.minor == 10:
        py_version = "cp310"
    elif version.major == 3 and version.minor == 11:
        py_version = "cp311"
    elif version.major == 3 and version.minor == 8:
        py_version = "cp38"
    else:
        py_version = f"cp{version.major}{version.minor}"
    
    # 确定架构
    if arch == "64bit":
        arch_tag = "win_amd64"
    else:
        arch_tag = "win32"
    
    return py_version, arch_tag

def download_talib_wheel():
    """下载TA-Lib预编译文件"""
    py_version, arch_tag = get_python_info()
    
    print(f"🔍 检测到Python版本: {py_version}, 架构: {arch_tag}")
    
    # TA-Lib预编译文件下载链接
    base_url = "https://download.lfd.uci.edu/pythonlibs/archived/"
    filename = f"TA_Lib-0.4.24-{py_version}-{py_version}-{arch_tag}.whl"
    download_url = base_url + filename
    
    print(f"📥 下载TA-Lib: {filename}")
    print(f"🔗 下载地址: {download_url}")
    
    try:
        response = requests.get(download_url, timeout=60)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ 下载完成: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

def install_talib_wheel(filename):
    """安装TA-Lib wheel文件"""
    try:
        print(f"🔧 安装TA-Lib...")
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', filename], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ TA-Lib安装成功!")
            return True
        else:
            print(f"❌ TA-Lib安装失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 安装异常: {e}")
        return False

def test_talib():
    """测试TA-Lib"""
    try:
        import talib
        import numpy as np
        
        # 简单测试
        close = np.random.randn(100)
        sma = talib.SMA(close, timeperiod=5)
        
        print("✅ TA-Lib测试成功!")
        print(f"📊 TA-Lib版本: {talib.__version__}")
        return True
        
    except ImportError:
        print("❌ TA-Lib导入失败")
        return False
    except Exception as e:
        print(f"❌ TA-Lib测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 TA-Lib安装助手")
    print("=" * 40)
    
    # 检查是否已安装
    try:
        import talib
        print("✅ TA-Lib已经安装!")
        test_talib()
        return
    except ImportError:
        print("📦 TA-Lib未安装，开始安装...")
    
    # 方法1: 尝试pip直接安装
    print("\n🔄 方法1: 尝试pip直接安装...")
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'TA-Lib'], 
                               capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("✅ pip直接安装成功!")
            if test_talib():
                return
    except:
        print("❌ pip直接安装失败")
    
    # 方法2: 下载预编译文件
    print("\n🔄 方法2: 下载预编译文件...")
    filename = download_talib_wheel()
    
    if filename and os.path.exists(filename):
        if install_talib_wheel(filename):
            if test_talib():
                # 清理下载文件
                os.remove(filename)
                print("🧹 清理下载文件")
                return
    
    # 方法3: 手动下载指导
    print("\n🔄 方法3: 手动下载安装")
    py_version, arch_tag = get_python_info()
    
    print("请手动下载TA-Lib:")
    print("1. 访问: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
    print(f"2. 下载对应版本: TA_Lib-0.4.24-{py_version}-{py_version}-{arch_tag}.whl")
    print("3. 运行安装: pip install 下载的文件名.whl")
    
    print("\n🔄 方法4: 使用conda安装")
    print("如果您使用conda环境:")
    print("conda install -c conda-forge ta-lib")

if __name__ == "__main__":
    main()