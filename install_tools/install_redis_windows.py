#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Windows Redis安装助手
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path
import time

def download_redis_windows():
    """下载Redis Windows版本"""
    print("📥 下载Redis Windows版本...")
    
    # Redis Windows版本下载链接
    redis_url = "https://github.com/MicrosoftArchive/redis/releases/download/win-3.2.100/Redis-x64-3.2.100.zip"
    redis_zip = "Redis-x64-3.2.100.zip"
    
    try:
        print(f"正在下载: {redis_url}")
        urllib.request.urlretrieve(redis_url, redis_zip)
        print(f"✅ 下载完成: {redis_zip}")
        return redis_zip
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

def extract_redis(zip_file):
    """解压Redis"""
    print("📂 解压Redis...")
    
    extract_dir = "redis-server"
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print(f"✅ 解压完成: {extract_dir}")
        
        # 清理下载文件
        os.remove(zip_file)
        
        return extract_dir
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return None

def start_redis_server(redis_dir):
    """启动Redis服务器"""
    print("🚀 启动Redis服务器...")
    
    redis_exe = os.path.join(redis_dir, "redis-server.exe")
    
    if not os.path.exists(redis_exe):
        print(f"❌ Redis可执行文件不存在: {redis_exe}")
        return None
    
    try:
        # 创建Redis配置文件
        config_content = """
# Redis配置文件
port 6379
bind 127.0.0.1
timeout 0
save 900 1
save 300 10
save 60 10000
rdbcompression yes
dbfilename dump.rdb
dir ./
"""
        
        config_file = os.path.join(redis_dir, "redis.conf")
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        # 启动Redis服务器
        print("启动Redis服务器...")
        process = subprocess.Popen([redis_exe, config_file])
        
        # 等待Redis启动
        time.sleep(3)
        
        print("✅ Redis服务器已启动")
        print(f"📍 进程ID: {process.pid}")
        print("🔗 连接地址: localhost:6379")
        
        return process
        
    except Exception as e:
        print(f"❌ 启动Redis失败: {e}")
        return None

def test_redis_connection():
    """测试Redis连接"""
    print("🧪 测试Redis连接...")
    
    try:
        import redis
        
        # 连接Redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # 测试连接
        r.ping()
        
        # 简单测试
        r.set('test_key', 'test_value')
        value = r.get('test_key')
        
        if value == 'test_value':
            print("✅ Redis连接测试成功")
            return True
        else:
            print("❌ Redis数据读写测试失败")
            return False
            
    except ImportError:
        print("❌ redis-py包未安装")
        print("请运行: pip install redis")
        return False
    except Exception as e:
        print(f"❌ Redis连接失败: {e}")
        return False

def install_redis_py():
    """安装redis-py包"""
    print("📦 安装redis-py包...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'redis'], check=True)
        print("✅ redis-py安装成功")
        return True
    except Exception as e:
        print(f"❌ redis-py安装失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 Redis Windows安装助手")
    print("=" * 40)
    
    # 1. 安装redis-py包
    if not install_redis_py():
        return
    
    # 2. 检查是否已有Redis服务器
    if test_redis_connection():
        print("✅ Redis已经在运行!")
        return
    
    # 3. 下载Redis
    zip_file = download_redis_windows()
    if not zip_file:
        print("❌ 下载Redis失败")
        return
    
    # 4. 解压Redis
    redis_dir = extract_redis(zip_file)
    if not redis_dir:
        print("❌ 解压Redis失败")
        return
    
    # 5. 启动Redis服务器
    process = start_redis_server(redis_dir)
    if not process:
        print("❌ 启动Redis失败")
        return
    
    # 6. 测试连接
    if test_redis_connection():
        print("\n🎉 Redis安装和启动成功!")
        print("\n📝 下一步:")
        print("1. 保持这个窗口开启(Redis服务器在运行)")
        print("2. 在新窗口运行: python start.py")
        print("\n⚠️ 注意: 关闭此窗口会停止Redis服务")
        
        # 保持Redis运行
        try:
            print("\n按Ctrl+C停止Redis服务...")
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 停止Redis服务...")
            process.terminate()
            process.wait()
            print("✅ Redis服务已停止")
    else:
        print("❌ Redis测试失败")
        if process:
            process.terminate()

if __name__ == "__main__":
    main()