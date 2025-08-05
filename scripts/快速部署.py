#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速部署脚本 - Docker跨网络部署专用
适用于原电脑和服务器不在同一局域网的场景
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

class QuickDeploy:
    """快速部署助手"""
    
    def __init__(self):
        self.project_dir = Path.cwd()
        
    def check_docker(self) -> bool:
        """检查Docker环境"""
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Docker已安装: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        
        print("❌ Docker未安装")
        print("📝 请先安装Docker:")
        print("   Ubuntu: curl -fsSL https://get.docker.com | sh")
        print("   CentOS: yum install -y docker")
        return False
    
    def check_docker_compose(self) -> bool:
        """检查Docker Compose"""
        for cmd in [["docker-compose", "--version"], ["docker", "compose", "version"]]:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ Docker Compose已安装")
                    return True
            except FileNotFoundError:
                continue
        
        print("❌ Docker Compose未安装")
        return False
    
    def setup_env_file(self, data_source_type: str, data_url: str):
        """设置环境变量文件"""
        env_content = f"""# Docker跨网络部署配置
DATA_SOURCE_TYPE={data_source_type}
DATA_PACKAGE_URL={data_url}
AUTO_DOWNLOAD_DATA=true

# 可选配置
# REDIS_PASSWORD=your_password
# API_PORT=8000
# WEB_PORT=8501
"""
        
        with open(".env", "w") as f:
            f.write(env_content)
        
        print(f"✅ 环境配置文件已创建")
        print(f"   数据源类型: {data_source_type}")
        print(f"   数据URL: {data_url}")
    
    def deploy(self, data_source_type: str, data_url: str):
        """执行部署"""
        print("🚀 开始Docker跨网络部署...")
        print("=" * 50)
        
        # 检查环境
        if not self.check_docker():
            return False
        
        if not self.check_docker_compose():
            return False
        
        # 设置环境变量
        self.setup_env_file(data_source_type, data_url)
        
        # 检查配置文件
        compose_file = "docker-compose.distributed.yml"
        if not Path(compose_file).exists():
            print(f"❌ 未找到配置文件: {compose_file}")
            return False
        
        # 启动服务
        print("🐳 启动Docker服务...")
        cmd = ["docker-compose", "-f", compose_file, "up", "-d"]
        
        # 尝试新版本命令
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("尝试新版本Docker Compose...")
            cmd = ["docker", "compose", "-f", compose_file, "up", "-d"]
            result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("✅ Docker服务启动成功")
            
            # 等待服务启动
            print("⏳ 等待服务启动...")
            import time
            time.sleep(10)
            
            # 显示服务状态
            print("📊 服务状态:")
            subprocess.run(["docker-compose", "-f", compose_file, "ps"])
            
            print("\n🎉 部署完成！")
            print("📱 访问地址:")
            print("   Web界面: http://localhost:8501")
            print("   API文档: http://localhost:8000/docs")
            print()
            print("🔧 管理命令:")
            print("   查看日志: docker-compose -f docker-compose.distributed.yml logs -f")
            print("   重启服务: docker-compose -f docker-compose.distributed.yml restart")
            print("   停止服务: docker-compose -f docker-compose.distributed.yml down")
            
            return True
        else:
            print("❌ Docker服务启动失败")
            return False
    
    def show_logs(self):
        """显示服务日志"""
        compose_file = "docker-compose.distributed.yml"
        subprocess.run(["docker-compose", "-f", compose_file, "logs", "-f"])
    
    def stop_services(self):
        """停止服务"""
        compose_file = "docker-compose.distributed.yml"
        result = subprocess.run(["docker-compose", "-f", compose_file, "down"])
        if result.returncode == 0:
            print("✅ 服务已停止")
        else:
            print("❌ 停止服务失败")

def get_data_source_info():
    """获取数据源信息"""
    print("📊 配置数据源:")
    print("1. 百度网盘")
    print("2. 阿里云OSS")
    print("3. 手动指定URL")
    
    while True:
        choice = input("请选择数据源类型 [1-3]: ").strip()
        
        if choice == "1":
            data_url = input("请输入百度网盘分享链接: ").strip()
            if data_url:
                return "baidu", data_url
        elif choice == "2":
            bucket = input("请输入OSS bucket名称: ").strip()
            if bucket:
                data_url = f"oss://{bucket}/stock-data/"
                return "aliyun", data_url
        elif choice == "3":
            data_url = input("请输入数据包URL: ").strip()
            if data_url:
                return "custom", data_url
        
        print("❌ 输入无效，请重试")

def main():
    parser = argparse.ArgumentParser(description="Docker跨网络快速部署")
    parser.add_argument("--data-type", choices=["baidu", "aliyun", "custom"], help="数据源类型")
    parser.add_argument("--data-url", help="数据包URL")
    parser.add_argument("--logs", action="store_true", help="查看服务日志")
    parser.add_argument("--stop", action="store_true", help="停止服务")
    
    args = parser.parse_args()
    
    deployer = QuickDeploy()
    
    if args.logs:
        deployer.show_logs()
        return
    
    if args.stop:
        deployer.stop_services()
        return
    
    print("🐳 AI股市预测系统 - Docker跨网络部署")
    print("=" * 50)
    
    # 获取数据源信息
    if args.data_type and args.data_url:
        data_source_type = args.data_type
        data_url = args.data_url
    else:
        data_source_type, data_url = get_data_source_info()
    
    # 执行部署
    if deployer.deploy(data_source_type, data_url):
        print("\n🎊 部署成功！")
    else:
        print("\n❌ 部署失败")
        print("💡 请检查错误信息并重试")

if __name__ == "__main__":
    main()