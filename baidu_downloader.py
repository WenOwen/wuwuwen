#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百度网盘下载助手 - 自动检测和安装百度网盘下载的部署包
"""

import os
import sys
import subprocess
import tarfile
import zipfile
from pathlib import Path
import platform
import argparse

class BaiduDownloadHelper:
    """百度网盘下载助手"""
    
    def __init__(self):
        self.current_dir = Path(".")
        self.system = platform.system().lower()
        
    def find_deployment_packages(self) -> list:
        """查找部署包"""
        packages = []
        
        # 查找压缩包
        for pattern in ["wuwuquant_*.tar.gz", "wuwuquant_*.zip"]:
            packages.extend(self.current_dir.glob(pattern))
        
        # 按修改时间排序，最新的在前
        packages.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return packages
    
    def extract_package(self, package_path: Path) -> Path:
        """解压部署包"""
        print(f"📦 解压部署包: {package_path.name}")
        
        # 确定解压目录名
        if package_path.suffix == ".gz":
            extract_dir = package_path.name.replace(".tar.gz", "")
        else:
            extract_dir = package_path.stem
        
        extract_path = Path(extract_dir)
        
        # 如果目录已存在，询问是否覆盖
        if extract_path.exists():
            response = input(f"目录 {extract_dir} 已存在，是否覆盖? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("已取消解压")
                return None
            
            import shutil
            shutil.rmtree(extract_path)
        
        # 解压文件
        try:
            if package_path.suffix == ".gz":
                with tarfile.open(package_path, "r:gz") as tar:
                    tar.extractall()
            else:
                with zipfile.ZipFile(package_path, 'r') as zip_ref:
                    zip_ref.extractall()
            
            print(f"✅ 解压完成: {extract_dir}")
            return extract_path
            
        except Exception as e:
            print(f"❌ 解压失败: {e}")
            return None
    
    def check_system_requirements(self) -> bool:
        """检查系统要求"""
        print("🔍 检查系统要求...")
        
        # 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            print(f"❌ Python版本过低: {python_version.major}.{python_version.minor}")
            print("   请安装Python 3.8或更高版本")
            return False
        
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # 检查pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         capture_output=True, check=True)
            print("✅ pip已安装")
        except:
            print("❌ pip未安装")
            return False
        
        # 检查可用空间
        import shutil
        free_space = shutil.disk_usage(".").free / (1024**3)  # GB
        if free_space < 5:
            print(f"⚠️  磁盘空间不足: {free_space:.1f}GB 可用")
            print("   建议至少有5GB可用空间")
        else:
            print(f"✅ 磁盘空间: {free_space:.1f}GB 可用")
        
        return True
    
    def run_installation(self, extract_path: Path) -> bool:
        """运行安装过程"""
        print("🚀 开始安装...")
        
        # 进入解压目录
        original_dir = Path.cwd()
        os.chdir(extract_path)
        
        try:
            # 根据系统选择安装脚本
            if self.system == "windows":
                script_path = Path("install.bat")
                if script_path.exists():
                    print("🔧 运行Windows安装脚本...")
                    result = subprocess.run([str(script_path)], shell=True)
                else:
                    print("⚠️  未找到Windows安装脚本，使用手动安装")
                    return self._manual_install()
            else:
                script_path = Path("install.sh")
                if script_path.exists():
                    print("🔧 运行Linux/Mac安装脚本...")
                    # 设置执行权限
                    os.chmod(script_path, 0o755)
                    result = subprocess.run(["bash", str(script_path)])
                else:
                    print("⚠️  未找到安装脚本，使用手动安装")
                    return self._manual_install()
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ 安装脚本执行失败: {e}")
            print("🔧 尝试手动安装...")
            return self._manual_install()
        
        finally:
            os.chdir(original_dir)
    
    def _manual_install(self) -> bool:
        """手动安装"""
        print("🔧 开始手动安装...")
        
        try:
            # 1. 解压训练数据
            data_archive = Path("training_data.tar.gz")
            if data_archive.exists():
                print("📊 解压训练数据...")
                with tarfile.open(data_archive, "r:gz") as tar:
                    tar.extractall()
                print("✅ 训练数据解压完成")
            else:
                print("⚠️  未找到训练数据包")
            
            # 2. 检查数据完整性
            data_dir = Path("datas_em")
            if data_dir.exists():
                csv_count = len(list(data_dir.glob("*.csv")))
                print(f"📊 检测到 {csv_count} 个数据文件")
                if csv_count < 100:
                    print("⚠️  数据文件较少，可能不完整")
            
            # 3. 安装Python依赖
            requirements_file = Path("requirements.txt")
            if requirements_file.exists():
                print("📦 安装Python依赖...")
                
                # 使用国内镜像加速
                pip_cmd = [
                    sys.executable, "-m", "pip", "install", 
                    "-r", str(requirements_file),
                    "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
                ]
                
                result = subprocess.run(pip_cmd)
                if result.returncode != 0:
                    print("⚠️  使用默认源重试...")
                    pip_cmd = [
                        sys.executable, "-m", "pip", "install", 
                        "-r", str(requirements_file)
                    ]
                    result = subprocess.run(pip_cmd)
                
                if result.returncode == 0:
                    print("✅ 依赖安装完成")
                else:
                    print("❌ 依赖安装失败")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ 手动安装失败: {e}")
            return False
    
    def start_service(self, extract_path: Path):
        """启动服务"""
        print("🚀 准备启动服务...")
        
        os.chdir(extract_path)
        
        # 查找启动脚本
        startup_scripts = ["start.py", "quick_start.py", "streamlit_app.py"]
        
        for script in startup_scripts:
            script_path = Path(script)
            if script_path.exists():
                print(f"📋 找到启动脚本: {script}")
                print("\n" + "="*50)
                print("🎉 安装完成！")
                print("="*50)
                print("📝 启动方法：")
                print(f"   cd {extract_path}")
                print(f"   python {script}")
                print()
                print("📱 访问地址：")
                print("   Web界面: http://localhost:8501")
                print("   API文档: http://localhost:8000/docs")
                print()
                
                # 询问是否立即启动
                response = input("是否立即启动服务? [Y/n]: ")
                if response.lower() in ['', 'y', 'yes']:
                    print("🚀 启动服务...")
                    subprocess.run([sys.executable, script])
                
                return
        
        print("⚠️  未找到启动脚本")
        print("请手动运行: python start.py")
    
    def show_package_list(self, packages: list):
        """显示包列表"""
        print("📦 找到以下部署包：")
        print("-" * 40)
        
        for i, package in enumerate(packages):
            file_size = package.stat().st_size / (1024 * 1024)  # MB
            mod_time = package.stat().st_mtime
            import datetime
            mod_date = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
            
            print(f"{i+1}. {package.name}")
            print(f"   大小: {file_size:.1f} MB")
            print(f"   时间: {mod_date}")
            print()
    
    def process_deployment(self, package_path: Path = None):
        """处理部署"""
        # 检查系统要求
        if not self.check_system_requirements():
            return False
        
        # 查找或使用指定的包
        if package_path:
            packages = [package_path]
        else:
            packages = self.find_deployment_packages()
        
        if not packages:
            print("❌ 未找到部署包")
            print("📝 请确保已从百度网盘下载 wuwuquant_*.tar.gz 或 wuwuquant_*.zip 文件")
            return False
        
        # 选择包
        if len(packages) == 1:
            selected_package = packages[0]
            print(f"📦 找到部署包: {selected_package.name}")
        else:
            self.show_package_list(packages)
            
            try:
                choice = input("请选择部署包编号 [1]: ").strip()
                if not choice:
                    choice = "1"
                
                index = int(choice) - 1
                if 0 <= index < len(packages):
                    selected_package = packages[index]
                else:
                    print("❌ 选择无效")
                    return False
            except ValueError:
                print("❌ 输入无效")
                return False
        
        # 解压包
        extract_path = self.extract_package(selected_package)
        if not extract_path:
            return False
        
        # 运行安装
        if self.run_installation(extract_path):
            self.start_service(extract_path)
            return True
        else:
            print("❌ 安装失败")
            return False

def main():
    parser = argparse.ArgumentParser(description="百度网盘下载助手")
    parser.add_argument("--package", help="指定部署包路径")
    parser.add_argument("--check-only", action="store_true", help="只检查系统要求")
    
    args = parser.parse_args()
    
    helper = BaiduDownloadHelper()
    
    print("📁 AI股市预测系统 - 百度网盘下载助手")
    print("=" * 50)
    
    if args.check_only:
        helper.check_system_requirements()
        return
    
    package_path = Path(args.package) if args.package else None
    
    if helper.process_deployment(package_path):
        print("\n🎊 部署成功完成！")
    else:
        print("\n❌ 部署失败")
        print("💡 请检查错误信息并重试")

if __name__ == "__main__":
    main()