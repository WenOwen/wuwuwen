#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百度网盘上传助手 - 创建适合百度网盘的部署包
"""

import os
import tarfile
import zipfile
import shutil
from pathlib import Path
import json
from datetime import datetime
import argparse

class BaiduUploadHelper:
    """百度网盘数据包上传助手 - 只打包数据，不打包代码"""
    
    def __init__(self, data_dir: str = "datas_em"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
    def create_data_package(self, format_type: str = "tar.gz") -> str:
        """创建纯数据包（不包含代码）"""
        print("📦 创建股票数据包...")
        
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"stock_data_{timestamp}.{format_type}"
        
        # 直接打包数据文件
        csv_count = self._create_data_archive(package_name, format_type)
        
        # 显示结果
        self._show_data_package_info(package_name, csv_count)
        
        return package_name
    
    def _create_data_archive(self, package_name: str, format_type: str) -> int:
        """创建数据压缩包"""
        print("📊 压缩股票数据文件...")
        
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到CSV文件")
        
        csv_count = 0
        
        if format_type == "tar.gz":
            with tarfile.open(package_name, "w:gz") as tar:
                for csv_file in csv_files:
                    # 保持目录结构 datas_em/文件名
                    tar.add(csv_file, arcname=f"datas_em/{csv_file.name}")
                    csv_count += 1
                    if csv_count % 500 == 0:
                        print(f"已压缩 {csv_count} 个文件...")
        
        elif format_type == "zip":
            with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for csv_file in csv_files:
                    zipf.write(csv_file, f"datas_em/{csv_file.name}")
                    csv_count += 1
                    if csv_count % 500 == 0:
                        print(f"已压缩 {csv_count} 个文件...")
        
        # 计算压缩比
        original_size = sum(f.stat().st_size for f in csv_files)
        compressed_size = Path(package_name).stat().st_size
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        print(f"✅ 数据压缩完成:")
        print(f"   - 文件数量: {csv_count}")
        print(f"   - 原始大小: {original_size // 1024 // 1024} MB")
        print(f"   - 压缩后: {compressed_size // 1024 // 1024} MB")
        print(f"   - 压缩率: {compression_ratio:.1f}%")
        
        return csv_count
    
    def _show_data_package_info(self, package_path: str, csv_count: int):
        """显示数据包信息"""
        package_file = Path(package_path)
        file_size_mb = package_file.stat().st_size / 1024 / 1024
        
        print("\n" + "="*50)
        print("🎉 股票数据包创建完成！")
        print("="*50)
        print(f"📦 文件名: {package_path}")
        print(f"📊 文件大小: {file_size_mb:.1f} MB")
        print(f"📈 数据文件: {csv_count} 个CSV文件")
        print()
        print("📝 Docker部署使用方法：")
        print("1. 📤 上传到云存储")
        print("   - 百度网盘：上传并创建分享链接")
        print("   - 阿里云OSS：ossutil cp package.tar.gz oss://bucket/")
        print()
        print("2. 🐳 服务器Docker部署")
        print("   - git clone your-repo && cd wuwuquant")
        print("   - echo 'DATA_PACKAGE_URL=分享链接' > .env")
        print("   - python 快速部署.py")
        print()
        print("3. ✅ 自动完成")
        print("   - Docker容器自动下载数据包")
        print("   - 自动解压到 datas_em/ 目录")
        print("   - 启动Web界面: http://服务器IP:8501")
        print()
        print("💡 纯数据包，代码通过git获取，完美分离！")

def main():
    parser = argparse.ArgumentParser(description="股票数据包上传助手")
    parser.add_argument("--data-dir", default="datas_em", help="数据目录")
    parser.add_argument("--format", choices=["tar.gz", "zip"], default="tar.gz", help="压缩格式")
    
    args = parser.parse_args()
    
    print("📊 AI股市预测系统 - 数据包创建工具")
    print("=" * 50)
    print("🎯 功能：创建纯数据包（不包含代码）")
    print("🐳 用途：Docker分离式部署")
    print()
    
    try:
        helper = BaiduUploadHelper(args.data_dir)
        package_path = helper.create_data_package(args.format)
        
        print(f"\n🎊 数据包创建成功: {package_path}")
        print("📤 请上传到百度网盘或云存储！")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 请确保在项目根目录运行，且datas_em目录存在")
    except Exception as e:
        print(f"❌ 创建数据包失败: {e}")

if __name__ == "__main__":
    main()