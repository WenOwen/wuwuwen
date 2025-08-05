#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能数据下载器 - 支持从多种数据源下载训练数据
"""

import os
import requests
import zipfile
import tarfile
import boto3
from pathlib import Path
import logging
import time
from typing import List, Dict, Optional
import json
import hashlib
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDownloader:
    """智能数据下载器 - 支持跨网络数据获取"""
    
    def __init__(self, data_dir: str = "datas_em"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.config_file = Path("data_sources.json")
        
        # 从环境变量获取配置
        self.data_source_type = os.getenv("DATA_SOURCE_TYPE", "auto")
        self.data_package_url = os.getenv("DATA_PACKAGE_URL")
        
    def load_data_sources(self) -> Dict:
        """加载数据源配置"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 默认配置
        return {
            "sources": [
                {
                    "name": "http_server",
                    "type": "http",
                    "url": os.getenv("DATA_SOURCE_URL"),
                    "enabled": True
                },
                {
                    "name": "s3_bucket", 
                    "type": "s3",
                    "bucket": "your-bucket-name",
                    "prefix": "stock-data/",
                    "enabled": False
                },
                {
                    "name": "local_backup",
                    "type": "local",
                    "path": "/backup/datas_em",
                    "enabled": False
                }
            ],
            "files": {
                "required": ["*.csv"],
                "optional": ["models/*.pkl", "config/*.yaml"],
                "exclude": ["logs/*", "*.tmp"]
            }
        }
    
    def check_data_integrity(self) -> bool:
        """检查数据完整性"""
        csv_files = list(self.data_dir.glob("*.csv"))
        logger.info(f"发现 {len(csv_files)} 个CSV文件")
        
        if len(csv_files) < 100:  # 假设至少需要100个股票数据
            logger.warning("数据文件数量不足，需要下载更多数据")
            return False
            
        # 检查文件大小
        total_size = sum(f.stat().st_size for f in csv_files)
        logger.info(f"数据总大小: {total_size / 1024 / 1024:.2f} MB")
        
        if total_size < 10 * 1024 * 1024:  # 小于10MB认为数据不完整
            logger.warning("数据大小异常，可能不完整")
            return False
            
        return True
    
    def download_from_http(self, source: Dict) -> bool:
        """从HTTP服务器下载数据"""
        if not source.get("url"):
            logger.warning("HTTP数据源URL未配置")
            return False
            
        url = source["url"]
        logger.info(f"从HTTP服务器下载数据: {url}")
        
        try:
            # 支持多种文件格式
            for ext in ['.tar.gz', '.zip', '.tar']:
                download_url = f"{url}/stock_data{ext}"
                
                response = requests.head(download_url, timeout=10)
                if response.status_code == 200:
                    logger.info(f"找到数据包: {download_url}")
                    return self._download_and_extract(download_url, ext)
            
            # 尝试直接下载文件列表
            response = requests.get(f"{url}/filelist.json", timeout=10)
            if response.status_code == 200:
                file_list = response.json()
                return self._download_files_from_list(url, file_list)
                
        except Exception as e:
            logger.error(f"HTTP下载失败: {e}")
            
        return False
    
    def download_from_s3(self, source: Dict) -> bool:
        """从S3下载数据"""
        try:
            s3_client = boto3.client('s3')
            bucket = source["bucket"]
            prefix = source.get("prefix", "")
            
            logger.info(f"从S3下载数据: s3://{bucket}/{prefix}")
            
            # 列出对象
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                logger.warning("S3桶中没有找到数据文件")
                return False
            
            # 下载文件
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.csv'):
                    local_path = self.data_dir / Path(key).name
                    s3_client.download_file(bucket, key, str(local_path))
                    logger.info(f"下载文件: {key}")
            
            return True
            
        except Exception as e:
            logger.error(f"S3下载失败: {e}")
            return False
    
    def copy_from_local(self, source: Dict) -> bool:
        """从本地路径复制数据"""
        src_path = Path(source["path"])
        if not src_path.exists():
            logger.warning(f"本地数据路径不存在: {src_path}")
            return False
        
        logger.info(f"从本地复制数据: {src_path}")
        
        try:
            import shutil
            if src_path.is_dir():
                shutil.copytree(src_path, self.data_dir, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, self.data_dir)
            return True
        except Exception as e:
            logger.error(f"本地复制失败: {e}")
            return False
    
    def _download_and_extract(self, url: str, file_ext: str) -> bool:
        """下载并解压文件"""
        temp_file = f"temp_data{file_ext}"
        
        try:
            # 下载文件
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 解压文件
            if file_ext == '.zip':
                with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
            elif file_ext in ['.tar.gz', '.tar']:
                with tarfile.open(temp_file, 'r:*') as tar_ref:
                    tar_ref.extractall(self.data_dir)
            
            os.remove(temp_file)
            logger.info("数据下载并解压完成")
            return True
            
        except Exception as e:
            logger.error(f"下载解压失败: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False
    
    def _download_files_from_list(self, base_url: str, file_list: List[str]) -> bool:
        """根据文件列表下载"""
        success_count = 0
        
        for filename in file_list:
            if not filename.endswith('.csv'):
                continue
                
            try:
                file_url = f"{base_url}/{filename}"
                response = requests.get(file_url, timeout=30)
                response.raise_for_status()
                
                local_path = self.data_dir / filename
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                success_count += 1
                logger.info(f"下载文件: {filename}")
                
            except Exception as e:
                logger.warning(f"下载文件失败 {filename}: {e}")
        
        logger.info(f"成功下载 {success_count} 个文件")
        return success_count > 0
    
    def download_package_from_url(self, url: str) -> bool:
        """从URL下载部署包并解压"""
        logger.info(f"从URL下载部署包: {url}")
        
        # 确定下载文件名
        temp_file = "temp_deployment_package.tar.gz"
        
        try:
            # 下载文件
            if url.startswith("http"):
                # 直接HTTP下载
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                logger.error(f"不支持的URL类型: {url}")
                return False
            
            # 解压包
            return self._extract_deployment_package(temp_file)
            
        except Exception as e:
            logger.error(f"下载部署包失败: {e}")
            return False
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def _extract_deployment_package(self, package_file: str) -> bool:
        """解压部署包"""
        logger.info("解压部署包...")
        
        try:
            import tarfile
            with tarfile.open(package_file, 'r:gz') as tar:
                # 查找数据文件
                for member in tar.getmembers():
                    if member.name.endswith("training_data.tar.gz"):
                        # 提取并解压数据文件
                        tar.extract(member)
                        
                        # 解压训练数据
                        with tarfile.open(member.name, 'r:gz') as data_tar:
                            data_tar.extractall()
                        
                        # 清理临时文件
                        os.remove(member.name)
                        logger.info("训练数据解压完成")
                        return True
            
            logger.error("未在部署包中找到训练数据")
            return False
            
        except Exception as e:
            logger.error(f"解压部署包失败: {e}")
            return False

    def download_data(self) -> bool:
        """智能下载数据"""
        # 首先检查数据完整性
        if self.check_data_integrity():
            logger.info("数据已存在且完整，跳过下载")
            return True
        
        # 如果配置了部署包URL，优先使用
        if self.data_package_url:
            logger.info("使用配置的部署包URL下载数据")
            if self.download_package_from_url(self.data_package_url):
                return True
        
        logger.info("开始从数据源下载训练数据...")
        config = self.load_data_sources()
        
        for source in config["sources"]:
            if not source.get("enabled", False):
                continue
                
            logger.info(f"尝试数据源: {source['name']}")
            
            success = False
            if source["type"] == "http":
                success = self.download_from_http(source)
            elif source["type"] == "s3":
                success = self.download_from_s3(source)
            elif source["type"] == "local":
                success = self.copy_from_local(source)
            
            if success and self.check_data_integrity():
                logger.info(f"数据下载成功，来源: {source['name']}")
                return True
        
        logger.error("所有数据源都下载失败")
        return False

def main():
    """主函数"""
    downloader = DataDownloader()
    
    # 检查环境变量
    auto_download = os.getenv("AUTO_DOWNLOAD_DATA", "true").lower() == "true"
    
    if auto_download:
        success = downloader.download_data()
        if not success:
            logger.error("数据下载失败，程序退出")
            exit(1)
    else:
        logger.info("自动下载已禁用")
        if not downloader.check_data_integrity():
            logger.warning("数据不完整，建议手动下载")

if __name__ == "__main__":
    main()