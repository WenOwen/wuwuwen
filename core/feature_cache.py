# -*- coding: utf-8 -*-
"""
特征工程缓存系统
功能：为大规模股票特征工程提供高效缓存机制
"""

import os
import pickle
import hashlib
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class FeatureCache:
    """
    特征工程缓存系统
    - 基于文件的缓存存储
    - 支持版本控制和数据完整性检查
    - 自动过期机制
    - 批量缓存优化
    """
    
    def __init__(self, cache_dir: str = "cache/features", cache_days: int = 7):
        """
        初始化缓存系统
        
        Args:
            cache_dir: 缓存目录
            cache_days: 缓存有效期（天）
        """
        self.cache_dir = cache_dir
        self.cache_days = cache_days
        self.version = "v1.0"  # 特征版本，变更时会清理旧缓存
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 缓存元数据文件
        self.metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        self.metadata = self._load_metadata()
        
        # 统计信息
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _load_metadata(self) -> Dict:
        """加载缓存元数据"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载缓存元数据失败: {e}")
        
        return {
            "version": self.version,
            "created_time": datetime.now().isoformat(),
            "cache_entries": {}
        }
    
    def _save_metadata(self):
        """保存缓存元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存缓存元数据失败: {e}")
    
    def _generate_cache_key(self, stock_code: str, data_hash: str) -> str:
        """
        生成缓存键
        
        Args:
            stock_code: 股票代码
            data_hash: 原始数据哈希值
            
        Returns:
            缓存键
        """
        # 组合版本、股票代码和数据哈希
        key_string = f"{self.version}_{stock_code}_{data_hash}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """
        计算DataFrame的哈希值
        
        Args:
            df: 输入数据
            
        Returns:
            数据哈希值
        """
        # 使用DataFrame的关键信息生成哈希
        def convert_to_serializable(obj):
            """转换对象为可序列化格式"""
            if pd.isna(obj):
                return None
            elif hasattr(obj, 'isoformat'):  # datetime/Timestamp对象
                return obj.isoformat()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj
        
        key_info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'first_date': convert_to_serializable(df['交易日期'].iloc[0]) if '交易日期' in df.columns and len(df) > 0 else None,
            'last_date': convert_to_serializable(df['交易日期'].iloc[-1]) if '交易日期' in df.columns and len(df) > 0 else None,
            'first_close': convert_to_serializable(df['收盘价'].iloc[0]) if '收盘价' in df.columns and len(df) > 0 else None,
            'last_close': convert_to_serializable(df['收盘价'].iloc[-1]) if '收盘价' in df.columns and len(df) > 0 else None,
        }
        
        key_string = json.dumps(key_info, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self.metadata["cache_entries"]:
            return False
        
        cache_info = self.metadata["cache_entries"][cache_key]
        
        # 检查版本
        if cache_info.get("version") != self.version:
            return False
        
        # 检查过期时间
        cached_time = datetime.fromisoformat(cache_info["cached_time"])
        if datetime.now() - cached_time > timedelta(days=self.cache_days):
            return False
        
        # 检查文件是否存在
        cache_path = self._get_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return False
        
        return True
    
    def get_cached_features(self, stock_code: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        获取缓存的特征数据
        
        Args:
            stock_code: 股票代码
            df: 原始数据
            
        Returns:
            缓存的特征数据，如果不存在返回None
        """
        try:
            # 计算数据哈希
            data_hash = self._calculate_data_hash(df)
            cache_key = self._generate_cache_key(stock_code, data_hash)
            
            # 检查缓存是否有效
            if not self._is_cache_valid(cache_key):
                self.cache_misses += 1
                return None
            
            # 加载缓存数据
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.cache_hits += 1
            logger.info(f"✅ 缓存命中: {stock_code}")
            return cached_data["features"]
            
        except Exception as e:
            logger.warning(f"读取缓存失败 {stock_code}: {e}")
            self.cache_misses += 1
            return None
    
    def save_features_to_cache(self, stock_code: str, df_original: pd.DataFrame, 
                              df_features: pd.DataFrame):
        """
        保存特征数据到缓存
        
        Args:
            stock_code: 股票代码
            df_original: 原始数据
            df_features: 特征数据
        """
        try:
            # 计算数据哈希
            data_hash = self._calculate_data_hash(df_original)
            cache_key = self._generate_cache_key(stock_code, data_hash)
            
            # 准备缓存数据
            cache_data = {
                "stock_code": stock_code,
                "data_hash": data_hash,
                "features": df_features,
                "cached_time": datetime.now().isoformat(),
                "version": self.version
            }
            
            # 保存缓存文件
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # 更新元数据
            self.metadata["cache_entries"][cache_key] = {
                "stock_code": stock_code,
                "data_hash": data_hash,
                "cached_time": datetime.now().isoformat(),
                "version": self.version,
                "file_size": os.path.getsize(cache_path)
            }
            self._save_metadata()
            
            logger.info(f"💾 缓存保存: {stock_code}")
            
        except Exception as e:
            logger.error(f"保存缓存失败 {stock_code}: {e}")
    
    def clear_expired_cache(self):
        """清理过期缓存"""
        expired_keys = []
        current_time = datetime.now()
        
        for cache_key, cache_info in self.metadata["cache_entries"].items():
            cached_time = datetime.fromisoformat(cache_info["cached_time"])
            if current_time - cached_time > timedelta(days=self.cache_days):
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            try:
                # 删除缓存文件
                cache_path = self._get_cache_path(cache_key)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                
                # 从元数据中移除
                del self.metadata["cache_entries"][cache_key]
                
            except Exception as e:
                logger.error(f"清理过期缓存失败 {cache_key}: {e}")
        
        if expired_keys:
            self._save_metadata()
            logger.info(f"🗑️ 清理过期缓存: {len(expired_keys)} 个")
    
    def clear_all_cache(self):
        """清理所有缓存"""
        try:
            # 删除所有缓存文件
            for cache_key in list(self.metadata["cache_entries"].keys()):
                cache_path = self._get_cache_path(cache_key)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            
            # 重置元数据
            self.metadata = {
                "version": self.version,
                "created_time": datetime.now().isoformat(),
                "cache_entries": {}
            }
            self._save_metadata()
            
            logger.info("🗑️ 所有缓存已清理")
            
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_entries = len(self.metadata["cache_entries"])
        total_size = sum(entry.get("file_size", 0) for entry in self.metadata["cache_entries"].values())
        
        hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            "cache_entries": total_entries,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "version": self.version,
            "cache_dir": self.cache_dir
        }
    
    def print_cache_stats(self):
        """打印缓存统计信息"""
        stats = self.get_cache_stats()
        print("\n📊 特征缓存统计:")
        print(f"  缓存条目: {stats['cache_entries']}")
        print(f"  总大小: {stats['total_size_mb']:.2f} MB")
        print(f"  命中次数: {stats['cache_hits']}")
        print(f"  未命中次数: {stats['cache_misses']}")
        print(f"  命中率: {stats['hit_rate']:.2%}")
        print(f"  版本: {stats['version']}")
        print(f"  缓存目录: {stats['cache_dir']}")


class BatchFeatureProcessor:
    """
    批量特征处理器
    - 支持多进程并行处理
    - 智能缓存管理
    - 进度监控
    """
    
    def __init__(self, feature_engineer, cache_dir: str = "cache/features", 
                 cache_days: int = 7, parallel_workers: int = 4):
        """
        初始化批量处理器
        
        Args:
            feature_engineer: 特征工程实例
            cache_dir: 缓存目录
            cache_days: 缓存有效期
            parallel_workers: 并行工作进程数
        """
        self.feature_engineer = feature_engineer
        self.cache = FeatureCache(cache_dir, cache_days)
        self.parallel_workers = parallel_workers
        
    def process_stocks_with_cache(self, stock_codes: list, data_loader_func, 
                                 show_progress: bool = True) -> Dict[str, pd.DataFrame]:
        """
        批量处理股票特征（带缓存）
        
        Args:
            stock_codes: 股票代码列表
            data_loader_func: 数据加载函数，接受股票代码返回DataFrame
            show_progress: 是否显示进度
            
        Returns:
            股票代码到特征DataFrame的映射
        """
        results = {}
        cached_count = 0
        processed_count = 0
        
        # 清理过期缓存
        self.cache.clear_expired_cache()
        
        if show_progress:
            print(f"🚀 开始批量处理 {len(stock_codes)} 只股票的特征...")
        
        for i, stock_code in enumerate(stock_codes):
            try:
                if show_progress and i % 10 == 0:
                    print(f"进度: {i+1}/{len(stock_codes)} ({(i+1)/len(stock_codes)*100:.1f}%)")
                
                # 加载原始数据
                df_original = data_loader_func(stock_code)
                if df_original is None or len(df_original) == 0:
                    logger.warning(f"股票 {stock_code} 数据为空，跳过")
                    continue
                
                # 尝试从缓存获取
                df_features = self.cache.get_cached_features(stock_code, df_original)
                
                if df_features is not None:
                    # 缓存命中
                    cached_count += 1
                    results[stock_code] = df_features
                else:
                    # 缓存未命中，需要计算特征
                    if show_progress:
                        print(f"🔧 处理股票 {stock_code} 特征...")
                    
                    df_features = self.feature_engineer.create_all_features(df_original, stock_code)
                    
                    # 保存到缓存
                    self.cache.save_features_to_cache(stock_code, df_original, df_features)
                    
                    processed_count += 1
                    results[stock_code] = df_features
                
            except Exception as e:
                logger.error(f"处理股票 {stock_code} 失败: {e}")
                continue
        
        if show_progress:
            print(f"\n✅ 批量处理完成!")
            print(f"  总股票数: {len(stock_codes)}")
            print(f"  成功处理: {len(results)}")
            print(f"  缓存命中: {cached_count}")
            print(f"  新计算: {processed_count}")
            
            # 显示缓存统计
            self.cache.print_cache_stats()
        
        return results