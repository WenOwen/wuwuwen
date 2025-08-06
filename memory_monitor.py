# -*- coding: utf-8 -*-
"""
内存监控工具 - 实时监控训练过程中的内存使用情况
"""

import os
import psutil
import time
import logging
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold  # 内存使用警告阈值
        self.critical_threshold = critical_threshold  # 内存使用危险阈值
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_info()
        self.peak_memory = self.initial_memory
        self.memory_history = []
        
    def get_memory_info(self) -> Dict:
        """获取当前内存信息"""
        # 系统内存
        system_mem = psutil.virtual_memory()
        
        # 进程内存
        process_mem = self.process.memory_info()
        
        return {
            'system_total': system_mem.total,
            'system_used': system_mem.used,
            'system_available': system_mem.available,
            'system_percent': system_mem.percent,
            'process_rss': process_mem.rss,  # 物理内存
            'process_vms': process_mem.vms,  # 虚拟内存
            'timestamp': datetime.now()
        }
    
    def check_memory_status(self) -> str:
        """检查内存状态"""
        mem_info = self.get_memory_info()
        self.memory_history.append(mem_info)
        
        # 更新峰值内存
        if mem_info['process_rss'] > self.peak_memory['process_rss']:
            self.peak_memory = mem_info
        
        system_usage = mem_info['system_percent'] / 100
        
        if system_usage >= self.critical_threshold:
            return "CRITICAL"
        elif system_usage >= self.warning_threshold:
            return "WARNING"
        else:
            return "OK"
    
    def log_memory_status(self, context: str = ""):
        """记录内存状态"""
        mem_info = self.get_memory_info()
        status = self.check_memory_status()
        
        process_mb = mem_info['process_rss'] / 1024 / 1024
        system_gb = mem_info['system_used'] / 1024 / 1024 / 1024
        system_total_gb = mem_info['system_total'] / 1024 / 1024 / 1024
        
        message = (
            f"📊 内存状态 [{status}] {context}\n"
            f"   进程内存: {process_mb:.1f} MB\n"
            f"   系统内存: {system_gb:.1f}/{system_total_gb:.1f} GB ({mem_info['system_percent']:.1f}%)"
        )
        
        if status == "CRITICAL":
            logger.critical(message)
        elif status == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
    
    def get_memory_summary(self) -> Dict:
        """获取内存使用摘要"""
        current = self.get_memory_info()
        
        return {
            'initial_mb': self.initial_memory['process_rss'] / 1024 / 1024,
            'current_mb': current['process_rss'] / 1024 / 1024,
            'peak_mb': self.peak_memory['process_rss'] / 1024 / 1024,
            'growth_mb': (current['process_rss'] - self.initial_memory['process_rss']) / 1024 / 1024,
            'system_usage_percent': current['system_percent']
        }
    
    def suggest_optimizations(self) -> List[str]:
        """建议内存优化措施"""
        summary = self.get_memory_summary()
        suggestions = []
        
        if summary['system_usage_percent'] > 85:
            suggestions.append("系统内存使用率过高，建议减少批处理大小")
        
        if summary['growth_mb'] > 1000:
            suggestions.append("进程内存增长过多，建议增加垃圾回收频率")
        
        if summary['peak_mb'] > 2000:
            suggestions.append("峰值内存过高，建议使用数据采样或特征选择")
        
        return suggestions

def monitor_memory_during_training(func, *args, **kwargs):
    """装饰器：在训练过程中监控内存"""
    monitor = MemoryMonitor()
    monitor.log_memory_status("训练开始前")
    
    try:
        result = func(*args, **kwargs)
        monitor.log_memory_status("训练完成后")
        
        # 显示内存摘要
        summary = monitor.get_memory_summary()
        logger.info(f"\n📊 内存使用摘要:")
        logger.info(f"   初始内存: {summary['initial_mb']:.1f} MB")
        logger.info(f"   当前内存: {summary['current_mb']:.1f} MB")
        logger.info(f"   峰值内存: {summary['peak_mb']:.1f} MB")
        logger.info(f"   内存增长: {summary['growth_mb']:.1f} MB")
        logger.info(f"   系统使用率: {summary['system_usage_percent']:.1f}%")
        
        # 优化建议
        suggestions = monitor.suggest_optimizations()
        if suggestions:
            logger.info(f"\n💡 内存优化建议:")
            for suggestion in suggestions:
                logger.info(f"   - {suggestion}")
        
        return result
        
    except Exception as e:
        monitor.log_memory_status("训练异常终止")
        raise e

if __name__ == "__main__":
    # 测试内存监控
    monitor = MemoryMonitor()
    monitor.log_memory_status("测试开始")
    
    # 模拟内存使用
    import numpy as np
    data = []
    for i in range(10):
        data.append(np.random.rand(1000, 1000))
        monitor.log_memory_status(f"分配数据 {i+1}")
        time.sleep(1)
    
    # 清理内存
    del data
    import gc
    gc.collect()
    monitor.log_memory_status("清理内存后")
    
    # 显示摘要
    summary = monitor.get_memory_summary()
    print(f"内存摘要: {summary}")