# -*- coding: utf-8 -*-
"""
内存优化配置文件 - 针对大规模股票数据训练的内存优化设置
"""

# 内存优化配置
MEMORY_OPTIMIZATION_CONFIG = {
    # 数据处理配置
    "max_stocks_per_batch": 50,           # 每批处理的最大股票数
    "max_samples_per_stock": 500,         # 每只股票的最大样本数
    "max_total_samples": 50000,           # 总最大样本数
    "lookback_window_limit": 30,          # 限制回望窗口大小
    "data_limit_per_stock": 2000,         # 每只股票的最大数据行数
    
    # 特征工程配置
    "enable_feature_selection": True,     # 启用特征选择
    "feature_selection_ratio": 0.7,       # 保留特征的比例
    "enable_pca": False,                  # 是否启用PCA降维
    "pca_components": 0.95,               # PCA保留的方差比例
    
    # 内存监控配置
    "memory_warning_threshold": 0.75,     # 内存警告阈值
    "memory_critical_threshold": 0.85,    # 内存危险阈值
    "gc_frequency": 10,                   # 垃圾回收频率（每N个股票）
    
    # 训练配置
    "max_prediction_days": 1,             # 最大训练的预测天数（只训练1天预测）
    "disable_hyperparameter_optimization": True,  # 禁用超参数优化
    "use_lightweight_models": True,       # 使用轻量级模型
    "early_stopping_patience": 5,        # 早停耐心值
    
    # 缓存配置
    "disable_all_cache": True,            # 禁用所有缓存
    "enable_disk_cache": False,           # 禁用磁盘缓存
    
    # 并行处理配置
    "max_workers": 1,                     # 最大工作进程数
    "use_multiprocessing": False,         # 禁用多进程处理
}

def get_optimized_config():
    """获取优化后的配置"""
    return MEMORY_OPTIMIZATION_CONFIG.copy()

def apply_memory_optimizations():
    """应用内存优化设置"""
    import os
    import gc
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # 设置垃圾回收
    gc.set_threshold(700, 10, 10)  # 更频繁的垃圾回收
    
    # TensorFlow内存优化（如果使用）
    try:
        import tensorflow as tf
        tf.config.experimental.set_memory_growth(
            tf.config.list_physical_devices('GPU')[0], True
        )
    except:
        pass
    
    # PyTorch内存优化（如果使用）
    try:
        import torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except:
        pass

# 内存优化的数据类型配置
DTYPE_CONFIG = {
    "float_dtype": "float32",     # 使用float32而不是float64
    "int_dtype": "int32",         # 使用int32而不是int64
    "enable_mixed_precision": True,  # 启用混合精度
}

# 特征工程优化配置
FEATURE_CONFIG = {
    "essential_features": [
        # 只保留最重要的特征
        "收盘价", "成交量", "涨跌幅",
        "RSI", "MACD", "BOLL_upper", "BOLL_lower",
        "MA_5", "MA_10", "MA_20",
        "volume_ratio", "price_change_ratio"
    ],
    "skip_complex_features": [
        # 跳过计算复杂的特征
        "sector_correlation", "industry_momentum",
        "complex_technical_indicators"
    ]
}

def print_memory_config():
    """打印内存优化配置"""
    print("📊 内存优化配置:")
    for key, value in MEMORY_OPTIMIZATION_CONFIG.items():
        print(f"   {key}: {value}")
    
    print("\n🔧 数据类型配置:")
    for key, value in DTYPE_CONFIG.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    print_memory_config()
    apply_memory_optimizations()
    print("✅ 内存优化配置已应用")