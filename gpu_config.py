#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU配置模块 - 双RTX 3090优化配置
专门为双RTX 3090显卡配置优化
主要针对LightGBM GPU加速优化
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
import subprocess

# 配置日志
logger = logging.getLogger(__name__)

# RTX 3090 显卡规格
RTX_3090_SPECS = {
    'memory_gb': 24,        # 24GB 显存
    'cuda_cores': 10496,    # CUDA核心数
    'tensor_cores': 328,    # 第三代Tensor核心
    'memory_bandwidth': 936, # GB/s
    'fp16_performance': 35.6, # TFLOPS
    'tensor_performance': 142, # TFLOPS (稀疏)
    'boost_clock': 1695,    # MHz
    'base_clock': 1395,     # MHz
}

def setup_dual_gpu() -> Optional[Dict[str, Any]]:
    """
    配置双RTX 3090 GPU环境 - 专门为LightGBM优化
    
    Returns:
        GPU配置字典，包含LightGBM GPU参数，如果配置失败则返回None
    """
    try:
        # 检查CUDA和GPU可用性
        gpu_count = _get_gpu_count()
        
        if gpu_count < 1:
            logger.error("未检测到GPU，使用CPU模式")
            return None
        
        logger.info(f"检测到 {gpu_count} 块GPU")
        
        # 检查GPU型号
        gpu_info = _get_gpu_info()
        
        # 配置LightGBM GPU参数
        gpu_config = {
            'device_type': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,  # 使用第一块GPU作为主要设备
            'max_bin': 255,      # RTX 3090的大显存允许更大的bin数量
            'gpu_use_dp': True,  # 启用双精度（更准确）
            'num_gpu': min(gpu_count, 2),  # 最多使用2块GPU
        }
        
        # RTX 3090特定优化
        if any('RTX 3090' in info for info in gpu_info.values()):
            logger.info("检测到RTX 3090，应用专用优化")
            gpu_config.update({
                'max_bin': 511,      # 利用大显存使用更大的bin数量
                'feature_fraction': 0.8,  # 增加特征采样率
                'bagging_fraction': 0.8,  # 增加数据采样率
                'num_leaves': 255,   # 增加叶子节点数量
                'max_depth': 15,     # 增加树的深度
                'min_data_in_leaf': 10,  # 减少叶子节点最小数据量
            })
        
        # 设置环境变量优化GPU性能
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 非阻塞CUDA启动
        os.environ['CUDA_CACHE_DISABLE'] = '0'    # 启用CUDA缓存
        os.environ['NVIDIA_TF32_OVERRIDE'] = '1'  # 启用TF32加速
        
        # 如果有多个GPU，配置数据并行
        if gpu_count >= 2:
            logger.info("配置双GPU并行训练")
            gpu_config['parallel_threads'] = gpu_count * 8  # 每个GPU 8个线程
            
        logger.info("LightGBM GPU配置完成")
        logger.info(f"配置参数: {gpu_config}")
        
        return gpu_config
        
    except Exception as e:
        logger.error(f"GPU配置失败: {e}")
        return None

def _get_gpu_count() -> int:
    """获取可用GPU数量"""
    try:
        # 尝试使用nvidia-ml-py
        try:
            import pynvml
            pynvml.nvmlInit()
            return pynvml.nvmlDeviceGetCount()
        except:
            pass
        
        # 尝试使用nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            if result.returncode == 0:
                return len([line for line in result.stdout.strip().split('\n') if line.startswith('GPU')])
        except:
            pass
        
        # 尝试使用PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except:
            pass
        
        return 0
    except Exception as e:
        logger.error(f"获取GPU数量失败: {e}")
        return 0

def _get_gpu_info() -> Dict[str, str]:
    """获取GPU信息"""
    gpu_info = {}
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        index = parts[0]
                        name = parts[1]
                        memory = parts[2]
                        gpu_info[f'GPU_{index}'] = f"{name} ({memory})"
    except Exception as e:
        logger.warning(f"获取GPU信息失败: {e}")
    return gpu_info

def get_optimal_batch_size(base_size: int, gpu_count: int = 2) -> int:
    """
    根据RTX 3090的规格动态计算最优批次大小
    专门为LightGBM和股票数据训练优化
    
    Args:
        base_size: 基础批次大小
        gpu_count: GPU数量
    
    Returns:
        优化后的批次大小
    """
    # RTX 3090的大显存允许使用更大的批次
    memory_factor = RTX_3090_SPECS['memory_gb'] / 11  # 相对于RTX 2080 Ti (11GB)的倍数
    
    # 根据GPU数量和内存大小调整
    optimal_size = int(base_size * memory_factor)
    
    # LightGBM针对性优化
    if gpu_count >= 2:
        # 双GPU可以处理更大的数据集
        optimal_size = int(optimal_size * 1.5)
    
    # 针对不同的基础大小设置合理上限
    if base_size <= 32:
        max_size = 512    # 对于小批次，利用大显存
    elif base_size <= 64:
        max_size = 1024   # 对于中等批次
    else:
        max_size = 2048   # 对于大批次，RTX 3090可以处理
    
    # 确保批次大小合理且是32的倍数（内存对齐优化）
    optimal_size = min(optimal_size, max_size)
    optimal_size = max(optimal_size, base_size)  # 不小于基础大小
    optimal_size = ((optimal_size + 31) // 32) * 32
    
    logger.info(f"批次大小优化: {base_size} -> {optimal_size} (GPU数量: {gpu_count})")
    
    return optimal_size

def get_memory_info() -> Dict[str, Any]:
    """获取GPU内存信息"""
    try:
        memory_info = {}
        gpu_info = _get_gpu_info()
        
        for gpu_id, info in gpu_info.items():
            # 解析内存信息
            memory_str = info.split('(')[1].split(')')[0] if '(' in info else "Unknown"
            memory_gb = RTX_3090_SPECS['memory_gb']  # 默认RTX 3090规格
            
            if 'RTX 3090' in info:
                memory_gb = 24
            elif 'RTX 3080' in info:
                memory_gb = 10
            elif 'RTX 4090' in info:
                memory_gb = 24
            
            memory_info[gpu_id] = {
                'name': info,
                'total_memory_gb': memory_gb,
                'available_memory_gb': memory_gb - 2,  # 预留2GB
                'cuda_cores': RTX_3090_SPECS['cuda_cores'] if 'RTX 3090' in info else 'Unknown',
                'tensor_cores': RTX_3090_SPECS['tensor_cores'] if 'RTX 3090' in info else 'Unknown',
                'boost_clock_mhz': RTX_3090_SPECS['boost_clock'] if 'RTX 3090' in info else 'Unknown'
            }
        
        return memory_info
    except Exception as e:
        logger.error(f"获取GPU信息失败: {e}")
    
    return {}

def optimize_for_stock_training() -> Dict[str, Any]:
    """
    专门为股票训练优化的配置
    返回LightGBM专用参数
    """
    # 股票数据特点：时间序列数据，需要防止过拟合
    stock_config = {
        # 基础配置
        'objective': 'regression',
        'metric': ['rmse', 'mae'],
        'boosting_type': 'gbdt',
        'num_boost_round': 1000,
        
        # 学习相关
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1,
        
        # 正则化（防止过拟合）
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_gain_to_split': 0.02,
        'drop_rate': 0.1,
        
        # 早停
        'early_stopping_rounds': 100,
        
        # 随机种子
        'random_state': 42,
        'deterministic': True,
    }
    
    # 如果有GPU，添加GPU配置
    gpu_count = _get_gpu_count()
    if gpu_count > 0:
        gpu_config = setup_dual_gpu()
        if gpu_config:
            stock_config.update(gpu_config)
    
    logger.info("股票训练专用优化配置完成")
    return stock_config

def get_recommended_settings() -> Dict[str, Any]:
    """
    获取双RTX 3090的推荐设置
    专门为LightGBM和股票数据优化
    """
    gpu_count = _get_gpu_count()
    
    return {
        'batch_sizes': {
            'small_dataset': get_optimal_batch_size(64, gpu_count),   # 小数据集
            'medium_dataset': get_optimal_batch_size(128, gpu_count), # 中等数据集
            'large_dataset': get_optimal_batch_size(256, gpu_count),  # 大数据集
            'cv_training': get_optimal_batch_size(32, gpu_count),     # 交叉验证
        },
        'lightgbm_gpu_settings': {
            'device_type': 'gpu',
            'max_bin': 511 if gpu_count > 0 else 255,
            'gpu_use_dp': True,
            'num_gpu': min(gpu_count, 2),
        },
        'memory_settings': {
            'memory_limit_per_gpu_gb': 22,
            'total_gpu_memory_gb': RTX_3090_SPECS['memory_gb'] * gpu_count,
            'recommended_data_size_gb': RTX_3090_SPECS['memory_gb'] * gpu_count * 0.8,
        },
        'performance_settings': {
            'cuda_cores_total': RTX_3090_SPECS['cuda_cores'] * gpu_count,
            'tensor_cores_total': RTX_3090_SPECS['tensor_cores'] * gpu_count,
            'memory_bandwidth_total': RTX_3090_SPECS['memory_bandwidth'] * gpu_count,
        },
        'stock_training_optimized': True,
        'gpu_count': gpu_count,
    }

def get_lightgbm_gpu_params() -> Dict[str, Any]:
    """
    获取LightGBM GPU训练的完整参数配置
    """
    base_params = optimize_for_stock_training()
    gpu_params = setup_dual_gpu()
    
    if gpu_params:
        base_params.update(gpu_params)
        logger.info("LightGBM GPU参数配置完成")
    else:
        logger.warning("GPU不可用，使用CPU参数")
        base_params['device_type'] = 'cpu'
    
    return base_params

def auto_detect_and_setup():
    """自动检测并配置GPU环境"""
    logger.info("开始自动检测GPU环境...")
    
    # 检测NVIDIA-SMI
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("NVIDIA驱动检测成功")
            if "RTX 3090" in result.stdout:
                logger.info("✅ 检测到RTX 3090显卡")
                gpu_count = result.stdout.count("RTX 3090")
                logger.info(f"RTX 3090数量: {gpu_count}")
            else:
                logger.warning("⚠️ 未检测到RTX 3090，配置可能需要调整")
        else:
            logger.error("❌ NVIDIA驱动检测失败")
    except FileNotFoundError:
        logger.warning("❌ 未找到nvidia-smi命令")
    
    return setup_dual_gpu()

def check_lightgbm_gpu_support() -> bool:
    """
    检查LightGBM是否支持GPU
    """
    try:
        import lightgbm as lgb
        # 尝试创建一个简单的GPU训练集来测试
        train_data = lgb.Dataset([[1, 2], [3, 4]], label=[0, 1])
        params = {'device_type': 'gpu', 'objective': 'binary', 'verbose': -1}
        
        # 测试是否能成功使用GPU
        lgb.train(params, train_data, num_boost_round=1, valid_sets=[train_data], 
                 callbacks=[lgb.early_stopping(1)])
        logger.info("✅ LightGBM GPU支持检测成功")
        return True
    except Exception as e:
        logger.warning(f"❌ LightGBM GPU支持检测失败: {e}")
        return False

if __name__ == "__main__":
    # 测试配置
    print("=== 双RTX 3090 GPU配置测试 ===")
    print("专门为LightGBM股票训练优化\n")
    
    # 自动检测
    gpu_config = auto_detect_and_setup()
    
    print("🔍 GPU检测结果:")
    gpu_count = _get_gpu_count()
    print(f"  检测到GPU数量: {gpu_count}")
    
    if gpu_config:
        print("✅ GPU配置成功")
        print(f"  配置类型: LightGBM GPU优化")
        
        # 显示GPU详细信息
        gpu_info = _get_gpu_info()
        if gpu_info:
            print("\n💾 GPU详细信息:")
            for gpu_id, info in gpu_info.items():
                print(f"  {gpu_id}: {info}")
        
        # LightGBM GPU支持测试
        print("\n🧪 LightGBM GPU支持测试:")
        if check_lightgbm_gpu_support():
            print("  ✅ LightGBM GPU加速可用")
        else:
            print("  ❌ LightGBM GPU加速不可用")
        
        # 显示推荐设置
        settings = get_recommended_settings()
        print("\n📊 推荐设置:")
        for key, value in settings.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # 测试批次大小优化
        print("\n🔧 批次大小优化测试:")
        test_sizes = [16, 32, 48, 64, 128]
        for base in test_sizes:
            optimal = get_optimal_batch_size(base, gpu_count)
            print(f"  基础大小 {base:3d} -> 优化后 {optimal:4d}")
        
        # 显示完整的LightGBM参数
        print("\n⚙️ LightGBM GPU完整参数:")
        lgb_params = get_lightgbm_gpu_params()
        for key, value in lgb_params.items():
            print(f"  {key}: {value}")
            
    else:
        print("❌ GPU配置失败，将使用CPU模式")
        
    print("\n=== 配置测试完成 ===")
    
    # 保存配置到文件
    try:
        import json
        config_summary = {
            'gpu_count': _get_gpu_count(),
            'gpu_info': _get_gpu_info(),
            'recommended_settings': get_recommended_settings(),
            'lightgbm_params': get_lightgbm_gpu_params() if gpu_config else None,
            'timestamp': str(subprocess.run(['date'], capture_output=True, text=True).stdout.strip())
        }
        
        with open('gpu_config_summary.json', 'w', encoding='utf-8') as f:
            json.dump(config_summary, f, indent=2, ensure_ascii=False)
        print("📝 配置摘要已保存到 gpu_config_summary.json")
        
    except Exception as e:
        print(f"⚠️ 保存配置摘要失败: {e}")