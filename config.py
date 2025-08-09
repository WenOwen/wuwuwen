#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一配置文件
集中管理所有模型和训练相关的配置参数
"""

import torch
from typing import Dict, Any


class Config:
    """统一配置类"""
    
    # 数据配置
    DATA_CONFIG = {
        'data_dir': './data/processed_v2',
        'train_ratio': 0.8,
        'batch_size': 512,  # 增大批次大小以充分利用GPU，适合大数据集
        'shuffle_train': True,
        'num_workers': 8,   # 增加数据加载进程数
        'pin_memory': True, # 加速GPU数据传输
        'persistent_workers': True,  # 保持worker进程存活
        # 数据信息（从data_info.json获取）
        'num_samples': 2430962,
        'num_stocks': 5219,
        'sequence_length': 30,
        'feature_dims': {
            'stock': 16,
            'sector': 6,
            'index': 5,
            'sentiment': 2,
            'total': 29
        }
    }
    
    # 模型配置
    MODEL_CONFIG = {
        'input_dim': 29,  # 从data_info.json确认的特征维度
        'hidden_dim': 256,  # 增大隐藏层维度，适应大规模数据
        'num_layers': 4,    # 增加LSTM层数以增强模型表达能力
        'dropout': 0.2,     # 降低dropout，因为数据量大不容易过拟合
        'output_dim': 1,
        'task_type': 'regression',  # 'regression' or 'classification'
        'use_multihead_attention': True,
        'num_attention_heads': 8,   # 多头注意力头数
        # 新增配置项
        'gradient_checkpointing': False,  # 梯度检查点，节省显存
        'layer_norm': True,               # 添加层归一化
        'residual_connections': True,     # 残差连接
    }
    
    # 训练配置
    TRAINING_CONFIG = {
        'num_epochs': 50,              # 增加训练轮数，大数据集需要更多epochs
        'learning_rate': 5e-4,         # 降低学习率，大模型需要更稳定的训练
        'weight_decay': 1e-5,          # 减小权重衰减，避免欠拟合
        'gradient_clip_norm': 1.0,
        
        # 早停配置
        'patience': 10,                # 增加耐心值，大数据集收敛较慢
        'early_stop_metric': 'ic',     # 'ic', 'loss', 'rank_ic'
        'early_stop_mode': 'max',      # 'max' for ic, 'min' for loss
        'min_delta': 1e-6,             # 最小改善阈值
        
        # 学习率调度
        'scheduler_type': 'cosine',    # 'cosine', 'step', 'plateau', 'none'
        'scheduler_params': {
            'eta_min': 1e-6,           # for cosine
            'T_max': 50,               # cosine周期
            'step_size': 15,           # for step
            'gamma': 0.7,              # for step
            'factor': 0.5,             # for plateau
            'patience': 5,             # for plateau
        },
        
        # 损失函数配置
        'loss_type': 'weighted_mse',   # 使用加权MSE，更适合股票数据的不平衡性
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        
        # 新增训练配置
        'warmup_epochs': 5,            # 学习率预热轮数
        'accumulate_grad_batches': 1,  # 梯度累积步数
        'validate_every_n_epochs': 2,  # 验证频率
        'save_every_n_epochs': 10,     # 保存频率
    }
    
    # 评估配置
    EVALUATION_CONFIG = {
        'metrics': ['ic', 'rank_ic', 'mse', 'r2', 'sharpe_ratio'],
        'save_predictions': True,
        'save_attention_weights': True,
        'plot_results': True,
        # 新增评估配置
        'eval_batch_size': 1024,       # 评估时使用更大的批次大小
        'top_k_for_sharpe': 500,       # 计算夏普比率时选择的top股票数
        'ic_rolling_window': 252,      # IC滚动窗口大小（一年）
        'sector_analysis': True,       # 是否进行行业分析
        'feature_importance': True,    # 是否计算特征重要性
    }
    
    # 输出配置
    OUTPUT_CONFIG = {
        'results_dir': './results_improved',
        'save_model': True,
        'save_history': True,
        'save_plots': True,
        'plot_dpi': 300,
        'verbose': True,
        'log_interval': 100,  # 每多少个batch打印一次
    }
    
    # 设备配置
    DEVICE_CONFIG = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'gpu_id': 0,
        'mixed_precision': True,   # 启用混合精度训练，适合大模型
        'compile_model': False,    # 是否使用torch.compile（PyTorch 2.0+）
        'dataloader_num_workers': 8,  # 数据加载器工作进程数
        'pin_memory': True,        # 固定内存，加速GPU传输
        # 内存优化
        'gradient_checkpointing': False,  # 梯度检查点
        'empty_cache_every_n_steps': 100,  # 每N步清空GPU缓存
    }
    
    @classmethod
    def get_all_config(cls) -> Dict[str, Any]:
        """获取所有配置"""
        return {
            'data': cls.DATA_CONFIG,
            'model': cls.MODEL_CONFIG,
            'training': cls.TRAINING_CONFIG,
            'evaluation': cls.EVALUATION_CONFIG,
            'output': cls.OUTPUT_CONFIG,
            'device': cls.DEVICE_CONFIG,
        }
    
    @classmethod
    def update_config(cls, config_type: str, updates: Dict[str, Any]):
        """更新特定类型的配置"""
        if config_type == 'data':
            cls.DATA_CONFIG.update(updates)
        elif config_type == 'model':
            cls.MODEL_CONFIG.update(updates)
        elif config_type == 'training':
            cls.TRAINING_CONFIG.update(updates)
        elif config_type == 'evaluation':
            cls.EVALUATION_CONFIG.update(updates)
        elif config_type == 'output':
            cls.OUTPUT_CONFIG.update(updates)
        elif config_type == 'device':
            cls.DEVICE_CONFIG.update(updates)
        else:
            raise ValueError(f"未知的配置类型: {config_type}")
    
    @classmethod
    def print_config(cls):
        """打印所有配置"""
        print("=" * 60)
        print("当前配置:")
        print("=" * 60)
        
        all_config = cls.get_all_config()
        for config_type, config_dict in all_config.items():
            print(f"\n【{config_type.upper()}配置】")
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
        print("=" * 60)
    
    @classmethod
    def validate_config(cls):
        """验证配置的合理性"""
        errors = []
        
        # 验证训练配置
        if cls.TRAINING_CONFIG['learning_rate'] <= 0:
            errors.append("learning_rate必须大于0")
        
        if cls.TRAINING_CONFIG['num_epochs'] <= 0:
            errors.append("num_epochs必须大于0")
        
        if cls.DATA_CONFIG['batch_size'] <= 0:
            errors.append("batch_size必须大于0")
        
        # 验证模型配置
        if cls.MODEL_CONFIG['hidden_dim'] <= 0:
            errors.append("hidden_dim必须大于0")
        
        if cls.MODEL_CONFIG['num_layers'] <= 0:
            errors.append("num_layers必须大于0")
        
        if not 0 <= cls.MODEL_CONFIG['dropout'] <= 1:
            errors.append("dropout必须在[0,1]范围内")
        
        # 验证数据配置
        if not 0 < cls.DATA_CONFIG['train_ratio'] < 1:
            errors.append("train_ratio必须在(0,1)范围内")
        
        if errors:
            raise ValueError("配置验证失败:\n" + "\n".join(f"- {error}" for error in errors))
        
        print("✓ 配置验证通过")


# 预定义的配置方案
class ConfigPresets:
    """预定义的配置方案"""
    @staticmethod
    def optimized_for_stock_data():
        """针对股票数据优化的配置"""
        Config.update_config('model', {
            'hidden_dim': 256,
            'num_layers': 4,
            'dropout': 0.15,  # 股票数据噪声较大，适度正则化
            'use_multihead_attention': True,
            'num_attention_heads': 8,
        })
        Config.update_config('training', {
            'num_epochs': 60,
            'learning_rate': 3e-4,
            'weight_decay': 1e-5,
            'loss_type': 'weighted_mse',
            'patience': 12,
            'scheduler_type': 'cosine',
        })
        Config.update_config('data', {
            'batch_size': 512,
            'num_workers': 8,
        })
        Config.update_config('device', {
            'mixed_precision': True,
        })
    
    @staticmethod
    def memory_efficient():
        """内存高效配置（适合显存不足的情况）"""
        Config.update_config('model', {
            'hidden_dim': 128,
            'num_layers': 3,
            'gradient_checkpointing': True,
        })
        Config.update_config('data', {
            'batch_size': 256,
            'num_workers': 4,
        })
        Config.update_config('training', {
            'accumulate_grad_batches': 2,  # 梯度累积模拟更大批次
        })
        Config.update_config('device', {
            'mixed_precision': True,
            'empty_cache_every_n_steps': 50,
        })
    
    @staticmethod
    def high_performance():
        """高性能配置（适合高端GPU）"""
        Config.update_config('model', {
            'hidden_dim': 512,
            'num_layers': 6,
            'dropout': 0.1,
            'num_attention_heads': 16,
        })
        Config.update_config('data', {
            'batch_size': 1024,
            'num_workers': 12,
        })
        Config.update_config('training', {
            'learning_rate': 1e-4,
            'num_epochs': 80,
            'patience': 15,
        })


if __name__ == "__main__":
    # 测试配置
    print("默认配置:")
    Config.print_config()
    
    print("\n验证配置...")
    Config.validate_config()
    
    print("\n应用快速测试预设...")
    ConfigPresets.high_performance()
    Config.print_config()