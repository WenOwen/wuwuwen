# 配置文件使用说明

## 概述
已成功将分散在多个文件中的配置参数统一到 `config.py` 文件中，实现了配置的集中管理。

## 文件结构

### config.py - 统一配置文件
- **Config类**: 包含所有配置参数的主类
- **ConfigPresets类**: 预定义的配置方案

### 配置分类

#### 1. 数据配置 (DATA_CONFIG)
```python
'data_dir': './data/processed_v2',  # 数据目录
'train_ratio': 0.8,                 # 训练集比例
'batch_size': 128,                  # 批次大小
'shuffle_train': True,              # 是否打乱训练数据
'num_workers': 4,                   # 数据加载器工作进程数
```

#### 2. 模型配置 (MODEL_CONFIG)
```python
'input_dim': 27,                    # 输入特征维度
'hidden_dim': 64,                   # LSTM隐藏层维度
'num_layers': 2,                    # LSTM层数
'dropout': 0.3,                     # Dropout率
'use_multihead_attention': True,    # 是否使用多头注意力
'num_attention_heads': 8,           # 注意力头数
```

#### 3. 训练配置 (TRAINING_CONFIG)
```python
'num_epochs': 30,                   # 训练轮数
'learning_rate': 1e-3,              # 学习率
'weight_decay': 1e-4,               # 权重衰减
'patience': 5,                      # 早停耐心值
'scheduler_type': 'cosine',         # 学习率调度器类型
'gradient_clip_norm': 1.0,          # 梯度裁剪范数
```

#### 4. 输出配置 (OUTPUT_CONFIG)
```python
'results_dir': './results_improved', # 结果保存目录
'save_model': True,                  # 是否保存模型
'save_history': True,                # 是否保存训练历史
'log_interval': 100,                 # 日志打印间隔
```

## 使用方法

### 1. 基本使用
```python
from config import Config

# 创建训练器时使用配置
trainer = ImprovedTrainer()  # 自动使用默认配置

# 或者传入自定义配置
config = Config.get_all_config()
trainer = ImprovedTrainer(config)
```

### 2. 修改配置
```python
# 修改特定配置
Config.update_config('training', {
    'learning_rate': 5e-4,
    'num_epochs': 50
})

# 修改数据配置
Config.update_config('data', {
    'batch_size': 256
})
```

### 3. 使用预设配置
```python
from config import ConfigPresets

# 快速测试配置
ConfigPresets.quick_test()

# 生产环境配置
ConfigPresets.production()

# 大模型配置
ConfigPresets.large_model()
```

### 4. 配置验证和查看
```python
# 验证配置合理性
Config.validate_config()

# 打印当前配置
Config.print_config()
```

## 重构内容

### train.py 重构
- ✅ 移除硬编码的训练参数
- ✅ 从配置文件读取所有参数
- ✅ 支持不同类型的学习率调度器
- ✅ 灵活的早停配置
- ✅ 可配置的日志输出间隔

### model.py 重构
- ✅ 保留向后兼容性
- ✅ 添加废弃警告
- ✅ 测试代码使用新配置系统

## 配置对比

### 修复前的问题
| 参数 | model.py | train.py | 问题 |
|------|----------|----------|------|
| learning_rate | 5e-4 | 1e-3 | 不一致 |
| batch_size | 256 | 128 | 不一致 |
| patience | 30 | 5 | 不一致 |

### 修复后的统一配置
所有参数现在都在 `config.py` 中统一管理，消除了配置冲突。

## 优势

1. **配置集中**: 所有参数在一个文件中管理
2. **消除冲突**: 不再有重复或冲突的配置
3. **易于维护**: 修改参数只需在一个地方
4. **预设支持**: 提供常用的配置预设
5. **验证机制**: 自动验证配置合理性
6. **向后兼容**: 保持原有代码的兼容性

## 测试结果

✅ 配置文件正常加载和验证
✅ 模型测试通过
✅ 预设配置功能正常
✅ 配置验证机制工作正常

现在可以安全地使用统一配置系统进行训练！