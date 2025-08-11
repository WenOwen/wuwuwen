# LightGBM股票数据训练解决方案

## 🎯 项目概述

本项目为您创建了一套完整的基于LightGBM的股票预测模型训练解决方案，专门针对professional_parquet文件夹中的股票数据进行了优化。

## 📁 创建的文件清单

### 核心脚本
1. **`stock_data_processor.py`** - 股票数据预处理脚本
   - 自动加载parquet格式股票数据
   - 智能识别股票代码、日期、目标列
   - 创建时序特征和统计特征
   - 生成适合机器学习的训练数据

2. **`lightgbm_stock_train.py`** - 股票专用LightGBM训练脚本
   - 集成数据预处理和模型训练
   - 支持时序数据分割
   - 针对股票数据优化的模型参数
   - 完整的评估和保存功能

3. **`demo_stock_training.py`** - 演示脚本
   - 创建模拟股票数据用于测试
   - 验证完整训练流程
   - 适合初次使用和测试

### 配置文件
4. **`config/train/lightGBM_stock_train.yaml`** - 股票训练专用配置
   - 针对股票数据优化的参数设置
   - 支持自动数据预处理
   - 时序分割和特征工程配置
   - 股票特有的评估指标

### 文档
5. **`股票训练指南.md`** - 详细使用指南
   - 快速开始教程
   - 配置说明
   - 常见问题解答

6. **`README_股票训练解决方案.md`** - 本文档

## 🚀 快速开始

### 方法1: 使用现有数据训练（推荐）

如果您已有professional_parquet文件夹中的股票数据：

```bash
# 一键训练（自动数据预处理+模型训练）
python lightgbm_stock_train.py --config config/train/lightGBM_stock_train.yaml
```

### 方法2: 演示模式

如果您想先测试整个流程：

```bash
# 创建演示数据并准备训练
python demo_stock_training.py

# 运行演示训练
python lightgbm_stock_train.py --config config/train/demo_config.yaml
```

### 方法3: 分步执行

如果您希望分步执行：

```bash
# 步骤1: 数据预处理
python stock_data_processor.py --data_dir data/professional_parquet

# 步骤2: 模型训练
python lightgbm_stock_train.py --config config/train/lightGBM_stock_train.yaml
```

## 🎯 核心特性

### 数据处理特性
- ✅ **自动识别**: 智能识别股票代码、日期、目标列
- ✅ **时序特征**: 创建多天历史回看特征
- ✅ **统计特征**: 生成均值、标准差、最值等统计特征
- ✅ **技术指标**: 支持RSI、MACD、布林带等技术指标
- ✅ **数据验证**: 完整的数据质量检查

### 训练特性
- ✅ **时序分割**: 专门针对股票数据的时序分割方式
- ✅ **鲁棒标准化**: 使用robust标准化处理异常值
- ✅ **异常值处理**: Winsorize方法处理极端值
- ✅ **特征选择**: 基于重要性和相关性的特征筛选
- ✅ **超参数优化**: 支持Optuna自动调参

### 评估特性
- ✅ **多元指标**: RMSE、MAE、MAPE、R²等
- ✅ **方向准确率**: 股票特有的涨跌方向预测准确率
- ✅ **分层评估**: 按涨跌幅范围分层分析
- ✅ **可视化**: 特征重要性、预测效果图表

## 📊 输出结果

### 模型文件 (models/lightgbm_stock/stock_training_001/)
```
├── lightgbm_stock_model.pkl      # 训练好的模型
├── scaler.pkl                    # 数据标准化器  
├── feature_names.json            # 特征信息
└── feature_names_simple.txt      # 特征列表
```

### 结果文件 (results/lightgbm_stock/stock_training_001/)
```
├── metrics.json                  # 评估指标
├── predictions.csv               # 预测结果
├── feature_importance.csv        # 特征重要性
└── top_features.txt              # 重要特征排序
```

## ⚙️ 主要配置项

### 数据配置
```yaml
data:
  source_data:
    parquet_dir: "./data/professional_parquet"  # 原始数据路径
    auto_process: true                          # 自动预处理
  stock_specific:
    time_series:
      lookback_days: 5    # 历史回看天数
      target_days: 1      # 预测天数
```

### 模型配置  
```yaml
lightgbm:
  basic_params:
    learning_rate: 0.05    # 较低学习率提高稳定性
    num_leaves: 63         # 适中复杂度
  advanced_params:
    max_depth: 6          # 防止过拟合
    lambda_l1: 0.1        # L1正则化
    lambda_l2: 0.1        # L2正则化
```

### 训练配置
```yaml
training:
  data_split:
    time_series_split: true  # 时序分割（重要）
    test_size: 0.2
    validation_size: 0.1
```

## 📈 性能指标示例

```
TEST集评估结果:
  rmse: 0.0234           # 均方根误差
  mae: 0.0187            # 平均绝对误差
  mape: 8.45%            # 平均绝对百分比误差
  r2_score: 0.723        # 决定系数
  directional_accuracy: 68.9%  # 方向准确率
```

## 🔧 自定义配置

### 修改预测目标
```yaml
# 在配置文件中修改
data:
  stock_specific:
    time_series:
      target_days: 3      # 改为预测3天后的收益
```

### 调整特征工程
```yaml
data:
  preprocessing:
    feature_engineering:
      statistical_features:
        - "mean"
        - "std" 
        - "max"
        - "min"
        - "skew"      # 添加偏度特征
```

### 超参数优化
```yaml
hyperparameter_tuning:
  enabled: true
  optuna_config:
    n_trials: 50         # 增加试验次数
    timeout: 3600        # 1小时超时
```

## 🚨 注意事项

1. **数据要求**: 确保professional_parquet文件夹包含有效的股票数据
2. **内存需求**: 大量股票数据可能需要较多内存，可调整chunk_size
3. **时序性**: 股票数据务必使用时序分割，避免数据泄露
4. **异常值**: 股票数据包含较多异常值，建议使用robust标准化

## 🆘 故障排除

### 问题1: 找不到数据文件
```bash
# 检查数据目录是否存在
ls -la data/professional_parquet/

# 如果不存在，运行数据整合脚本
python data_processing/professional_data_integration.py
```

### 问题2: 内存不足
```yaml
# 在配置文件中减小批处理大小
misc:
  memory_optimization:
    enabled: true
    chunk_size: 5000
```

### 问题3: 训练时间过长
```yaml
# 减少超参数优化试验
hyperparameter_tuning:
  optuna_config:
    n_trials: 10
```

## 📞 使用支持

- 详细教程: 查看 `股票训练指南.md`
- 演示运行: 执行 `python demo_stock_training.py`
- 配置参考: 查看 `config/train/lightGBM_stock_train.yaml`

## 🎉 开始使用

准备好开始了吗？运行以下命令：

```bash
# 如果有现成数据
python lightgbm_stock_train.py

# 如果想先测试
python demo_stock_training.py
```

祝您使用愉快！如有问题，请参考相关文档或检查日志文件。