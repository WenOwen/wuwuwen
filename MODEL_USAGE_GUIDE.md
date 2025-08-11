# 🎯 LightGBM股票预测模型使用指南

## 📊 模型性能总结

基于56万样本、5158只股票的大规模时序数据训练，经过精细调优后获得最佳性能：

| 指标 | 精细调优结果 | 相比原始改进 |
|------|-------------|-------------|
| **RMSE** | 1.7819 | ✅ +4.67% |
| **MAE** | 1.2607 | ✅ +3.27% |
| **R²** | 0.2790 | ✅ +2.61% |
| **MAPE** | 132.22% | ✅ +1.44% |

## 🚀 推荐使用配置

**最佳配置文件**: `config/train/lightGBM_train_fine_tuned.yaml`

### 核心优化特性：
- ✅ **渐进式特征选择**: 从868个特征优化到431个最重要特征
- ✅ **智能异常值处理**: IQR方法，阈值2.5，有效移除2.1万个异常样本
- ✅ **Standard标准化**: 比Robust标准化更适合股票数据
- ✅ **平衡的模型复杂度**: 防止过拟合同时保持预测能力

## 📈 使用步骤

### 1. 快速测试（推荐）
```bash
# 运行快速对比测试，验证配置效果
python quick_model_comparison.py
```

### 2. 完整训练（生产使用）
```bash
# 使用精细调优配置进行训练
python lightgbm_train.py --config config/train/lightGBM_train_fine_tuned.yaml
```

### 3. 超参数优化（最佳性能）
```bash
# 基于精细调优配置进行超参数优化（需要40分钟）
python run_final_optimization.py
```

## 🛠️ 模型部署建议

### 训练输出文件
每次训练会在独立文件夹中生成：
```
models/lightgbm_fine_tuned/training_fine_tuned_001/
├── lightgbm_stock_model_fine_tuned.pkl    # 主模型文件
├── scaler.pkl                              # 数据标准化器
├── feature_names.json                      # 详细特征信息
└── feature_names_simple.txt               # 特征列表

results/lightgbm_fine_tuned/training_fine_tuned_001/
├── metrics.json                            # 评估指标
├── predictions.csv                         # 预测结果
├── feature_importance.csv                  # 特征重要性
└── top_features.txt                        # 前20重要特征
```

### 预测使用示例
```python
import joblib
import pandas as pd
import numpy as np

# 加载模型和预处理器
model = joblib.load('models/lightgbm_fine_tuned/training_fine_tuned_001/lightgbm_stock_model_fine_tuned.pkl')
scaler = joblib.load('models/lightgbm_fine_tuned/training_fine_tuned_001/scaler.pkl')

# 加载新数据并预处理
# X_new = your_preprocessing_pipeline(raw_data)
# X_scaled = scaler.transform(X_new)

# 进行预测
# predictions = model.predict(X_scaled)
```

## 📋 数据要求

### 输入数据格式
- **样本数**: 建议 > 10万样本获得最佳性能
- **特征数**: 840个时序窗口特征（30时间步 × 28特征类型）
- **数据格式**: CSV文件，包含 `target` 和 `stock_code` 列

### 特征结构（时序窗口）
```
step_00_个股_开盘价, step_00_个股_收盘价, ..., step_29_个股_收盘价
step_00_行业_开盘价, step_00_行业_收盘价, ..., step_29_行业_收盘价
step_00_指数_上证指数, step_00_指数_沪深300, ..., step_29_指数_中证500
step_00_情绪_sentiment_0, step_00_情绪_sentiment_1, ..., step_29_情绪_sentiment_1
```

## ⚡ 性能优化要点

### 成功的关键因素
1. **异常值处理**: 移除而非调整异常值效果更好
2. **特征选择**: 渐进式选择比激进删减保留更多有用信息
3. **标准化方法**: Standard标准化比Robust更适合股票数据
4. **样本平衡**: 合理的异常值阈值平衡数据质量和数量

### 避免的误区
- ❌ 过度的特征删减（避免从840→200的激进删减）
- ❌ 异常值调整策略（winsorize效果不如移除）
- ❌ 过于复杂的模型参数（容易过拟合）

## 🔧 进一步优化建议

### 短期改进（1-2天）
1. **启用超参数优化**: 使用 `run_final_optimization.py` 
2. **特征工程优化**: 增加更多技术指标
3. **时序验证**: 实施时序分割验证

### 中期改进（1-2周）
1. **模型集成**: 结合多个模型提升稳定性
2. **在线学习**: 支持模型增量更新
3. **特征重要性分析**: 深入分析最重要的特征

### 长期改进（1个月+）
1. **深度学习模型**: 尝试LSTM、Transformer等
2. **多目标预测**: 同时预测涨跌幅和趋势
3. **强化学习**: 结合交易策略优化

## 📞 问题反馈

如果在使用过程中遇到问题：
1. 检查数据格式是否符合要求
2. 确认配置文件路径正确
3. 查看日志文件获取详细错误信息
4. 参考快速对比测试验证环境

---

**最后更新**: 2025-08-10  
**最佳配置**: lightGBM_train_fine_tuned.yaml  
**性能基准**: RMSE 1.7819, MAE 1.2607, R² 0.2790