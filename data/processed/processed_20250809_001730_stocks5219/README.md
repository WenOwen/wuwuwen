# 股票数据处理结果

## 处理信息
- 处理时间: 2025-08-09 00:19:39
- 股票数量: 5219 只
- 样本数量: 2,430,962 个
- 特征维度: 29 维

## 文件说明
- `X_features.npy`: 特征数据，形状 (2430962, 30, 29)
- `y_targets.npy`: 目标数据（收益率），形状 (2430962,)
- `stock_codes.json`: 对应的股票代码列表
- `data_info.json`: 详细的数据信息和元数据
- `README.md`: 本说明文件

## 特征构成
- 个股特征: 16 维 (OHLCV + 技术指标等)
- 行业特征: 6 维 (所属行业的OHLCV等)
- 指数特征: 5 维 (主要市场指数)
- 情绪特征: 2 维 (涨停强度、连板强度)

## 使用方法
```python
import numpy as np
import json

# 加载数据
X = np.load('X_features.npy')
y = np.load('y_targets.npy')

# 加载股票代码
with open('stock_codes.json', 'r', encoding='utf-8') as f:
    stock_codes = json.load(f)
```
