# 股票数据整合处理系统

🚀 一个完整的股票数据整合处理系统，用于将个股、行业、概念和指数数据整合为按交易日分区的Parquet文件。

## 📁 项目结构

```
.
├── data/                           # 数据根目录
│   ├── datas_em/                   # 个股历史数据 (CSV格式)
│   ├── datas_sector_historical/    # 板块历史数据
│   │   ├── 股票板块映射表.csv       # 股票与板块映射关系
│   │   ├── 行业板块_全部历史/       # 行业板块历史数据
│   │   └── 概念板块_全部历史/       # 概念板块历史数据
│   ├── datas_index/                # 指数数据
│   ├── financial_csv/              # 财务数据
│   └── integrated_parquet/         # 输出的整合数据 (Parquet格式)
├── data_integration_processor.py   # 主数据处理脚本
├── run_data_integration.py         # 简化运行脚本
├── data_validator.py               # 数据验证工具
├── data_query_tools.py             # 数据查询工具
├── requirements.txt                # Python依赖包
└── README.md                      # 本文档
```

## 🎯 核心功能

### 1. 数据整合
- ✅ **个股数据**: 保留所有原始特征 (开盘价、收盘价、成交量等)
- ✅ **行业数据**: 自动重命名冲突列为 `{行业名称}_{特征名}` 格式
- ✅ **概念数据**: 使用One-Hot编码处理多概念映射
- ✅ **指数数据**: 合并相关指数信息
- ✅ **映射关系**: 基于股票板块映射表建立关联

### 2. 输出格式
- 📊 **按交易日分区**: 每个文件命名为 `YYYY-MM-DD.parquet`
- 🏷️ **Symbol索引**: 以股票代码为行索引
- 💾 **Parquet格式**: 高效的列式存储，支持快速查询
- 📋 **元数据**: 包含完整的数据统计和列信息

### 3. 数据特征
- 🔢 **个股特征**: 交易日期、OHLCV、涨跌幅、换手率等
- 🏭 **行业特征**: 各行业板块的市场表现数据
- 💡 **概念特征**: 独热编码的概念标识 (`概念_{概念名称}`)
- 📈 **指数特征**: 相关市场指数数据
- 🌍 **基础信息**: 股票名称、所属行业、地区等

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保数据文件按以下结构组织：

```
data/
├── datas_em/                    # 个股CSV文件 (如 sh600000.csv)
├── datas_sector_historical/     
│   ├── 股票板块映射表.csv        # 必需的映射表
│   ├── 行业板块_全部历史/        # 行业板块CSV文件
│   └── 概念板块_全部历史/        # 概念板块CSV文件
└── datas_index/                 # 指数数据 (可选)
```

### 3. 运行处理

```bash
# 方式1: 使用简化脚本
python run_data_integration.py

# 方式2: 直接运行主脚本
python data_integration_processor.py
```

### 4. 验证结果

```bash
# 验证数据质量
python data_validator.py

# 测试查询功能
python data_query_tools.py
```

## 📊 数据使用示例

### 加载单日数据

```python
import pandas as pd

# 加载2024年1月1日的数据
df = pd.read_parquet('data/integrated_parquet/2024-01-01.parquet')

# 查看数据结构
print(f"股票数量: {len(df)}")
print(f"特征数量: {len(df.columns)}")
print(df.head())
```

### 使用查询工具

```python
from data_query_tools import DataQueryTools

# 创建查询工具
query = DataQueryTools()

# 获取股票历史数据
stock_data = query.get_stock_history('sh600000', '2024-01-01', '2024-01-31')

# 搜索新能源概念股票
new_energy_stocks = query.search_stocks_by_concept('新能源车', '2024-01-01')

# 搜索银行业股票
bank_stocks = query.search_stocks_by_industry('银行', '2024-01-01')

# 获取市场统计
stats = query.get_summary_statistics('2024-01-01')
print(stats)
```

### 时间序列分析

```python
import pandas as pd
from pathlib import Path

# 加载多日数据进行时间序列分析
data_dir = Path('data/integrated_parquet')
files = sorted(data_dir.glob('2024-01-*.parquet'))

all_data = []
for file in files:
    df = pd.read_parquet(file)
    df['交易日期'] = file.stem  # 添加日期列
    all_data.append(df)

# 合并数据
time_series_data = pd.concat(all_data, ignore_index=True)

# 分析特定股票的时间序列
stock_ts = time_series_data[time_series_data.index == 'sh600000']
print(stock_ts[['交易日期', '收盘价', '涨跌幅']].head())
```

## 🔧 高级配置

### 自定义处理参数

```python
from data_integration_processor import DataIntegrationProcessor

# 创建自定义处理器
processor = DataIntegrationProcessor(data_root='your_data_path')

# 自定义概念处理
processor.concept_columns = ['自定义概念1', '自定义概念2']

# 运行处理
processor.run_integration()
```

### 增量更新

```python
# 只处理新增的交易日数据
# 适用于日常数据更新场景

processor = DataIntegrationProcessor()
# 可以通过修改 load_stock_data 方法来实现增量处理
```

## 📋 数据结构说明

### 输出Parquet文件结构

| 列名类型 | 示例 | 说明 |
|---------|------|------|
| 基础信息 | `股票名称`, `所属行业`, `地区` | 股票基本信息 |
| 交易数据 | `开盘价`, `收盘价`, `成交量`, `涨跌幅` | 个股交易特征 |
| 概念标识 | `概念_新能源车`, `概念_人工智能` | One-Hot编码的概念标识 |
| 行业数据 | `银行_开盘价`, `银行_成交量` | 所属行业的市场数据 |
| 概念数据 | `新能源车板块_开盘价` | 相关概念板块的市场数据 |
| 指数数据 | `上证指数_收盘价` | 相关市场指数数据 |

### 映射表格式

股票板块映射表 (`股票板块映射表.csv`) 的格式：

```csv
股票代码,股票名称,所属行业,概念板块,地区
sh600000,浦发银行,银行,"HS300_,转债标的,机构重仓",上海
```

## 🛠️ 故障排查

### 常见问题

1. **内存不足**
   - 概念数据文件过多时可能导致内存问题
   - 解决：增加系统内存或分批处理概念数据

2. **文件编码问题**
   - CSV文件编码不是UTF-8
   - 解决：检查并转换文件编码

3. **日期格式问题**
   - 数据文件中日期格式不一致
   - 解决：检查并统一日期格式

4. **缺失依赖**
   ```bash
   pip install pandas pyarrow numpy tqdm
   ```

### 日志信息

处理过程中会显示详细的进度信息：
- ✅ 成功操作
- ⚠️ 警告信息  
- ❌ 错误信息

## 📈 性能优化

### 建议的系统配置

- **内存**: 16GB+ (处理大量概念数据时)
- **存储**: SSD推荐 (提高I/O性能)
- **CPU**: 多核处理器 (并行处理)

### 优化建议

1. **分批处理**: 对于大量数据，可以分批次处理
2. **并行化**: 可以并行处理不同日期的数据
3. **压缩**: Parquet文件使用Snappy压缩，平衡性能和大小

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License

---

💡 **提示**: 首次运行可能需要较长时间，特别是处理概念数据时。建议在服务器或性能较好的机器上运行。

🔍 **问题反馈**: 如果遇到任何问题，请查看控制台输出的详细错误信息，并检查数据文件的格式和完整性。