# 测试股票池
TEST_STOCKS = [
    "sh600519",  # 贵州茅台
    "sz000001",  # 平安银行
    "sz000002",  # 万科A
    "sh600036",  # 招商银行
    "sz000858"   # 五粮液
]

# 系统配置
SYSTEM_CONFIG = {
    "sequence_length": 60,
    "prediction_days": [1, 3, 5],
    "min_data_points": 100,
    "batch_size": 32
}
