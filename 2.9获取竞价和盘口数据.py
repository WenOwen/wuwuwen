from samequant_functions import Spider_func

# 初始化Spider_func实例
s_f_1 = Spider_func()

# 测试股票代码
stock_code = 'sh600519'  # 贵州茅台

print("=== 获取五档买卖盘口数据 ===")
df_bid_ask = s_f_1.get_stock_bid_ask_data_from_eastmoney(stock_code=stock_code)
if not df_bid_ask.empty:
    print("五档买卖盘口数据:")
    print(df_bid_ask.T)  # 转置显示
    print()

print("=== 获取当日分时数据 ===")
df_minute_1day = s_f_1.get_stock_minute_data_from_eastmoney(stock_code=stock_code, ndays=1)
if not df_minute_1day.empty:
    print(f"当日分时数据行数: {len(df_minute_1day)}")
    print("最新10条分时数据:")
    print(df_minute_1day.tail(10))
    print()

print("=== 获取5日分时数据 ===")
df_minute_5day = s_f_1.get_stock_minute_data_from_eastmoney(stock_code=stock_code, ndays=5)
if not df_minute_5day.empty:
    print(f"5日分时数据行数: {len(df_minute_5day)}")
    print("最新10条分时数据:")
    print(df_minute_5day.tail(10))
    print()

print("=== 获取开盘集合竞价数据（专门方法）===")
df_opening_auction = s_f_1.get_stock_opening_auction_data_from_eastmoney(stock_code=stock_code)
if not df_opening_auction.empty:
    print(f"开盘集合竞价数据行数: {len(df_opening_auction)}")
    print("开盘竞价数据:")
    print(df_opening_auction.head(10))
    print()

print("=== 获取开盘集合竞价数据（通用方法）===")
df_auction = s_f_1.get_stock_auction_data_from_eastmoney(stock_code=stock_code, auction_type='开盘集合竞价')
if not df_auction.empty:
    print(f"开盘集合竞价数据行数: {len(df_auction)}")
    print("竞价数据:")
    print(df_auction.head(10))
    print()

print("=== 获取收盘集合竞价数据 ===")
df_closing_auction = s_f_1.get_stock_auction_data_from_eastmoney(stock_code=stock_code, auction_type='收盘集合竞价')
if not df_closing_auction.empty:
    print(f"收盘集合竞价数据行数: {len(df_closing_auction)}")
    print("收盘竞价数据:")
    print(df_closing_auction.head(10))
    print()

print("=== 获取分时成交明细 ===")
df_trade_detail = s_f_1.get_stock_auction_data_from_eastmoney(stock_code=stock_code, auction_type='分时成交')
if not df_trade_detail.empty:
    print(f"分时成交明细行数: {len(df_trade_detail)}")
    print("最新10条成交明细:")
    print(df_trade_detail.head(10))
    print()

# 测试另一个股票
stock_code_2 = 'sz000001'  # 平安银行

print(f"=== 获取{stock_code_2}的五档买卖数据 ===")
df_bid_ask_2 = s_f_1.get_stock_bid_ask_data_from_eastmoney(stock_code=stock_code_2)
if not df_bid_ask_2.empty:
    print("平安银行五档买卖盘口数据:")
    print(df_bid_ask_2.T)
    print()