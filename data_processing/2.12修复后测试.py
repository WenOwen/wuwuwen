# -*- coding: utf-8 -*-
# @老师微信:samequant
# @网站:打板哥网 www.dabange.com
# @更多源码下载地址: https://dabange.com/download
# @有偿服务：量化课程、量化数据、策略代写、实盘对接...

"""
修复错误后的测试脚本
主要测试之前出现错误的功能
"""

from samequant_functions import Spider_func
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# 初始化Spider_func实例
s_f_1 = Spider_func()

# 测试股票代码
stock_code = 'sh600519'  # 贵州茅台

print("🔧 【错误修复测试】")
print("=" * 50)

print("\n1. 测试资金流向数据获取（修复JSON解析错误）")
print("-" * 40)

# 测试当日资金流向
try:
    df_flow_today = s_f_1.get_stock_money_flow_from_eastmoney(stock_code, '当日')
    if not df_flow_today.empty:
        print(f"✅ 当日资金流向数据获取成功: {len(df_flow_today)}条")
        print("前3条数据:")
        print(df_flow_today.head(3))
    else:
        print("⚠️ 当日资金流向数据为空（可能是非交易时间）")
except Exception as e:
    print(f"❌ 当日资金流向获取失败: {e}")

print()

# 测试历史资金流向
try:
    df_flow_hist = s_f_1.get_stock_money_flow_from_eastmoney(stock_code, '历史')
    if not df_flow_hist.empty:
        print(f"✅ 历史资金流向数据获取成功: {len(df_flow_hist)}条")
        print("最新3条数据:")
        print(df_flow_hist.head(3))
    else:
        print("⚠️ 历史资金流向数据为空")
except Exception as e:
    print(f"❌ 历史资金流向获取失败: {e}")

print()

print("2. 测试集合竞价数据获取（修复列数不匹配错误）")
print("-" * 40)

# 测试开盘集合竞价数据（新功能）
try:
    df_opening_auction = s_f_1.get_stock_opening_auction_data_from_eastmoney(stock_code)
    if not df_opening_auction.empty:
        print(f"✅ 开盘集合竞价数据获取成功: {len(df_opening_auction)}条")
        print(f"数据列数: {len(df_opening_auction.columns)}")
        print("列名:", list(df_opening_auction.columns))
        print("前3条数据:")
        print(df_opening_auction.head(3))
    else:
        print("⚠️ 开盘集合竞价数据为空（可能是非交易时间）")
except Exception as e:
    print(f"❌ 开盘集合竞价数据获取失败: {e}")

print()

# 测试收盘集合竞价数据
try:
    df_closing_auction = s_f_1.get_stock_auction_data_from_eastmoney(stock_code, '收盘集合竞价')
    if not df_closing_auction.empty:
        print(f"✅ 收盘集合竞价数据获取成功: {len(df_closing_auction)}条")
        print(f"数据列数: {len(df_closing_auction.columns)}")
        print("列名:", list(df_closing_auction.columns))
        print("前3条数据:")
        print(df_closing_auction.head(3))
    else:
        print("⚠️ 收盘集合竞价数据为空（可能是非交易时间）")
except Exception as e:
    print(f"❌ 收盘集合竞价数据获取失败: {e}")

print()

# 测试分时成交数据
try:
    df_trade_detail = s_f_1.get_stock_auction_data_from_eastmoney(stock_code, '分时成交')
    if not df_trade_detail.empty:
        print(f"✅ 分时成交数据获取成功: {len(df_trade_detail)}条")
        print(f"数据列数: {len(df_trade_detail.columns)}")
        print("列名:", list(df_trade_detail.columns))
        print("前3条数据:")
        print(df_trade_detail.head(3))
    else:
        print("⚠️ 分时成交数据为空（可能是非交易时间）")
except Exception as e:
    print(f"❌ 分时成交数据获取失败: {e}")

print()

print("3. 测试正常功能（验证没有被破坏）")
print("-" * 40)

# 测试技术指标
try:
    df_macd = s_f_1.get_stock_technical_indicators_from_eastmoney(stock_code, 'MACD')
    if not df_macd.empty:
        print(f"✅ MACD技术指标正常: {len(df_macd)}条")
    else:
        print("❌ MACD技术指标获取失败")
except Exception as e:
    print(f"❌ MACD技术指标出错: {e}")

# 测试实时数据
try:
    df_realtime = s_f_1.get_stock_realtime_data_from_eastmoney(stock_code)
    if not df_realtime.empty:
        print(f"✅ 实时数据正常: {len(df_realtime.columns)}个字段")
    else:
        print("❌ 实时数据获取失败")
except Exception as e:
    print(f"❌ 实时数据出错: {e}")

# 测试五档买卖
try:
    df_bid_ask = s_f_1.get_stock_bid_ask_data_from_eastmoney(stock_code)
    if not df_bid_ask.empty:
        print(f"✅ 五档买卖数据正常: {len(df_bid_ask.columns)}个字段")
    else:
        print("❌ 五档买卖数据获取失败")
except Exception as e:
    print(f"❌ 五档买卖数据出错: {e}")

# 测试板块数据
try:
    df_industry = s_f_1.get_industry_data_from_eastmoney()
    if not df_industry.empty:
        print(f"✅ 行业板块数据正常: {len(df_industry)}个行业")
    else:
        print("❌ 行业板块数据获取失败")
except Exception as e:
    print(f"❌ 行业板块数据出错: {e}")

print()
print("=" * 50)
print("🎉 修复测试完成！")
print("=" * 50)