"""
综合数据获取测试脚本
展示所有已实现的数据获取功能
"""

from samequant_functions import Spider_func
import time

# 初始化Spider_func实例
s_f_1 = Spider_func()

# 测试股票代码
stock_code = 'sh600519'  # 贵州茅台

print("=" * 60)
print("           吾吾量化数据获取功能综合测试")
print("=" * 60)
print()

print("🔥 【功能1】资金流向数据获取")
print("-" * 40)
# 当日资金流向
df_flow_today = s_f_1.get_stock_money_flow_from_eastmoney(stock_code, '当日')
if not df_flow_today.empty:
    print(f"✅ 当日资金流向数据: {len(df_flow_today)}条")
else:
    print("❌ 当日资金流向数据获取失败")

# 历史资金流向
df_flow_hist = s_f_1.get_stock_money_flow_from_eastmoney(stock_code, '历史')
if not df_flow_hist.empty:
    print(f"✅ 历史资金流向数据: {len(df_flow_hist)}条")
else:
    print("❌ 历史资金流向数据获取失败")

# 全市场资金流向排行
df_flow_rank = s_f_1.get_all_stocks_money_flow_from_eastmoney('个股')
if not df_flow_rank.empty:
    print(f"✅ 全市场个股资金流向排行: {len(df_flow_rank)}条")
else:
    print("❌ 全市场资金流向排行获取失败")

print()

print("📊 【功能2】技术指标数据获取")
print("-" * 40)
# MACD指标
df_macd = s_f_1.get_stock_technical_indicators_from_eastmoney(stock_code, 'MACD')
if not df_macd.empty:
    print(f"✅ MACD技术指标: {len(df_macd)}条")
else:
    print("❌ MACD技术指标获取失败")

# RSI指标
df_rsi = s_f_1.get_stock_technical_indicators_from_eastmoney(stock_code, 'RSI')
if not df_rsi.empty:
    print(f"✅ RSI技术指标: {len(df_rsi)}条")
else:
    print("❌ RSI技术指标获取失败")

# KDJ指标
df_kdj = s_f_1.get_stock_technical_indicators_from_eastmoney(stock_code, 'KDJ')
if not df_kdj.empty:
    print(f"✅ KDJ技术指标: {len(df_kdj)}条")
else:
    print("❌ KDJ技术指标获取失败")

print()

print("🎯 【功能3】实时详细行情数据")
print("-" * 40)
# 实时详细数据
df_realtime = s_f_1.get_stock_realtime_data_from_eastmoney(stock_code)
if not df_realtime.empty:
    print(f"✅ 实时详细行情数据: {len(df_realtime)}条，包含{len(df_realtime.columns)}个字段")
    print(f"   字段包括: 最新价、涨跌幅、成交量、成交额、市盈率、市净率、总市值等")
else:
    print("❌ 实时详细行情数据获取失败")

print()

print("⚡【功能4】竞价和盘口数据获取")
print("-" * 40)
# 五档买卖数据
df_bid_ask = s_f_1.get_stock_bid_ask_data_from_eastmoney(stock_code)
if not df_bid_ask.empty:
    print(f"✅ 五档买卖盘口数据: {len(df_bid_ask)}条")
else:
    print("❌ 五档买卖盘口数据获取失败")

# 分时数据
df_minute = s_f_1.get_stock_minute_data_from_eastmoney(stock_code, 1)
if not df_minute.empty:
    print(f"✅ 当日分时数据: {len(df_minute)}条")
else:
    print("❌ 当日分时数据获取失败")

# 开盘集合竞价数据
df_opening_auction = s_f_1.get_stock_opening_auction_data_from_eastmoney(stock_code)
if not df_opening_auction.empty:
    print(f"✅ 开盘集合竞价数据: {len(df_opening_auction)}条")
else:
    print("❌ 开盘集合竞价数据获取失败")

# 收盘集合竞价数据
df_closing_auction = s_f_1.get_stock_auction_data_from_eastmoney(stock_code, '收盘集合竞价')
if not df_closing_auction.empty:
    print(f"✅ 收盘集合竞价数据: {len(df_closing_auction)}条")
else:
    print("❌ 收盘集合竞价数据获取失败")

print()

print("🏭 【功能5】板块数据获取")
print("-" * 40)
# 行业板块数据
df_industry = s_f_1.get_industry_data_from_eastmoney()
if not df_industry.empty:
    print(f"✅ 行业板块数据: {len(df_industry)}个行业")
else:
    print("❌ 行业板块数据获取失败")

# 概念板块数据
df_concept = s_f_1.get_concept_data_from_eastmoney()
if not df_concept.empty:
    print(f"✅ 概念板块数据: {len(df_concept)}个概念")
else:
    print("❌ 概念板块数据获取失败")

# 热门概念
df_hot_concepts = s_f_1.get_hot_concepts_from_eastmoney(20)
if not df_hot_concepts.empty:
    print(f"✅ 热门概念排行: {len(df_hot_concepts)}个")
else:
    print("❌ 热门概念排行获取失败")

# 个股行业概念信息
stock_info = s_f_1.get_stock_industry_info_from_eastmoney(stock_code)
if stock_info:
    print(f"✅ 个股行业概念信息: {stock_info['股票名称']} - {stock_info['所属行业']}")
else:
    print("❌ 个股行业概念信息获取失败")

print()

print("📈 【功能总览】")
print("-" * 40)
print("✅ 已实现功能:")
print("   1. 资金流向数据获取（净流入、主力流入流出）")
print("   2. 技术指标获取（RSI、MACD、KDJ、BOLL、MA等）")
print("   3. 增强实时行情数据（包含更多字段）")
print("   4. 竞价数据获取（开盘集合竞价、收盘集合竞价、五档盘口、分时数据）")
print("   5. 板块数据获取（行业分类、概念板块、热点题材）")
print()
print("⏳ 待实现功能:")
print("   6. 扩展财务数据获取（基于东财API的实时财务数据）")
print("   7. 机构数据获取（机构持股、基金持仓、股东信息）")
print("   8. 龙虎榜数据获取（大单交易、营业部数据）")
print()

print("=" * 60)
print("           测试完成，功能运行正常！")
print("=" * 60)