# -*- coding: utf-8 -*-
# @老师微信:samequant
# @网站:打板哥网 www.dabange.com
# @更多源码下载地址: https://dabange.com/download
# @有偿服务：量化课程、量化数据、策略代写、实盘对接...

from samequant_functions import Spider_func

# 初始化Spider_func实例
s_f_1 = Spider_func()

# 测试获取单只股票的资金流向数据
stock_code = 'sh600519'  # 贵州茅台

print("=== 获取贵州茅台当日资金流向数据 ===")
df_today_flow = s_f_1.get_stock_money_flow_from_eastmoney(stock_code=stock_code, period='当日')
print(f"当日资金流向数据行数: {len(df_today_flow)}")
if not df_today_flow.empty:
    print(df_today_flow.head())
    print()

print("=== 获取贵州茅台历史资金流向数据 ===")
df_hist_flow = s_f_1.get_stock_money_flow_from_eastmoney(stock_code=stock_code, period='历史')
print(f"历史资金流向数据行数: {len(df_hist_flow)}")
if not df_hist_flow.empty:
    print(df_hist_flow.head())
    print()

print("=== 获取全市场个股资金流向排行 ===")
df_stock_flow_rank = s_f_1.get_all_stocks_money_flow_from_eastmoney(flow_type='个股')
print(f"个股资金流向排行数据行数: {len(df_stock_flow_rank)}")
if not df_stock_flow_rank.empty:
    print("主力净流入排行前10:")
    print(df_stock_flow_rank.head(10))
    print()

print("=== 获取行业资金流向排行 ===")
df_industry_flow = s_f_1.get_all_stocks_money_flow_from_eastmoney(flow_type='行业')
print(f"行业资金流向数据行数: {len(df_industry_flow)}")
if not df_industry_flow.empty:
    print("行业资金流向排行前10:")
    print(df_industry_flow.head(10))
    print()

print("=== 获取概念板块资金流向排行 ===")
df_concept_flow = s_f_1.get_all_stocks_money_flow_from_eastmoney(flow_type='概念')
print(f"概念板块资金流向数据行数: {len(df_concept_flow)}")
if not df_concept_flow.empty:
    print("概念板块资金流向排行前10:")
    print(df_concept_flow.head(10))