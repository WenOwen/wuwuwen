# -*- coding: utf-8 -*-
# @老师微信:samequant
# @网站:打板哥网 www.dabange.com
# @更多源码下载地址: https://dabange.com/download
# @有偿服务：量化课程、量化数据、策略代写、实盘对接...

"""
专门获取开盘集合竞价数据的测试脚本
集合竞价时间：9:15-9:25
"""

from samequant_functions import Spider_func
import datetime

# 初始化Spider_func实例
s_f_1 = Spider_func()

# 测试股票代码
stock_codes = ['sh600519', 'sz000001', 'sz000002']  # 茅台、平安银行、万科A

print("📊 【开盘集合竞价数据获取测试】")
print("=" * 60)

# 显示当前时间和集合竞价时间段说明
now = datetime.datetime.now()
current_time = now.strftime("%H:%M:%S")
print(f"当前时间: {current_time}")
print()
print("📅 集合竞价时间段说明:")
print("   ⏰ 9:15-9:20: 可以挂单和撤单")
print("   ⏰ 9:20-9:25: 可以挂单但不能撤单")
print("   ⏰ 9:25:00: 集合竞价撮合，确定开盘价")
print("   ⏰ 9:25-9:30: 可以挂单但不能成交")
print()

# 判断当前是否在集合竞价时间段
if "09:15" <= current_time <= "09:25":
    print("🟢 当前处于开盘集合竞价时间段，可获取实时竞价数据")
elif "09:00" <= current_time <= "09:15":
    print("🟡 即将进入集合竞价时间段")
elif "09:25" <= current_time <= "09:30":
    print("🟡 集合竞价结束，即将开盘")
else:
    print("🔴 当前不在集合竞价时间段，将获取历史成交数据")

print()

for i, stock_code in enumerate(stock_codes, 1):
    print(f"【股票 {i}】{stock_code}")
    print("-" * 40)
    
    # 使用新的专门方法获取开盘集合竞价数据
    print("🔥 使用专门的开盘集合竞价方法:")
    try:
        df_opening_auction = s_f_1.get_stock_opening_auction_data_from_eastmoney(stock_code)
        if not df_opening_auction.empty:
            print(f"✅ 开盘集合竞价数据获取成功: {len(df_opening_auction)}条")
            print(f"   数据列数: {len(df_opening_auction.columns)}")
            print(f"   列名: {list(df_opening_auction.columns)}")
            
            # 显示数据类型统计
            if '数据类型' in df_opening_auction.columns:
                data_type_counts = df_opening_auction['数据类型'].value_counts()
                print(f"   数据类型统计: {dict(data_type_counts)}")
            
            print("   前5条数据:")
            print(df_opening_auction.head())
            
            # 如果有集合竞价数据，单独显示
            if '数据类型' in df_opening_auction.columns:
                auction_data = df_opening_auction[df_opening_auction['数据类型'] == '开盘集合竞价']
                if not auction_data.empty:
                    print(f"\n   📈 纯集合竞价数据 ({len(auction_data)}条):")
                    print(auction_data)
        else:
            print("⚠️ 开盘集合竞价数据为空")
    except Exception as e:
        print(f"❌ 开盘集合竞价数据获取失败: {e}")
    
    print()
    
    # 使用通用方法对比
    print("🔄 使用通用集合竞价方法对比:")
    try:
        df_general_auction = s_f_1.get_stock_auction_data_from_eastmoney(stock_code, '开盘集合竞价')
        if not df_general_auction.empty:
            print(f"✅ 通用方法获取成功: {len(df_general_auction)}条")
        else:
            print("⚠️ 通用方法数据为空")
    except Exception as e:
        print(f"❌ 通用方法获取失败: {e}")
    
    print()
    
    # 获取收盘集合竞价数据作为对比
    print("🕐 获取收盘集合竞价数据作为对比:")
    try:
        df_closing_auction = s_f_1.get_stock_auction_data_from_eastmoney(stock_code, '收盘集合竞价')
        if not df_closing_auction.empty:
            print(f"✅ 收盘集合竞价数据: {len(df_closing_auction)}条")
        else:
            print("⚠️ 收盘集合竞价数据为空")
    except Exception as e:
        print(f"❌ 收盘集合竞价获取失败: {e}")
    
    print("=" * 60)
    if i < len(stock_codes):
        print()

print()
print("💡 使用说明:")
print("1. 最佳使用时间：交易日 9:15-9:25")
print("2. 数据包含：竞价时间、价格、成交量、成交额等")
print("3. 自动筛选：只显示 9:15-9:25 时间段的数据")
print("4. 时间提示：自动判断当前是否在集合竞价时间段")
print()
print("🎯 专门方法：get_stock_opening_auction_data_from_eastmoney()")
print("🔧 通用方法：get_stock_auction_data_from_eastmoney(stock_code, '开盘集合竞价')")
print()
print("🎉 开盘集合竞价数据获取测试完成！")