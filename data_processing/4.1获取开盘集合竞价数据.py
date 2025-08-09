# -*- coding: utf-8 -*-
# @老师微信:samequant
# @网站:涨停客量化zhangtingke.com
# @更多源码下载地址: https://zhangtingke.com/download
# @优化版本：适配新版本 samequant_functions_new.py

"""
专门获取开盘集合竞价数据的测试脚本
集合竞价时间：9:15-9:25
适配新版本接口
"""

import datetime
import pandas as pd
import requests
import time
import random
from samequant_functions_new import get_spider_client, Config

def get_stock_opening_auction_data(stock_code: str) -> pd.DataFrame:
    """
    获取开盘集合竞价数据（适配新版本接口）
    """
    try:
        # 处理股票代码
        if len(stock_code) == 6:
            symbol = stock_code
        elif len(stock_code) == 8:
            symbol = stock_code[2:]
        else:
            return pd.DataFrame()

        if symbol[0] == '6':
            market = '1'
        elif symbol[0] in ['0', '3']:
            market = '0'
        else:
            market = '0'
        secid = market + '.' + symbol

        # 构建东财集合竞价API URL
        url = f'http://push2.eastmoney.com/api/qt/stock/details/get?ut=fa5fd1943c7b386f172d6893dbfba10b&invt=2&fltt=1&secid={secid}&fields1=f1,f2,f3,f4&fields2=f51,f52,f53,f54,f55,f56&pos=-0&_={int(time.time() * 1000)}'
        
        headers = {'User-Agent': random.choice(Config.USER_AGENTS)}
        response = requests.get(url=url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'data' in data and data['data'] and 'details' in data['data']:
                details = data['data']['details']
                
                if details:
                    # 解析数据
                    auction_data = []
                    for detail in details:
                        parts = detail.split(',')
                        if len(parts) >= 6:
                            auction_data.append({
                                '时间': parts[0],
                                '价格': float(parts[1]) if parts[1] != '--' else 0,
                                '成交量': int(parts[2]) if parts[2] != '--' else 0,
                                '成交额': float(parts[3]) if parts[3] != '--' else 0,
                                '性质': parts[4] if len(parts) > 4 else '',
                                '涨跌': float(parts[5]) if len(parts) > 5 and parts[5] != '--' else 0
                            })
                    
                    if auction_data:
                        df = pd.DataFrame(auction_data)
                        
                        # 筛选集合竞价时间段的数据
                        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
                        df['完整时间'] = current_date + ' ' + df['时间']
                        df['完整时间'] = pd.to_datetime(df['完整时间'])
                        
                        # 筛选9:15-9:25的数据
                        morning_start = pd.to_datetime(f"{current_date} 09:15:00")
                        morning_end = pd.to_datetime(f"{current_date} 09:25:59")
                        
                        auction_df = df[(df['完整时间'] >= morning_start) & (df['完整时间'] <= morning_end)]
                        
                        if not auction_df.empty:
                            auction_df['数据类型'] = '开盘集合竞价'
                            return auction_df[['时间', '价格', '成交量', '成交额', '性质', '涨跌', '数据类型']]
                
        return pd.DataFrame()
        
    except Exception as e:
        print(f"获取集合竞价数据异常: {e}")
        return pd.DataFrame()


def get_stock_realtime_quote(stock_code: str) -> pd.DataFrame:
    """
    获取股票实时行情数据（新版本接口）
    """
    try:
        spider = get_spider_client()
        
        # 使用新版本的实时数据获取方法
        df = spider.eastmoney_client.get_stock_realtime_data([stock_code])
        
        if not df.empty:
            # 应用字段映射
            df = spider.rename_dataframe_columns(df, Config.FIELD_MAPPINGS['stock_basic'])
            
            # 转换数值列
            numeric_cols = ['最新价', '涨跌幅', '涨跌额', '成交量', '成交额', '最高价', '最低价', 
                           '开盘价', '总市值', '流通市值', '换手率', '市盈率', '市净率']
            df = spider.convert_numeric_columns(df, numeric_cols)
            
            return df
        
        return pd.DataFrame()
        
    except Exception as e:
        print(f"获取实时行情数据异常: {e}")
        return pd.DataFrame()


def main():
    """主函数"""
    # 测试股票代码
    stock_codes = ['sh600519', 'sz000001', 'sz000002']  # 茅台、平安银行、万科A

    print("📊 【开盘集合竞价数据获取测试】（适配新版本接口）")
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
        
        # 获取集合竞价数据
        print("🔥 获取开盘集合竞价数据:")
        try:
            df_opening_auction = get_stock_opening_auction_data(stock_code)
            if not df_opening_auction.empty:
                print(f"✅ 开盘集合竞价数据获取成功: {len(df_opening_auction)}条")
                print(f"   数据列数: {len(df_opening_auction.columns)}")
                print(f"   列名: {list(df_opening_auction.columns)}")
                
                print("   前5条数据:")
                print(df_opening_auction.head())
                
                # 显示集合竞价统计
                if not df_opening_auction.empty:
                    total_volume = df_opening_auction['成交量'].sum()
                    total_amount = df_opening_auction['成交额'].sum()
                    avg_price = df_opening_auction['价格'].mean()
                    print(f"\n   📈 集合竞价统计:")
                    print(f"      总成交量: {total_volume:,.0f}")
                    print(f"      总成交额: {total_amount:,.2f}")
                    print(f"      平均价格: {avg_price:.2f}")
            else:
                print("⚠️ 开盘集合竞价数据为空")
        except Exception as e:
            print(f"❌ 开盘集合竞价数据获取失败: {e}")
        
        print()
        
        # 获取实时行情数据作为对比
        print("📈 获取实时行情数据:")
        try:
            df_realtime = get_stock_realtime_quote(stock_code)
            if not df_realtime.empty:
                print(f"✅ 实时行情数据获取成功")
                print("   关键数据:")
                if '股票代码' in df_realtime.columns:
                    code = df_realtime['股票代码'].iloc[0] if not df_realtime.empty else 'N/A'
                    name = df_realtime.get('股票名称', pd.Series(['N/A'])).iloc[0]
                    price = df_realtime.get('最新价', pd.Series([0])).iloc[0]
                    change = df_realtime.get('涨跌幅', pd.Series([0])).iloc[0]
                    volume = df_realtime.get('成交量', pd.Series([0])).iloc[0]
                    
                    print(f"      股票: {name} ({code})")
                    print(f"      最新价: {price:.2f}")
                    print(f"      涨跌幅: {change:.2f}%")
                    print(f"      成交量: {volume:,.0f}")
            else:
                print("⚠️ 实时行情数据为空")
        except Exception as e:
            print(f"❌ 实时行情数据获取失败: {e}")
        
        print("=" * 60)
        if i < len(stock_codes):
            print()

    print()
    print("💡 使用说明:")
    print("1. 最佳使用时间：交易日 9:15-9:25")
    print("2. 数据包含：竞价时间、价格、成交量、成交额等")
    print("3. 自动筛选：只显示 9:15-9:25 时间段的数据")
    print("4. 时间提示：自动判断当前是否在集合竞价时间段")
    print("5. 新版本接口：适配 samequant_functions_new.py")
    print()
    print("🎯 集合竞价方法：get_stock_opening_auction_data()")
    print("🔧 实时行情方法：get_stock_realtime_quote()")
    print()
    print("🎉 开盘集合竞价数据获取测试完成！")


if __name__ == '__main__':
    main()