# -*- coding: utf-8 -*-
"""
测试修改后的集合竞价数据获取功能
验证能否正确获取9:15-9:25期间的真实成交量
"""

import sys
import os
import importlib.util
import datetime

# 动态导入带中文名的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '6.1获取历史竞价数据.py')

spec = importlib.util.spec_from_file_location("auction_module", module_path)
auction_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auction_module)

# 获取类
HistoricalAuctionData = auction_module.HistoricalAuctionData

def test_auction_data():
    """测试集合竞价数据获取"""
    print("🔍 测试集合竞价数据获取功能")
    print("=" * 60)
    
    # 初始化
    auction_tool = HistoricalAuctionData()
    
    # 测试股票
    test_stock = 'sh600519'  # 茅台
    
    # 设置日期范围（最近几个交易日）
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=5)).strftime('%Y-%m-%d')
    
    print(f"📊 测试股票: {test_stock}")
    print(f"📅 日期范围: {start_date} 至 {end_date}")
    print()
    
    # 获取集合竞价数据
    df = auction_tool.get_historical_auction_data(
        stock_code=test_stock,
        start_date=start_date,
        end_date=end_date
    )
    
    if not df.empty:
        print("✅ 数据获取成功！")
        print(f"📈 获取到 {len(df)} 条记录")
        print()
        
        # 显示关键字段
        print("🔍 关键数据字段检查:")
        print("="*60)
        
        for idx, row in df.head().iterrows():
            print(f"日期: {row['日期']}")
            print(f"  开盘竞价价格: {row['开盘竞价价格']}")
            print(f"  昨日收盘价: {row['昨日收盘价']}")
            print(f"  竞价涨跌幅: {row['竞价涨跌幅(%)']}%")
            print(f"  集合竞价成交量: {row['集合竞价成交量']:,}")
            print(f"  集合竞价成交额: {row['集合竞价成交额']:,.2f}")
            print(f"  全日成交量: {row['全日成交量']:,}")
            print(f"  全日成交额: {row['全日成交额']:,.2f}")
            print(f"  集合竞价成交量 vs 全日成交量 占比: {(row['集合竞价成交量'] / row['全日成交量'] * 100):.2f}%" if row['全日成交量'] > 0 else "N/A")
            print("-" * 40)
        
        # 统计分析
        print("\n📊 数据统计分析:")
        print("="*60)
        
        # 集合竞价成交量统计
        auction_volumes = df[df['集合竞价成交量'] > 0]['集合竞价成交量']
        if not auction_volumes.empty:
            print(f"有集合竞价成交的天数: {len(auction_volumes)} / {len(df)}")
            print(f"集合竞价平均成交量: {auction_volumes.mean():,.0f}")
            print(f"集合竞价最大成交量: {auction_volumes.max():,.0f}")
            print(f"集合竞价最小成交量: {auction_volumes.min():,.0f}")
        else:
            print("⚠️ 所有日期的集合竞价成交量都为0")
        
        # 集合竞价成交额统计
        auction_amounts = df[df['集合竞价成交额'] > 0]['集合竞价成交额']
        if not auction_amounts.empty:
            print(f"集合竞价平均成交额: {auction_amounts.mean():,.2f}")
            print(f"集合竞价最大成交额: {auction_amounts.max():,.2f}")
        else:
            print("⚠️ 所有日期的集合竞价成交额都为0")
        
        # 竞价涨跌幅统计
        print(f"竞价涨跌幅 - 平均: {df['竞价涨跌幅(%)'].mean():.2f}%")
        print(f"竞价涨跌幅 - 最大: {df['竞价涨跌幅(%)'].max():.2f}%")
        print(f"竞价涨跌幅 - 最小: {df['竞价涨跌幅(%)'].min():.2f}%")
        
        # 数据质量检查
        print("\n🔍 数据质量检查:")
        print("="*60)
        
        zero_auction_volume_count = len(df[df['集合竞价成交量'] == 0])
        zero_auction_amount_count = len(df[df['集合竞价成交额'] == 0])
        
        print(f"集合竞价成交量为0的记录: {zero_auction_volume_count} / {len(df)}")
        print(f"集合竞价成交额为0的记录: {zero_auction_amount_count} / {len(df)}")
        
        if zero_auction_volume_count == len(df):
            print("❌ 所有记录的集合竞价成交量都为0，可能存在数据获取问题")
        elif zero_auction_volume_count > len(df) * 0.8:
            print("⚠️ 大部分记录的集合竞价成交量为0，建议检查数据源")
        else:
            print("✅ 数据质量正常，成功获取到集合竞价成交量")
        
        # 保存测试结果
        save_path = auction_tool.save_auction_data(df, test_stock, '修改测试')
        print(f"\n💾 测试数据已保存到: {save_path}")
        
        # 进行分析
        print("\n📈 数据分析:")
        print("="*60)
        analysis = auction_tool.analyze_auction_patterns(df)
        for key, value in analysis.items():
            print(f"{key}: {value}")
        
    else:
        print("❌ 数据获取失败")
    
    print("\n" + "="*60)
    print("🎉 测试完成")

if __name__ == "__main__":
    test_auction_data()