#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试板块历史数据获取功能
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

try:
    from core.samequant_functions import Spider_func
    print("✅ 成功导入 Spider_func")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def test_single_sector_historical():
    """测试单个板块历史数据获取"""
    print("\n🧪 测试单个板块历史数据获取...")
    
    spider = Spider_func()
    
    # 测试获取证券板块最近10天的数据
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    
    print(f"获取日期范围: {start_date} 至 {end_date}")
    
    try:
        # 测试是否有新增的历史数据获取函数
        if hasattr(spider, 'get_historical_sector_data_from_eastmoney'):
            print("✅ 找到历史数据获取函数")
            
            # 获取证券板块历史数据
            df = spider.get_historical_sector_data_from_eastmoney(
                sector_code='BK0473',  # 证券板块
                sector_type='industry',
                start_date=start_date,
                end_date=end_date,
                period='daily'
            )
            
            if not df.empty:
                print(f"✅ 成功获取历史数据 {len(df)} 条")
                print("数据样例:")
                print(df.head(3))
                return True
            else:
                print("⚠️ 获取的数据为空")
                return False
        else:
            print("❌ 未找到历史数据获取函数")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_fund_flow_history():
    """测试资金流向历史数据获取"""
    print("\n🧪 测试资金流向历史数据获取...")
    
    spider = Spider_func()
    
    try:
        if hasattr(spider, 'get_sector_fund_flow_history'):
            print("✅ 找到资金流向历史数据获取函数")
            
            # 获取证券板块资金流向数据
            df = spider.get_sector_fund_flow_history(
                sector_code='BK0473',  # 证券板块
                sector_type='industry',
                days=5
            )
            
            if not df.empty:
                print(f"✅ 成功获取资金流向数据 {len(df)} 条")
                print("数据样例:")
                print(df[['日期', '主力净流入', '主力净流入占比']].head(3))
                return True
            else:
                print("⚠️ 获取的资金流向数据为空")
                return False
        else:
            print("❌ 未找到资金流向历史数据获取函数")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_current_functions():
    """测试现有的当前数据获取功能"""
    print("\n🧪 测试现有功能...")
    
    spider = Spider_func()
    
    try:
        # 测试获取行业板块当前数据
        industry_df = spider.get_industry_data_from_eastmoney()
        if not industry_df.empty:
            print(f"✅ 获取行业板块数据成功: {len(industry_df)} 个板块")
        else:
            print("⚠️ 行业板块数据为空")
        
        # 测试获取概念板块当前数据
        concept_df = spider.get_concept_data_from_eastmoney()
        if not concept_df.empty:
            print(f"✅ 获取概念板块数据成功: {len(concept_df)} 个板块")
        else:
            print("⚠️ 概念板块数据为空")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试现有功能失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 板块历史数据获取功能测试")
    print("=" * 50)
    
    # 测试现有功能
    result1 = test_current_functions()
    
    # 测试新功能
    result2 = test_single_sector_historical()
    result3 = test_fund_flow_history()
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    print(f"现有功能: {'✅ 通过' if result1 else '❌ 失败'}")
    print(f"历史数据获取: {'✅ 通过' if result2 else '❌ 失败'}")
    print(f"资金流向历史: {'✅ 通过' if result3 else '❌ 失败'}")
    
    if all([result1, result2, result3]):
        print("\n🎉 所有测试通过！板块历史数据获取功能已成功集成")
    else:
        print("\n⚠️ 部分测试失败，请检查相关功能")
    
    print("\n💡 使用提示:")
    print("1. 运行 'python data_processing/获取板块历史数据.py' 使用专用脚本")
    print("2. 运行 'python data_processing/获取板块数据并保存CSV.py' 选择菜单项2")
    print("3. 查看 'docs/板块历史数据获取使用指南.md' 了解详细用法")

if __name__ == "__main__":
    main()