# -*- coding: utf-8 -*-
"""
调试板块数据整合问题
"""

import pandas as pd
import os

def debug_integration():
    sector_mapping_file = "/home/wangkai/6tdisk/wht/wuwuwen/data/datas_sector/股票板块映射表.csv"
    
    print("🔍 调试板块数据整合...")
    
    # 读取原始数据
    df_mapping = pd.read_csv(sector_mapping_file, encoding='utf-8-sig')
    print(f"原始数据列: {df_mapping.columns.tolist()}")
    print(f"原始数据形状: {df_mapping.shape}")
    
    # 创建测试额外数据
    extra_rows = []
    extra_stocks = ['test001', 'test002']
    
    for stock_code in extra_stocks:
        extra_rows.append({
            'stock_code': stock_code,
            'stock_name': f'股票{stock_code}',
            'industry': '测试行业',
            'primary_concept': '测试概念',
            'all_concepts': '测试概念',
            'region': '测试地区'
        })
    
    df_extra = pd.DataFrame(extra_rows)
    print(f"额外数据列: {df_extra.columns.tolist()}")
    print(f"额外数据形状: {df_extra.shape}")
    
    # 合并数据
    df_combined = pd.concat([df_mapping, df_extra], ignore_index=True)
    print(f"合并后列: {df_combined.columns.tolist()}")
    print(f"合并后形状: {df_combined.shape}")
    print(f"列名重复检查: {df_combined.columns.duplicated().any()}")
    
    # 检查industry列
    if 'industry' in df_combined.columns:
        print(f"industry列存在")
        try:
            industry_stats = df_combined['industry'].value_counts()
            print(f"行业统计成功: {len(industry_stats)}个行业")
        except Exception as e:
            print(f"行业统计失败: {e}")
    
    if '所属行业' in df_combined.columns:
        print(f"所属行业列存在")
        try:
            industry_stats = df_combined['所属行业'].value_counts()
            print(f"所属行业统计成功: {len(industry_stats)}个行业")
        except Exception as e:
            print(f"所属行业统计失败: {e}")

if __name__ == "__main__":
    debug_integration()