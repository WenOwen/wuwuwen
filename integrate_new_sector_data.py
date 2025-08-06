# -*- coding: utf-8 -*-
"""
新板块数据整合脚本
将股票板块映射表.csv与datas_em中的股票数据整合，并生成训练用的板块映射文件
"""

import pandas as pd
import os
import json
from typing import Dict, List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrate_sector_data():
    """整合新的板块数据"""
    
    # 数据路径
    sector_mapping_file = "/home/wangkai/6tdisk/wht/wuwuwen/data/datas_sector/股票板块映射表.csv"
    datas_em_dir = "/home/wangkai/6tdisk/wht/wuwuwen/data/datas_em"
    
    # 其他板块数据文件
    industry_data_file = "/home/wangkai/6tdisk/wht/wuwuwen/data/datas_sector/行业板块数据.csv"
    concept_data_file = "/home/wangkai/6tdisk/wht/wuwuwen/data/datas_sector/概念板块数据.csv"
    hot_concepts_file = "/home/wangkai/6tdisk/wht/wuwuwen/data/datas_sector/热门概念排行.csv"
    
    # 输出路径
    output_dir = "/home/wangkai/6tdisk/wht/wuwuwen/data"
    
    logger.info("🚀 开始整合新的板块数据...")
    
    # 1. 读取股票板块映射表
    logger.info("📊 读取股票板块映射表...")
    df_mapping = pd.read_csv(sector_mapping_file, encoding='utf-8-sig')
    logger.info(f"✅ 板块映射表: {len(df_mapping)}只股票")
    
    # 2. 检查datas_em中对应的股票数据，并找到额外的股票
    logger.info("📊 检查datas_em中的股票数据...")
    available_stocks = []
    missing_stocks = []
    extra_stocks = []
    
    # 检查映射表中的股票
    mapping_stocks = set(df_mapping['股票代码'].values)
    
    for _, row in df_mapping.iterrows():
        stock_code = row['股票代码']
        stock_file = os.path.join(datas_em_dir, f"{stock_code}.csv")
        
        if os.path.exists(stock_file):
            available_stocks.append(stock_code)
        else:
            missing_stocks.append(stock_code)
    
    # 检查datas_em中是否有额外的股票
    if os.path.exists(datas_em_dir):
        all_files = [f for f in os.listdir(datas_em_dir) if f.endswith('.csv')]
        all_stock_codes = [f.replace('.csv', '') for f in all_files]
        
        for stock_code in all_stock_codes:
            if stock_code not in mapping_stocks:
                extra_stocks.append(stock_code)
    
    logger.info(f"✅ 映射表中有数据的股票: {len(available_stocks)}只")
    logger.info(f"📊 datas_em中额外的股票: {len(extra_stocks)}只")
    if missing_stocks:
        logger.warning(f"⚠️  映射表中缺失数据的股票: {len(missing_stocks)}只")
        logger.warning(f"缺失股票列表: {missing_stocks[:10]}...")  # 只显示前10只
    if extra_stocks:
        logger.info(f"📋 额外股票示例: {extra_stocks[:10]}...")  # 只显示前10只
    
    # 3. 创建训练用的板块映射文件
    logger.info("📊 创建训练用的板块映射文件...")
    
    # 首先重命名原有数据的列
    df_mapping_renamed = df_mapping.rename(columns={
        '股票代码': 'stock_code',
        '股票名称': 'stock_name', 
        '所属行业': 'industry',
        '概念板块': 'all_concepts',
        '地区': 'region'
    })
    
    # 处理概念板块数据 - 提取主要概念
    def extract_primary_concept(concepts_str):
        if pd.isna(concepts_str) or concepts_str == '':
            return '无概念'
        concepts = concepts_str.split(',')
        # 过滤掉一些通用概念，保留特色概念
        filtered_concepts = [c.strip() for c in concepts 
                           if c.strip() not in ['融资融券', '沪股通', '富时罗素', '标准普尔', 'MSCI中国']]
        return filtered_concepts[0] if filtered_concepts else concepts[0].strip()
    
    # 添加主要概念列
    df_mapping_renamed['primary_concept'] = df_mapping_renamed['all_concepts'].apply(extract_primary_concept)
    
    # 过滤出有数据的股票
    df_training = df_mapping_renamed[df_mapping_renamed['stock_code'].isin(available_stocks)].copy()
    
    # 4. 为额外的股票创建默认映射
    if extra_stocks:
        logger.info(f"📊 为{len(extra_stocks)}只额外股票创建默认板块映射...")
        
        extra_rows = []
        for stock_code in extra_stocks:
            # 根据股票代码推断基本信息
            if stock_code.startswith('sh'):
                if stock_code.startswith('sh688'):
                    industry = '科创板'
                    primary_concept = '科创板'
                elif stock_code.startswith('sh601'):
                    industry = '大盘股'
                    primary_concept = '蓝筹股'
                else:
                    industry = '沪市主板'
                    primary_concept = '主板'
                region = '上海'
            elif stock_code.startswith('sz'):
                if stock_code.startswith('sz300'):
                    industry = '创业板'
                    primary_concept = '创业板'
                elif stock_code.startswith('sz301'):
                    industry = '创业板'
                    primary_concept = '创业板注册制'
                elif stock_code.startswith('sz002'):
                    industry = '中小板'
                    primary_concept = '中小板'
                else:
                    industry = '深市主板'
                    primary_concept = '主板'
                region = '深圳'
            else:
                industry = '其他'
                primary_concept = '其他'
                region = '未知'
            
            extra_rows.append({
                'stock_code': stock_code,
                'stock_name': f'股票{stock_code}',
                'industry': industry,
                'primary_concept': primary_concept,
                'all_concepts': primary_concept,
                'region': region
            })
        
        # 添加额外股票到训练数据
        df_extra = pd.DataFrame(extra_rows)
        df_training = pd.concat([df_training, df_extra], ignore_index=True)
        
        logger.info(f"✅ 已添加{len(extra_stocks)}只额外股票的默认映射")
    
    # 保存训练用映射文件
    training_file = os.path.join(output_dir, "股票板块映射_训练用.csv")
    df_training[['stock_code', 'stock_name', 'industry', 'primary_concept', 'all_concepts', 'region']].to_csv(
        training_file, index=False, encoding='utf-8-sig'
    )
    logger.info(f"✅ 训练用板块映射已保存: {training_file}")
    
    # 4. 读取并整合其他板块数据（用作特征）
    logger.info("📊 整合其他板块数据作为特征...")
    
    additional_features = {}
    
    # 行业板块数据
    if os.path.exists(industry_data_file):
        df_industry = pd.read_csv(industry_data_file, encoding='utf-8-sig')
        industry_features = {}
        for _, row in df_industry.iterrows():
            industry_features[row['行业名称']] = {
                'industry_return': row['涨跌幅'],
                'industry_volume': row['成交量'],
                'industry_amount': row['成交额'],
                'industry_net_inflow': row['主力净流入'],
                'industry_net_inflow_ratio': row['主力净流入占比']
            }
        additional_features['industry_features'] = industry_features
        logger.info(f"✅ 行业特征: {len(industry_features)}个行业")
    
    # 概念板块数据
    if os.path.exists(concept_data_file):
        df_concept = pd.read_csv(concept_data_file, encoding='utf-8-sig')
        concept_features = {}
        for _, row in df_concept.iterrows():
            concept_features[row['概念名称']] = {
                'concept_return': row['涨跌幅'],
                'concept_volume': row['成交量'],
                'concept_amount': row['成交额'],
                'concept_net_inflow': row['主力净流入'],
                'concept_net_inflow_ratio': row['主力净流入占比']
            }
        additional_features['concept_features'] = concept_features
        logger.info(f"✅ 概念特征: {len(concept_features)}个概念")
    
    # 热门概念排行
    if os.path.exists(hot_concepts_file):
        df_hot = pd.read_csv(hot_concepts_file, encoding='utf-8-sig')
        hot_concepts = {}
        for idx, row in df_hot.iterrows():
            hot_concepts[row['概念名称']] = {
                'hot_rank': idx + 1,
                'hot_return': row['涨跌幅'],
                'hot_net_inflow': row['主力净流入'],
                'up_count': row['上涨家数'],
                'down_count': row['下跌家数']
            }
        additional_features['hot_concepts'] = hot_concepts
        logger.info(f"✅ 热门概念: {len(hot_concepts)}个概念")
    
    # 保存额外特征数据
    features_file = os.path.join(output_dir, "板块特征数据.json")
    with open(features_file, 'w', encoding='utf-8') as f:
        json.dump(additional_features, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"✅ 板块特征数据已保存: {features_file}")
    
    # 5. 生成统计报告
    logger.info("📊 生成整合报告...")
    
    # 统计各行业股票数量
    industry_stats = df_training['industry'].value_counts()
    logger.info(f"📈 行业分布统计:")
    for industry, count in industry_stats.head(10).items():
        logger.info(f"  {industry}: {count}只")
    
    # 统计各地区股票数量
    region_stats = df_training['region'].value_counts()
    logger.info(f"📈 地区分布统计:")
    for region, count in region_stats.head(10).items():
        logger.info(f"  {region}: {count}只")
    
    # 生成摘要文件
    summary_file = os.path.join(output_dir, "板块数据整合摘要.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("新板块数据整合摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"整合时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"数据统计:\n")
        f.write(f"- 总股票数: {len(df_training)}只\n")
        f.write(f"- 行业数: {df_training['industry'].nunique()}个\n")
        f.write(f"- 地区数: {df_training['region'].nunique()}个\n")
        f.write(f"- 原有映射股票: {len(available_stocks)}只\n")
        f.write(f"- 额外补充股票: {len(extra_stocks)}只\n")
        f.write(f"- 缺失数据股票: {len(missing_stocks)}只\n\n")
        
        f.write("生成的文件:\n")
        f.write(f"- 股票板块映射_训练用.csv: 训练用的主要映射文件\n")
        f.write(f"- 板块特征数据.json: 行业和概念的额外特征\n")
        f.write(f"- 板块数据整合摘要.txt: 本文件\n\n")
        
        f.write("行业分布 (前10名):\n")
        for industry, count in industry_stats.head(10).items():
            f.write(f"- {industry}: {count}只\n")
        
        f.write("\n地区分布 (前10名):\n")
        for region, count in region_stats.head(10).items():
            f.write(f"- {region}: {count}只\n")
    
    logger.info(f"✅ 整合摘要已保存: {summary_file}")
    logger.info("🎉 新板块数据整合完成！")
    
    return {
        'training_file': training_file,
        'features_file': features_file,
        'summary_file': summary_file,
        'available_stocks': len(available_stocks),
        'extra_stocks': len(extra_stocks),
        'missing_stocks': len(missing_stocks),
        'total_stocks': len(df_training),
        'total_industries': df_training['industry'].nunique(),
        'total_regions': df_training['region'].nunique()
    }

if __name__ == "__main__":
    result = integrate_sector_data()
    print("\n整合结果:")
    print(f"✅ 原有映射股票: {result['available_stocks']}只")
    print(f"📊 额外补充股票: {result['extra_stocks']}只")
    print(f"🎯 总股票数: {result['total_stocks']}只")
    print(f"⚠️  缺失股票: {result['missing_stocks']}只") 
    print(f"📊 行业数: {result['total_industries']}个")
    print(f"📊 地区数: {result['total_regions']}个")
    print(f"\n📁 生成的文件:")
    print(f"- {result['training_file']}")
    print(f"- {result['features_file']}")
    print(f"- {result['summary_file']}")