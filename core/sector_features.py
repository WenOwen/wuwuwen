# -*- coding: utf-8 -*-
"""
板块特征工程模块
利用新的板块数据为股票添加板块相关特征
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SectorFeatureEngineer:
    """
    板块特征工程类
    基于新的板块数据生成板块相关特征
    """
    
    def __init__(self, features_file: str = "data/板块特征数据.json"):
        """
        初始化板块特征工程器
        
        Args:
            features_file: 板块特征数据文件路径
        """
        self.features_file = features_file
        self.sector_features = {}
        self._load_sector_features()
    
    def _load_sector_features(self):
        """加载板块特征数据"""
        if os.path.exists(self.features_file):
            try:
                with open(self.features_file, 'r', encoding='utf-8') as f:
                    self.sector_features = json.load(f)
                
                industry_count = len(self.sector_features.get('industry_features', {}))
                concept_count = len(self.sector_features.get('concept_features', {}))
                hot_count = len(self.sector_features.get('hot_concepts', {}))
                
                logger.info(f"✅ 加载板块特征数据: {industry_count}个行业, {concept_count}个概念, {hot_count}个热门概念")
                
            except Exception as e:
                logger.error(f"❌ 加载板块特征数据失败: {str(e)}")
                self.sector_features = {}
        else:
            logger.warning(f"⚠️ 板块特征数据文件不存在: {self.features_file}")
            self.sector_features = {}
    
    def add_sector_features(self, df: pd.DataFrame, stock_code: str, 
                           stock_sector_info: Dict) -> pd.DataFrame:
        """
        为股票数据添加板块特征
        
        Args:
            df: 股票数据DataFrame
            stock_code: 股票代码
            stock_sector_info: 股票板块信息
            
        Returns:
            添加了板块特征的DataFrame
        """
        if df.empty or not self.sector_features:
            return df
        
        df_result = df.copy()
        
        # 获取股票的行业和概念信息
        industry = stock_sector_info.get('sector', '')
        primary_concept = stock_sector_info.get('primary_concept', '')
        all_concepts = stock_sector_info.get('all_concepts', '')
        region = stock_sector_info.get('region', '')
        
        # 1. 行业特征
        industry_features = self.sector_features.get('industry_features', {})
        if industry in industry_features:
            industry_data = industry_features[industry]
            df_result['sector_industry_return'] = industry_data.get('industry_return', 0)
            df_result['sector_industry_volume'] = industry_data.get('industry_volume', 0)
            df_result['sector_industry_amount'] = industry_data.get('industry_amount', 0)
            df_result['sector_industry_net_inflow'] = industry_data.get('industry_net_inflow', 0)
            df_result['sector_industry_net_inflow_ratio'] = industry_data.get('industry_net_inflow_ratio', 0)
        else:
            # 默认值
            df_result['sector_industry_return'] = 0
            df_result['sector_industry_volume'] = 0
            df_result['sector_industry_amount'] = 0
            df_result['sector_industry_net_inflow'] = 0
            df_result['sector_industry_net_inflow_ratio'] = 0
        
        # 2. 主要概念特征
        concept_features = self.sector_features.get('concept_features', {})
        if primary_concept in concept_features:
            concept_data = concept_features[primary_concept]
            df_result['sector_concept_return'] = concept_data.get('concept_return', 0)
            df_result['sector_concept_volume'] = concept_data.get('concept_volume', 0)
            df_result['sector_concept_amount'] = concept_data.get('concept_amount', 0)
            df_result['sector_concept_net_inflow'] = concept_data.get('concept_net_inflow', 0)
            df_result['sector_concept_net_inflow_ratio'] = concept_data.get('concept_net_inflow_ratio', 0)
        else:
            # 默认值
            df_result['sector_concept_return'] = 0
            df_result['sector_concept_volume'] = 0
            df_result['sector_concept_amount'] = 0
            df_result['sector_concept_net_inflow'] = 0
            df_result['sector_concept_net_inflow_ratio'] = 0
        
        # 3. 热门概念特征
        hot_concepts = self.sector_features.get('hot_concepts', {})
        hot_rank = 999  # 默认排名很低
        hot_return = 0
        hot_net_inflow = 0
        
        # 检查是否包含热门概念
        if all_concepts:
            concepts_list = [c.strip() for c in all_concepts.split(',')]
            for concept in concepts_list:
                if concept in hot_concepts:
                    hot_data = hot_concepts[concept]
                    concept_rank = hot_data.get('hot_rank', 999)
                    if concept_rank < hot_rank:  # 取最高的排名（数字最小）
                        hot_rank = concept_rank
                        hot_return = hot_data.get('hot_return', 0)
                        hot_net_inflow = hot_data.get('hot_net_inflow', 0)
        
        df_result['sector_hot_rank'] = hot_rank
        df_result['sector_hot_return'] = hot_return
        df_result['sector_hot_net_inflow'] = hot_net_inflow
        df_result['sector_is_hot'] = 1 if hot_rank <= 20 else 0  # 前20名算热门
        
        # 4. 地区特征（简单的地区编码）
        region_mapping = {
            '广东': 1, '浙江': 2, '江苏': 3, '北京': 4, '上海': 5,
            '山东': 6, '安徽': 7, '福建': 8, '四川': 9, '湖南': 10
        }
        df_result['sector_region_code'] = region_mapping.get(region, 0)
        
        # 5. 板块相对强度特征
        # 股票相对于行业的强度
        if len(df_result) > 1:
            stock_return = df_result['涨跌幅'].iloc[-1] if '涨跌幅' in df_result.columns else 0
            industry_return = df_result['sector_industry_return'].iloc[-1]
            df_result['sector_relative_strength'] = stock_return - industry_return
        else:
            df_result['sector_relative_strength'] = 0
        
        # 6. 概念数量特征
        concept_count = len([c.strip() for c in all_concepts.split(',')]) if all_concepts else 0
        df_result['sector_concept_count'] = concept_count
        
        return df_result
    
    def get_feature_names(self) -> List[str]:
        """获取板块特征名称列表"""
        return [
            'sector_industry_return',
            'sector_industry_volume', 
            'sector_industry_amount',
            'sector_industry_net_inflow',
            'sector_industry_net_inflow_ratio',
            'sector_concept_return',
            'sector_concept_volume',
            'sector_concept_amount', 
            'sector_concept_net_inflow',
            'sector_concept_net_inflow_ratio',
            'sector_hot_rank',
            'sector_hot_return',
            'sector_hot_net_inflow',
            'sector_is_hot',
            'sector_region_code',
            'sector_relative_strength',
            'sector_concept_count'
        ]
    
    def get_feature_info(self) -> Dict[str, str]:
        """获取特征信息描述"""
        return {
            'sector_industry_return': '所属行业涨跌幅',
            'sector_industry_volume': '所属行业成交量',
            'sector_industry_amount': '所属行业成交额',
            'sector_industry_net_inflow': '所属行业主力净流入',
            'sector_industry_net_inflow_ratio': '所属行业主力净流入占比',
            'sector_concept_return': '主要概念涨跌幅',
            'sector_concept_volume': '主要概念成交量',
            'sector_concept_amount': '主要概念成交额',
            'sector_concept_net_inflow': '主要概念主力净流入',
            'sector_concept_net_inflow_ratio': '主要概念主力净流入占比',
            'sector_hot_rank': '热门概念排名',
            'sector_hot_return': '热门概念涨跌幅',
            'sector_hot_net_inflow': '热门概念主力净流入',
            'sector_is_hot': '是否为热门概念',
            'sector_region_code': '地区编码',
            'sector_relative_strength': '相对行业强度',
            'sector_concept_count': '概念数量'
        }

if __name__ == "__main__":
    # 测试板块特征工程
    engineer = SectorFeatureEngineer()
    
    # 模拟股票数据
    test_data = {
        '交易日期': ['2024-01-01', '2024-01-02'],
        '收盘价': [100, 105],
        '涨跌幅': [0, 5]
    }
    df = pd.DataFrame(test_data)
    
    # 模拟股票板块信息
    stock_info = {
        'sector': '半导体',
        'primary_concept': 'AI芯片',
        'all_concepts': 'AI芯片,人工智能,芯片概念',
        'region': '广东'
    }
    
    # 添加板块特征
    df_with_features = engineer.add_sector_features(df, 'sz000001', stock_info)
    
    print("添加板块特征后的数据:")
    print(df_with_features.head())
    
    print("\n板块特征列表:")
    for feature in engineer.get_feature_names():
        print(f"- {feature}: {engineer.get_feature_info()[feature]}")