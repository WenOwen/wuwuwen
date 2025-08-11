#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合数据整合处理脚本
处理个股、行业板块、概念板块、指数数据，生成按日期分区的Parquet文件

作者: AI Assistant
日期: 2025-01-27
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import re
from datetime import datetime
import warnings
from tqdm import tqdm
import logging

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataProcessor:
    def __init__(self, data_root_path):
        """
        初始化数据处理器
        
        Args:
            data_root_path: 数据根目录路径
        """
        self.data_root = Path(data_root_path)
        self.output_dir = self.data_root / "processed_parquet"
        self.output_dir.mkdir(exist_ok=True)
        
        # 数据路径
        self.em_data_path = self.data_root / "datas_em"
        self.sector_data_path = self.data_root / "datas_sector_historical"
        self.index_data_path = self.data_root / "datas_index" / "datas_index"
        self.mapping_file = self.sector_data_path / "股票板块映射表.csv"
        
        # 初始化数据存储
        self.stock_mapping = None
        self.concept_mapping = None
        self.all_concepts = set()
        self.processed_dates = set()
        
    def load_stock_mapping(self):
        """读取股票板块映射表，建立映射关系"""
        logger.info("正在读取股票板块映射表...")
        
        try:
            mapping_df = pd.read_csv(self.mapping_file, encoding='utf-8')
            logger.info(f"成功读取映射表，共{len(mapping_df)}条记录")
            
            # 建立股票到行业的映射
            self.stock_mapping = dict(zip(mapping_df['股票代码'], mapping_df['所属行业']))
            
            # 建立股票到概念的映射
            self.concept_mapping = {}
            all_concepts = set()
            
            for _, row in mapping_df.iterrows():
                stock_code = row['股票代码']
                concepts_str = str(row['概念板块'])
                
                if pd.notna(concepts_str) and concepts_str != 'nan':
                    # 处理概念板块字符串，去掉引号并分割
                    concepts_str = concepts_str.strip('"').strip("'")
                    concepts = [c.strip() for c in concepts_str.split(',') if c.strip()]
                    self.concept_mapping[stock_code] = concepts
                    all_concepts.update(concepts)
                else:
                    self.concept_mapping[stock_code] = []
            
            self.all_concepts = sorted(list(all_concepts))
            logger.info(f"识别出{len(self.all_concepts)}个不同的概念板块")
            
        except Exception as e:
            logger.error(f"读取映射表失败: {e}")
            raise
    
    def process_em_data(self):
        """处理个股数据"""
        logger.info("正在处理个股数据...")
        
        em_files = list(self.em_data_path.glob("*.csv"))
        logger.info(f"发现{len(em_files)}个个股数据文件")
        
        # 存储所有个股数据，按日期分组
        date_stock_data = {}
        
        for file_path in tqdm(em_files, desc="处理个股数据"):
            try:
                # 从文件名提取股票代码
                stock_code = file_path.stem
                
                # 读取数据
                df = pd.read_csv(file_path, encoding='utf-8')
                
                # 标准化日期列名
                if '交易日期' in df.columns:
                    df['date'] = pd.to_datetime(df['交易日期'])
                elif '日期' in df.columns:
                    df['date'] = pd.to_datetime(df['日期'])
                else:
                    logger.warning(f"文件{file_path}缺少日期列，跳过")
                    continue
                
                # 添加股票代码列
                df['symbol'] = stock_code
                
                # 重命名列以避免冲突
                column_mapping = {}
                for col in df.columns:
                    if col not in ['date', 'symbol']:
                        column_mapping[col] = f"stock_{col}"
                
                df.rename(columns=column_mapping, inplace=True)
                
                # 按日期分组存储
                for _, row in df.iterrows():
                    date_str = row['date'].strftime('%Y-%m-%d')
                    
                    if date_str not in date_stock_data:
                        date_stock_data[date_str] = []
                    
                    # 转换为字典，排除日期列
                    row_dict = row.drop('date').to_dict()
                    date_stock_data[date_str].append(row_dict)
                    
            except Exception as e:
                logger.error(f"处理文件{file_path}时出错: {e}")
                continue
        
        logger.info(f"个股数据处理完成，覆盖{len(date_stock_data)}个交易日")
        return date_stock_data
    
    def process_industry_data(self):
        """处理行业板块数据"""
        logger.info("正在处理行业板块数据...")
        
        industry_path = self.sector_data_path / "行业板块_全部历史"
        industry_files = list(industry_path.glob("*_daily_历史数据.csv"))
        
        # 排除汇总文件
        industry_files = [f for f in industry_files if not f.name.startswith("所有")]
        
        logger.info(f"发现{len(industry_files)}个行业板块数据文件")
        
        date_industry_data = {}
        
        for file_path in tqdm(industry_files, desc="处理行业板块数据"):
            try:
                # 从文件名提取板块名称
                filename = file_path.stem
                match = re.match(r'^(.+?)\(BK\d+\)_daily_历史数据$', filename)
                if match:
                    sector_name = match.group(1)
                else:
                    logger.warning(f"无法解析文件名: {filename}")
                    continue
                
                # 读取数据
                df = pd.read_csv(file_path, encoding='utf-8')
                
                if '日期' not in df.columns:
                    logger.warning(f"文件{file_path}缺少日期列，跳过")
                    continue
                
                df['date'] = pd.to_datetime(df['日期'])
                
                # 删除标识特征
                columns_to_drop = ['日期', '板块代码', '板块类型', '板块名称']
                feature_columns = [col for col in df.columns if col not in columns_to_drop]
                
                # 重命名特征列
                column_mapping = {}
                for col in feature_columns:
                    column_mapping[col] = f"{sector_name}_{col}"
                
                df.rename(columns=column_mapping, inplace=True)
                
                # 按日期分组存储
                for _, row in df.iterrows():
                    date_str = row['date'].strftime('%Y-%m-%d')
                    
                    if date_str not in date_industry_data:
                        date_industry_data[date_str] = {}
                    
                    # 存储该行业在该日期的数据
                    row_dict = row.drop('date').to_dict()
                    for key, value in row_dict.items():
                        date_industry_data[date_str][key] = value
                        
            except Exception as e:
                logger.error(f"处理文件{file_path}时出错: {e}")
                continue
        
        logger.info(f"行业板块数据处理完成，覆盖{len(date_industry_data)}个交易日")
        return date_industry_data
    
    def process_concept_data(self):
        """处理概念板块数据，生成One-Hot编码"""
        logger.info("正在处理概念板块数据...")
        
        concept_path = self.sector_data_path / "概念板块_全部历史"
        concept_files = list(concept_path.glob("*_daily_历史数据.csv"))
        
        # 排除汇总文件
        concept_files = [f for f in concept_files if not f.name.startswith("所有")]
        
        logger.info(f"发现{len(concept_files)}个概念板块数据文件")
        
        # 收集所有概念板块的收盘价数据
        concept_prices = {}
        
        for file_path in tqdm(concept_files, desc="处理概念板块数据"):
            try:
                # 从文件名提取概念名称
                filename = file_path.stem
                match = re.match(r'^(.+?)\(BK\d+\)_daily_历史数据$', filename)
                if match:
                    concept_name = match.group(1)
                else:
                    logger.warning(f"无法解析概念文件名: {filename}")
                    continue
                
                # 读取数据
                df = pd.read_csv(file_path, encoding='utf-8')
                
                if '日期' not in df.columns or '收盘价' not in df.columns:
                    logger.warning(f"文件{file_path}缺少必要列，跳过")
                    continue
                
                df['date'] = pd.to_datetime(df['日期'])
                
                # 存储该概念的收盘价数据
                concept_prices[concept_name] = {}
                for _, row in df.iterrows():
                    date_str = row['date'].strftime('%Y-%m-%d')
                    concept_prices[concept_name][date_str] = row['收盘价']
                    
            except Exception as e:
                logger.error(f"处理概念文件{file_path}时出错: {e}")
                continue
        
        logger.info(f"概念板块数据处理完成，识别{len(concept_prices)}个概念")
        return concept_prices
    
    def process_index_data(self):
        """处理指数数据"""
        logger.info("正在处理指数数据...")
        
        index_files = list(self.index_data_path.glob("*.csv"))
        logger.info(f"发现{len(index_files)}个指数数据文件")
        
        date_index_data = {}
        
        for file_path in tqdm(index_files, desc="处理指数数据"):
            try:
                # 从文件名提取指数代码
                index_code = file_path.stem
                
                # 读取数据
                df = pd.read_csv(file_path, encoding='utf-8')
                
                if '交易日期' not in df.columns:
                    logger.warning(f"指数文件{file_path}缺少交易日期列，跳过")
                    continue
                
                df['date'] = pd.to_datetime(df['交易日期'])
                
                # 重命名列以避免冲突
                column_mapping = {}
                for col in df.columns:
                    if col not in ['date', '交易日期']:
                        column_mapping[col] = f"{index_code}_{col}"
                
                df.rename(columns=column_mapping, inplace=True)
                
                # 按日期分组存储
                for _, row in df.iterrows():
                    date_str = row['date'].strftime('%Y-%m-%d')
                    
                    if date_str not in date_index_data:
                        date_index_data[date_str] = {}
                    
                    # 存储该指数在该日期的数据
                    row_dict = row.drop('date').to_dict()
                    for key, value in row_dict.items():
                        date_index_data[date_str][key] = value
                        
            except Exception as e:
                logger.error(f"处理指数文件{file_path}时出错: {e}")
                continue
        
        logger.info(f"指数数据处理完成，覆盖{len(date_index_data)}个交易日")
        return date_index_data
    
    def create_concept_features_for_stock(self, stock_code, date_str, concept_prices):
        """为指定股票创建概念板块的One-Hot编码特征"""
        concept_features = {}
        
        # 获取该股票对应的概念
        stock_concepts = self.concept_mapping.get(stock_code, [])
        
        # 为每个概念创建特征
        for concept in self.all_concepts:
            # One-Hot编码
            concept_features[f"concept_{concept}_indicator"] = 1 if concept in stock_concepts else 0
            
            # 如果股票属于该概念，添加概念板块的价格特征
            if concept in stock_concepts and concept in concept_prices:
                concept_data = concept_prices[concept].get(date_str)
                if concept_data is not None:
                    concept_features[f"concept_{concept}_price"] = concept_data
                else:
                    concept_features[f"concept_{concept}_price"] = np.nan
            else:
                concept_features[f"concept_{concept}_price"] = np.nan
        
        return concept_features
    
    def integrate_and_export_data(self, stock_data, industry_data, concept_prices, index_data):
        """整合所有数据并按日期导出Parquet文件"""
        logger.info("正在整合数据并生成Parquet文件...")
        
        # 获取所有日期
        all_dates = set()
        all_dates.update(stock_data.keys())
        all_dates.update(industry_data.keys())
        all_dates.update(index_data.keys())
        
        logger.info(f"识别出{len(all_dates)}个交易日")
        
        for date_str in tqdm(sorted(all_dates), desc="生成Parquet文件"):
            try:
                # 获取该日期的个股数据
                daily_stocks = stock_data.get(date_str, [])
                if not daily_stocks:
                    continue
                
                # 转换为DataFrame
                daily_df = pd.DataFrame(daily_stocks)
                daily_df.set_index('symbol', inplace=True)
                
                # 添加行业板块数据
                industry_daily = industry_data.get(date_str, {})
                for feature, value in industry_daily.items():
                    daily_df[feature] = value
                
                # 添加指数数据
                index_daily = index_data.get(date_str, {})
                for feature, value in index_daily.items():
                    daily_df[feature] = value
                
                # 为每个股票添加概念板块特征
                concept_features_list = []
                for stock_code in daily_df.index:
                    concept_features = self.create_concept_features_for_stock(
                        stock_code, date_str, concept_prices
                    )
                    concept_features_list.append(concept_features)
                
                # 将概念特征添加到DataFrame
                if concept_features_list:
                    concept_df = pd.DataFrame(concept_features_list, index=daily_df.index)
                    daily_df = pd.concat([daily_df, concept_df], axis=1)
                
                # 保存为Parquet文件
                output_file = self.output_dir / f"{date_str}.parquet"
                daily_df.to_parquet(output_file, compression='snappy')
                
                logger.debug(f"已保存 {date_str}.parquet，包含{len(daily_df)}只股票，{len(daily_df.columns)}个特征")
                
            except Exception as e:
                logger.error(f"处理日期{date_str}时出错: {e}")
                continue
        
        logger.info(f"数据整合完成！文件保存至: {self.output_dir}")
    
    def run(self):
        """运行完整的数据处理流程"""
        logger.info("开始执行数据处理流程...")
        
        # 1. 读取映射表
        logger.info("步骤1: 读取股票板块映射表")
        self.load_stock_mapping()
        
        # 2. 处理各类数据
        logger.info("步骤2: 处理个股数据")
        stock_data = self.process_em_data()
        
        logger.info("步骤3: 处理行业板块数据")
        industry_data = self.process_industry_data()
        
        logger.info("步骤4: 处理概念板块数据")
        concept_prices = self.process_concept_data()
        
        logger.info("步骤5: 处理指数数据")
        index_data = self.process_index_data()
        
        # 3. 整合并导出
        logger.info("步骤6: 整合数据并导出Parquet文件")
        self.integrate_and_export_data(stock_data, industry_data, concept_prices, index_data)
        
        logger.info("数据处理流程全部完成！")
        
        # 输出统计信息
        parquet_files = list(self.output_dir.glob("*.parquet"))
        logger.info(f"共生成{len(parquet_files)}个Parquet文件")
        
        if parquet_files:
            # 读取一个示例文件查看结构
            sample_df = pd.read_parquet(parquet_files[0])
            logger.info(f"数据结构示例 - 股票数量: {len(sample_df)}, 特征数量: {len(sample_df.columns)}")
            logger.info(f"特征列举例: {list(sample_df.columns[:10])}")


def main():
    """主函数"""
    # 设置数据根目录
    data_root = "/home/wangkai/6tdisk/wht/wuwuwen/data"
    
    # 创建处理器并运行
    processor = StockDataProcessor(data_root)
    processor.run()


if __name__ == "__main__":
    main()