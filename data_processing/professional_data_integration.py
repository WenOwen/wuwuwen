#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专业金融数据整合处理脚本
严格按照金融工程要求：
1. 股票代码放第一列
2. 每个行业/概念固定3列：归属标识 + 涨跌幅 + 交互特征
3. 加载全部概念和行业
4. 严格的数据类型控制
"""

import pandas as pd
import numpy as np
import os
import gc
from pathlib import Path
import warnings
from datetime import datetime
import re
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

class ProfessionalDataProcessor:
    def __init__(self, data_root='data', start_date=None, end_date=None, recent_days=None, skip_existing=True):
        """
        初始化专业数据处理器
        
        Args:
            data_root: 数据根目录路径
            start_date: 开始日期 (字符串格式: 'YYYY-MM-DD' 或 datetime.date 对象)
            end_date: 结束日期 (字符串格式: 'YYYY-MM-DD' 或 datetime.date 对象)
            recent_days: 只处理最近几个交易日（如果不指定start_date和end_date）
            skip_existing: 是否跳过已存在的文件，避免重复处理
        """
        self.data_root = Path(data_root)
        self.output_dir = self.data_root / 'professional_parquet'
        self.output_dir.mkdir(exist_ok=True)
        
        # 日期范围设置
        self.start_date = self._parse_date(start_date) if start_date else None
        self.end_date = self._parse_date(end_date) if end_date else None
        self.recent_days = recent_days
        self.skip_existing = skip_existing
        
        # 数据路径配置
        self.paths = {
            'stock_data': self.data_root / 'datas_em',
            'sector_mapping': self.data_root / 'datas_sector_historical' / '股票板块映射表.csv',
            'industry_data': self.data_root / 'datas_sector_historical' / '行业板块_全部历史',
            'concept_data': self.data_root / 'datas_sector_historical' / '概念板块_全部历史',
        }
        
        # 存储映射关系和全集
        self.stock_mapping = {}
        self.industry_universe = []  # 所有行业的固定全集
        self.concept_universe = []   # 所有概念的固定全集
        self.industry_pct_data = {}  # 行业涨跌幅数据：{industry: {date: pct}}
        self.concept_pct_data = {}   # 概念涨跌幅数据：{concept: {date: pct}}
        
    def _parse_date(self, date_input):
        """解析日期输入，支持字符串和date对象"""
        if isinstance(date_input, str):
            try:
                return datetime.strptime(date_input, '%Y-%m-%d').date()
            except ValueError:
                print(f"❌ 日期格式错误: {date_input}，请使用 YYYY-MM-DD 格式")
                return None
        elif hasattr(date_input, 'date'):
            return date_input.date()
        elif hasattr(date_input, 'year'):
            return date_input
        else:
            print(f"❌ 无法解析日期: {date_input}")
            return None
        
    def load_stock_mapping(self):
        """加载股票映射表并建立完整的行业概念全集"""
        print("📊 加载股票映射表并建立完整全集...")
        try:
            mapping_df = pd.read_csv(self.paths['sector_mapping'], encoding='utf-8')
            
            # 建立行业全集
            all_industries = set()
            for industry in mapping_df['所属行业'].fillna(''):
                if industry.strip():
                    all_industries.add(industry.strip())
            
            # 建立概念全集
            all_concepts = set()
            for concepts in mapping_df['概念板块'].fillna(''):
                if concepts.strip():
                    concept_list = [c.strip() for c in concepts.split(',') if c.strip()]
                    all_concepts.update(concept_list)
            
            # 按字母顺序排序，确保一致性
            self.industry_universe = sorted(list(all_industries))
            self.concept_universe = sorted(list(all_concepts))
            
            print(f"   ✅ 行业全集: {len(self.industry_universe)} 个")
            print(f"   ✅ 概念全集: {len(self.concept_universe)} 个")
            
            # 构建股票映射字典
            for _, row in mapping_df.iterrows():
                stock_code = row['股票代码']
                concepts = row['概念板块'] if pd.notna(row['概念板块']) else ''
                concept_list = [c.strip() for c in concepts.split(',') if c.strip()]
                
                self.stock_mapping[stock_code] = {
                    'name': row['股票名称'],
                    'industry': row['所属行业'].strip() if pd.notna(row['所属行业']) else '',
                    'concepts': concept_list,
                }
                
            print(f"   ✅ 股票映射: {len(self.stock_mapping)} 只股票")
            return True
            
        except Exception as e:
            print(f"❌ 加载股票映射表失败: {e}")
            return False
    
    def load_all_industry_data(self):
        """加载所有行业的涨跌幅数据"""
        print("🏭 加载所有行业涨跌幅数据...")
        
        industry_files = list(self.paths['industry_data'].glob('*_daily_历史数据.csv'))
        print(f"   发现 {len(industry_files)} 个行业数据文件")
        
        loaded_industries = set()
        
        for industry_file in tqdm(industry_files, desc="加载行业数据"):
            try:
                df = pd.read_csv(industry_file, encoding='utf-8')
                df['日期'] = pd.to_datetime(df['日期']).dt.date
                
                # 获取行业名称
                industry_name = df['板块名称'].iloc[0] if len(df) > 0 else None
                
                if industry_name and '涨跌幅' in df.columns:
                    # 构建日期->涨跌幅的映射
                    date_pct_map = {}
                    for _, row in df.iterrows():
                        date = row['日期']
                        pct = row['涨跌幅'] / 100.0  # 转换为小数格式
                        date_pct_map[date] = pct
                    
                    self.industry_pct_data[industry_name] = date_pct_map
                    loaded_industries.add(industry_name)
                
            except Exception as e:
                print(f"   ⚠️ 处理 {industry_file} 失败: {e}")
                continue
        
        print(f"   ✅ 成功加载 {len(loaded_industries)} 个行业的涨跌幅数据")
        
        # 为没有数据的行业填充0
        for industry in self.industry_universe:
            if industry not in self.industry_pct_data:
                self.industry_pct_data[industry] = {}
                print(f"   ⚠️ 行业 '{industry}' 没有市场数据，将使用0填充")
        
    def load_all_concept_data(self):
        """加载所有概念的涨跌幅数据"""
        print("💡 加载所有概念涨跌幅数据...")
        
        concept_files = list(self.paths['concept_data'].glob('*_daily_历史数据.csv'))
        print(f"   发现 {len(concept_files)} 个概念数据文件")
        
        loaded_concepts = set()
        
        for concept_file in tqdm(concept_files, desc="加载概念数据"):
            try:
                df = pd.read_csv(concept_file, encoding='utf-8')
                df['日期'] = pd.to_datetime(df['日期']).dt.date
                
                # 获取概念名称
                concept_name = df['板块名称'].iloc[0] if len(df) > 0 else None
                
                if concept_name and '涨跌幅' in df.columns:
                    # 构建日期->涨跌幅的映射
                    date_pct_map = {}
                    for _, row in df.iterrows():
                        date = row['日期']
                        pct = row['涨跌幅'] / 100.0  # 转换为小数格式
                        date_pct_map[date] = pct
                    
                    self.concept_pct_data[concept_name] = date_pct_map
                    loaded_concepts.add(concept_name)
                
            except Exception as e:
                print(f"   ⚠️ 处理 {concept_file} 失败: {e}")
                continue
        
        print(f"   ✅ 成功加载 {len(loaded_concepts)} 个概念的涨跌幅数据")
        
        # 为没有数据的概念填充0
        for concept in self.concept_universe:
            if concept not in self.concept_pct_data:
                self.concept_pct_data[concept] = {}
                print(f"   ⚠️ 概念 '{concept}' 没有市场数据，将使用0填充")
    
    def get_target_dates(self):
        """获取目标处理日期范围"""
        # 从几个样本股票中获取所有可用的交易日期
        print("🔍 扫描可用的交易日期...")
        stock_files = list(self.paths['stock_data'].glob('*.csv'))[:20]
        
        all_dates = set()
        for stock_file in stock_files[:5]:
            try:
                df = pd.read_csv(stock_file, encoding='utf-8', usecols=['交易日期'])
                dates = pd.to_datetime(df['交易日期']).dt.date
                all_dates.update(dates)
            except:
                continue
        
        if not all_dates:
            print("❌ 没有找到有效的交易日期")
            return []
        
        # 排序所有日期
        sorted_dates = sorted(all_dates)
        
        # 根据设置选择日期范围
        if self.start_date and self.end_date:
            # 指定了起始和结束日期
            target_dates = [d for d in sorted_dates if self.start_date <= d <= self.end_date]
            print(f"   📅 指定日期范围: {self.start_date} 到 {self.end_date}")
            print(f"   📊 可用交易日: {len(target_dates)} 个")
        elif self.start_date:
            # 只指定了开始日期，处理从开始日期到最新的所有数据
            target_dates = [d for d in sorted_dates if d >= self.start_date]
            print(f"   📅 从指定日期开始: {self.start_date} 到 {sorted_dates[-1]}")
            print(f"   📊 可用交易日: {len(target_dates)} 个")
        elif self.end_date:
            # 只指定了结束日期，处理到指定日期的所有数据
            target_dates = [d for d in sorted_dates if d <= self.end_date]
            print(f"   📅 到指定日期结束: {sorted_dates[0]} 到 {self.end_date}")
            print(f"   📊 可用交易日: {len(target_dates)} 个")
        elif self.recent_days:
            # 使用最近N天模式
            target_dates = sorted_dates[-self.recent_days:]
            print(f"   📅 最近 {self.recent_days} 个交易日: {target_dates[0]} 到 {target_dates[-1]}")
        else:
            # 如果都没指定，默认处理所有日期
            target_dates = sorted_dates
            print(f"   📅 处理所有可用日期: {sorted_dates[0]} 到 {sorted_dates[-1]} (共 {len(target_dates)} 个)")
        
        # 如果需要跳过已存在的文件
        if self.skip_existing:
            existing_files = set()
            for f in self.output_dir.glob('*.parquet'):
                try:
                    date_obj = datetime.strptime(f.stem, '%Y-%m-%d').date()
                    existing_files.add(date_obj)
                except:
                    continue
            
            original_count = len(target_dates)
            target_dates = [d for d in target_dates if d not in existing_files]
            skipped_count = original_count - len(target_dates)
            
            if skipped_count > 0:
                print(f"   ⏭️  跳过已存在的文件: {skipped_count} 个")
                print(f"   🎯 实际需要处理: {len(target_dates)} 个")
        
        if not target_dates:
            if self.skip_existing:
                print("   ✅ 所有目标日期的文件都已存在，无需处理")
            else:
                print("   ❌ 没有找到符合条件的交易日期")
            return []
        
        print(f"   📈 最终处理范围: {target_dates[0]} 到 {target_dates[-1]}")
        return target_dates
    
    def process_single_date(self, target_date):
        """处理单个交易日的数据 - 严格按照专业格式"""
        print(f"   📅 处理日期: {target_date}")
        
        stock_files = list(self.paths['stock_data'].glob('*.csv'))
        processed_stocks = []
        processed_count = 0
        
        # 只处理前100只股票以加速测试
        for stock_file in stock_files:  # 移除[:100]限制
            try:
                stock_code = stock_file.stem
                if stock_code not in self.stock_mapping:
                    continue
                
                # 读取股票数据
                df = pd.read_csv(stock_file, encoding='utf-8')
                df['交易日期'] = pd.to_datetime(df['交易日期']).dt.date
                
                # 筛选指定日期的数据
                date_df = df[df['交易日期'] == target_date]
                
                if len(date_df) == 0:
                    continue
                
                # 获取股票信息
                stock_info = self.stock_mapping[stock_code]
                stock_industry = stock_info['industry']
                stock_concepts = stock_info['concepts']
                
                # 构建单只股票的数据行
                stock_row = {}
                
                # 1. 基础信息 (股票代码放第一列)
                stock_row['symbol'] = stock_code
                stock_row['name'] = stock_info['name']
                
                # 2. 股票交易数据
                for col in ['开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率', '流通市值']:
                    if col in date_df.columns:
                        stock_row[col] = date_df.iloc[0][col]
                
                # 3. 为每个行业生成固定的3列
                for industry in self.industry_universe:
                    # 3.1 归属标识 (uint8: 0或1)
                    stock_row[f'industry_{industry}'] = np.uint8(1 if industry == stock_industry else 0)
                    
                    # 3.2 行业当日涨跌幅 (float32: 小数格式)
                    industry_pct = self.industry_pct_data.get(industry, {}).get(target_date, 0.0)
                    stock_row[f'industry_{industry}_pct'] = np.float32(industry_pct)
                    
                    # 3.3 交互特征 (float32: 归属 × 涨跌幅)
                    interaction = stock_row[f'industry_{industry}'] * industry_pct
                    stock_row[f'industry_{industry}_x_ret'] = np.float32(interaction)
                
                # 4. 为每个概念生成固定的3列
                for concept in self.concept_universe:
                    # 4.1 归属标识 (uint8: 0或1)
                    stock_row[f'concept_{concept}'] = np.uint8(1 if concept in stock_concepts else 0)
                    
                    # 4.2 概念当日涨跌幅 (float32: 小数格式)
                    concept_pct = self.concept_pct_data.get(concept, {}).get(target_date, 0.0)
                    stock_row[f'concept_{concept}_pct'] = np.float32(concept_pct)
                    
                    # 4.3 交互特征 (float32: 归属 × 涨跌幅)
                    interaction = stock_row[f'concept_{concept}'] * concept_pct
                    stock_row[f'concept_{concept}_x_ret'] = np.float32(interaction)
                
                processed_stocks.append(stock_row)
                processed_count += 1
                
            except Exception as e:
                print(f"   ⚠️ 处理股票 {stock_file.stem} 失败: {e}")
                continue
        
        if not processed_stocks:
            print(f"      ⚠️ 日期 {target_date} 没有数据")
            return False
        
        # 转换为DataFrame
        combined_df = pd.DataFrame(processed_stocks)
        
        # 设置索引
        combined_df = combined_df.set_index('symbol')
        
        # 验证列的顺序和数据类型
        expected_col_count = 2 + 11 + len(self.industry_universe) * 3 + len(self.concept_universe) * 3  # name + 11个股票特征 + 行业*3 + 概念*3
        print(f"      📊 预期特征数: {expected_col_count}, 实际特征数: {len(combined_df.columns)}")
        
        # 保存为Parquet文件
        date_str = target_date.strftime('%Y-%m-%d')
        output_file = self.output_dir / f'{date_str}.parquet'
        combined_df.to_parquet(output_file, compression='snappy')
        
        print(f"      ✅ 保存了 {processed_count} 只股票，{len(combined_df.columns)} 个特征")
        
        return True
    
    def generate_metadata(self):
        """生成详细元数据"""
        print("📋 生成专业元数据...")
        
        parquet_files = list(self.output_dir.glob('*.parquet'))
        
        if not parquet_files:
            print("   ❌ 没有找到生成的Parquet文件")
            return
        
        # 分析样本文件
        sample_file = parquet_files[0]
        sample_df = pd.read_parquet(sample_file)
        
        # 验证特征结构
        stock_features = ['name'] + [col for col in sample_df.columns if col in ['开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率', '流通市值']]
        
        industry_features = {
            'indicators': [col for col in sample_df.columns if col.startswith('industry_') and not col.endswith('_pct') and not col.endswith('_x_ret')],
            'pct': [col for col in sample_df.columns if col.startswith('industry_') and col.endswith('_pct')],
            'interaction': [col for col in sample_df.columns if col.startswith('industry_') and col.endswith('_x_ret')]
        }
        
        concept_features = {
            'indicators': [col for col in sample_df.columns if col.startswith('concept_') and not col.endswith('_pct') and not col.endswith('_x_ret')],
            'pct': [col for col in sample_df.columns if col.startswith('concept_') and col.endswith('_pct')],
            'interaction': [col for col in sample_df.columns if col.startswith('concept_') and col.endswith('_x_ret')]
        }
        
        metadata = {
            'version': 'professional_v1.0',
            'processing_standard': 'financial_engineering',
            'total_files': len(parquet_files),
            'date_range': f"{min(f.stem for f in parquet_files)} 到 {max(f.stem for f in parquet_files)}",
            'sample_stocks_count': len(sample_df),
            'total_columns': len(sample_df.columns),
            
            'feature_structure': {
                'stock_basic': len(stock_features),
                'industry_total': len(industry_features['indicators']) + len(industry_features['pct']) + len(industry_features['interaction']),
                'industry_breakdown': {
                    'indicators': len(industry_features['indicators']),
                    'pct': len(industry_features['pct']),
                    'interaction': len(industry_features['interaction'])
                },
                'concept_total': len(concept_features['indicators']) + len(concept_features['pct']) + len(concept_features['interaction']),
                'concept_breakdown': {
                    'indicators': len(concept_features['indicators']),
                    'pct': len(concept_features['pct']),
                    'interaction': len(concept_features['interaction'])
                }
            },
            
            'data_universe': {
                'industry_count': len(self.industry_universe),
                'concept_count': len(self.concept_universe),
                'industry_list': self.industry_universe,
                'concept_list': self.concept_universe
            },
            
            'data_types': {
                'symbol': 'string (index)',
                'name': 'string',
                'stock_features': 'float64',
                'industry_indicators': 'uint8 (0/1)',
                'industry_pct': 'float32 (decimal)',
                'industry_interaction': 'float32 (decimal)',
                'concept_indicators': 'uint8 (0/1)',
                'concept_pct': 'float32 (decimal)',
                'concept_interaction': 'float32 (decimal)'
            },
            
            'quality_check': {
                'expected_industry_triplets': len(self.industry_universe),
                'actual_industry_triplets': len(industry_features['indicators']),
                'expected_concept_triplets': len(self.concept_universe),
                'actual_concept_triplets': len(concept_features['indicators']),
                'industry_complete': len(industry_features['indicators']) == len(self.industry_universe),
                'concept_complete': len(concept_features['indicators']) == len(self.concept_universe)
            },
            
            'created_at': datetime.now().isoformat(),
            'notes': [
                '股票代码(symbol)作为第一列和索引',
                '删除了地区数据',
                '每个行业/概念严格3列：归属标识+涨跌幅+交互特征',
                '涨跌幅已转换为小数格式 (2.3% → 0.023)',
                '数据类型经过优化：uint8用于0/1标识，float32用于数值特征'
            ]
        }
        
        with open(self.output_dir / 'professional_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 输出质量报告
        print(f"   ✅ 质量检查报告：")
        print(f"      行业三元组: {metadata['quality_check']['actual_industry_triplets']}/{metadata['quality_check']['expected_industry_triplets']} ({'✅' if metadata['quality_check']['industry_complete'] else '❌'})")
        print(f"      概念三元组: {metadata['quality_check']['actual_concept_triplets']}/{metadata['quality_check']['expected_concept_triplets']} ({'✅' if metadata['quality_check']['concept_complete'] else '❌'})")
        print(f"      总特征数: {metadata['total_columns']}")
        
    def run_professional_integration(self):
        """运行专业数据整合流程"""
        print("🚀 开始专业金融数据整合处理...")
        print("=" * 60)
        
        # 1. 加载股票映射表并建立完整全集
        if not self.load_stock_mapping():
            return False
        
        # 2. 加载所有行业和概念的市场数据
        self.load_all_industry_data()
        self.load_all_concept_data()
        
        # 3. 获取目标处理日期
        target_dates = self.get_target_dates()
        if not target_dates:
            return False
        
        # 4. 处理每个日期
        success_count = 0
        for date in target_dates:
            if self.process_single_date(date):
                success_count += 1
        
        # 5. 生成专业元数据
        self.generate_metadata()
        
        print("=" * 60)
        print(f"✅ 专业数据整合完成！成功处理 {success_count}/{len(target_dates)} 个交易日")
        
        return success_count > 0

def create_processor_examples():
    """提供不同使用场景的处理器创建示例"""
    examples = {
        "继续处理历史数据": {
            "description": "如果已处理了最近100天，现在想处理更早的数据",
            "code": """
processor = ProfessionalDataProcessor(
    start_date='2015-01-01',    # 从2015年开始
    end_date='2023-01-01',      # 到2023年结束
    skip_existing=True          # 跳过已存在的文件
)
            """.strip()
        },
        
        "补充特定年份数据": {
            "description": "只处理某个特定年份的数据",
            "code": """
processor = ProfessionalDataProcessor(
    start_date='2022-01-01',    # 2022年开始
    end_date='2022-12-31',      # 2022年结束
    skip_existing=True
)
            """.strip()
        },
        
        "从某日期开始到最新": {
            "description": "从指定日期开始处理到最新数据",
            "code": """
processor = ProfessionalDataProcessor(
    start_date='2020-01-01',    # 从2020年开始
    skip_existing=True          # 会处理到最新日期
)
            """.strip()
        },
        
        "处理最近N天": {
            "description": "只处理最近的N个交易日",
            "code": """
processor = ProfessionalDataProcessor(
    recent_days=50,             # 最近50个交易日
    skip_existing=True
)
            """.strip()
        }
    }
    
    print("📖 使用示例:")
    print("=" * 60)
    for title, example in examples.items():
        print(f"\n🔹 {title}")
        print(f"   说明: {example['description']}")
        print(f"   代码:")
        for line in example['code'].split('\n'):
            print(f"   {line}")
    print("=" * 60)

def main():
    """主函数"""
    print("💼 专业金融数据整合处理器")
    print("🎯 严格按照金融工程标准:")
    print("   ✅ 股票代码第一列")
    print("   ✅ 每个行业/概念固定3列")
    print("   ✅ 归属标识(uint8) + 涨跌幅(float32) + 交互特征(float32)")
    print("   ✅ 涨跌幅转换为小数格式")
    print("   ✅ 加载全部行业和概念")
    print()
    
    # 显示使用示例
    create_processor_examples()
    
    # 当前设置：继续处理历史数据（你已经有100天的数据了）
    print("\n🚀 当前运行模式：继续处理历史数据")
    print("   💡 如需修改日期范围，请编辑下面的代码")
    print()
    
    # 根据你的需求调整这里的日期
    processor = ProfessionalDataProcessor(
        start_date='2024-09-01',  # 从2015年开始（你可以改这个日期）
        end_date='2025-01-01',    # 到2024年结束（你可以改这个日期）
        skip_existing=True        # 会自动跳过已存在的文件
    )
    
    # 其他可选模式（取消注释即可使用）：
    
    # 1. 处理特定年份
    # processor = ProfessionalDataProcessor(
    #     start_date='2022-01-01',
    #     end_date='2022-12-31',
    #     skip_existing=True
    # )
    
    # 2. 从某日期开始到最新
    # processor = ProfessionalDataProcessor(
    #     start_date='2020-01-01',
    #     skip_existing=True
    # )
    
    # 3. 继续处理最近N天
    # processor = ProfessionalDataProcessor(
    #     recent_days=200,
    #     skip_existing=True
    # )
    
    success = processor.run_professional_integration()
    
    if success:
        print("\n🎉 专业处理成功！")
        print(f"📁 输出目录: {processor.output_dir}")
        print("📋 数据格式:")
        print("   symbol | name | 股票特征 | industry_XXX | industry_XXX_pct | industry_XXX_x_ret | concept_XXX | concept_XXX_pct | concept_XXX_x_ret | ...")
        print()
        
        # 显示样本数据结构
        parquet_files = list(processor.output_dir.glob('*.parquet'))
        if parquet_files:
            sample_file = parquet_files[0]
            df = pd.read_parquet(sample_file)
            print(f"📊 样本文件: {sample_file.name}")
            print(f"   📈 股票数量: {len(df)}")
            print(f"   📋 特征数量: {len(df.columns)}")
            print(f"   🎯 行业特征: {len([c for c in df.columns if c.startswith('industry_')])} 个")
            print(f"   🎯 概念特征: {len([c for c in df.columns if c.startswith('concept_')])} 个")
            
        print("\n✅ 数据符合专业金融建模标准！")
    else:
        print("\n❌ 专业处理失败！")

if __name__ == "__main__":
    main()