# -*- coding: utf-8 -*-
"""
股票板块映射模块
功能：建立股票代码到板块的映射关系，支持板块效应分析
"""

import pandas as pd
from typing import Dict, List, Optional
import json
import os

class StockSectorMapping:
    """
    股票板块映射类
    负责管理股票代码与板块的映射关系
    """
    
    def __init__(self, mapping_file: str = None):
        # 优先使用新整合的真实板块数据
        csv_files = [
            "data/股票板块映射_训练用.csv",
            "../data/股票板块映射_训练用.csv", 
            "股票板块映射_训练用.csv",
            "sector_data/股票板块映射_训练用.csv",
            "../sector_data/股票板块映射_训练用.csv"
        ]
        
        self.mapping_file = None
        self.csv_mapping_file = None
        
        # 寻找CSV映射文件
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                self.csv_mapping_file = csv_file
                print(f"✅ 发现真实板块数据: {csv_file}")
                break
        
        # 如果没有CSV文件，使用传统JSON方式
        if self.csv_mapping_file is None:
            self.mapping_file = mapping_file or "stock_sector_data.json"
            print(f"⚠️ 未找到CSV板块数据，使用传统方式: {self.mapping_file}")
        
        self.stock_to_sector = {}
        self.sector_to_stocks = {}
        self.stock_to_id = {}
        self.sector_to_id = {}
        self.id_to_stock = {}
        self.id_to_sector = {}
        
        # 加载或创建映射数据
        self._load_or_create_mapping()
    
    def _create_default_mapping(self) -> Dict:
        """创建默认的股票板块映射"""
        return {
            # 科技股
            "sz000001": {"sector": "科技", "sector_en": "technology", "name": "平安银行"},
            "sz000002": {"sector": "房地产", "sector_en": "real_estate", "name": "万科A"},
            "sz000858": {"sector": "食品饮料", "sector_en": "food_beverage", "name": "五粮液"},
            "sz000876": {"sector": "新能源", "sector_en": "new_energy", "name": "新希望"},
            
            # 银行股
            "sh600036": {"sector": "银行", "sector_en": "banking", "name": "招商银行"},
            "sh601318": {"sector": "保险", "sector_en": "insurance", "name": "中国平安"},
            "sh600519": {"sector": "食品饮料", "sector_en": "food_beverage", "name": "贵州茅台"},
            "sh600887": {"sector": "电力", "sector_en": "power", "name": "伊利股份"},
            
            # 新能源汽车
            "sz002594": {"sector": "新能源汽车", "sector_en": "new_energy_vehicle", "name": "比亚迪"},
            "sh600104": {"sector": "汽车", "sector_en": "automobile", "name": "上汽集团"},
            
            # 医药股
            "sz000661": {"sector": "医药", "sector_en": "pharmaceutical", "name": "长春高新"},
            "sh600276": {"sector": "医药", "sector_en": "pharmaceutical", "name": "恒瑞医药"},
            
            # 科技股
            "sz000063": {"sector": "科技", "sector_en": "technology", "name": "中兴通讯"},
            "sh600570": {"sector": "科技", "sector_en": "technology", "name": "恒生电子"},
            
            # 周期股
            "sh600019": {"sector": "钢铁", "sector_en": "steel", "name": "宝钢股份"},
            "sh601899": {"sector": "煤炭", "sector_en": "coal", "name": "紫金矿业"},
            
            # 消费股
            "sh600809": {"sector": "零售", "sector_en": "retail", "name": "山西汾酒"},
            "sz000895": {"sector": "零售", "sector_en": "retail", "name": "双汇发展"},
            
            # 创业板
            "sz300059": {"sector": "医药", "sector_en": "pharmaceutical", "name": "东方财富"},
            "sz300274": {"sector": "科技", "sector_en": "technology", "name": "阳光电源"},
            "sz300750": {"sector": "新能源", "sector_en": "new_energy", "name": "宁德时代"},
        }
    
    def _auto_detect_sector_from_code(self, stock_code: str) -> Dict:
        """根据股票代码自动推断板块信息"""
        # 基于股票代码的一些启发式规则
        default_info = {
            "sector": "其他", 
            "sector_en": "others", 
            "name": f"股票{stock_code}"
        }
        
        # 根据前缀推断市场
        if stock_code.startswith("sh"):
            # 上海股票
            if stock_code.startswith("sh600"):
                # 沪市主板
                code_num = int(stock_code[5:])
                if 600000 <= code_num <= 600099:
                    default_info.update({"sector": "银行", "sector_en": "banking"})
                elif 600100 <= code_num <= 600199:
                    default_info.update({"sector": "汽车", "sector_en": "automobile"})
                elif 600500 <= code_num <= 600599:
                    default_info.update({"sector": "食品饮料", "sector_en": "food_beverage"})
                elif 600800 <= code_num <= 600899:
                    default_info.update({"sector": "消费", "sector_en": "consumer"})
            elif stock_code.startswith("sh601"):
                # 多为大盘蓝筹
                default_info.update({"sector": "蓝筹", "sector_en": "blue_chip"})
        
        elif stock_code.startswith("sz"):
            # 深圳股票
            if stock_code.startswith("sz000"):
                # 深市主板
                default_info.update({"sector": "主板", "sector_en": "main_board"})
            elif stock_code.startswith("sz002"):
                # 中小板
                default_info.update({"sector": "中小板", "sector_en": "sme_board"})
            elif stock_code.startswith("sz300"):
                # 创业板
                default_info.update({"sector": "创业板", "sector_en": "growth_board"})
            elif stock_code.startswith("sz301"):
                # 创业板注册制
                default_info.update({"sector": "创业板", "sector_en": "growth_board"})
        
        return default_info
    
    def _load_or_create_mapping(self):
        """加载或创建映射数据"""
        
        # 优先使用CSV数据
        if self.csv_mapping_file and os.path.exists(self.csv_mapping_file):
            mapping_data = self._load_csv_mapping()
        # 回退到JSON数据
        elif self.mapping_file and os.path.exists(self.mapping_file):
            mapping_data = self._load_json_mapping()
        else:
            print("⚠️ 未找到任何映射文件，使用默认映射")
            mapping_data = self._create_default_mapping()
            if self.mapping_file:
                self.save_mapping(mapping_data)
        
        self._build_mappings(mapping_data)
    
    def _load_csv_mapping(self) -> Dict:
        """从CSV文件加载真实的板块映射数据"""
        try:
            df = pd.read_csv(self.csv_mapping_file, encoding='utf-8-sig')
            print(f"✅ 加载新整合的板块数据: {self.csv_mapping_file}")
            print(f"📊 板块映射统计: {len(df)}只股票, {df['industry'].nunique()}个行业, {df['primary_concept'].nunique()}个主要概念")
            
            mapping_data = {}
            for _, row in df.iterrows():
                stock_code = row['stock_code']
                mapping_data[stock_code] = {
                    'sector': row['industry'],  # 使用行业作为主要板块
                    'sector_en': self._translate_sector_to_english(row['industry']),
                    'name': row['stock_name'],
                    'primary_concept': row['primary_concept'],
                    'all_concepts': row['all_concepts'],
                    'region': row['region']
                }
            
            return mapping_data
            
        except Exception as e:
            print(f"❌ 加载CSV映射文件失败: {str(e)}")
            return self._create_default_mapping()
    
    def _load_json_mapping(self) -> Dict:
        """从JSON文件加载映射数据"""
        try:
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            print(f"✅ 加载股票板块映射文件: {self.mapping_file}")
            print(f"📊 映射统计: {len(mapping_data)}只股票, {len(set(item['sector'] for item in mapping_data.values()))}个板块")
            return mapping_data
        except Exception as e:
            print(f"❌ 加载JSON映射文件失败: {str(e)}")
            return self._create_default_mapping()
    
    def _translate_sector_to_english(self, chinese_sector: str) -> str:
        """将中文板块名称翻译为英文"""
        translation_map = {
            '银行': 'banking',
            '证券': 'securities', 
            '保险': 'insurance',
            '房地产': 'real_estate',
            '食品饮料': 'food_beverage',
            '医药生物': 'pharmaceutical',
            '电子': 'electronics',
            '计算机': 'computer',
            '通信': 'communication',
            '汽车': 'automobile',
            '机械设备': 'machinery',
            '电力设备': 'power_equipment',
            '新能源': 'new_energy',
            '化工': 'chemical',
            '建筑材料': 'construction_materials',
            '钢铁': 'steel',
            '有色金属': 'non_ferrous_metals',
            '煤炭': 'coal',
            '石油石化': 'petrochemical',
            '交通运输': 'transportation',
            '公用事业': 'utilities',
            '商业贸易': 'commercial_trade',
            '休闲服务': 'leisure_services',
            '纺织服装': 'textile_clothing',
            '轻工制造': 'light_manufacturing',
            '农林牧渔': 'agriculture',
            '综合': 'diversified',
            '国防军工': 'defense',
            '传媒': 'media',
            '家用电器': 'home_appliances',
            '建筑装饰': 'construction_decoration'
        }
        
        return translation_map.get(chinese_sector, 'others')
    
    def _build_mappings(self, mapping_data: Dict):
        """构建各种映射关系"""
        # 保存原始映射数据
        self.original_mapping_data = mapping_data
        
        # 构建股票到板块映射
        for stock_code, info in mapping_data.items():
            self.stock_to_sector[stock_code] = info["sector"]
        
        # 构建板块到股票映射
        for stock_code, sector in self.stock_to_sector.items():
            if sector not in self.sector_to_stocks:
                self.sector_to_stocks[sector] = []
            self.sector_to_stocks[sector].append(stock_code)
        
        # 构建ID映射（用于模型embedding）
        unique_stocks = sorted(list(self.stock_to_sector.keys()))
        unique_sectors = sorted(list(set(self.stock_to_sector.values())))
        
        # 股票ID映射
        for i, stock in enumerate(unique_stocks):
            self.stock_to_id[stock] = i
            self.id_to_stock[i] = stock
        
        # 板块ID映射
        for i, sector in enumerate(unique_sectors):
            self.sector_to_id[sector] = i
            self.id_to_sector[i] = sector
        
        print(f"📊 映射统计: {len(unique_stocks)}只股票, {len(unique_sectors)}个板块")
    
    def get_stock_info(self, stock_code: str) -> Dict:
        """获取股票信息"""
        if stock_code in self.stock_to_sector:
            # 从原始映射数据中获取完整信息
            full_info = self.original_mapping_data.get(stock_code, {})
            return {
                "stock_code": stock_code,
                "sector": self.stock_to_sector[stock_code],
                "stock_id": self.stock_to_id[stock_code],
                "sector_id": self.sector_to_id[self.stock_to_sector[stock_code]],
                "name": full_info.get("name", f"股票{stock_code}"),
                "primary_concept": full_info.get("primary_concept", ""),
                "all_concepts": full_info.get("all_concepts", ""),
                "region": full_info.get("region", "")
            }
        else:
            # 自动推断
            auto_info = self._auto_detect_sector_from_code(stock_code)
            print(f"🔍 自动推断股票 {stock_code} 的板块信息: {auto_info['sector']}")
            
            # 动态添加到映射中
            self.add_stock(stock_code, auto_info["sector"], auto_info["name"])
            return self.get_stock_info(stock_code)
    
    def add_stock(self, stock_code: str, sector: str, name: str = ""):
        """添加新股票"""
        # 更新映射
        self.stock_to_sector[stock_code] = sector
        
        if sector not in self.sector_to_stocks:
            self.sector_to_stocks[sector] = []
        self.sector_to_stocks[sector].append(stock_code)
        
        # 重新构建ID映射
        unique_stocks = sorted(list(self.stock_to_sector.keys()))
        unique_sectors = sorted(list(set(self.stock_to_sector.values())))
        
        # 重建股票ID映射
        self.stock_to_id.clear()
        self.id_to_stock.clear()
        for i, stock in enumerate(unique_stocks):
            self.stock_to_id[stock] = i
            self.id_to_stock[i] = stock
        
        # 重建板块ID映射
        self.sector_to_id.clear()
        self.id_to_sector.clear()
        for i, sector in enumerate(unique_sectors):
            self.sector_to_id[sector] = i
            self.id_to_sector[i] = sector
        
        print(f"➕ 添加股票: {stock_code} -> {sector}")
    
    def get_sector_stocks(self, sector: str) -> List[str]:
        """获取板块内所有股票"""
        return self.sector_to_stocks.get(sector, [])
    
    def get_all_sectors(self) -> List[str]:
        """获取所有板块"""
        return list(self.sector_to_stocks.keys())
    
    def get_all_stocks(self) -> List[str]:
        """获取所有股票"""
        return list(self.stock_to_sector.keys())
    
    def save_mapping(self, mapping_data: Dict = None):
        """保存映射到文件"""
        if mapping_data is None:
            # 从当前状态重构mapping_data
            mapping_data = {}
            for stock_code, sector in self.stock_to_sector.items():
                mapping_data[stock_code] = {
                    "sector": sector,
                    "sector_en": sector.lower().replace(" ", "_"),
                    "name": f"股票{stock_code}"
                }
        
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 保存股票板块映射到: {self.mapping_file}")
    
    def get_sector_correlation_matrix(self, stock_returns: pd.DataFrame) -> pd.DataFrame:
        """计算板块间相关性矩阵"""
        # stock_returns: DataFrame with columns as stock codes
        sector_returns = {}
        
        for sector in self.get_all_sectors():
            sector_stocks = self.get_sector_stocks(sector)
            # 获取板块内股票的平均收益率
            available_stocks = [s for s in sector_stocks if s in stock_returns.columns]
            if available_stocks:
                sector_returns[sector] = stock_returns[available_stocks].mean(axis=1)
        
        if sector_returns:
            sector_df = pd.DataFrame(sector_returns)
            return sector_df.corr()
        else:
            return pd.DataFrame()

    def print_mapping_summary(self):
        """打印映射摘要"""
        print("\n📋 股票板块映射摘要:")
        print("=" * 50)
        
        for sector in sorted(self.get_all_sectors()):
            stocks = self.get_sector_stocks(sector)
            print(f"🏷️  {sector} ({len(stocks)}只):")
            for stock in sorted(stocks)[:5]:  # 只显示前5只
                print(f"   {stock}")
            if len(stocks) > 5:
                print(f"   ... 还有{len(stocks)-5}只")
            print()


if __name__ == "__main__":
    # 测试代码
    mapping = StockSectorMapping()
    mapping.print_mapping_summary()
    
    # 测试获取股票信息
    test_codes = ['sh600519', 'sz000001', 'sz301636']
    
    for code in test_codes:
        info = mapping.get_stock_info(code)
        print(f"股票 {code}: {info}")