# -*- coding: utf-8 -*-
# @微信:samequant
# @网站:涨停客量化zhangtingke.com
# @更多源码下载地址: https://zhangtingke.com/download
# @优化版本：消除重复代码，提高可维护性

import os
import pandas as pd
import requests
import time
import random
import datetime
import json
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod

# 导入处理 - 支持直接运行和模块导入
try:
    from ..get_opening_calendar_from_szse.get_opening_calendar_from_szse import Get_stock_opening_calendar
except ImportError:
    # 直接运行时的导入
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    try:
        from get_opening_calendar_from_szse.get_opening_calendar_from_szse import Get_stock_opening_calendar
    except ImportError:
        # 如果仍然无法导入，提供一个简单的替代实现
        class Get_stock_opening_calendar:
            def __init__(self):
                pass
            def get_calendar(self):
                return []

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 6000)


class Config:
    """配置类，统一管理所有配置信息"""
    
    # 网络请求配置
    DEFAULT_TIMEOUT = 30
    DEFAULT_MAX_RETRIES = 10
    DEFAULT_SLEEP_TIME = 1
    
    # 用户代理
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ]
    
    # API 基础URL
    EASTMONEY_BASE_URLS = {
        'quote': 'http://push2.eastmoney.com/api/qt',
        'history': 'http://push2his.eastmoney.com/api/qt',
        'list': 'http://push2.eastmoney.com/api/qt/clist',
        'stock': 'http://push2.eastmoney.com/api/qt/stock'
    }
    
    # 上交所和深交所配置
    SSE_CONFIG = {
        'headers': {
            'X-Requested-With': 'XMLHttpRequest',
            'User-Agent': USER_AGENTS[0],
            'Referer': 'http://www.sse.com.cn/assortment/stock/list/share/'
        },
        'base_url': 'http://query.sse.com.cn'
    }
    
    SZSE_CONFIG = {
        'headers': {
            'user-agent': USER_AGENTS[0]
        },
        'base_url': 'http://www.szse.cn/api/report/ShowReport'
    }
    
    # 字段映射配置
    FIELD_MAPPINGS = {
        'stock_basic': {
            'f12': '股票代码', 'f14': '股票名称', 'f2': '最新价', 'f3': '涨跌幅',
            'f4': '涨跌额', 'f5': '成交量', 'f6': '成交额', 'f7': '振幅',
            'f15': '最高价', 'f16': '最低价', 'f17': '开盘价', 'f18': '昨收价',
            'f20': '总市值', 'f21': '流通市值', 'f8': '换手率', 'f9': '市盈率',
            'f23': '市净率', 'f11': '涨速', 'f10': '量比'
        },
        'money_flow': {
            'f62': '主力净流入', 'f184': '主力净流入占比',
            'f66': '超大单净流入', 'f69': '超大单净流入占比',
            'f72': '大单净流入', 'f75': '大单净流入占比',
            'f78': '中单净流入', 'f81': '小单净流入'
        },
        'sector': {
            'f12': '代码', 'f14': '名称', 'f2': '最新价', 'f3': '涨跌幅',
            'f4': '涨跌额', 'f5': '成交量', 'f6': '成交额', 'f7': '振幅',
            'f84': '总股本', 'f124': '更新时间'
        }
    }


class BaseNetworkClient:
    """基础网络客户端，统一处理网络请求"""
    
    def __init__(self):
        self.session = requests.Session()
    
    def make_request(self, url: str, headers: Dict = None, params: Dict = None, 
                    max_retries: int = Config.DEFAULT_MAX_RETRIES, 
                    sleep_time: float = Config.DEFAULT_SLEEP_TIME,
                    timeout: int = Config.DEFAULT_TIMEOUT) -> requests.Response:
        """
        统一的网络请求方法
        """
        if headers is None:
            headers = {'User-Agent': random.choice(Config.USER_AGENTS)}
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    url=url, 
                    headers=headers, 
                    params=params,
                    timeout=timeout
                )
                response.raise_for_status()
                return response
                
            except Exception as e:
                print(f'请求失败 (尝试 {attempt + 1}/{max_retries}): {e}')
                if attempt < max_retries - 1:
                    time.sleep(sleep_time)
                else:
                    raise ValueError(f'网络请求失败，已达到最大重试次数: {e}')
    
    def download_file(self, url: str, save_path: str, headers: Dict = None, 
                     params: Dict = None) -> bool:
        """
        下载文件到指定路径
        """
        try:
            response = self.make_request(url, headers, params)
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=10000):
                    if chunk:
                        f.write(chunk)
            
            print(f'文件下载成功: {save_path}')
            return True
            
        except Exception as e:
            print(f'文件下载失败: {e}')
            return False


class BaseDataProcessor:
    """基础数据处理类"""
    
    @staticmethod
    def symbol_to_stock_code(symbol: Union[str, int]) -> str:
        """
        统一的股票代码格式化方法
        """
        if len(str(symbol)) < 6:
            symbol = str(symbol).rjust(6, '0')
        elif len(str(symbol)) == 6:
            symbol = str(symbol)
        else:
            return str(symbol)
        
        if symbol.startswith('6'):
            return 'sh' + symbol
        elif symbol.startswith(('0', '3')):
            return 'sz' + symbol
        elif symbol.startswith(('4', '8', '9')):
            return 'bj' + symbol
        else:
            return symbol
    
    @staticmethod
    def rename_dataframe_columns(df: pd.DataFrame, field_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        统一的DataFrame列重命名方法
        """
        # 只重命名存在的列
        available_mappings = {k: v for k, v in field_mapping.items() if k in df.columns}
        return df.rename(columns=available_mappings)
    
    @staticmethod
    def convert_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        统一的数值列转换方法
        """
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    @staticmethod
    def clean_and_save_dataframe(df: pd.DataFrame, save_path: str, 
                                encoding: str = 'utf-8') -> bool:
        """
        统一的DataFrame清理和保存方法
        """
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存文件
            df.to_csv(save_path, index=False, encoding=encoding)
            print(f'数据保存成功: {save_path} (共{len(df)}条记录)')
            return True
            
        except Exception as e:
            print(f'数据保存失败: {e}')
            return False


class EastmoneyAPIClient(BaseNetworkClient):
    """东方财富API客户端，统一处理东方财富相关的API调用"""
    
    def __init__(self):
        super().__init__()
        self.base_urls = Config.EASTMONEY_BASE_URLS
    
    def _get_secid(self, stock_code: str) -> str:
        """获取secid格式"""
        if len(stock_code) == 6:
            symbol = stock_code
        elif len(stock_code) == 8:
            symbol = stock_code[2:]
        else:
            return ""
        
        if symbol[0] == '6':
            return f"1.{symbol}"
        elif symbol[0] in ['0', '3']:
            return f"0.{symbol}"
        else:
            return f"0.{symbol}"
    
    def _get_timestamp(self) -> int:
        """获取13位时间戳"""
        return int(time.time() * 1000)
    
    def get_stock_realtime_data(self, stock_codes: Union[str, List[str]], 
                               fields: str = None) -> pd.DataFrame:
        """
        获取股票实时数据的统一方法
        """
        if isinstance(stock_codes, str):
            stock_codes = [stock_codes]
        
        if fields is None:
            fields = 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152'
        
        # 构建请求参数
        fs_param = 'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048'
        
        all_data = []
        page_size = 200
        
        for page in range(1, 50):  # 最多50页
            params = {
                'pn': page,
                'pz': page_size,
                'po': '1',
                'np': '1',
                'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                'fltt': '2',
                'invt': '2',
                'fid': 'f3',
                'fs': fs_param,
                'fields': fields,
                '_': self._get_timestamp()
            }
            
            url = f"{self.base_urls['list']}/get"
            
            try:
                response = self.make_request(url, params=params)
                data = response.json()
                
                if 'data' in data and data['data'] and 'diff' in data['data']:
                    page_data = data['data']['diff']
                    if not page_data:
                        break
                    all_data.extend(page_data)
                    
                    if len(page_data) < page_size:
                        break
                else:
                    break
                    
                time.sleep(0.1)  # 避免请求过频
                
            except Exception as e:
                print(f'获取第{page}页数据失败: {e}')
                break
        
        if all_data:
            df = pd.DataFrame(all_data)
            # 统一处理股票代码格式
            if 'f12' in df.columns:
                df['f12'] = df['f12'].apply(BaseDataProcessor.symbol_to_stock_code)
            return df
        
        return pd.DataFrame()
    
    def get_sector_data(self, sector_type: str = 'industry', 
                       sort_field: str = 'f3') -> pd.DataFrame:
        """
        获取板块数据的统一方法
        """
        if sector_type == 'industry':
            fs_param = 'm:90+t:2+f:!50'
        elif sector_type == 'concept':
            fs_param = 'm:90+t:3+f:!50'
        else:
            raise ValueError("sector_type must be 'industry' or 'concept'")
        
        all_data = []
        page_size = 50
        
        fields = 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f62,f184,f66,f69,f72,f75,f78,f81,f82,f84,f85,f86,f87,f124'
        
        page = 1
        while True:
            params = {
                'pn': page,
                'pz': page_size,
                'po': '1',
                'np': '1',
                'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                'fltt': '2',
                'invt': '2',
                'fid': sort_field,
                'fs': fs_param,
                'fields': fields,
                '_': self._get_timestamp()
            }
            
            url = f"{self.base_urls['list']}/get"
            
            try:
                response = self.make_request(url, params=params)
                data = response.json()
                
                if 'data' in data and data['data'] and 'diff' in data['data']:
                    page_data = data['data']['diff']
                    if not page_data:
                        break
                    all_data.extend(page_data)
                    
                    if len(page_data) < page_size:
                        break
                else:
                    break
                    
                page += 1
                time.sleep(0.1)  # 避免请求过频
                
            except Exception as e:
                print(f'获取第{page}页板块数据失败: {e}')
                break
        
        if all_data:
            df = pd.DataFrame(all_data)
            print(f'✅ 成功获取 {len(df)} 个{sector_type}板块数据')
            return df
        
        return pd.DataFrame()


class OptimizedDownloadStocksList(BaseNetworkClient, BaseDataProcessor):
    """优化后的股票列表下载类"""
    
    def __init__(self):
        super().__init__()
        self.file_full_dir = os.path.dirname(os.path.abspath(__file__))
        self.path_dir = os.path.join(self.file_full_dir, 'stockcode_list')
        self.error_dir = os.path.join(self.file_full_dir, 'error_txt')
        
        # 创建目录
        os.makedirs(self.path_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)
        
        self.all_stocklist_path = os.path.join(self.path_dir, 'all_stock_list.csv')
    
    def _download_and_process_excel(self, url: str, filename: str, 
                                   column_mappings: Dict[str, str],
                                   market_prefix: str,
                                   status: str = '正常交易',
                                   extra_params: Dict = None) -> bool:
        """
        统一的Excel下载和处理方法
        """
        file_path = os.path.join(self.path_dir, f"{filename}.xls")
        csv_path = os.path.join(self.path_dir, f"{filename}.csv")
        
        try:
            # 下载文件
            headers = Config.SSE_CONFIG['headers'] if 'sse' in url else Config.SZSE_CONFIG['headers']
            if not self.download_file(url, file_path, headers, extra_params):
                return False
            
            # 处理Excel文件
            df = pd.read_excel(file_path, dtype=str)
            
            if df.empty:
                print(f'{filename} 数据为空')
                return False
            
            # 重命名列
            df = self.rename_dataframe_columns(df, column_mappings)
            
            # 添加市场前缀
            if '股票代码' in df.columns:
                df['股票代码'] = market_prefix + df['股票代码']
            
            # 处理日期列
            date_columns = ['上市日期', '终止/暂停上市日期']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
            
            # 添加状态列
            df['上市状态'] = status
            if '终止/暂停上市日期' not in df.columns:
                df['终止/暂停上市日期'] = None
            
            # 选择需要的列
            required_columns = ['股票代码', '名称', '上市日期', '上市状态', '终止/暂停上市日期']
            df = df[[col for col in required_columns if col in df.columns]]
            
            # 保存CSV
            return self.clean_and_save_dataframe(df, csv_path)
            
        except Exception as e:
            print(f'{filename} 处理失败: {e}')
            return False
    
    def download_sh_stocks(self) -> bool:
        """下载上交所股票列表"""
        # 主板A股
        main_url = 'http://query.sse.com.cn//sseQuery/commonExcelDd.do?sqlId=COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L&type=inParams&CSRC_CODE=&STOCK_CODE=&REG_PROVINCE=&STOCK_TYPE=1&COMPANY_STATUS=2,4,5,7,8'
        main_mappings = {'A股代码': '股票代码', '证券简称': '名称'}
        
        # 科创板
        kcb_url = 'http://query.sse.com.cn//sseQuery/commonExcelDd.do?sqlId=COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L&type=inParams&CSRC_CODE=&STOCK_CODE=&REG_PROVINCE=&STOCK_TYPE=8&COMPANY_STATUS=2,4,5,7,8'
        
        success1 = self._download_and_process_excel(main_url, '上交所主板A股', main_mappings, 'sh')
        success2 = self._download_and_process_excel(kcb_url, '上交所科创板', main_mappings, 'sh')
        
        return success1 and success2
    
    def download_sz_stocks(self) -> bool:
        """下载深交所股票列表"""
        url = Config.SZSE_CONFIG['base_url']
        params = {'SHOWTYPE': 'xlsx', 'CATALOGID': '1110', 'TABKEY': 'tab1', 'random': random.random()}
        mappings = {'A股简称': '名称', 'A股代码': '股票代码', 'A股上市日期': '上市日期'}
        
        return self._download_and_process_excel(url, '深交所A股列表', mappings, 'sz', '正常交易', params)
    
    def merge_all_stocks(self) -> pd.DataFrame:
        """合并所有股票列表"""
        all_files = [
            '深交所A股列表.csv', '上交所主板A股.csv', '上交所科创板.csv'
        ]
        
        all_dfs = []
        for filename in all_files:
            file_path = os.path.join(self.path_dir, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, dtype={'股票代码': str}, encoding='utf-8')
                all_dfs.append(df)
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df['上市日期'] = pd.to_datetime(combined_df['上市日期'])
            combined_df = combined_df.sort_values(['股票代码', '上市日期']).reset_index(drop=True)
            
            self.clean_and_save_dataframe(combined_df, self.all_stocklist_path)
            return combined_df
        
        return pd.DataFrame()
    
    def main(self) -> pd.DataFrame:
        """主执行方法"""
        print("开始下载股票列表...")
        
        # 下载各个市场的股票列表
        self.download_sh_stocks()
        time.sleep(1)
        self.download_sz_stocks()
        time.sleep(1)
        
        # 合并所有列表
        return self.merge_all_stocks()


class OptimizedSpiderFunc(BaseDataProcessor):
    """优化后的爬虫功能类"""
    
    def __init__(self):
        self.file_full_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(self.file_full_dir)
        self.stock_hisdata_dir = os.path.join(project_root, 'data', 'datas_em')
        os.makedirs(self.stock_hisdata_dir, exist_ok=True)
        
        # 初始化API客户端
        self.eastmoney_client = EastmoneyAPIClient()
    
    def get_realtime_market_data(self) -> pd.DataFrame:
        """获取实时市场数据"""
        df = self.eastmoney_client.get_stock_realtime_data([])
        
        if not df.empty:
            # 应用字段映射
            df = self.rename_dataframe_columns(df, Config.FIELD_MAPPINGS['stock_basic'])
            
            # 添加交易日期
            try:
                calendar = Get_stock_opening_calendar()
                recent_date = calendar.get_recent_trade_date()
                df['交易日期'] = pd.to_datetime(recent_date).strftime("%Y-%m-%d")
            except:
                df['交易日期'] = datetime.datetime.now().strftime("%Y-%m-%d")
            
            # 过滤北交所股票
            df = df[df['股票代码'].str[:2] != 'bj']
            
            # 转换数值列
            numeric_cols = ['最新价', '涨跌幅', '涨跌额', '成交量', '成交额', '最高价', '最低价', 
                           '开盘价', '总市值', '流通市值', '换手率', '市盈率', '市净率', '涨速']
            df = self.convert_numeric_columns(df, numeric_cols)
            
            # 成交量转换（手转股）
            if '成交量' in df.columns:
                df['成交量'] = df['成交量'] * 100
        
        return df
    
    def get_industry_data(self, sort_field: str = 'f3') -> pd.DataFrame:
        """获取行业板块数据"""
        df = self.eastmoney_client.get_sector_data('industry', sort_field)
        
        if not df.empty:
            # 选择并重命名列
            select_cols = ['f12', 'f14', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f15', 'f16', 'f17', 'f18', 'f20', 'f21', 'f62', 'f184', 'f84', 'f124']
            available_cols = [col for col in select_cols if col in df.columns]
            df = df[available_cols]
            
            mappings = {
                'f12': '行业代码', 'f14': '行业名称', 'f2': '最新价', 'f3': '涨跌幅',
                'f4': '涨跌额', 'f5': '成交量', 'f6': '成交额', 'f7': '振幅',
                'f15': '最高价', 'f16': '最低价', 'f17': '开盘价', 'f18': '昨收价',
                'f20': '总市值', 'f21': '流通市值', 'f62': '主力净流入', 'f184': '主力净流入占比',
                'f84': '总股本', 'f124': '更新时间'
            }
            
            df = self.rename_dataframe_columns(df, mappings)
            
            # 转换数值列
            numeric_cols = ['最新价', '涨跌幅', '涨跌额', '成交量', '成交额', '振幅', '最高价', '最低价', 
                           '开盘价', '昨收价', '总市值', '流通市值', '主力净流入', '主力净流入占比', '总股本']
            df = self.convert_numeric_columns(df, numeric_cols)
        
        return df
    
    def get_concept_data(self, sort_field: str = 'f3') -> pd.DataFrame:
        """获取概念板块数据"""
        df = self.eastmoney_client.get_sector_data('concept', sort_field)
        
        if not df.empty:
            # 选择并重命名列
            select_cols = ['f12', 'f14', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f15', 'f16', 'f17', 'f18', 'f20', 'f21', 'f62', 'f184', 'f84', 'f124']
            available_cols = [col for col in select_cols if col in df.columns]
            df = df[available_cols]
            
            mappings = {
                'f12': '概念代码', 'f14': '概念名称', 'f2': '最新价', 'f3': '涨跌幅',
                'f4': '涨跌额', 'f5': '成交量', 'f6': '成交额', 'f7': '振幅',
                'f15': '最高价', 'f16': '最低价', 'f17': '开盘价', 'f18': '昨收价',
                'f20': '总市值', 'f21': '流通市值', 'f62': '主力净流入', 'f184': '主力净流入占比',
                'f84': '总股本', 'f124': '更新时间'
            }
            
            df = self.rename_dataframe_columns(df, mappings)
            
            # 转换数值列
            numeric_cols = ['最新价', '涨跌幅', '涨跌额', '成交量', '成交额', '振幅', '最高价', '最低价', 
                           '开盘价', '昨收价', '总市值', '流通市值', '主力净流入', '主力净流入占比', '总股本']
            df = self.convert_numeric_columns(df, numeric_cols)
        
        return df
    
    def get_stock_history_data(self, stock_code: str, period: str = '日', fqt: str = '0') -> pd.DataFrame:
        """
        获取个股历史数据
        :param stock_code: 股票代码，如 'sh600519' 或 '600519'
        :param period: 数据周期，'日', '周', '5分钟', '30分钟', '1分钟'
        :param fqt: 复权类型，'0'不复权, '1'前复权, '2'后复权
        :return: 包含历史K线数据的DataFrame
        """
        if len(stock_code) == 6:
            symbol = stock_code
        elif len(stock_code) == 8:
            symbol = stock_code[2:]
        else:
            return pd.DataFrame()

        if symbol[0] == '6':
            market = '1'
        elif symbol[0] == '0' or symbol[0] == '3':
            market = '0'
        else:
            market = '0'
        secid = market + '.' + symbol

        # 设置K线周期
        period_map = {'周': '102', '日': '101', '5分钟': '5', '30分钟': '30', '1分钟': '1'}
        klt = period_map.get(period, '101')

        # 构建API URL
        url = f'http://22.push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt={klt}&fqt={fqt}&end=20500101&lmt=20000&_={int(time.time() * 1000)}'
        
        try:
            headers = {'User-Agent': random.choice(Config.USER_AGENTS)}
            resp = requests.get(url=url, headers=headers, timeout=30)
            content = resp.content.decode('utf-8')
            
            if 'data' in content:
                data = resp.json()['data']['klines']
                lst_data = []

                for i in data:
                    lst_i = i.split(',')
                    lst_data.append(lst_i)
                
                df = pd.DataFrame(lst_data)
                df.columns = ['交易日期', '开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                
                # 转换数值列
                numeric_cols = ['开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                df = self.convert_numeric_columns(df, numeric_cols)
                
                return df
            else:
                print(f'获取历史行情出错！股票代码: {stock_code}')
                return pd.DataFrame()
                
        except Exception as e:
            print(f'获取历史行情异常！股票代码: {stock_code}, 错误: {str(e)}')
            return pd.DataFrame()

    def get_stock_history_data_with_market_cap_from_163(self, stock_code: str, start_date: str = '20000101', end_date: str = None) -> pd.DataFrame:
        """
        从网易163获取包含流通市值的历史数据
        :param stock_code: 股票代码，如 'sh600519' 或 '600519'
        :param start_date: 开始日期，格式：YYYYMMDD
        :param end_date: 结束日期，格式：YYYYMMDD，默认为今天
        :return: 包含历史K线数据和流通市值的DataFrame
        """
        import datetime
        
        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y%m%d')
        
        # 处理股票代码
        if len(stock_code) == 8 and stock_code.startswith(('sh', 'sz')):
            code = stock_code[2:]
            code_prefix = stock_code[:2]
        elif len(stock_code) == 6:
            code = stock_code
            # 根据代码判断市场
            if code[0] == '6':
                code_prefix = 'sh'
            else:
                code_prefix = 'sz'
        else:
            return pd.DataFrame()
        
        # 网易163的市场代码（与东财相反）
        if code[0] == '6':
            market_code = '0'  # 上交所
        else:
            market_code = '1'  # 深交所
        
        # 构建网易163 API URL - 包含流通市值字段MCAP
        url = f"http://quotes.money.163.com/service/chddata.html?code={market_code}{code}&start={start_date}&end={end_date}&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP"
        
        try:
            headers = {'User-Agent': random.choice(Config.USER_AGENTS)}
            response = requests.get(url=url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # 将响应内容保存到临时文件并读取
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            try:
                # 读取CSV数据
                df = pd.read_csv(temp_path, dtype={'股票代码': str}, sep=',', encoding='gbk')
                
                if len(df) >= 1:
                    # 处理数据
                    df = df.replace('None', 0)
                    df = df.fillna(value=0)
                    
                    # 重命名列名
                    rename_dict = {'日期': '交易日期', '前收盘': '前收盘价', '成交金额': '成交额'}
                    df.rename(columns=rename_dict, inplace=True)
                    
                    # 处理股票代码列
                    if '股票代码' in df.columns:
                        df['股票代码'] = df['股票代码'].str.strip()
                        df['股票代码'] = code_prefix + df['股票代码'].str[1:]
                    else:
                        df['股票代码'] = stock_code
                    
                    # 转换数值列
                    numeric_cols = ['收盘价', '最高价', '最低价', '开盘价', '前收盘价', '涨跌额', '涨跌幅', '换手率', '成交量', '成交额', '总市值', '流通市值']
                    available_numeric_cols = [col for col in numeric_cols if col in df.columns]
                    df[available_numeric_cols] = df[available_numeric_cols].astype(float)
                    
                    # 处理日期列
                    if '交易日期' in df.columns:
                        df['交易日期'] = pd.to_datetime(df['交易日期'])
                        df = df.sort_values(by=['交易日期'], ascending=True)
                        df.reset_index(drop=True, inplace=True)
                        # 将日期转换为字符串格式，与东财API保持一致
                        df['交易日期'] = df['交易日期'].dt.strftime('%Y-%m-%d')
                    
                    return df
                else:
                    return pd.DataFrame()
                    
            finally:
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            print(f'从网易163获取历史数据异常！股票代码: {stock_code}, 错误: {str(e)}')
            return pd.DataFrame()

    def get_stock_history_data_with_real_market_cap(self, stock_code: str, period: str = '日', fqt: str = '0') -> pd.DataFrame:
        """
        获取包含真实流通市值的历史数据
        通过获取当前流通股本，计算历史各日的流通市值
        :param stock_code: 股票代码，如 'sh600519' 或 '600519'
        :param period: 数据周期，'日', '周', '5分钟', '30分钟', '1分钟'
        :param fqt: 复权类型，'0'不复权, '1'前复权, '2'后复权
        :return: 包含历史K线数据和流通市值的DataFrame
        """
        # 先获取基础历史数据
        df_history = self.get_stock_history_data(stock_code, period, fqt)
        
        if df_history.empty:
            return df_history
        
        # 获取当前股票的流通股本信息
        try:
            # 直接使用东财单股票查询API获取流通市值
            if len(stock_code) == 6:
                symbol = stock_code
            elif len(stock_code) == 8:
                symbol = stock_code[2:]
            else:
                df_history['流通市值'] = None
                return df_history
            
            if symbol[0] == '6':
                market = '1'
            elif symbol[0] in ['0', '3']:
                market = '0'
            else:
                market = '0'
            secid = market + '.' + symbol
            
            # 使用东财单股票查询API获取流通股本
            info_url = f'http://push2.eastmoney.com/api/qt/stock/get?ut=fa5fd1943c7b386f172d6893dbfba10b&invt=2&fltt=1&fields=f84,f85&secid={secid}&_={int(time.time() * 1000)}'
            
            headers = {'User-Agent': random.choice(Config.USER_AGENTS)}
            resp = requests.get(url=info_url, headers=headers, timeout=30)
            
            if resp.status_code == 200:
                info_data = resp.json()
                if 'data' in info_data and info_data['data']:
                    stock_info = info_data['data']
                    liutong_gub = stock_info.get('f85', 0)  # 流通股本（股）
                    
                    if liutong_gub:
                        try:
                            liutong_gub = float(liutong_gub)
                            if liutong_gub > 0:
                                # 直接用流通股本计算流通市值 = 收盘价 * 流通股本
                                df_history['流通市值'] = df_history['收盘价'] * liutong_gub
                                
                                # 转换数值列
                                numeric_cols = ['开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率', '流通市值']
                                df_history = self.convert_numeric_columns(df_history, numeric_cols)
                                
                                return df_history
                            else:
                                # 如果流通股本为0，设置流通市值为空
                                df_history['流通市值'] = None
                                return df_history
                        except (ValueError, TypeError):
                            # 如果转换失败，设置流通市值为空
                            df_history['流通市值'] = None
                            return df_history
                    else:
                        # 如果获取不到流通股本数据，设置流通市值为空
                        df_history['流通市值'] = None
                        return df_history
                else:
                    # 如果API返回无效数据，设置流通市值为空
                    df_history['流通市值'] = None
                    return df_history
            else:
                # 如果API请求失败，设置流通市值为空
                df_history['流通市值'] = None
                return df_history
                
        except Exception as e:
            # 如果获取流通股本失败，设置流通市值为空但不影响其他数据
            df_history['流通市值'] = None
            print(f'获取流通市值信息失败 {stock_code}: {e}')
            return df_history

    def download_all_stocks_history_data(self) -> bool:
        """
        下载所有A股历史数据到data/datas_em目录
        """
        print("🔄 开始下载所有A股历史数据...")
        
        try:
            # 获取股票列表
            downloader = OptimizedDownloadStocksList()
            stock_list_path = downloader.all_stocklist_path
            
            # 检查股票列表文件是否存在
            if not os.path.exists(stock_list_path):
                print("❌ 股票列表文件不存在，请先运行股票列表下载")
                return False
            
            # 读取股票列表
            df_stocks = pd.read_csv(stock_list_path, dtype={'股票代码': str}, encoding='utf-8')
            total_stocks = len(df_stocks)
            
            print(f"📊 共需下载 {total_stocks} 只股票的历史数据")
            
            success_count = 0
            error_count = 0
            
            for i, row in df_stocks.iterrows():
                try:
                    code = row['股票代码']
                    print(f"🔄 正在处理: {code} ({i+1}/{total_stocks})")
                    
                    # 获取历史数据（包含真实流通市值，基于当前流通股本计算）
                    df_code = self.get_stock_history_data_with_real_market_cap(stock_code=code)
                    
                    if not df_code.empty:
                        # 保存数据
                        save_path = os.path.join(self.stock_hisdata_dir, f'{code}.csv')
                        self.clean_and_save_dataframe(df_code, save_path)
                        
                        success_count += 1
                        print(f'✅ {code} 数据保存完成')
                        
                        # 添加延迟避免请求过频
                        time.sleep(0.5)
                    else:
                        print(f'❌ {code} 未获取到有效数据')
                        error_count += 1
                        
                except Exception as e:
                    print(f'❌ {code} 处理出错: {e}')
                    error_count += 1
                    continue
            
            print(f"\n📊 下载完成统计:")
            print(f"   ✅ 成功: {success_count} 只")
            print(f"   ❌ 失败: {error_count} 只")
            print(f"   📁 数据保存目录: {self.stock_hisdata_dir}")
            
            return success_count > 0
            
        except Exception as e:
            print(f"❌ 批量下载过程中出现错误: {e}")
            return False

    def get_hot_concepts_from_eastmoney(self, limit: int = 50) -> pd.DataFrame:
        """
        获取热门概念板块排行
        :param limit: 返回数量限制
        :return: 包含热门概念的DataFrame
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }
        
        timestamp = int(time.time() * 1000)
        
        # 热门概念API（按涨跌幅排序）
        fields = 'f12,f14,f2,f3,f4,f5,f6,f7,f62,f184,f104,f105,f140,f141,f136'
        url = f'http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz={limit}&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:90+t:3+f:!50&fields={fields}&_={timestamp}'

        try:
            import requests
            resp = requests.get(url=url, headers=headers)
            resp_json = resp.json()
            
            if 'data' in resp_json and resp_json['data'] is not None:
                df = pd.DataFrame(resp_json['data']['diff'])
                
                if not df.empty:
                    # 选择并重命名列
                    df = df[['f12', 'f14', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f62', 'f184', 'f104', 'f105']]
                    
                    rename_dict = {
                        'f12': '概念代码',
                        'f14': '概念名称', 
                        'f2': '最新指数',
                        'f3': '涨跌幅',
                        'f4': '涨跌额',
                        'f5': '成交量',
                        'f6': '成交额',
                        'f7': '振幅',
                        'f62': '主力净流入',
                        'f184': '主力净流入占比',
                        'f104': '上涨家数',
                        'f105': '下跌家数'
                    }
                    
                    df.rename(columns=rename_dict, inplace=True)
                    
                    # 转换数据类型
                    numeric_cols = ['最新指数', '涨跌幅', '涨跌额', '成交量', '成交额', '振幅', '主力净流入', '主力净流入占比', '上涨家数', '下跌家数']
                    df[numeric_cols] = df[numeric_cols].astype(float)
                    
                    return df
            
            print('获取热门概念数据失败')
            return pd.DataFrame()
            
        except Exception as e:
            print(f'获取热门概念数据出错：{e}')
            return pd.DataFrame()

    def get_stock_industry_info_from_eastmoney(self, stock_code: str) -> dict:
        """
        获取个股所属行业和概念信息
        :param stock_code: 股票代码，如 'sh600519' 或 '600519'
        :return: 包含行业和概念信息的字典
        """
        if len(stock_code) == 6:
            symbol = stock_code
        elif len(stock_code) == 8:
            symbol = stock_code[2:]
        else:
            return {}

        if symbol[0] == '6':
            market = '1'
        elif symbol[0] == '0' or symbol[0] == '3':
            market = '0'
        else:
            market = '0'
        secid = market + '.' + symbol

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }

        timestamp = int(time.time() * 1000)
        
        try:
            import requests
            # 第一步：获取基本信息
            basic_url = f'http://push2.eastmoney.com/api/qt/stock/get?secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields=f57,f58,f84,f85,f86,f87,f127&_={timestamp}'
            
            resp = requests.get(url=basic_url, headers=headers)
            resp_json = resp.json()
            
            stock_info = {
                '股票代码': stock_code,
                '股票名称': '',
                '所属行业': '',
                '概念板块': '',
                '地区': '',
                '总股本': 0,
                '流通股': 0,
                '每股收益': 0,
                '每股净资产': 0
            }
            
            if 'data' in resp_json and resp_json['data'] is not None:
                data = resp_json['data']
                
                stock_info.update({
                    '股票名称': data.get('f58', ''),
                    '所属行业': data.get('f127', ''),
                    '总股本': data.get('f84', 0),
                    '流通股': data.get('f85', 0),
                    '每股收益': data.get('f86', 0),
                    '每股净资产': data.get('f87', 0)
                })
            
            # 第二步：使用多种方法获取概念板块和地区信息
            concept_found = False
            region_found = False
            
            # 方法1：使用完整字段的股票详情API
            try:
                detail_url = f'http://push2.eastmoney.com/api/qt/stock/get?secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields=f116,f117,f127,f128,f136,f173&_={timestamp}'
                
                detail_resp = requests.get(url=detail_url, headers=headers)
                if detail_resp.status_code == 200:
                    detail_data = detail_resp.json()
                    
                    if 'data' in detail_data and detail_data['data']:
                        data = detail_data['data']
                        
                        # 尝试多个可能的字段
                        concept_fields = ['f128', 'f116', 'f173']
                        region_fields = ['f136', 'f117']
                        
                        # 处理地区信息（f128字段实际是地区板块）
                        if 'f128' in data:
                            region_board = data['f128']
                            if region_board and str(region_board) not in ['-', '0', '']:
                                # 从"贵州板块"提取"贵州"
                                region_name = str(region_board).replace('板块', '')
                                stock_info['地区'] = region_name
                                region_found = True
                        
                        # 尝试获取真正的概念板块信息
                        if not concept_found:
                            try:
                                # 使用个股所属概念板块API
                                concept_api_url = f'http://push2.eastmoney.com/api/qt/slist/get?spt=1&fltt=2&invt=2&pi=0&pz=200&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fields=f12,f14,f103&fid=f3&secid={secid}&_={timestamp}'
                                
                                concept_resp = requests.get(url=concept_api_url, headers=headers)
                                if concept_resp.status_code == 200:
                                    concept_data = concept_resp.json()
                                    
                                    if 'data' in concept_data and concept_data['data'] and 'diff' in concept_data['data']:
                                        concepts = concept_data['data']['diff']
                                        
                                        # 查找当前股票的概念信息
                                        for item in concepts:
                                            if item.get('f12') == symbol:  # 找到当前股票
                                                concept_list = item.get('f103', '')
                                                if concept_list and concept_list != '-':
                                                    stock_info['概念板块'] = concept_list
                                                    concept_found = True
                                                    break
                            except Exception as e:
                                pass
                        
                        # 如果还没找到概念板块，设为空
                        if not concept_found:
                            stock_info['概念板块'] = ''
                
            except Exception as e:
                pass
            
            # 方法2：如果还没找到，从股票列表中查找
            if not concept_found or not region_found:
                try:
                    # 使用包含更多字段的股票列表API
                    list_url = f'http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:6+f:!2,m:0+t:13+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2,m:0+t:7+f:!2,m:1+t:3+f:!2&fields=f12,f14,f116,f117,f127,f128,f136,f173&_={timestamp}'
                    
                    list_resp = requests.get(url=list_url, headers=headers)
                    if list_resp.status_code == 200:
                        list_data = list_resp.json()
                        
                        if 'data' in list_data and list_data['data'] and 'diff' in list_data['data']:
                            stocks = list_data['data']['diff']
                            
                            # 查找当前股票
                            for stock in stocks:
                                if stock.get('f12') == symbol:  # f12是股票代码
                                    # 处理地区信息（f128是地区板块）
                                    if not region_found and 'f128' in stock:
                                        region_board = stock['f128']
                                        if region_board and str(region_board) not in ['-', '0', '']:
                                            # 从"贵州板块"提取"贵州"
                                            region_name = str(region_board).replace('板块', '')
                                            stock_info['地区'] = region_name
                                            region_found = True
                                    
                                    # 如果还没找到概念板块，尝试获取
                                    if not concept_found:
                                        try:
                                            # 使用个股所属概念板块API
                                            concept_api_url = f'http://push2.eastmoney.com/api/qt/slist/get?spt=1&fltt=2&invt=2&pi=0&pz=200&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fields=f12,f14,f103&fid=f3&secid={secid}&_={timestamp}'
                                            
                                            concept_resp = requests.get(url=concept_api_url, headers=headers)
                                            if concept_resp.status_code == 200:
                                                concept_data = concept_resp.json()
                                                
                                                if 'data' in concept_data and concept_data['data'] and 'diff' in concept_data['data']:
                                                    concepts = concept_data['data']['diff']
                                                    
                                                    # 查找当前股票的概念信息
                                                    for item in concepts:
                                                        if item.get('f12') == symbol:  # 找到当前股票
                                                            concept_list = item.get('f103', '')
                                                            if concept_list and concept_list != '-':
                                                                stock_info['概念板块'] = concept_list
                                                                concept_found = True
                                                                break
                                        except Exception as e:
                                            pass
                                    
                                    # 如果还是没找到，设为空
                                    if not concept_found:
                                        stock_info['概念板块'] = ''
                                    
                                    break
                
                except Exception as e:
                    pass
            
            # 如果还是没有获取到，使用默认值
            if not stock_info['概念板块']:
                stock_info['概念板块'] = ''
            if not stock_info['地区']:
                stock_info['地区'] = ''
                
            return stock_info
            
        except Exception as e:
            print(f'获取{stock_code}行业概念信息出错：{e}')
            return {}

    def get_historical_sector_data_from_eastmoney(self, sector_code: str, sector_type: str = 'industry', 
                                                  trading_days: int = 30, period: str = 'daily', 
                                                  is_incremental: bool = False, save_dir: str = None) -> pd.DataFrame:
        """
        获取板块历史数据
        :param sector_code: 板块代码
        :param sector_type: 板块类型，'industry'行业板块 或 'concept'概念板块
        :param trading_days: 获取最近多少个交易日，默认30个交易日
        :param period: 数据周期，'daily'日线 'weekly'周线 'monthly'月线
        :param is_incremental: 是否为增量更新模式
        :param save_dir: 保存目录，用于增量更新时查找现有数据
        :return: 包含历史数据的DataFrame
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
            'Referer': 'http://quote.eastmoney.com/'
        }
        
        # 增量更新逻辑
        start_date_override = None
        if is_incremental and save_dir:
            # 查找现有数据文件
            import os
            sector_name = f"板块{sector_code}"  # 临时名称，可能需要获取真实名称
            possible_filenames = [
                f"{sector_name}({sector_code})_daily_历史数据.csv",
                f"{sector_code}_daily_历史数据.csv"
            ]
            
            existing_file = None
            for filename in possible_filenames:
                file_path = os.path.join(save_dir, filename)
                if os.path.exists(file_path):
                    existing_file = file_path
                    break
            
            if existing_file:
                try:
                    existing_df = pd.read_csv(existing_file, encoding='utf-8-sig')
                    if len(existing_df) > 0 and '日期' in existing_df.columns:
                        latest_date = existing_df['日期'].max()
                        # 从最新日期的下一天开始获取
                        latest_dt = datetime.datetime.strptime(latest_date, '%Y-%m-%d')
                        next_day = latest_dt + datetime.timedelta(days=1)
                        start_date_override = next_day.strftime('%Y%m%d')
                        print(f"增量更新模式: 从 {latest_date} 之后开始获取数据")
                except Exception as e:
                    print(f"读取现有数据失败，使用常规获取模式: {e}")
        
        try:
            import requests
            # 计算日期范围（基于交易日数量）
            end_date = datetime.datetime.now().strftime('%Y%m%d')
            
            # 使用增量更新的起始日期覆盖
            if start_date_override:
                start_date = start_date_override
                print(f"增量更新数据，日期范围: {start_date} 至 {end_date}")
            elif trading_days is None or trading_days >= 1000:
                # 获取全部历史数据，设置一个很早的开始日期
                start_date = '20100101'  # 从2010年开始，涵盖大部分板块的历史
                print(f"获取全部历史数据，日期范围: {start_date} 至 {end_date}")
            else:
                # 为了确保获取到足够的交易日，实际获取天数要多一些（考虑周末和节假日）
                actual_days = trading_days * 2  # 大约两倍天数确保包含足够交易日
                start_dt = datetime.datetime.now() - datetime.timedelta(days=actual_days)
                start_date = start_dt.strftime('%Y%m%d')
                print(f"获取最近{trading_days}个交易日数据，日期范围: {start_date} 至 {end_date}")
            
            # 根据周期设置参数
            klt_map = {'daily': '101', 'weekly': '102', 'monthly': '103'}
            klt = klt_map.get(period, '101')
            
            # 根据板块类型设置secid前缀
            if sector_type == 'industry':
                secid = f"90.{sector_code}"  # 行业板块使用90前缀
            else:
                secid = f"90.{sector_code}"  # 概念板块也使用90前缀
            
            # 构建历史数据API URL
            if trading_days is None or trading_days >= 1000:
                # 获取全部历史数据，不限制数量
                url = f'http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt={klt}&fqt=1&beg={start_date}&end={end_date}&_={int(time.time() * 1000)}'
            else:
                # 获取指定数量的数据
                url = f'http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt={klt}&fqt=1&beg={start_date}&end={end_date}&lmt={trading_days}&_={int(time.time() * 1000)}'
            
            print(f"正在获取板块 {sector_code} 的历史数据...")
            print(f"请求URL: {url}")
            
            resp = requests.get(url=url, headers=headers, timeout=30)
            
            if resp.status_code != 200:
                print(f"请求失败，状态码: {resp.status_code}")
                return pd.DataFrame()
                
            resp_json = resp.json()
            
            if 'data' in resp_json and resp_json['data'] is not None:
                klines = resp_json['data'].get('klines', [])
                
                if not klines:
                    print(f"板块 {sector_code} 无历史数据")
                    return pd.DataFrame()
                
                # 解析K线数据
                data_list = []
                for kline in klines:
                    # K线数据格式：日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
                    parts = kline.split(',')
                    if len(parts) >= 11:
                        data_list.append({
                            '日期': parts[0],
                            '开盘价': float(parts[1]) if parts[1] != '-' else 0,
                            '收盘价': float(parts[2]) if parts[2] != '-' else 0,
                            '最高价': float(parts[3]) if parts[3] != '-' else 0,
                            '最低价': float(parts[4]) if parts[4] != '-' else 0,
                            '成交量': float(parts[5]) if parts[5] != '-' else 0,
                            '成交额': float(parts[6]) if parts[6] != '-' else 0,
                            '振幅': float(parts[7]) if parts[7] != '-' else 0,
                            '涨跌幅': float(parts[8]) if parts[8] != '-' else 0,
                            '涨跌额': float(parts[9]) if parts[9] != '-' else 0,
                            '换手率': float(parts[10]) if parts[10] != '-' else 0
                        })
                
                if data_list:
                    df = pd.DataFrame(data_list)
                    # 添加板块信息
                    df['板块代码'] = sector_code
                    df['板块类型'] = sector_type
                    
                    # 按日期排序
                    df = df.sort_values('日期').reset_index(drop=True)
                    
                    print(f"成功获取板块 {sector_code} 历史数据 {len(df)} 条")
                    return df
                else:
                    print(f"板块 {sector_code} 数据解析失败")
                    return pd.DataFrame()
            else:
                print(f"板块 {sector_code} 响应数据格式错误")
                return pd.DataFrame()
                
        except Exception as e:
            print(f'获取板块 {sector_code} 历史数据出错：{e}')
            return pd.DataFrame()

    def get_all_sectors_historical_data(self, sector_type: str = 'industry', 
                                       trading_days: int = 30, period: str = 'daily', 
                                       save_dir: str = None, is_incremental: bool = False) -> dict:
        """
        批量获取所有板块的历史数据
        :param sector_type: 板块类型，'industry'行业板块 或 'concept'概念板块
        :param trading_days: 获取最近多少个交易日，默认30个交易日
        :param period: 数据周期，'daily'日线 'weekly'周线 'monthly'月线
        :param save_dir: 保存目录，如果提供则保存CSV文件
        :param is_incremental: 是否为增量更新模式
        :return: 包含所有板块数据的字典
        """
        print(f"开始批量获取{sector_type}板块历史数据...")
        
        # 首先获取所有板块列表
        if sector_type == 'industry':
            sectors_df = self.get_industry_data()
            code_col = '行业代码'
            name_col = '行业名称'
        else:
            sectors_df = self.get_concept_data()
            code_col = '概念代码'
            name_col = '概念名称'
        
        if sectors_df.empty:
            print(f"获取{sector_type}板块列表失败")
            return {}
        
        # 创建保存目录
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
        
        all_data = {}
        successful_count = 0
        total_count = len(sectors_df)
        
        print(f"共找到 {total_count} 个{sector_type}板块")
        
        for index, row in sectors_df.iterrows():
            sector_code = row[code_col]
            sector_name = row[name_col]
            
            try:
                print(f"进度: {index + 1}/{total_count} - 正在获取 {sector_name}({sector_code}) 历史数据...")
                
                # 获取历史数据
                historical_df = self.get_historical_sector_data_from_eastmoney(
                    sector_code=sector_code,
                    sector_type=sector_type,
                    trading_days=trading_days,
                    period=period,
                    is_incremental=is_incremental,
                    save_dir=save_dir if is_incremental else None
                )
                
                if not historical_df.empty:
                    # 添加板块名称
                    historical_df['板块名称'] = sector_name
                    all_data[sector_code] = historical_df
                    successful_count += 1
                    
                    # 保存单个板块数据
                    if save_dir:
                        filename = f"{sector_name}({sector_code})_{period}_历史数据.csv"
                        filepath = os.path.join(save_dir, filename)
                        historical_df.to_csv(filepath, index=False, encoding='utf-8-sig')
                        print(f"已保存: {filepath}")
                    
                else:
                    print(f"警告: {sector_name}({sector_code}) 无历史数据")
                
                # 添加延迟避免请求过于频繁
                time.sleep(0.2)
                
            except Exception as e:
                print(f"错误: 获取 {sector_name}({sector_code}) 数据失败: {e}")
                continue
        
        print(f"\n批量获取完成！成功: {successful_count}/{total_count}")
        
        # 合并所有数据并保存
        if all_data and save_dir:
            try:
                combined_df = pd.concat(all_data.values(), ignore_index=True)
                combined_filename = f"所有{sector_type}板块_{period}_历史数据汇总.csv"
                combined_filepath = os.path.join(save_dir, combined_filename)
                combined_df.to_csv(combined_filepath, index=False, encoding='utf-8-sig')
                print(f"已保存汇总文件: {combined_filepath} (共 {len(combined_df)} 条记录)")
            except Exception as e:
                print(f"保存汇总文件失败: {e}")
        
        return all_data

class OptimizedCommonFunctions:
    """优化后的通用功能类"""
    
    @staticmethod
    def get_file_list_in_directory(path: str, filetype: str = '.csv') -> List[str]:
        """
        获取指定目录下所有指定类型的文件列表
        """
        file_list = []
        
        if not os.path.exists(path):
            print(f"目录不存在: {path}")
            return file_list
        
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(filetype):
                    file_list.append(file[:8] if len(file) >= 8 else file)
        
        return sorted(file_list)
    
    @staticmethod
    def batch_process_files(directory: str, process_func, filetype: str = '.csv'):
        """
        批量处理文件的通用方法
        """
        files = OptimizedCommonFunctions.get_file_list_in_directory(directory, filetype)
        
        results = []
        for file in files:
            try:
                result = process_func(os.path.join(directory, file))
                results.append(result)
            except Exception as e:
                print(f"处理文件 {file} 失败: {e}")
        
        return results


# 主要使用的类实例
def get_stock_downloader():
    """获取股票下载器实例"""
    return OptimizedDownloadStocksList()

def get_spider_client():
    """获取爬虫客户端实例"""
    return OptimizedSpiderFunc()

def get_common_functions():
    """获取通用功能实例"""
    return OptimizedCommonFunctions()


if __name__ == '__main__':
    # 使用示例
    print("=== 优化版本 samequant_functions ===")
    
    # 下载股票列表
    downloader = get_stock_downloader()
    all_stocks = downloader.main()
    print(f"获取到 {len(all_stocks)} 只股票")
    exit()
    # 获取实时行情
    spider = get_spider_client()
    df = spider.get_realtime_market_data()
    print(f"获取实时行情数据: {len(df)} 条")
    
    # 获取行业数据
    industry_df = spider.get_industry_data()
    print(f"获取行业数据: {len(industry_df)} 条")
    
    # 获取概念数据
    concept_df = spider.get_concept_data()
    print(f"获取概念数据: {len(concept_df)} 条")