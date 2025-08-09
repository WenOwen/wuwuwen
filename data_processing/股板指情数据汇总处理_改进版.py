#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理模块 - 改进版
添加了日期范围控制功能
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import glob
import warnings
from datetime import datetime
import pickle
import json

warnings.filterwarnings('ignore')


class ImprovedDataProcessorV2:
    """改进的数据处理器V2：每只股票只使用所属行业特征 + 日期范围控制"""
    
    def __init__(self, data_base_path: str = "./data"):
        self.data_base_path = data_base_path
        self.stock_path = os.path.join(data_base_path, "datas_em")
        self.industry_path = os.path.join(data_base_path, "datas_sector_historical/行业板块_全部历史")
        self.index_path = os.path.join(data_base_path, "datas_index/datas_index")
        self.concept_path = os.path.join(data_base_path, "datas_sector_historical/概念板块_全部历史")
        
        # 股票行业映射（需要根据实际情况构建）
        self.stock_sector_mapping = self._create_stock_sector_mapping()
        
        # 行业代码到文件名的映射
        self.sector_file_mapping = self._create_sector_file_mapping()
        
    def _create_stock_sector_mapping(self) -> Dict[str, str]:
        """
        从真实的股票板块映射表创建股票到行业的映射关系
        """
        mapping = {}
        
        # 读取真实的股票板块映射表
        mapping_file = os.path.join(self.data_base_path, "datas_sector_historical/股票板块映射表.csv")
        
        if not os.path.exists(mapping_file):
            print(f"警告：映射文件不存在 {mapping_file}，使用默认映射")
            return self._create_default_mapping()
        
        try:
            df = pd.read_csv(mapping_file, encoding='utf-8')
            
            for _, row in df.iterrows():
                stock_code = str(row['股票代码']).lower()  # 转为小写
                sector = row['所属行业']
                
                # 标准化股票代码格式，去掉可能的前缀
                if stock_code.startswith('sh') or stock_code.startswith('sz'):
                    clean_code = stock_code[2:]  # 去掉sh/sz前缀
                else:
                    clean_code = stock_code
                
                # 同时保存原始格式和清理后的格式
                mapping[stock_code] = sector
                mapping[clean_code] = sector
                
                # 也保存我们数据文件中的格式 (如sz301559)
                if len(clean_code) == 6:
                    if clean_code.startswith('0') or clean_code.startswith('3'):
                        file_format = f"sz{clean_code}"
                    else:
                        file_format = f"sh{clean_code}"
                    mapping[file_format] = sector
            
            print(f"从映射表加载了 {len(df)} 个股票的行业映射")
            print(f"映射字典包含 {len(mapping)} 个条目")
            
            # 显示一些行业统计
            sector_counts = {}
            for sector in mapping.values():
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            print("主要行业分布:")
            for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {sector}: {count} 只股票")
                
        except Exception as e:
            print(f"读取映射文件出错: {e}")
            return self._create_default_mapping()
            
        return mapping
    
    def _create_default_mapping(self) -> Dict[str, str]:
        """创建默认的股票到行业映射（备用方案）"""
        mapping = {}
        stock_files = glob.glob(os.path.join(self.stock_path, "*.csv"))
        
        # 默认行业分配
        default_sectors = ['银行', '医药商业', '软件开发', '专用设备', '食品饮料', 
                          '房地产开发', '石油行业', '化学制品', '航运港口']
        
        for i, stock_file in enumerate(stock_files):
            stock_code = os.path.basename(stock_file).replace('.csv', '')
            sector = default_sectors[i % len(default_sectors)]
            mapping[stock_code] = sector
            
        print(f"使用默认映射创建了 {len(mapping)} 个股票的行业映射")
        return mapping
    
    def _create_sector_file_mapping(self) -> Dict[str, str]:
        """创建行业名称到文件路径的映射"""
        mapping = {}
        industry_files = glob.glob(os.path.join(self.industry_path, "*.csv"))
        
        for file_path in industry_files:
            filename = os.path.basename(file_path)
            sector_name = filename.split('(')[0]  # 提取行业名称
            mapping[sector_name] = file_path
            
        print(f"找到 {len(mapping)} 个行业板块文件")
        return mapping
    
    def load_individual_stock_data(self, limit_stocks: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        加载个股数据，保持每只股票独立
        """
        print("正在加载个股数据...")
        
        stock_files = glob.glob(os.path.join(self.stock_path, "*.csv"))
        if limit_stocks:
            stock_files = stock_files[:limit_stocks]
            
        print(f"找到 {len(stock_files)} 个股票文件")
        
        stock_data_dict = {}
        
        for i, file_path in enumerate(stock_files):
            if i % 100 == 0:
                progress = (i+1) / len(stock_files) * 100
                print(f"处理进度: {i+1}/{len(stock_files)} ({progress:.1f}%)")
                
            try:
                stock_code = os.path.basename(file_path).replace('.csv', '')
                df = pd.read_csv(file_path, encoding='utf-8')
                
                # 标准化列名和索引
                df['交易日期'] = pd.to_datetime(df['交易日期'])
                df = df.drop_duplicates(subset=['交易日期']).sort_values('交易日期')
                df = df.set_index('交易日期')
                
                # 计算技术指标
                df = self._calculate_technical_indicators(df)
                
                # 选择个股特征（16维）
                feature_cols = [
                    '开盘价', '收盘价', '最高价', '最低价', '成交量', 
                    '涨跌幅', '换手率', '流通市值', 'RSI', 'MACD', 'MACD_signal', 
                    'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'ROC'
                ]
                
                # 只保留存在的列
                available_cols = [col for col in feature_cols if col in df.columns]
                df_selected = df[available_cols].copy()
                
                # 确保16维，不足的用0填充
                while len(df_selected.columns) < 16:
                    df_selected[f'feature_{len(df_selected.columns)}'] = 0
                
                # 超过16维则截取前16维
                if len(df_selected.columns) > 16:
                    df_selected = df_selected.iloc[:, :16]
                
                stock_data_dict[stock_code] = df_selected
                
            except Exception as e:
                print(f"处理股票文件 {file_path} 时出错: {e}")
                continue
        
        print(f"成功加载 {len(stock_data_dict)} 只股票的数据")
        return stock_data_dict
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标（与之前相同）"""
        try:
            # RSI
            df['RSI'] = self._calculate_rsi(df['收盘价'])
            
            # MACD
            macd_data = self._calculate_macd(df['收盘价'])
            df['MACD'] = macd_data['MACD']
            df['MACD_signal'] = macd_data['MACD_signal']
            
            # 布林带
            bb_data = self._calculate_bollinger_bands(df['收盘价'])
            df['BB_upper'] = bb_data['upper']
            df['BB_middle'] = bb_data['middle']
            df['BB_lower'] = bb_data['lower']
            
            # ATR
            df['ATR'] = self._calculate_atr(df['最高价'], df['最低价'], df['收盘价'])
            
            # ROC
            df['ROC'] = df['收盘价'].pct_change(periods=12) * 100
            
        except Exception as e:
            print(f"计算技术指标时出错: {e}")
            
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """计算MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        
        return {
            'MACD': macd,
            'MACD_signal': macd_signal
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict:
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """计算ATR"""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def load_sector_data(self) -> Dict[str, pd.DataFrame]:
        """加载行业板块数据，保持每个行业独立"""
        print("正在加载行业板块数据...")
        
        sector_data_dict = {}
        
        for sector_name, file_path in self.sector_file_mapping.items():
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                df['日期'] = pd.to_datetime(df['日期'])
                
                # 去重并排序
                df = df.drop_duplicates(subset=['日期']).sort_values('日期')
                df = df.set_index('日期')
                
                # 选择行业特征（6维：OHLCV + 涨跌幅）
                feature_cols = ['开盘价', '收盘价', '最高价', '最低价', '成交量', '涨跌幅']
                available_cols = [col for col in feature_cols if col in df.columns]
                
                if len(available_cols) >= 4:  # 至少要有OHLC
                    df_selected = df[available_cols].copy()
                    
                    # 确保6维
                    while len(df_selected.columns) < 6:
                        df_selected[f'sector_feature_{len(df_selected.columns)}'] = 0
                    
                    if len(df_selected.columns) > 6:
                        df_selected = df_selected.iloc[:, :6]
                    
                    sector_data_dict[sector_name] = df_selected
                    
            except Exception as e:
                print(f"处理行业文件 {file_path} 时出错: {e}")
                continue
        
        print(f"成功加载 {len(sector_data_dict)} 个行业板块数据")
        return sector_data_dict
    
    def load_index_data(self) -> pd.DataFrame:
        """加载指数数据（5维）"""
        print("正在加载指数数据...")
        
        # 主要指数映射
        index_mapping = {
            'zs000001.csv': '上证指数',
            'zs000300.csv': '沪深300',
            'zs399001.csv': '深证成指',
            'zs399006.csv': '创业板指',
            'zs000905.csv': '中证500'
        }
        
        all_index_data = []
        
        for filename, index_name in index_mapping.items():
            file_path = os.path.join(self.index_path, filename)
            
            if not os.path.exists(file_path):
                continue
                
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                df['交易日期'] = pd.to_datetime(df['交易日期'])
                
                # 去重并排序
                df = df.drop_duplicates(subset=['交易日期']).sort_values('交易日期')
                df = df.set_index('交易日期')
                
                # 只取一个主要特征（收盘价）
                df_selected = df[['收盘价']].copy()
                df_selected.columns = [f"{index_name}"]
                
                all_index_data.append(df_selected)
                
            except Exception as e:
                print(f"处理指数文件 {filename} 时出错: {e}")
                continue
        
        if all_index_data:
            # 合并所有指数数据
            combined_data = all_index_data[0]
            for df in all_index_data[1:]:
                combined_data = combined_data.join(df, how='outer')
            
            # 确保5维
            while len(combined_data.columns) < 5:
                combined_data[f'index_feature_{len(combined_data.columns)}'] = 0
            
            if len(combined_data.columns) > 5:
                combined_data = combined_data.iloc[:, :5]
                
            combined_data = combined_data.sort_index()
        else:
            # 创建空数据
            combined_data = pd.DataFrame()
        
        print(f"指数数据形状: {combined_data.shape}")
        return combined_data
    
    def load_sentiment_data(self) -> pd.DataFrame:
        """
        加载情绪数据：昨日连板和涨停板块（2维）
        """
        print("正在加载情绪数据...")
        
        # 寻找连板和涨停相关板块
        concept_files = glob.glob(os.path.join(self.concept_path, "*.csv"))
        
        limit_up_files = []
        consecutive_files = []
        
        for file_path in concept_files:
            filename = os.path.basename(file_path)
            if '涨停' in filename:
                limit_up_files.append(file_path)
            if '连板' in filename or '连续' in filename:
                consecutive_files.append(file_path)
        

        # 处理涨停数据
        limit_up_data = self._process_sentiment_concept_data(limit_up_files[:1], "涨停强度")
        
        # 处理连板数据
        consecutive_data = self._process_sentiment_concept_data(consecutive_files[:1], "连板强度")
        
        # 合并为2维情绪数据
        if not limit_up_data.empty and not consecutive_data.empty:
            sentiment_data = pd.concat([limit_up_data, consecutive_data], axis=1)
        elif not limit_up_data.empty:
            sentiment_data = limit_up_data
            sentiment_data['连板强度'] = 0  # 添加第二维
        elif not consecutive_data.empty:
            sentiment_data = consecutive_data
            sentiment_data['涨停强度'] = 0  # 添加第一维
        else:
            # 创建空数据
            dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
            sentiment_data = pd.DataFrame({
                '涨停强度': np.zeros(len(dates)),
                '连板强度': np.zeros(len(dates))
            }, index=dates)
        
        # 确保2维
        while len(sentiment_data.columns) < 2:
            sentiment_data[f'sentiment_feature_{len(sentiment_data.columns)}'] = 0
        
        if len(sentiment_data.columns) > 2:
            sentiment_data = sentiment_data.iloc[:, :2]
        
        print(f"情绪数据形状: {sentiment_data.shape}")
        return sentiment_data
    
    def _process_sentiment_concept_data(self, file_list: List[str], feature_name: str) -> pd.DataFrame:
        """处理情绪概念数据"""
        if not file_list:
            return pd.DataFrame()
        
        all_data = []
        
        for file_path in file_list:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                df['日期'] = pd.to_datetime(df['日期'])
                
                # 去重并排序
                df = df.drop_duplicates(subset=['日期']).sort_values('日期')
                df = df.set_index('日期')
                
                # 使用涨跌幅作为强度指标
                if '涨跌幅' in df.columns:
                    df_selected = df[['涨跌幅']].copy()
                    df_selected.columns = [feature_name]
                    all_data.append(df_selected)
                
            except Exception as e:
                print(f"处理情绪文件 {file_path} 时出错: {e}")
                continue
        
        if all_data:
            # 取平均值
            combined_data = pd.concat(all_data, axis=1)
            result = combined_data.mean(axis=1).to_frame(feature_name)
            return result
        
        return pd.DataFrame()
    
    def create_training_samples(self, stock_data_dict: Dict[str, pd.DataFrame],
                              sector_data_dict: Dict[str, pd.DataFrame],
                              index_data: pd.DataFrame,
                              sentiment_data: pd.DataFrame,
                              window_size: int = 30,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        创建训练样本：每只股票使用其所属行业的特征，支持日期范围控制
        
        Args:
            stock_data_dict: 股票数据字典
            sector_data_dict: 行业数据字典  
            index_data: 指数数据
            sentiment_data: 情绪数据
            window_size: 滑动窗口大小
            start_date: 开始日期，格式'YYYY-MM-DD'，None表示使用全部数据
            end_date: 结束日期，格式'YYYY-MM-DD'，None表示使用全部数据
        """
        print("正在创建训练样本...")
        
        # 转换日期参数
        start_dt = pd.to_datetime(start_date) if start_date else None
        end_dt = pd.to_datetime(end_date) if end_date else None
        
        if start_dt or end_dt:
            date_range_str = f"从 {start_date if start_date else '最早'} 到 {end_date if end_date else '最新'}"
            print(f"📅 日期范围限制: {date_range_str}")
        
        all_samples = []
        all_targets = []
        all_stock_codes = []
        date_stats = {'min_date': None, 'max_date': None, 'total_dates': 0}
        
        for stock_code, stock_df in stock_data_dict.items():
            try:
                # 获取该股票所属的行业
                sector_name = self.stock_sector_mapping.get(stock_code, '银行')  # 默认银行
                
                # 寻找匹配的行业数据
                sector_df = None
                for sector_key, sector_data in sector_data_dict.items():
                    if sector_name in sector_key:
                        sector_df = sector_data
                        break
                
                if sector_df is None:
                    # 如果没找到匹配的行业，使用第一个可用的行业数据
                    if sector_data_dict:
                        sector_df = list(sector_data_dict.values())[0]
                    else:
                        # 创建空的行业数据
                        sector_df = pd.DataFrame(
                            np.zeros((len(stock_df), 6)),
                            index=stock_df.index,
                            columns=[f'sector_feature_{i}' for i in range(6)]
                        )
                
                # 数据对齐
                common_dates = stock_df.index.intersection(sector_df.index)
                common_dates = common_dates.intersection(index_data.index)
                common_dates = common_dates.intersection(sentiment_data.index)
                
                # 按日期范围过滤
                if start_dt or end_dt:
                    if start_dt:
                        common_dates = common_dates[common_dates >= start_dt]
                    if end_dt:
                        common_dates = common_dates[common_dates <= end_dt]
                
                # 更新日期统计
                if len(common_dates) > 0:
                    current_min = common_dates.min()
                    current_max = common_dates.max()
                    
                    if date_stats['min_date'] is None or current_min < date_stats['min_date']:
                        date_stats['min_date'] = current_min
                    if date_stats['max_date'] is None or current_max > date_stats['max_date']:
                        date_stats['max_date'] = current_max
                    date_stats['total_dates'] = max(date_stats['total_dates'], len(common_dates))
                
                if len(common_dates) < window_size + 1:
                    continue
                
                # 重新索引并排序
                common_dates = common_dates.sort_values()
                stock_aligned = stock_df.reindex(common_dates).fillna(method='ffill').fillna(0)
                sector_aligned = sector_df.reindex(common_dates).fillna(method='ffill').fillna(0)
                index_aligned = index_data.reindex(common_dates).fillna(method='ffill').fillna(0)
                sentiment_aligned = sentiment_data.reindex(common_dates).fillna(method='ffill').fillna(0)
                
                # 创建滑动窗口样本
                for i in range(len(common_dates) - window_size):
                    # 特征窗口
                    window_start = i
                    window_end = i + window_size
                    
                    # 拼接特征：个股(16) + 行业(6) + 指数(5) + 情绪(2) = 29维
                    stock_window = stock_aligned.iloc[window_start:window_end].values  # (30, 16)
                    sector_window = sector_aligned.iloc[window_start:window_end].values  # (30, 6)
                    index_window = index_aligned.iloc[window_start:window_end].values  # (30, 5)
                    sentiment_window = sentiment_aligned.iloc[window_start:window_end].values  # (30, 2)
                    
                    # 合并特征
                    combined_window = np.hstack([
                        stock_window, sector_window, index_window, sentiment_window
                    ])  # (30, 29)
                    
                    # 目标：下一天的股票收益率
                    target_idx = window_end
                    if target_idx < len(stock_aligned) and '涨跌幅' in stock_aligned.columns:
                        target = stock_aligned.iloc[target_idx]['涨跌幅']
                    else:
                        # 计算收益率
                        if target_idx < len(stock_aligned) and '收盘价' in stock_aligned.columns:
                            current_price = stock_aligned.iloc[window_end - 1]['收盘价']
                            next_price = stock_aligned.iloc[target_idx]['收盘价']
                            target = (next_price - current_price) / current_price * 100
                        else:
                            continue
                    
                    all_samples.append(combined_window)
                    all_targets.append(target)
                    all_stock_codes.append(stock_code)
                    
            except Exception as e:
                print(f"处理股票 {stock_code} 时出错: {e}")
                continue
        
        if not all_samples:
            raise ValueError("没有生成任何训练样本")
        
        X = np.array(all_samples)
        y = np.array(all_targets)
        
        print(f"✅ 生成训练样本: X={X.shape}, y={y.shape}")
        print(f"📊 特征维度: 个股(16) + 行业(6) + 指数(5) + 情绪(2) = {X.shape[-1]}维")
        print(f"📈 样本统计: 涉及{len(set(all_stock_codes))}只股票")
        
        # 打印日期范围信息
        if date_stats['min_date'] and date_stats['max_date']:
            print(f"📅 实际数据日期范围: {date_stats['min_date'].strftime('%Y-%m-%d')} 到 {date_stats['max_date'].strftime('%Y-%m-%d')}")
            print(f"📊 最大有效交易日数量: {date_stats['total_dates']} 天")
        
        return X, y, all_stock_codes
    
    def save_processed_data(self, X: np.ndarray, y: np.ndarray, 
                       stock_codes: List[str], save_base_dir: str = "./data/processed", 
                       create_unique_folder: bool = True,
                       date_range_info: Optional[Dict] = None) -> str:
        """保存处理后的数据
        
        Args:
            X: 特征数据
            y: 目标数据  
            stock_codes: 股票代码列表
            save_base_dir: 基础保存目录
            create_unique_folder: 是否创建唯一标识文件夹
            date_range_info: 日期范围信息
            
        Returns:
            str: 实际保存路径
        """
        
        if create_unique_folder:
            # 创建唯一标识符：时间戳 + 股票数量 + 日期范围
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_stocks = len(set(stock_codes))
            
            # 添加日期范围到文件夹名
            if date_range_info and date_range_info.get('start_date') or date_range_info.get('end_date'):
                start_str = date_range_info.get('start_date', '').replace('-', '')
                end_str = date_range_info.get('end_date', '').replace('-', '')
                date_suffix = f"_{start_str}to{end_str}" if start_str or end_str else ""
            else:
                date_suffix = ""
            
            folder_name = f"processed_{timestamp}_stocks{unique_stocks}{date_suffix}"
            save_dir = os.path.join(save_base_dir, folder_name)
        else:
            save_dir = save_base_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"正在保存处理后的数据到: {save_dir}")
        
        # 保存numpy数组
        np.save(os.path.join(save_dir, "X_features.npy"), X)
        np.save(os.path.join(save_dir, "y_targets.npy"), y)
        
        # 保存股票代码列表
        with open(os.path.join(save_dir, "stock_codes.json"), 'w', encoding='utf-8') as f:
            json.dump(stock_codes, f, ensure_ascii=False, indent=2)
        
        # 保存数据信息
        info = {
            'X_shape': X.shape,
            'y_shape': y.shape,
            'feature_dims': {
                'stock': 16,
                'sector': 6,
                'index': 5,
                'sentiment': 2,
                'total': X.shape[-1]
            },
            'num_samples': X.shape[0],
            'num_stocks': len(set(stock_codes)),
            'processing_time': datetime.now().isoformat(),
            'save_path': save_dir,
            'folder_name': os.path.basename(save_dir) if create_unique_folder else 'default',
            'date_range': date_range_info or {}
        }
        
        with open(os.path.join(save_dir, "data_info.json"), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        # 创建一个README文件说明数据内容
        date_info_str = ""
        if date_range_info and (date_range_info.get('start_date') or date_range_info.get('end_date')):
            start = date_range_info.get('start_date', '最早')
            end = date_range_info.get('end_date', '最新')
            date_info_str = f"- 日期范围: {start} 到 {end}\n"
        
        readme_content = f"""# 股票数据处理结果 (改进版V2)

## 处理信息
- 处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 股票数量: {len(set(stock_codes))} 只
- 样本数量: {X.shape[0]:,} 个
- 特征维度: {X.shape[-1]} 维
{date_info_str}
## 文件说明
- `X_features.npy`: 特征数据，形状 {X.shape}
- `y_targets.npy`: 目标数据（收益率），形状 {y.shape}
- `stock_codes.json`: 对应的股票代码列表
- `data_info.json`: 详细的数据信息和元数据
- `README.md`: 本说明文件

## 特征构成 (时序窗口: 30天)
- 个股特征: 16 维 (OHLCV + 技术指标等)
- 行业特征: 6 维 (所属行业的OHLCV等)
- 指数特征: 5 维 (主要市场指数)
- 情绪特征: 2 维 (涨停强度、连板强度)

## 改进版特点
- ✅ 支持日期范围控制
- ✅ 唯一文件夹命名 (带时间戳和日期范围)
- ✅ 详细的数据统计信息
- ✅ 自动生成说明文档

## 使用方法
```python
import numpy as np
import json

# 加载数据
X = np.load('X_features.npy')
y = np.load('y_targets.npy')

# 加载股票代码
with open('stock_codes.json', 'r', encoding='utf-8') as f:
    stock_codes = json.load(f)

# 加载详细信息
with open('data_info.json', 'r', encoding='utf-8') as f:
    data_info = json.load(f)

print(f"数据形状: {X.shape}")
print(f"目标形状: {y.shape}")
print(f"股票数量: {data_info['num_stocks']}")
```
"""
        
        with open(os.path.join(save_dir, "README.md"), 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✅ 数据已成功保存到: {save_dir}")
        print("📁 保存的文件:")
        print("  - X_features.npy: 特征数据")
        print("  - y_targets.npy: 目标数据")
        print("  - stock_codes.json: 股票代码列表")
        print("  - data_info.json: 数据信息")
        print("  - README.md: 说明文档")
        
        return save_dir


def main(limit_stocks=None, start_date=None, end_date=None):
    """
    主处理流程
    
    Args:
        limit_stocks: 限制股票数量，None表示加载全部股票
        start_date: 开始日期，格式'YYYY-MM-DD'
        end_date: 结束日期，格式'YYYY-MM-DD'
    """
    print("🚀 开始改进版数据处理 (V2 - 支持日期范围控制)...")
    
    if limit_stocks is None:
        print("⚠️  注意：将处理全部股票数据（约5400+只），预计需要较长时间和大量内存")
        print("💡 建议：如果内存不足，可以设置limit_stocks参数限制股票数量")
        print("📊 预计内存需求：16GB+ RAM，处理时间：30-60分钟")
    else:
        print(f"📊 限制处理股票数量：{limit_stocks} 只")
    
    if start_date or end_date:
        date_info = f"从 {start_date if start_date else '最早'} 到 {end_date if end_date else '最新'}"
        print(f"📅 日期范围限制：{date_info}")
    
    # 创建改进的数据处理器
    processor = ImprovedDataProcessorV2(data_base_path="./data")
    
    try:
        # 1. 加载个股数据
        print(f"\n🔄 步骤1/6: 加载个股数据...")
        stock_data_dict = processor.load_individual_stock_data(limit_stocks=limit_stocks)
        
        # 2. 加载行业板块数据
        print(f"\n🔄 步骤2/6: 加载行业板块数据...")
        sector_data_dict = processor.load_sector_data()
        
        # 3. 加载指数数据
        print(f"\n🔄 步骤3/6: 加载指数数据...")
        index_data = processor.load_index_data()
        
        # 4. 加载情绪数据
        print(f"\n🔄 步骤4/6: 加载情绪数据...")
        sentiment_data = processor.load_sentiment_data()
        
        # 5. 创建训练样本（添加日期范围控制）
        print(f"\n🔄 步骤5/6: 创建训练样本...")
        print("⏳ 这一步可能需要较长时间，请耐心等待...")
        X, y, stock_codes = processor.create_training_samples(
            stock_data_dict, sector_data_dict, index_data, sentiment_data,
            start_date=start_date, end_date=end_date
        )
        
        # 6. 保存处理后的数据
        print(f"\n🔄 步骤6/6: 保存处理后的数据...")
        date_range_info = {'start_date': start_date, 'end_date': end_date}
        save_dir = processor.save_processed_data(X, y, stock_codes, 
                                               date_range_info=date_range_info)
        
        print("\n✅ 改进版数据处理完成！")
        print(f"📊 最终数据形状:")
        print(f"  特征数据: {X.shape}")
        print(f"  目标数据: {y.shape}")
        print(f"  特征维度: {X.shape[-1]} (个股16 + 行业6 + 指数5 + 情绪2)")
        print(f"  涉及股票: {len(set(stock_codes))} 只")
        print(f"🗂️  保存位置: {save_dir}")
        
        # 数据统计
        print(f"\n📈 数据统计:")
        print(f"  目标均值: {np.mean(y):.4f}")
        print(f"  目标标准差: {np.std(y):.4f}")
        print(f"  目标范围: [{np.min(y):.4f}, {np.max(y):.4f}]")
        
    except Exception as e:
        print(f"❌ 数据处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def print_usage():
    """打印使用说明"""
    print("\n💡 用法：python 股板指情数据汇总处理_改进版.py [选项]")
    print("\n📝 选项:")
    print("  数字           限制股票数量，如 100")
    print("  --start=日期   开始日期，格式 YYYY-MM-DD")
    print("  --end=日期     结束日期，格式 YYYY-MM-DD")
    print("\n📝 使用示例:")
    print("  python 股板指情数据汇总处理_改进版.py                           # 处理全部股票，全部日期")
    print("  python 股板指情数据汇总处理_改进版.py 100                       # 处理100只股票，全部日期")
    print("  python 股板指情数据汇总处理_改进版.py --start=2023-01-01        # 全部股票，从2023年开始")
    print("  python 股板指情数据汇总处理_改进版.py --end=2023-12-31          # 全部股票，到2023年结束")
    print("  python 股板指情数据汇总处理_改进版.py 100 --start=2023-01-01 --end=2023-12-31  # 100只股票，2023年数据")
    print("\n🌟 改进版特色:")
    print("  ✅ 支持精确的日期范围控制")
    print("  ✅ 自动创建唯一时间戳文件夹")
    print("  ✅ 详细的数据统计和说明文档")
    print("  ✅ 更好的错误处理和进度显示")


if __name__ == "__main__":
    import sys
    
    # 命令行参数处理
    if len(sys.argv) > 1:
        try:
            limit_stocks = None
            start_date = None
            end_date = None
            
            # 解析命令行参数
            for i, arg in enumerate(sys.argv[1:], 1):
                if arg.startswith('--start='):
                    start_date = arg.split('=')[1]
                elif arg.startswith('--end='):
                    end_date = arg.split('=')[1]
                elif arg == '--help' or arg == '-h':
                    print_usage()
                    sys.exit(0)
                elif arg.isdigit():
                    limit_stocks = int(arg)
                else:
                    print(f"❌ 未知参数: {arg}")
                    print_usage()
                    sys.exit(1)
            
            # 验证日期格式
            if start_date:
                try:
                    pd.to_datetime(start_date)
                except:
                    print(f"❌ 错误：开始日期格式不正确: {start_date}")
                    print("💡 日期格式应为: YYYY-MM-DD，如 2023-01-01")
                    sys.exit(1)
            
            if end_date:
                try:
                    pd.to_datetime(end_date)
                except:
                    print(f"❌ 错误：结束日期格式不正确: {end_date}")
                    print("💡 日期格式应为: YYYY-MM-DD，如 2023-12-31")
                    sys.exit(1)
            
            # 打印参数信息
            print("📋 运行参数:")
            print(f"  股票数量限制: {limit_stocks if limit_stocks else '全部'}")
            print(f"  开始日期: {start_date if start_date else '最早'}")
            print(f"  结束日期: {end_date if end_date else '最新'}")
            
            main(limit_stocks=limit_stocks, start_date=start_date, end_date=end_date)
            
        except ValueError as e:
            print(f"❌ 参数错误: {e}")
            print_usage()
    else:
        # 默认处理全部数据
        main(limit_stocks=None, start_date=None, end_date=None)