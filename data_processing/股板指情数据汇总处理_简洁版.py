#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票数据处理 - 简洁版
专注核心功能：配置文件控制特征选择，简化处理流程
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import glob
import warnings
from datetime import datetime
import json
import yaml

warnings.filterwarnings('ignore')


class StockDataProcessor:
    """简洁的股票数据处理器"""
    
    def __init__(self, config_path: str = None):
        # 加载配置
        self.config = self.load_config(config_path) if config_path else {}
        
        # 设置路径
        base_path = self.config.get('paths', {}).get('data_base_path', './data')
        paths = self.config.get('paths', {})
        self.stock_path = os.path.join(base_path, paths.get('stock_path', 'datas_em'))
        self.sector_path = os.path.join(base_path, paths.get('industry_path', 'datas_sector_historical/行业板块_全部历史'))
        self.index_path = os.path.join(base_path, paths.get('index_path', 'datas_index/datas_index'))
        self.concept_path = os.path.join(base_path, paths.get('concept_path', 'datas_sector_historical/概念板块_全部历史'))
        
        # 特征配置
        self.features_config = self.config.get('features', {})
        self.processing_config = self.config.get('processing', {})
        self.output_config = self.config.get('output', {})
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            return {}
    
    def get_feature_columns(self, feature_type: str) -> List[str]:
        """获取指定类型的特征列"""
        config = self.features_config.get(feature_type, {})
        
        if feature_type == 'stock':
            # 基础特征
            basic = config.get('basic_features', [])
            
            # 技术指标
            tech = []
            if config.get('technical_indicators', {}).get('enabled', True):
                indicators = config.get('technical_indicators', {}).get('indicators', [])
                for ind in indicators:
                    name = ind.get('name')
                    if name == 'RSI':
                        tech.append('RSI')
                    elif name == 'MACD':
                        tech.extend(['MACD', 'MACD_signal'])
                    elif name == '布林带':
                        tech.extend(['BB_upper', 'BB_middle', 'BB_lower'])
                    elif name == 'ATR':
                        tech.append('ATR')
                    elif name == 'ROC':
                        tech.append('ROC')
            
            return basic + tech
            
        elif feature_type == 'sector':
            return config.get('features', ['开盘价', '收盘价', '最高价', '最低价', '成交量', '涨跌幅'])
        
        return []
    
    def load_stock_data(self) -> Dict[str, pd.DataFrame]:
        """加载股票数据"""
        if not self.features_config.get('stock', {}).get('enabled', True):
            return {}
        
        limit = self.processing_config.get('limit_stocks')
        files = glob.glob(os.path.join(self.stock_path, "*.csv"))
        if limit:
            files = files[:limit]
        
        feature_cols = self.get_feature_columns('stock')
        target_dims = self.features_config.get('stock', {}).get('target_dimensions', 16)
        
        stock_data = {}
        for file_path in files:
            try:
                code = os.path.basename(file_path).replace('.csv', '')
                df = pd.read_csv(file_path, encoding='utf-8')
                df['交易日期'] = pd.to_datetime(df['交易日期'])
                df = df.set_index('交易日期').sort_index()
                
                # 计算技术指标
                df = self.calculate_indicators(df)
                
                # 选择特征
                available_cols = [col for col in feature_cols if col in df.columns]
                df_selected = df[available_cols].copy() if available_cols else df.iloc[:, :target_dims].copy()
                
                # 调整维度
                while len(df_selected.columns) < target_dims:
                    df_selected[f'feature_{len(df_selected.columns)}'] = 0
                df_selected = df_selected.iloc[:, :target_dims]
                
                stock_data[code] = df_selected
                
            except Exception:
                continue
        
        return stock_data
    
    def load_sector_data(self) -> Dict[str, pd.DataFrame]:
        """加载行业数据"""
        if not self.features_config.get('sector', {}).get('enabled', True):
            return {}
        
        files = glob.glob(os.path.join(self.sector_path, "*.csv"))
        feature_cols = self.get_feature_columns('sector')
        target_dims = self.features_config.get('sector', {}).get('target_dimensions', 6)
        
        sector_data = {}
        for file_path in files:
            try:
                name = os.path.basename(file_path).split('(')[0]
                df = pd.read_csv(file_path, encoding='utf-8')
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.set_index('日期').sort_index()
                
                # 选择特征
                available_cols = [col for col in feature_cols if col in df.columns]
                if len(available_cols) >= 4:
                    df_selected = df[available_cols].copy()
                    
                    # 调整维度
                    while len(df_selected.columns) < target_dims:
                        df_selected[f'feature_{len(df_selected.columns)}'] = 0
                    df_selected = df_selected.iloc[:, :target_dims]
                    
                    sector_data[name] = df_selected
                    
            except Exception:
                continue
        
        return sector_data
    
    def load_index_data(self) -> pd.DataFrame:
        """加载指数数据"""
        if not self.features_config.get('index', {}).get('enabled', True):
            return pd.DataFrame()
        
        index_mapping = self.features_config.get('index', {}).get('index_mapping', {
            'zs000001.csv': '上证指数',
            'zs000300.csv': '沪深300',
            'zs399001.csv': '深证成指',
            'zs399006.csv': '创业板指',
            'zs000905.csv': '中证500'
        })
        
        target_dims = self.features_config.get('index', {}).get('target_dimensions', 5)
        all_data = []
        
        for filename, name in index_mapping.items():
            file_path = os.path.join(self.index_path, filename)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                    df['交易日期'] = pd.to_datetime(df['交易日期'])
                    df = df.set_index('交易日期').sort_index()
                    df_selected = df[['收盘价']].copy()
                    df_selected.columns = [name]
                    all_data.append(df_selected)
                except Exception:
                    continue
        
        if all_data:
            combined = all_data[0]
            for df in all_data[1:]:
                combined = combined.join(df, how='outer')
            
            # 调整维度
            while len(combined.columns) < target_dims:
                combined[f'index_{len(combined.columns)}'] = 0
            combined = combined.iloc[:, :target_dims]
            
            return combined.fillna(0)
        
        return pd.DataFrame()
    
    def load_sentiment_data(self) -> pd.DataFrame:
        """加载情绪数据"""
        if not self.features_config.get('sentiment', {}).get('enabled', True):
            return pd.DataFrame()
        
        files = glob.glob(os.path.join(self.concept_path, "*.csv"))
        target_dims = self.features_config.get('sentiment', {}).get('target_dimensions', 2)
        
        # 简化：只取前两个概念文件作为情绪特征
        sentiment_data = []
        for file_path in files[:target_dims]:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.set_index('日期').sort_index()
                if '涨跌幅' in df.columns:
                    sentiment_data.append(df[['涨跌幅']])
            except Exception:
                continue
        
        if sentiment_data:
            combined = pd.concat(sentiment_data, axis=1)
            combined.columns = [f'sentiment_{i}' for i in range(len(combined.columns))]
            
            # 调整维度
            while len(combined.columns) < target_dims:
                combined[f'sentiment_{len(combined.columns)}'] = 0
            combined = combined.iloc[:, :target_dims]
            
            return combined.fillna(0)
        
        return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        try:
            # RSI
            delta = df['收盘价'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['收盘价'].ewm(span=12).mean()
            exp2 = df['收盘价'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            
            # 布林带
            rolling_mean = df['收盘价'].rolling(20).mean()
            rolling_std = df['收盘价'].rolling(20).std()
            df['BB_upper'] = rolling_mean + (rolling_std * 2)
            df['BB_middle'] = rolling_mean
            df['BB_lower'] = rolling_mean - (rolling_std * 2)
            
            # ATR
            high_low = df['最高价'] - df['最低价']
            high_close = np.abs(df['最高价'] - df['收盘价'].shift())
            low_close = np.abs(df['最低价'] - df['收盘价'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            
            # ROC
            df['ROC'] = df['收盘价'].pct_change(12) * 100
            
        except Exception:
            pass
        
        return df
    
    def create_samples(self, stock_data: Dict, sector_data: Dict, index_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """创建训练样本"""
        window_size = self.processing_config.get('window_size', 30)
        start_date = self.processing_config.get('date_range', {}).get('start_date')
        end_date = self.processing_config.get('date_range', {}).get('end_date')
        
        all_samples = []
        all_targets = []
        all_codes = []
        
        for code, stock_df in stock_data.items():
            try:
                # 数据对齐
                dates = stock_df.index
                
                if not sector_data and not index_data.empty:
                    dates = dates.intersection(index_data.index)
                elif sector_data and index_data.empty:
                    # 找到对应行业数据
                    sector_df = None
                    for name, sdf in sector_data.items():
                        sector_df = sdf
                        break
                    if sector_df is not None:
                        dates = dates.intersection(sector_df.index)
                elif sector_data and not index_data.empty:
                    sector_df = None
                    for name, sdf in sector_data.items():
                        sector_df = sdf
                        break
                    if sector_df is not None:
                        dates = dates.intersection(sector_df.index)
                    dates = dates.intersection(index_data.index)
                
                if not sentiment_data.empty:
                    dates = dates.intersection(sentiment_data.index)
                
                # 日期范围过滤
                if start_date:
                    dates = dates[dates >= pd.to_datetime(start_date)]
                if end_date:
                    dates = dates[dates <= pd.to_datetime(end_date)]
                
                if len(dates) < window_size + 1:
                    continue
                
                dates = dates.sort_values()
                
                # 对齐数据
                stock_aligned = stock_df.reindex(dates).fillna(0)
                
                feature_list = []
                if sector_data:
                    for name, sdf in sector_data.items():
                        sector_aligned = sdf.reindex(dates).fillna(0)
                        break
                else:
                    sector_aligned = None
                
                if not index_data.empty:
                    index_aligned = index_data.reindex(dates).fillna(0)
                else:
                    index_aligned = None
                
                if not sentiment_data.empty:
                    sentiment_aligned = sentiment_data.reindex(dates).fillna(0)
                else:
                    sentiment_aligned = None
                
                # 创建样本
                for i in range(len(dates) - window_size):
                    window_start = i
                    window_end = i + window_size
                    
                    # 拼接特征
                    features = []
                    
                    # 个股特征
                    if self.features_config.get('stock', {}).get('enabled', True):
                        stock_window = stock_aligned.iloc[window_start:window_end].values
                        features.append(stock_window)
                    
                    # 行业特征
                    if self.features_config.get('sector', {}).get('enabled', True) and sector_aligned is not None:
                        sector_window = sector_aligned.iloc[window_start:window_end].values
                        features.append(sector_window)
                    
                    # 指数特征
                    if self.features_config.get('index', {}).get('enabled', True) and index_aligned is not None:
                        index_window = index_aligned.iloc[window_start:window_end].values
                        features.append(index_window)
                    
                    # 情绪特征
                    if self.features_config.get('sentiment', {}).get('enabled', True) and sentiment_aligned is not None:
                        sentiment_window = sentiment_aligned.iloc[window_start:window_end].values
                        features.append(sentiment_window)
                    
                    if features:
                        combined = np.hstack(features)
                        
                        # 目标
                        target_idx = window_end
                        if '涨跌幅' in stock_aligned.columns:
                            target = stock_aligned.iloc[target_idx]['涨跌幅']
                        else:
                            current_price = stock_aligned.iloc[window_end - 1]['收盘价']
                            next_price = stock_aligned.iloc[target_idx]['收盘价']
                            target = (next_price - current_price) / current_price * 100
                        
                        all_samples.append(combined)
                        all_targets.append(target)
                        all_codes.append(code)
                
            except Exception:
                continue
        
        if all_samples:
            return np.array(all_samples), np.array(all_targets), all_codes
        else:
            raise ValueError("没有生成样本")
    
    def save_data(self, X: np.ndarray, y: np.ndarray, codes: List[str]) -> str:
        """保存数据"""
        save_dir = self.output_config.get('save_base_dir', './data/processed')
        custom_name = self.output_config.get('custom_folder_name')
        
        if custom_name:
            final_dir = os.path.join(save_dir, custom_name)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_dir = os.path.join(save_dir, f"processed_{timestamp}")
        
        os.makedirs(final_dir, exist_ok=True)
        
        # 保存文件
        np.save(os.path.join(final_dir, "X_features.npy"), X)
        np.save(os.path.join(final_dir, "y_targets.npy"), y)
        
        with open(os.path.join(final_dir, "stock_codes.json"), 'w', encoding='utf-8') as f:
            json.dump(codes, f, ensure_ascii=False, indent=2)
        
        # 简单信息文件
        info = {
            'X_shape': X.shape,
            'y_shape': y.shape,
            'num_stocks': len(set(codes)),
            'processing_time': datetime.now().isoformat()
        }
        
        with open(os.path.join(final_dir, "data_info.json"), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        return final_dir


def main(config_path: str = None):
    """主处理流程"""
    # 用户输入配置文件路径
    config_path = "config/datas_process/data_process_simple.yaml"   
  
  
    print(f"使用配置: {config_path if config_path else '默认设置'}")
    
    # 创建处理器
    processor = StockDataProcessor(config_path)
    
    try:
        # 加载数据
        print("加载股票数据...")
        stock_data = processor.load_stock_data()
        
        print("加载行业数据...")
        sector_data = processor.load_sector_data()
        
        print("加载指数数据...")
        index_data = processor.load_index_data()
        
        print("加载情绪数据...")
        sentiment_data = processor.load_sentiment_data()
        
        # 创建样本
        print("创建训练样本...")
        X, y, codes = processor.create_samples(stock_data, sector_data, index_data, sentiment_data)
        
        # 保存结果
        print("保存数据...")
        save_dir = processor.save_data(X, y, codes)
        
        print(f"完成! 数据形状: X{X.shape}, y{y.shape}")
        print(f"保存位置: {save_dir}")
        
    except Exception as e:
        print(f"处理失败: {e}")


if __name__ == "__main__":
    import sys
    
    config_path = None
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("用法: python 股板指情数据汇总处理_简洁版.py [配置文件路径]")
            print("示例: python 股板指情数据汇总处理_简洁版.py config/datas_process/data_process_1.yaml")
            sys.exit(0)
        else:
            config_path = sys.argv[1]
    
    main(config_path)