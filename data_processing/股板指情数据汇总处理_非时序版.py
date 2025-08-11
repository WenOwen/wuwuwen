#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票数据处理 - 非时序版
专注当日特征：提取每个交易日的特征，列名包含日期，不使用时序窗口
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import glob
import warnings
from datetime import datetime, timedelta
import json
import yaml

warnings.filterwarnings('ignore')


class NonTimeSeriesStockDataProcessor:
    """非时序股票数据处理器"""
    
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
        
        # 非时序特定配置
        self.include_date_in_columns = self.processing_config.get('include_date_in_columns', False)
        self.date_format = self.processing_config.get('date_format', '%Y%m%d')
        self.target_prediction_days = self.processing_config.get('target_prediction_days', 1)
        
        # 输出组织配置
        self.group_by_date = self.processing_config.get('group_by_date', True)
        self.month_folder_format = self.processing_config.get('month_folder_format', '%y%m')
        self.daily_file_format = self.processing_config.get('daily_file_format', '%Y%m%d')
        self.only_save_full_data = self.output_config.get('only_save_full_data', True)
    
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
    
    def create_daily_samples_with_feature_names(self, stock_data: Dict, sector_data: Dict, index_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
        """创建每日样本并生成特征名称（不含日期）"""
        start_date = self.processing_config.get('date_range', {}).get('start_date')
        end_date = self.processing_config.get('date_range', {}).get('end_date')
        
        all_samples = []
        all_targets = []
        all_codes = []
        all_dates = []      # 存储每个样本对应的日期
        feature_names = []  # 存储特征名称
        feature_names_generated = False
        
        # 将行业数据转换为列表，便于按股票分配
        sector_list = list(sector_data.items()) if sector_data else []
        
        for stock_idx, (code, stock_df) in enumerate(stock_data.items()):
            try:
                # 数据对齐
                dates = stock_df.index
                
                # 为当前股票分配行业数据（基于股票索引循环分配）
                current_sector_df = None
                current_sector_name = None
                if sector_list:
                    sector_idx = stock_idx % len(sector_list)  # 循环分配行业数据
                    current_sector_name, current_sector_df = sector_list[sector_idx]
                
                # 对齐日期索引
                if current_sector_df is not None and not index_data.empty:
                    dates = dates.intersection(current_sector_df.index)
                    dates = dates.intersection(index_data.index)
                elif current_sector_df is not None:
                    dates = dates.intersection(current_sector_df.index)
                elif not index_data.empty:
                    dates = dates.intersection(index_data.index)
                
                if not sentiment_data.empty:
                    dates = dates.intersection(sentiment_data.index)
                
                # 日期范围过滤
                if start_date:
                    dates = dates[dates >= pd.to_datetime(start_date)]
                if end_date:
                    dates = dates[dates <= pd.to_datetime(end_date)]
                
                if len(dates) < 1:  # 修改：只需要至少1天数据
                    continue
                
                dates = dates.sort_values()
                
                # 对齐数据
                stock_aligned = stock_df.reindex(dates).fillna(0)
                
                # 使用为当前股票分配的行业数据
                if current_sector_df is not None:
                    sector_aligned = current_sector_df.reindex(dates).fillna(0)
                    if stock_idx < 3:  # 只为前3个股票显示行业分配信息
                        print(f"📈 股票 {code} 分配行业数据: {current_sector_name}")
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
                
                # 创建每日样本 - 处理所有日期
                for i in range(len(dates)):
                    current_date = dates[i]
                    
                    # 当天的特征
                    daily_features = []
                    current_feature_names = []  # 当前样本的特征名称
                    
                    # 记录当前日期（用于文件分组）
                    date_str = current_date.strftime(self.date_format)
                    
                    # 个股特征 - 当天
                    if self.features_config.get('stock', {}).get('enabled', True):
                        stock_values = stock_aligned.iloc[i].values
                        daily_features.extend(stock_values)
                        
                        # 生成个股特征名称 (只在第一次生成)
                        if not feature_names_generated:
                            for col_name in stock_aligned.columns:
                                feature_name = f"个股_{col_name}"
                                current_feature_names.append(feature_name)
                    
                    # 行业特征 - 当天
                    if self.features_config.get('sector', {}).get('enabled', True) and sector_aligned is not None:
                        sector_values = sector_aligned.iloc[i].values
                        daily_features.extend(sector_values)
                        
                        # 生成行业特征名称
                        if not feature_names_generated:
                            for col_name in sector_aligned.columns:
                                feature_name = f"行业_{col_name}"
                                current_feature_names.append(feature_name)
                    
                    # 指数特征 - 当天
                    if self.features_config.get('index', {}).get('enabled', True) and index_aligned is not None:
                        index_values = index_aligned.iloc[i].values
                        daily_features.extend(index_values)
                        
                        # 生成指数特征名称
                        if not feature_names_generated:
                            for col_name in index_aligned.columns:
                                feature_name = f"指数_{col_name}"
                                current_feature_names.append(feature_name)
                    
                    # 情绪特征 - 当天
                    if self.features_config.get('sentiment', {}).get('enabled', True) and sentiment_aligned is not None:
                        sentiment_values = sentiment_aligned.iloc[i].values
                        daily_features.extend(sentiment_values)
                        
                        # 生成情绪特征名称
                        if not feature_names_generated:
                            for col_name in sentiment_aligned.columns:
                                feature_name = f"情绪_{col_name}"
                                current_feature_names.append(feature_name)
                    
                    if daily_features:
                        combined = np.array(daily_features)
                        
                        # 保存特征名称 (只在第一次)
                        if not feature_names_generated:
                            feature_names = current_feature_names
                            feature_names_generated = True
                        
                        # 不再计算target，只保存特征数据
                        all_samples.append(combined)
                        all_targets.append(0)  # 占位符，后续会被移除
                        all_codes.append(code)
                        all_dates.append(current_date)
                
            except Exception as e:
                print(f"处理股票 {code} 时出错: {e}")
                continue
        
        if all_samples:
            return np.array(all_samples), np.array(all_targets), all_codes, feature_names, all_dates
        else:
            raise ValueError("没有生成样本")
    
    def save_data_with_meaningful_names(self, X: np.ndarray, y: np.ndarray, codes: List[str], feature_names: List[str] = None, dates: List = None) -> str:
        """按日期分组保存数据到年月文件夹"""
        if dates is None:
            raise ValueError("dates参数不能为空")
            
        save_base_dir = self.output_config.get('save_base_dir', './data/datas_final')
        
        # 处理X数据 (非时序数据应该是2D的)
        if len(X.shape) == 2:
            X_processed = X
            if feature_names is None or len(feature_names) != X.shape[1]:
                feature_names = [f'feature_{i:02d}' for i in range(X.shape[1])]
        else:
            raise ValueError(f"期望2D数据，但得到{len(X.shape)}D数据: {X.shape}")
        
        # 按日期分组数据
        data_by_date = {}
        for i, (sample, target, code, date) in enumerate(zip(X_processed, y, codes, dates)):
            date_key = date.strftime(self.daily_file_format)
            if date_key not in data_by_date:
                data_by_date[date_key] = {
                    'samples': [],
                    'targets': [],
                    'codes': [],
                    'date_obj': date
                }
            data_by_date[date_key]['samples'].append(sample)
            data_by_date[date_key]['targets'].append(target)
            data_by_date[date_key]['codes'].append(code)
        
        saved_files = []
        
        # 为每个日期创建文件
        for date_key, date_data in data_by_date.items():
            date_obj = date_data['date_obj']
            
            # 创建年月文件夹 (例如: 2508 表示2025年8月)
            month_folder = date_obj.strftime(self.month_folder_format)
            month_dir = os.path.join(save_base_dir, month_folder)
            os.makedirs(month_dir, exist_ok=True)
            
            # 创建当日的DataFrame
            daily_X = np.array(date_data['samples'])
            daily_y = np.array(date_data['targets'])
            daily_codes = date_data['codes']
            
            # 创建完整数据DataFrame（stock_code作为第一列，不包含target）
            full_df = pd.DataFrame()
            full_df['stock_code'] = daily_codes  # 第一列：股票代码
            
            # 添加特征列
            feature_df = pd.DataFrame(daily_X, columns=feature_names)
            full_df = pd.concat([full_df, feature_df], axis=1)
            
            # 保存当日的full_data.csv
            daily_file_path = os.path.join(month_dir, f"{date_key}.csv")
            full_df.to_csv(daily_file_path, index=False, encoding='utf-8')
            saved_files.append(daily_file_path)
            
            print(f"📅 {date_key}: 保存了 {len(daily_codes)} 个样本")
        
        # 在基础目录保存特征名称信息（供参考）
        os.makedirs(save_base_dir, exist_ok=True)
        
        # 保存特征名称信息
        with open(os.path.join(save_base_dir, "feature_names.json"), 'w', encoding='utf-8') as f:
            json.dump(feature_names, f, ensure_ascii=False, indent=2)
        
        # 保存处理信息
        info = {
            'processing_time': datetime.now().isoformat(),
            'total_samples': len(X_processed),
            'total_stocks': len(set(codes)),
            'total_dates': len(data_by_date),
            'feature_count': len(feature_names),
            'month_folder_format': self.month_folder_format,
            'daily_file_format': self.daily_file_format,
            'output_structure': 'yearly_monthly_folders',
            'file_type': 'daily_features_csv',
            'columns_order': ['stock_code'] + feature_names,
            'has_target': False,
            'date_range': {
                'start': min(dates).strftime('%Y-%m-%d'),
                'end': max(dates).strftime('%Y-%m-%d')
            },
            'files_created': [os.path.relpath(f, save_base_dir) for f in saved_files]
        }
        
        with open(os.path.join(save_base_dir, "processing_info.json"), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print("=" * 60)
        print("📊 数据保存完成:")
        print(f"   📁 基础目录: {save_base_dir}")
        print(f"   📅 日期范围: {info['date_range']['start']} ~ {info['date_range']['end']}")
        print(f"   🗂️  月份文件夹: {len(set(date.strftime(self.month_folder_format) for date in dates))} 个")
        print(f"   📄 CSV文件: {len(saved_files)} 个")
        print(f"   🏷️  特征数量: {len(feature_names)} 个")
        print(f"   📋 列结构: stock_code (第1列) + {len(feature_names)}个特征列")
        print(f"   🎯 不包含target列 (由训练脚本定义)")
        print()
        
        # 显示文件夹结构示例
        month_folders = set(date.strftime(self.month_folder_format) for date in dates)
        print("📁 文件夹结构示例:")
        for month_folder in sorted(month_folders)[:3]:  # 只显示前3个月份
            month_files = [f for f in saved_files if f"/{month_folder}/" in f]
            print(f"   {save_base_dir}/{month_folder}/")
            for file_path in month_files[:3]:  # 每个月份只显示前3个文件
                filename = os.path.basename(file_path)
                print(f"     ├── {filename}")
            if len(month_files) > 3:
                print(f"     └── ... 还有 {len(month_files) - 3} 个文件")
        
        # 显示特征名称示例
        print("\n🏷️  CSV文件结构示例:")
        print("   第1列: stock_code")
        feature_types = {}
        for name in feature_names:
            if name.startswith('个股_'):
                feature_type = '个股'
            elif name.startswith('行业_'):
                feature_type = '行业'
            elif name.startswith('指数_'):
                feature_type = '指数'
            elif name.startswith('情绪_'):
                feature_type = '情绪'
            else:
                feature_type = '其他'
            
            if feature_type not in feature_types:
                feature_types[feature_type] = []
            feature_types[feature_type].append(name)
        
        col_index = 2  # 从第2列开始
        for feature_type, features_in_type in feature_types.items():
            examples = features_in_type[:3]
            if len(features_in_type) > 3:
                examples_str = f"{', '.join(examples)}... (+{len(features_in_type)-3}个)"
            else:
                examples_str = ', '.join(examples)
            print(f"   第{col_index}-{col_index+len(features_in_type)-1}列: {feature_type}特征 ({examples_str})")
            col_index += len(features_in_type)
        
        return save_base_dir


def main(config_path: str = None):
    """主处理流程 - 非时序版"""
    if config_path is None:
        config_path = "config/datas_process/data_process_non_timeseries.yaml"
    
    print(f"使用配置: {config_path}")
    print("=" * 60)
    print("🚀 非时序股票数据处理开始")
    print("=" * 60)
    
    # 创建处理器
    processor = NonTimeSeriesStockDataProcessor(config_path)
    
    try:
        # 加载数据
        print("📈 加载股票数据...")
        stock_data = processor.load_stock_data()
        print(f"   加载了 {len(stock_data)} 只股票")
        
        print("🏭 加载行业数据...")
        sector_data = processor.load_sector_data()
        print(f"   加载了 {len(sector_data)} 个行业")
        
        print("📊 加载指数数据...")
        index_data = processor.load_index_data()
        print(f"   加载了 {len(index_data.columns) if not index_data.empty else 0} 个指数")
        
        print("💭 加载情绪数据...")
        sentiment_data = processor.load_sentiment_data()
        print(f"   加载了 {len(sentiment_data.columns) if not sentiment_data.empty else 0} 个情绪指标")
        
        # 创建样本
        print("🔧 创建非时序样本...")
        X, y, codes, feature_names, dates = processor.create_daily_samples_with_feature_names(
            stock_data, sector_data, index_data, sentiment_data
        )
        
        print(f"   生成了 {len(X)} 个样本")
        print(f"   每个样本包含 {len(feature_names)} 个特征")
        print(f"   日期范围: {min(dates).strftime('%Y-%m-%d')} ~ {max(dates).strftime('%Y-%m-%d')}")
        
        # 保存结果
        print("💾 保存数据...")
        save_dir = processor.save_data_with_meaningful_names(X, y, codes, feature_names, dates)
        
        print("=" * 60)
        print("🎉 处理完成!")
        print(f"📊 数据形状: X{X.shape}")
        print(f"🔢 特征数量: {len(feature_names)}")
        print(f"📁 保存位置: {save_dir}")
        print(f"🎯 输出格式: 每日CSV文件，第1列为stock_code，其余为特征列")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    config_path = None
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("用法: python 股板指情数据汇总处理_非时序版.py [配置文件路径]")
            print("示例: python 股板指情数据汇总处理_非时序版.py config/datas_process/data_process_non_timeseries.yaml")
            print("")
            print("特点:")
            print("  - 非时序数据：每个样本只包含当天的特征")
            print("  - 列名包含日期：便于识别数据时间点")
            print("  - 预测未来：预测N天后的涨跌幅")
            sys.exit(0)
        else:
            config_path = sys.argv[1]
    
    main(config_path)