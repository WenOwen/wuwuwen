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
    
    def create_samples_with_feature_names(self, stock_data: Dict, sector_data: Dict, index_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """创建训练样本并生成有意义的特征名称"""
        window_size = self.processing_config.get('window_size', 30)
        start_date = self.processing_config.get('date_range', {}).get('start_date')
        end_date = self.processing_config.get('date_range', {}).get('end_date')
        
        all_samples = []
        all_targets = []
        all_codes = []
        feature_names = []  # 存储特征名称
        feature_names_generated = False
        
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
                    
                    # 收集每个时间步的数据
                    time_step_features = []
                    current_feature_names = []  # 当前样本的特征名称
                    
                    # 按时间步组织特征（而不是按特征类型）
                    for t in range(window_size):
                        step_features = []
                        step_feature_names = []
                        
                        # 个股特征 - 当前时间步
                        if self.features_config.get('stock', {}).get('enabled', True):
                            stock_values = stock_aligned.iloc[window_start + t].values
                            step_features.extend(stock_values)
                            
                            # 生成个股特征名称 (只在第一次生成)
                            if not feature_names_generated:
                                for col_name in stock_aligned.columns:
                                    step_feature_names.append(f'step_{t:02d}_个股_{col_name}')
                        
                        # 行业特征 - 当前时间步
                        if self.features_config.get('sector', {}).get('enabled', True) and sector_aligned is not None:
                            sector_values = sector_aligned.iloc[window_start + t].values
                            step_features.extend(sector_values)
                            
                            # 生成行业特征名称
                            if not feature_names_generated:
                                for col_name in sector_aligned.columns:
                                    step_feature_names.append(f'step_{t:02d}_行业_{col_name}')
                        
                        # 指数特征 - 当前时间步
                        if self.features_config.get('index', {}).get('enabled', True) and index_aligned is not None:
                            index_values = index_aligned.iloc[window_start + t].values
                            step_features.extend(index_values)
                            
                            # 生成指数特征名称
                            if not feature_names_generated:
                                for col_name in index_aligned.columns:
                                    step_feature_names.append(f'step_{t:02d}_指数_{col_name}')
                        
                        # 情绪特征 - 当前时间步
                        if self.features_config.get('sentiment', {}).get('enabled', True) and sentiment_aligned is not None:
                            sentiment_values = sentiment_aligned.iloc[window_start + t].values
                            step_features.extend(sentiment_values)
                            
                            # 生成情绪特征名称
                            if not feature_names_generated:
                                for col_name in sentiment_aligned.columns:
                                    step_feature_names.append(f'step_{t:02d}_情绪_{col_name}')
                        
                        # 添加当前时间步的所有特征
                        time_step_features.extend(step_features)
                        if not feature_names_generated:
                            current_feature_names.extend(step_feature_names)
                    
                    if time_step_features:
                        combined = np.array(time_step_features)
                        
                        # 保存特征名称 (只在第一次)
                        if not feature_names_generated:
                            feature_names = current_feature_names
                            feature_names_generated = True
                        
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
            return np.array(all_samples), np.array(all_targets), all_codes, feature_names
        else:
            raise ValueError("没有生成样本")
    
    def save_data_with_meaningful_names(self, X: np.ndarray, y: np.ndarray, codes: List[str], feature_names: List[str] = None) -> str:
        """保存数据并使用有意义的特征名称"""
        save_dir = self.output_config.get('save_base_dir', './data/processed')
        custom_name = self.output_config.get('custom_folder_name')
        
        if custom_name:
            final_dir = os.path.join(save_dir, custom_name)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_dir = os.path.join(save_dir, f"processed_{timestamp}")
        
        os.makedirs(final_dir, exist_ok=True)
        
        # 处理X数据
        if len(X.shape) == 3:
            samples, timesteps, features = X.shape
            X_flattened = X.reshape(samples, timesteps * features)
            
            # 使用传入的特征名称，如果没有则生成默认名称
            if feature_names is None or len(feature_names) != X_flattened.shape[1]:
                feature_names = []
                for t in range(timesteps):
                    for f in range(features):
                        feature_names.append(f'timestep_{t:02d}_feature_{f:02d}')
        
        else:
            # 如果已经是2D数组
            X_flattened = X
            if feature_names is None or len(feature_names) != X.shape[1]:
                feature_names = [f'feature_{i:02d}' for i in range(X.shape[1])]
        
        # 创建特征DataFrame
        X_df = pd.DataFrame(X_flattened, columns=feature_names)
        
        # 添加股票代码列
        X_df['stock_code'] = codes
        
        # 创建目标DataFrame
        y_df = pd.DataFrame({
            'stock_code': codes,
            'target': y
        })
        
        # 创建完整数据DataFrame（特征+目标+股票代码）
        full_df = X_df.copy()
        full_df['target'] = y
        
        # 保存CSV文件
        print("保存特征数据...")
        X_df.to_csv(os.path.join(final_dir, "X_features.csv"), index=False, encoding='utf-8')
        
        print("保存目标数据...")
        y_df.to_csv(os.path.join(final_dir, "y_targets.csv"), index=False, encoding='utf-8')
        
        print("保存完整数据...")
        full_df.to_csv(os.path.join(final_dir, "full_data.csv"), index=False, encoding='utf-8')
        
        # 保存股票代码信息
        with open(os.path.join(final_dir, "stock_codes.json"), 'w', encoding='utf-8') as f:
            json.dump(codes, f, ensure_ascii=False, indent=2)
        
        # 保存特征名称信息
        with open(os.path.join(final_dir, "feature_names.json"), 'w', encoding='utf-8') as f:
            json.dump(feature_names, f, ensure_ascii=False, indent=2)
        
        # 保存数据信息
        info = {
            'original_X_shape': list(X.shape),
            'flattened_X_shape': list(X_flattened.shape),
            'y_shape': list(y.shape),
            'num_samples': len(codes),
            'num_stocks': len(set(codes)),
            'num_features': X_flattened.shape[1] if len(X_flattened.shape) > 1 else 1,
            'feature_names_count': len(feature_names),
            'processing_time': datetime.now().isoformat(),
            'output_format': 'CSV',
            'has_meaningful_names': True,
            'files_created': [
                'X_features.csv',
                'y_targets.csv', 
                'full_data.csv',
                'stock_codes.json',
                'feature_names.json',
                'data_info.json'
            ]
        }
        
        with open(os.path.join(final_dir, "data_info.json"), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"数据已保存为CSV格式:")
        print(f"  - 特征数据: X_features.csv ({X_flattened.shape[0]} 行 x {X_flattened.shape[1]} 列)")
        print(f"  - 目标数据: y_targets.csv ({len(y)} 行)")
        print(f"  - 完整数据: full_data.csv ({len(y)} 行 x {X_flattened.shape[1] + 2} 列)")
        print(f"  - 特征名称: feature_names.json ({len(feature_names)} 个)")
        
        # 显示特征名称示例，按时间步分组展示
        print(f"特征名称示例 (按时间步组织):")
        
        # 分析特征结构
        step_features = {}
        for i, name in enumerate(feature_names):
            if name.startswith('step_'):
                step_num = name.split('_')[1]
                if step_num not in step_features:
                    step_features[step_num] = []
                step_features[step_num].append((i, name))
        
        # 显示前3个时间步作为示例
        steps_to_show = sorted(step_features.keys())[:3]
        for step in steps_to_show:
            print(f"  {step}时间步:")
            features_in_step = step_features[step][:6]  # 只显示前6个特征
            for idx, name in features_in_step:
                print(f"    {idx:3d}: {name}")
            if len(step_features[step]) > 6:
                print(f"    ... 还有 {len(step_features[step]) - 6} 个特征")
            print()
        
        if len(steps_to_show) < len(step_features):
            remaining_steps = len(step_features) - len(steps_to_show)
            print(f"  ... 还有 {remaining_steps} 个时间步")
        
        # 统计每个时间步的特征数量
        if step_features:
            features_per_step = len(step_features[steps_to_show[0]]) if steps_to_show else 0
            print(f"特征结构: {len(step_features)} 个时间步 × {features_per_step} 个特征/时间步 = {len(feature_names)} 个总特征")
        
        return final_dir


def main(config_path: str = None):
    """主处理流程 - 修改版"""
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
        
        # 创建样本 - 使用新方法
        print("创建训练样本...")
        X, y, codes, feature_names = processor.create_samples_with_feature_names(
            stock_data, sector_data, index_data, sentiment_data
        )
        
        # 保存结果 - 使用新方法
        print("保存数据...")
        save_dir = processor.save_data_with_meaningful_names(X, y, codes, feature_names)
        
        print(f"完成! 数据形状: X{X.shape}, y{y.shape}")
        print(f"特征数量: {len(feature_names)}")
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