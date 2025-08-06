# -*- coding: utf-8 -*-
"""
批量特征处理脚本
演示如何使用缓存系统高效处理大量股票
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from core.feature_engineering import FeatureEngineering
from core.feature_cache import BatchFeatureProcessor
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_stock_data(stock_code: str, data_dir: str = "data/datas_em") -> pd.DataFrame:
    """
    加载股票数据
    """
    try:
        file_path = os.path.join(data_dir, f"{stock_code}.csv")
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return None
        
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 数据清理和标准化
        if '交易日期' in df.columns:
            df['交易日期'] = pd.to_datetime(df['交易日期'])
        
        # 确保基本列存在
        required_cols = ['收盘价', '开盘价', '最高价', '最低价', '成交量']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"股票 {stock_code} 缺少必要列: {missing_cols}")
            return None
        
        # 数据质量检查
        if len(df) < 60:  # 至少需要60天数据
            logger.warning(f"股票 {stock_code} 数据量不足: {len(df)} 天")
            return None
        
        return df.sort_values('交易日期').reset_index(drop=True)
        
    except Exception as e:
        logger.error(f"加载股票 {stock_code} 数据失败: {e}")
        return None

def get_stock_list(data_dir: str = "data/datas_em", limit: int = None) -> list:
    """
    获取股票列表
    """
    try:
        if not os.path.exists(data_dir):
            logger.error(f"数据目录不存在: {data_dir}")
            return []
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        stock_codes = [f.replace('.csv', '') for f in csv_files]
        
        if limit:
            stock_codes = stock_codes[:limit]
        
        logger.info(f"找到 {len(stock_codes)} 只股票")
        return stock_codes
        
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return []

def demo_traditional_processing(stock_codes: list, data_dir: str = "data/datas_em"):
    """
    演示传统处理方式（无缓存）
    """
    print("\n🐌 传统处理方式 (无缓存):")
    
    fe = FeatureEngineering(enable_cache=False)
    start_time = time.time()
    
    processed_count = 0
    for i, stock_code in enumerate(stock_codes):
        try:
            if i % 5 == 0:
                print(f"进度: {i+1}/{len(stock_codes)} ({(i+1)/len(stock_codes)*100:.1f}%)")
            
            df = load_stock_data(stock_code, data_dir)
            if df is None:
                continue
            
            df_features = fe.create_all_features(df, stock_code)
            processed_count += 1
            
        except Exception as e:
            logger.error(f"处理股票 {stock_code} 失败: {e}")
            continue
    
    end_time = time.time()
    
    print(f"✅ 传统方式完成:")
    print(f"  处理股票: {processed_count}")
    print(f"  总耗时: {end_time - start_time:.2f} 秒")
    print(f"  平均每只: {(end_time - start_time) / max(processed_count, 1):.2f} 秒")

def demo_cached_processing(stock_codes: list, data_dir: str = "data/datas_em"):
    """
    演示缓存处理方式
    """
    print("\n🚀 缓存处理方式:")
    
    fe = FeatureEngineering(enable_cache=True)
    processor = BatchFeatureProcessor(fe)
    
    start_time = time.time()
    
    # 第一次处理（建立缓存）
    def data_loader(stock_code):
        return load_stock_data(stock_code, data_dir)
    
    results = processor.process_stocks_with_cache(stock_codes, data_loader)
    
    first_run_time = time.time()
    
    print(f"\n✅ 第一次处理完成 (建立缓存):")
    print(f"  处理股票: {len(results)}")
    print(f"  总耗时: {first_run_time - start_time:.2f} 秒")
    print(f"  平均每只: {(first_run_time - start_time) / max(len(results), 1):.2f} 秒")
    
    # 第二次处理（使用缓存）
    print(f"\n🚀 第二次处理 (使用缓存):")
    second_start_time = time.time()
    
    results2 = processor.process_stocks_with_cache(stock_codes, data_loader)
    
    second_end_time = time.time()
    
    print(f"\n✅ 第二次处理完成 (使用缓存):")
    print(f"  处理股票: {len(results2)}")
    print(f"  总耗时: {second_end_time - second_start_time:.2f} 秒")
    print(f"  平均每只: {(second_end_time - second_start_time) / max(len(results2), 1):.2f} 秒")
    
    # 计算加速比
    if len(results) > 0:
        speedup = (first_run_time - start_time) / (second_end_time - second_start_time)
        print(f"\n📈 性能提升:")
        print(f"  加速比: {speedup:.1f}x")
        print(f"  时间节省: {((first_run_time - start_time) - (second_end_time - second_start_time)):.2f} 秒")

def main():
    """
    主函数
    """
    print("📊 批量特征处理性能测试")
    print("=" * 50)
    
    # 获取股票列表（限制数量用于演示）
    stock_codes = get_stock_list(limit=20)  # 先用20只股票测试
    
    if len(stock_codes) == 0:
        print("❌ 没有找到股票数据")
        return
    
    print(f"📋 将处理 {len(stock_codes)} 只股票:")
    print(f"  {', '.join(stock_codes[:10])}{'...' if len(stock_codes) > 10 else ''}")
    
    # 演示缓存处理方式
    demo_cached_processing(stock_codes)
    
    # 清理缓存并重置（可选）
    # print("\n🗑️ 清理缓存...")
    # fe = FeatureEngineering(enable_cache=True)
    # if fe.cache:
    #     fe.cache.clear_all_cache()
    
    print("\n🎉 性能测试完成!")

if __name__ == "__main__":
    main()