#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证股票列表文件和板块数据获取脚本
快速检查all_stock_list.csv文件和相关配置
"""

import os
import sys
import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_stock_list_file():
    """检查股票列表文件"""
    logger.info("🔍 检查股票列表文件...")
    
    stock_list_paths = [
        "data/stockcode_list/all_stock_list.csv",
        "../data/stockcode_list/all_stock_list.csv", 
        "stockcode_list/all_stock_list.csv",
        "all_stock_list.csv"
    ]
    
    stock_list_file = None
    for path in stock_list_paths:
        if os.path.exists(path):
            stock_list_file = path
            break
    
    if not stock_list_file:
        logger.error("❌ 未找到 all_stock_list.csv 文件")
        logger.info("📁 请确保文件位于以下任一位置:")
        for path in stock_list_paths:
            logger.info(f"   - {path}")
        return False
    
    try:
        df_stocks = pd.read_csv(stock_list_file, encoding='utf-8-sig')
        logger.info(f"✅ 成功找到股票列表文件: {stock_list_file}")
        logger.info(f"📊 文件总行数: {len(df_stocks)}")
        
        # 检查文件格式
        required_columns = ['股票代码', '名称', '上市状态']
        missing_columns = [col for col in required_columns if col not in df_stocks.columns]
        
        if missing_columns:
            logger.warning(f"⚠️ 缺少必要列: {missing_columns}")
            logger.info(f"📋 实际列名: {list(df_stocks.columns)}")
        else:
            logger.info("✅ 文件格式正确")
        
        # 统计股票状态
        if '上市状态' in df_stocks.columns:
            status_counts = df_stocks['上市状态'].value_counts()
            logger.info("📊 股票状态统计:")
            for status, count in status_counts.items():
                logger.info(f"   - {status}: {count}只")
            
            # 筛选正常交易的股票
            active_stocks = df_stocks[df_stocks['上市状态'] == '正常交易']
            logger.info(f"🎯 正常交易股票数: {len(active_stocks)}只")
            
            # 显示前10只股票作为示例
            logger.info("📋 正常交易股票示例 (前10只):")
            for _, stock in active_stocks.head(10).iterrows():
                logger.info(f"   - {stock['股票代码']}: {stock['名称']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 读取股票列表文件失败: {str(e)}")
        return False

def check_samequant_functions():
    """检查samequant_functions是否可用"""
    logger.info("🔍 检查samequant_functions模块...")
    
    # 添加项目路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, 'core'))
    
    try:
        from samequant_functions import Spider_func
        logger.info("✅ samequant_functions 模块导入成功")
        
        # 测试实例化
        s_f_1 = Spider_func()
        logger.info("✅ Spider_func 实例创建成功")
        
        # 测试一个简单的功能（获取行业数据）
        try:
            logger.info("🧪 测试获取行业数据功能...")
            df_industry = s_f_1.get_industry_data_from_eastmoney(sort_field='f3')
            if not df_industry.empty:
                logger.info(f"✅ 行业数据获取成功: {len(df_industry)}个行业")
                logger.info("📋 前5个行业示例:")
                for i, row in df_industry.head(5).iterrows():
                    logger.info(f"   - {row.get('行业名称', 'N/A')}: {row.get('涨跌幅', 'N/A')}")
            else:
                logger.warning("⚠️ 行业数据为空")
                
        except Exception as test_e:
            logger.warning(f"⚠️ 行业数据获取测试失败: {str(test_e)}")
            logger.info("💡 这可能是网络问题，不影响脚本的主要功能")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ 无法导入samequant_functions: {str(e)}")
        logger.info("📁 请确保samequant_functions.py在项目根目录")
        return False
    except Exception as e:
        logger.error(f"❌ samequant_functions测试失败: {str(e)}")
        return False

def check_output_directory():
    """检查输出目录"""
    logger.info("🔍 检查输出目录...")
    
    output_dir = "sector_data"
    
    if os.path.exists(output_dir):
        logger.info(f"✅ 输出目录已存在: {output_dir}")
        
        # 检查现有文件
        existing_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        if existing_files:
            logger.info(f"📁 发现 {len(existing_files)} 个现有CSV文件:")
            for file in existing_files:
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                logger.info(f"   - {file}: {file_size:.1f} KB")
        else:
            logger.info("📁 输出目录为空")
    else:
        logger.info(f"📁 输出目录不存在，将自动创建: {output_dir}")
    
    # 检查写权限
    try:
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        logger.info("✅ 输出目录写权限正常")
        return True
    except Exception as e:
        logger.error(f"❌ 输出目录写权限检查失败: {str(e)}")
        return False

def estimate_processing_time():
    """预估处理时间"""
    logger.info("⏱️ 处理时间预估...")
    
    # 基于经验值的时间预估
    time_per_stock = 0.5  # 每只股票约0.5秒（包含网络延迟和重试）
    
    stock_counts = [100, 500, 1000, 5000]
    logger.info("📊 不同模式的预估处理时间:")
    
    for count in stock_counts:
        estimated_seconds = count * time_per_stock
        estimated_minutes = estimated_seconds / 60
        estimated_hours = estimated_minutes / 60
        
        if estimated_hours >= 1:
            time_str = f"{estimated_hours:.1f}小时"
        elif estimated_minutes >= 1:
            time_str = f"{estimated_minutes:.1f}分钟"
        else:
            time_str = f"{estimated_seconds:.0f}秒"
        
        logger.info(f"   - {count}只股票: 约{time_str}")

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("🔧 股票板块数据获取 - 系统验证")
    logger.info("=" * 60)
    
    all_checks_passed = True
    
    # 1. 检查股票列表文件
    if not check_stock_list_file():
        all_checks_passed = False
    
    logger.info("")
    
    # 2. 检查samequant_functions模块
    if not check_samequant_functions():
        all_checks_passed = False
    
    logger.info("")
    
    # 3. 检查输出目录
    if not check_output_directory():
        all_checks_passed = False
    
    logger.info("")
    
    # 4. 预估处理时间
    estimate_processing_time()
    
    logger.info("")
    logger.info("=" * 60)
    
    if all_checks_passed:
        logger.info("🎉 所有检查通过！系统已准备就绪")
        logger.info("💡 现在可以运行: python data_processing/获取板块数据并保存CSV.py")
    else:
        logger.warning("⚠️ 部分检查未通过，请解决上述问题后再运行数据获取脚本")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()