#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
获取真实的板块数据并保存为CSV文件
基于原始的2.10获取板块数据.py，增加CSV保存功能
"""

import os
import sys
import pandas as pd
from datetime import datetime
import logging
import time

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'core'))

# 导入量化函数
try:
    from samequant_functions import Spider_func
except ImportError as e:
    print(f"❌ 无法导入samequant_functions: {str(e)}")
    print("请确保samequant_functions.py在项目根目录")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_real_sector_data():
    """获取真实的板块数据并保存"""
    logger.info("🚀 开始获取真实板块数据...")
    
    # 初始化Spider_func实例
    s_f_1 = Spider_func()
    
    # 创建保存目录
    save_dir = "data/datas_sector"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 获取行业板块数据
    logger.info("📊 获取行业板块数据...")
    try:
        df_industry = s_f_1.get_industry_data_from_eastmoney(sort_field='f3')
        if not df_industry.empty:
            industry_file = os.path.join(save_dir, "行业板块数据.csv")
            df_industry.to_csv(industry_file, index=False, encoding='utf-8-sig')
            logger.info(f"✅ 行业板块数据已保存: {industry_file} ({len(df_industry)}行)")
            print("涨幅排行前10的行业:")
            print(df_industry[['行业名称', '涨跌幅', '主力净流入', '成交额', '总市值']].head(10))
        else:
            logger.warning("⚠️ 行业板块数据为空")
    except Exception as e:
        logger.error(f"❌ 获取行业板块数据失败: {str(e)}")
        df_industry = pd.DataFrame()

    # 2. 获取概念板块数据
    logger.info("📊 获取概念板块数据...")
    try:
        df_concept = s_f_1.get_concept_data_from_eastmoney(sort_field='f3')
        if not df_concept.empty:
            concept_file = os.path.join(save_dir, "概念板块数据.csv")
            df_concept.to_csv(concept_file, index=False, encoding='utf-8-sig')
            logger.info(f"✅ 概念板块数据已保存: {concept_file} ({len(df_concept)}行)")
            print("涨幅排行前10的概念:")
            print(df_concept[['概念名称', '涨跌幅', '主力净流入', '成交额', '总市值', '流通市值']].head(10))
        else:
            logger.warning("⚠️ 概念板块数据为空")
    except Exception as e:
        logger.error(f"❌ 获取概念板块数据失败: {str(e)}")
        df_concept = pd.DataFrame()

    # 3. 获取热门概念排行
    logger.info("📊 获取热门概念排行...")
    try:
        df_hot_concepts = s_f_1.get_hot_concepts_from_eastmoney(limit=50)  # 增加到50个
        if not df_hot_concepts.empty:
            hot_concepts_file = os.path.join(save_dir, "热门概念排行.csv")
            df_hot_concepts.to_csv(hot_concepts_file, index=False, encoding='utf-8-sig')
            logger.info(f"✅ 热门概念排行已保存: {hot_concepts_file} ({len(df_hot_concepts)}行)")
            print("热门概念排行前20:")
            print(df_hot_concepts[['概念名称', '涨跌幅', '主力净流入', '上涨家数', '下跌家数']].head(20))
        else:
            logger.warning("⚠️ 热门概念数据为空")
    except Exception as e:
        logger.error(f"❌ 获取热门概念数据失败: {str(e)}")
        df_hot_concepts = pd.DataFrame()

    return df_industry, df_concept, df_hot_concepts

def load_all_stock_list():
    """从all_stock_list.csv加载完整的股票列表"""
    logger.info("📊 加载完整股票列表...")
    
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
        return []
    
    try:
        df_stocks = pd.read_csv(stock_list_file, encoding='utf-8-sig')
        logger.info(f"✅ 成功加载股票列表: {stock_list_file}")
        logger.info(f"📊 总股票数: {len(df_stocks)}")
        
        # 筛选正常交易的股票
        if '上市状态' in df_stocks.columns:
            active_stocks = df_stocks[df_stocks['上市状态'] == '正常交易']
            logger.info(f"📊 正常交易股票数: {len(active_stocks)}")
        else:
            active_stocks = df_stocks
            logger.warning("⚠️ 未找到上市状态列，使用全部股票")
        
        # 提取股票代码
        stock_codes = active_stocks['股票代码'].tolist()
        logger.info(f"📊 将获取 {len(stock_codes)} 只股票的板块信息")
        
        return stock_codes
        
    except Exception as e:
        logger.error(f"❌ 加载股票列表失败: {str(e)}")
        return []

def get_stock_sector_mapping(max_stocks=None, start_from=0, batch_size=50):
    """
    获取股票的板块映射信息
    
    Args:
        max_stocks: 最大处理股票数量，None表示处理全部
        start_from: 从第几只股票开始处理（用于断点续传）
        batch_size: 批次大小，每批保存一次中间结果
    """
    logger.info("📊 获取股票板块映射信息...")
    
    # 初始化Spider_func实例
    s_f_1 = Spider_func()
    
    # 从all_stock_list.csv加载股票代码
    all_stock_codes = load_all_stock_list()
    
    if not all_stock_codes:
        logger.error("❌ 无法获取股票列表，使用默认代码")
        # 使用一些常见的股票代码作为示例
        all_stock_codes = [
            'sh600519', 'sz000001', 'sz000002', 'sh600000', 'sz000858', 
            'sh600036', 'sz000858', 'sh600276', 'sz002594', 'sh601318',
            'sh600309', 'sz000063', 'sh603259', 'sz002415', 'sh600887'
        ]
    
    # 确定处理范围
    end_index = min(start_from + max_stocks, len(all_stock_codes)) if max_stocks else len(all_stock_codes)
    stock_codes = all_stock_codes[start_from:end_index]
    
    logger.info(f"📊 处理范围: {start_from+1} - {end_index} / {len(all_stock_codes)} 只股票")
    logger.info(f"📊 本批次处理: {len(stock_codes)} 只股票")
    
    # 检查是否有已保存的中间结果
    temp_file = "data/datas_sector/股票板块映射_临时.csv"
    existing_data = []
    if os.path.exists(temp_file):
        try:
            df_existing = pd.read_csv(temp_file, encoding='utf-8-sig')
            existing_data = df_existing.to_dict('records')
            existing_codes = set(df_existing['股票代码'].tolist())
            # 过滤掉已经处理过的股票
            stock_codes = [code for code in stock_codes if code not in existing_codes]
            logger.info(f"📊 发现中间结果，已处理 {len(existing_data)} 只股票")
            logger.info(f"📊 剩余待处理: {len(stock_codes)} 只股票")
        except Exception as e:
            logger.warning(f"⚠️ 读取中间结果失败: {str(e)}")
            existing_data = []
    
    # 获取股票板块信息
    stock_sector_mapping = existing_data.copy()  # 包含已有数据
    successful_count = len(existing_data)
    batch_count = 0
    
    # 创建保存目录
    os.makedirs("data/datas_sector", exist_ok=True)
    
    logger.info(f"🚀 开始获取 {len(stock_codes)} 只股票的板块信息...")
    logger.info(f"💾 每 {batch_size} 只股票保存一次中间结果")
    
    for i, stock_code in enumerate(stock_codes, 1):
        try:
            # 显示进度
            if i % 10 == 0:
                progress = (i / len(stock_codes)) * 100
                logger.info(f"📊 进度: {i}/{len(stock_codes)} ({progress:.1f}%) - 成功: {successful_count}")
            
            # 获取股票信息（添加重试机制）
            stock_info = None
            for retry in range(3):  # 最多重试3次
                try:
                    stock_info = s_f_1.get_stock_industry_info_from_eastmoney(stock_code=stock_code)
                    if stock_info:
                        break
                except Exception as retry_e:
                    if retry == 2:  # 最后一次重试
                        logger.warning(f"⚠️ 股票 {stock_code} 重试3次后仍失败: {str(retry_e)}")
                    else:
                        time.sleep(1)  # 重试前等待1秒
            
            if stock_info:
                mapping_info = {
                    '股票代码': stock_code,
                    '股票名称': stock_info.get('股票名称', ''),
                    '所属行业': stock_info.get('所属行业', ''),
                    '概念板块': stock_info.get('概念板块', ''),
                    '地区': stock_info.get('地区', ''),
                    '总股本': stock_info.get('总股本', ''),
                    '流通股': stock_info.get('流通股', '')
                }
                stock_sector_mapping.append(mapping_info)
                successful_count += 1
                
                # 显示前10个成功的结果
                if successful_count <= 10:
                    print(f"{stock_info['股票名称']}({stock_code}):")
                    print(f"  所属行业: {stock_info['所属行业']}")
                    print(f"  概念板块: {stock_info['概念板块']}")
                    print()
            else:
                logger.warning(f"⚠️ 无法获取股票 {stock_code} 的信息")
            
            # 批次保存中间结果
            batch_count += 1
            if batch_count >= batch_size:
                try:
                    df_temp = pd.DataFrame(stock_sector_mapping)
                    df_temp.to_csv(temp_file, index=False, encoding='utf-8-sig')
                    logger.info(f"💾 已保存中间结果: {len(stock_sector_mapping)} 条记录")
                    batch_count = 0
                except Exception as save_e:
                    logger.error(f"❌ 保存中间结果失败: {str(save_e)}")
            
            # 添加延迟避免请求太频繁
            time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"❌ 获取股票 {stock_code} 信息时出错: {str(e)}")
            continue
    
    # 最终保存（包含最后一批数据）
    if batch_count > 0 and stock_sector_mapping:
        try:
            df_temp = pd.DataFrame(stock_sector_mapping)
            df_temp.to_csv(temp_file, index=False, encoding='utf-8-sig')
            logger.info(f"💾 保存最终中间结果: {len(stock_sector_mapping)} 条记录")
        except Exception as save_e:
            logger.error(f"❌ 保存最终结果失败: {str(save_e)}")
    
    # 保存股票板块映射
    if stock_sector_mapping:
        df_stock_mapping = pd.DataFrame(stock_sector_mapping)
        
        # 保存完整映射
        save_dir = "data/datas_sector"
        os.makedirs(save_dir, exist_ok=True)
        mapping_file = os.path.join(save_dir, "股票板块映射表.csv")
        df_stock_mapping.to_csv(mapping_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"✅ 股票板块映射已保存: {mapping_file} ({len(df_stock_mapping)}行)")
        logger.info(f"📊 成功获取信息的股票: {successful_count}/{len(all_stock_codes[start_from:end_index])} ({successful_count/(len(all_stock_codes[start_from:end_index]))*100:.1f}%)")
        
        # 删除临时文件
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info("🗑️ 已删除临时文件")
        except Exception as e:
            logger.warning(f"⚠️ 删除临时文件失败: {str(e)}")
        
        return df_stock_mapping
    
    return pd.DataFrame()

def create_sector_summary():
    """创建板块数据摘要"""
    logger.info("📊 创建板块数据摘要...")
    
    save_dir = "data/datas_sector"
    if not os.path.exists(save_dir):
        logger.error("❌ data/datas_sector目录不存在，请先运行数据获取")
        return
    
    summary = {
        '数据获取时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '文件列表': []
    }
    
    # 检查生成的文件
    csv_files = [f for f in os.listdir(save_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        file_path = os.path.join(save_dir, csv_file)
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            file_info = {
                '文件名': csv_file,
                '行数': len(df),
                '列数': len(df.columns),
                '文件大小': f"{os.path.getsize(file_path) / 1024:.1f} KB"
            }
            summary['文件列表'].append(file_info)
            logger.info(f"✅ {csv_file}: {len(df)}行 x {len(df.columns)}列")
        except Exception as e:
            logger.error(f"❌ 读取文件 {csv_file} 失败: {str(e)}")
    
    # 保存摘要
    summary_file = os.path.join(save_dir, "数据摘要.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("板块数据获取摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"获取时间: {summary['数据获取时间']}\n\n")
        f.write("生成的文件:\n")
        for file_info in summary['文件列表']:
            f.write(f"- {file_info['文件名']}: {file_info['行数']}行, {file_info['文件大小']}\n")
    
    logger.info(f"✅ 数据摘要已保存: {summary_file}")

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("🚀 AI股市预测系统 - 真实板块数据获取")
    logger.info("=" * 60)
    
    # 配置选项
    print("请选择数据获取模式:")
    print("1. 快速测试模式 (100只股票)")
    print("2. 中等规模模式 (500只股票)")
    print("3. 大规模模式 (1000只股票)")
    print("4. 完整模式 (所有正常交易股票)")
    print("5. 自定义数量")
    
    try:
        choice = input("请选择 (1-5): ").strip()
        
        if choice == "1":
            max_stocks = 100
            logger.info("🧪 快速测试模式: 处理100只股票")
        elif choice == "2":
            max_stocks = 500
            logger.info("📊 中等规模模式: 处理500只股票")
        elif choice == "3":
            max_stocks = 1000
            logger.info("📈 大规模模式: 处理1000只股票")
        elif choice == "4":
            max_stocks = None
            logger.info("🎯 完整模式: 处理所有正常交易股票")
        elif choice == "5":
            try:
                max_stocks = int(input("请输入要处理的股票数量: ").strip())
                logger.info(f"🔧 自定义模式: 处理{max_stocks}只股票")
            except ValueError:
                logger.error("❌ 输入无效，使用快速测试模式")
                max_stocks = 100
        else:
            logger.info("🧪 默认使用快速测试模式: 处理100只股票")
            max_stocks = 100
            
    except KeyboardInterrupt:
        logger.info("用户取消操作")
        return
    except Exception:
        logger.info("🧪 使用快速测试模式: 处理100只股票")
        max_stocks = 100
    
    try:
        start_time = datetime.now()
        
        # 1. 获取板块数据
        logger.info("第一步: 获取板块行业数据...")
        df_industry, df_concept, df_hot_concepts = get_real_sector_data()
        
        # 2. 获取股票板块映射
        logger.info("第二步: 获取股票板块映射...")
        df_stock_mapping = get_stock_sector_mapping(
            max_stocks=max_stocks,
            start_from=0,
            batch_size=50
        )
        
        # 3. 创建数据摘要
        logger.info("第三步: 创建数据摘要...")
        create_sector_summary()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("🎉 所有板块数据获取完成！")
        logger.info(f"⏱️ 总耗时: {duration}")
        logger.info("📁 数据保存在 data/datas_sector/ 目录中")
        logger.info("💡 可以直接使用 股票板块映射表.csv 进行模型训练")
        
        # 显示获取统计
        if not df_stock_mapping.empty:
            unique_industries = df_stock_mapping['所属行业'].nunique()
            unique_concepts = df_stock_mapping['概念板块'].nunique()
            logger.info(f"📊 数据统计: {len(df_stock_mapping)}只股票, {unique_industries}个行业, {unique_concepts}个概念")
        
    except KeyboardInterrupt:
        logger.info("⏹️ 用户中断操作")
        logger.info("💾 中间结果已保存，可以继续运行来完成剩余工作")
    except Exception as e:
        logger.error(f"❌ 板块数据获取失败: {str(e)}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")

def continue_from_interruption():
    """从中断处继续获取数据"""
    logger.info("🔄 继续从中断处获取数据...")
    
    # 检查临时文件
    temp_file = "data/datas_sector/股票板块映射_临时.csv"
    if not os.path.exists(temp_file):
        logger.error("❌ 未找到中断的临时文件")
        return
    
    try:
        df_existing = pd.read_csv(temp_file, encoding='utf-8-sig')
        processed_count = len(df_existing)
        logger.info(f"📊 已处理 {processed_count} 只股票，继续处理剩余股票...")
        
        # 继续处理
        df_stock_mapping = get_stock_sector_mapping(
            max_stocks=None,  # 处理所有剩余
            start_from=0,     # 函数内部会自动跳过已处理的
            batch_size=50
        )
        
        create_sector_summary()
        logger.info("🎉 继续处理完成！")
        
    except Exception as e:
        logger.error(f"❌ 继续处理失败: {str(e)}")

if __name__ == "__main__":
    # 检查是否有中断的临时文件
    temp_file = "data/datas_sector/股票板块映射_临时.csv"
    if os.path.exists(temp_file):
        print("发现中断的临时文件，是否继续之前的工作？")
        continue_choice = input("输入 'c' 继续，或任意键重新开始: ").strip().lower()
        if continue_choice == 'c':
            continue_from_interruption()
        else:
            main()
    else:
        main()

