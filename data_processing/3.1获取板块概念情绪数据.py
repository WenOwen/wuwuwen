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

    # 2. 获取概念板块数据（完整的438个概念板块）
    logger.info("📊 获取完整概念板块数据...")
    try:
        df_concept = s_f_1.get_concept_data_from_eastmoney(sort_field='f3')
        if not df_concept.empty:
            concept_file = os.path.join(save_dir, "概念板块数据.csv")
            df_concept.to_csv(concept_file, index=False, encoding='utf-8-sig')
            logger.info(f"✅ 概念板块数据已保存: {concept_file} ({len(df_concept)}行)")
            logger.info(f"📊 成功获取完整的 {len(df_concept)} 个概念板块（预期438个）")
            print("涨幅排行前10的概念:")
            print(df_concept[['概念名称', '涨跌幅', '主力净流入', '成交额', '总市值', '流通市值']].head(10))
            
            # 显示获取统计
            up_count = len(df_concept[df_concept['涨跌幅'] > 0])
            down_count = len(df_concept[df_concept['涨跌幅'] < 0])
            flat_count = len(df_concept[df_concept['涨跌幅'] == 0])
            logger.info(f"📈 概念板块表现: 上涨{up_count}个, 下跌{down_count}个, 平盘{flat_count}个")
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

def get_latest_data_date(file_path):
    """
    获取数据文件中最新的日期
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        最新日期字符串 (YYYY-MM-DD格式) 或 None
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        if len(df) == 0:
            return None
        
        # 查找日期列
        date_columns = ['日期', 'date', '时间', 'datetime']
        date_col = None
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            logger.warning(f"⚠️ 文件 {file_path} 中未找到日期列")
            return None
        
        # 获取最新日期
        latest_date = df[date_col].max()
        # 转换为标准格式
        if pd.notna(latest_date):
            if isinstance(latest_date, str):
                return latest_date
            else:
                return latest_date.strftime('%Y-%m-%d')
        
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ 读取文件 {file_path} 最新日期失败: {str(e)}")
        return None

def check_stock_mapping_update_needed():
    """
    检查股票板块映射是否需要更新
    
    Returns:
        tuple: (需要更新, 最新日期, 说明信息)
    """
    mapping_file = "data/datas_sector_historical/股票板块映射表.csv"
    
    if not os.path.exists(mapping_file):
        return True, None, "映射文件不存在，需要全量获取"
    
    try:
        # 检查文件修改时间
        file_mtime = os.path.getmtime(mapping_file)
        file_date = datetime.fromtimestamp(file_mtime)
        current_date = datetime.now()
        days_diff = (current_date - file_date).days
        
        if days_diff >= 7:  # 超过7天则建议更新
            return True, file_date.strftime('%Y-%m-%d'), f"文件已{days_diff}天未更新，建议更新"
        else:
            return False, file_date.strftime('%Y-%m-%d'), f"文件较新（{days_diff}天前更新），无需更新"
            
    except Exception as e:
        logger.warning(f"⚠️ 检查映射文件状态失败: {str(e)}")
        return True, None, "无法检查文件状态，建议更新"

def get_stock_sector_mapping(max_stocks=None, start_from=0, batch_size=50, incremental=False):
    """
    获取股票的板块映射信息
    
    Args:
        max_stocks: 最大处理股票数量，None表示处理全部
        start_from: 从第几只股票开始处理（用于断点续传）
        batch_size: 批次大小，每批保存一次中间结果
        incremental: 是否为增量更新模式
    """
    logger.info("📊 获取股票板块映射信息...")
    
    # 增量更新模式检查
    if incremental:
        needs_update, last_date, message = check_stock_mapping_update_needed()
        logger.info(f"🔍 增量更新检查: {message}")
        
        if not needs_update:
            logger.info("✅ 股票映射数据较新，跳过更新")
            mapping_file = "data/datas_sector_historical/股票板块映射表.csv"
            if os.path.exists(mapping_file):
                return pd.read_csv(mapping_file, encoding='utf-8-sig')
            
        logger.info("🔄 启动增量更新模式")
    
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
    temp_file = "data/datas_sector_historical/股票板块映射_临时.csv"
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
    os.makedirs("data/datas_sector_historical", exist_ok=True)
    
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
                    '地区': stock_info.get('地区', '')
                }
                stock_sector_mapping.append(mapping_info)
                successful_count += 1
                
                # 显示前10个成功的结果
                if successful_count <= 10:
                    print(f"{stock_info['股票名称']}({stock_code}):")
                    print(f"  所属行业: {stock_info['所属行业']}")
                    print(f"  概念板块: {stock_info['概念板块']}")
                    if stock_info.get('地区'):
                        print(f"  地区: {stock_info['地区']}")
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
            
            # 每处理100个股票显示一次进度统计
            if i % 100 == 0:
                success_rate = (successful_count / i) * 100
                logger.info(f"📊 中期统计: 成功率 {success_rate:.1f}% ({successful_count}/{i})")
                
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
        save_dir = "data/datas_sector_historical"
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

def check_sector_data_update_needed(sector_name, sector_code, sector_type):
    """
    检查板块数据是否需要更新
    
    Args:
        sector_name: 板块名称
        sector_code: 板块代码
        sector_type: 板块类型
        
    Returns:
        tuple: (需要更新, 最新日期, 数据文件路径)
    """
    # 查找可能的数据文件
    possible_dirs = [
        "data/datas_sector_historical/行业板块_全部历史",
        "data/datas_sector_historical/概念板块_全部历史",
        "data/datas_sector_historical/行业板块",
        "data/datas_sector_historical/概念板块",
        "data/datas_sector_historical"
    ]
    
    possible_filenames = [
        f"{sector_name}({sector_code})_daily_历史数据.csv",
        f"{sector_name}_{sector_code}_daily_历史数据.csv",
        f"板块{sector_code}_daily_历史数据.csv",
        f"{sector_code}_daily_历史数据.csv"
    ]
    
    data_file = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            for filename in possible_filenames:
                file_path = os.path.join(dir_path, filename)
                if os.path.exists(file_path):
                    data_file = file_path
                    break
        if data_file:
            break
    
    if not data_file:
        return True, None, None
    
    # 获取最新日期
    latest_date = get_latest_data_date(data_file)
    if not latest_date:
        return True, None, data_file
    
    # 检查日期是否需要更新（超过1天）
    try:
        latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
        current_dt = datetime.now()
        days_diff = (current_dt - latest_dt).days
        
        if days_diff >= 1:  # 超过1天则需要更新
            return True, latest_date, data_file
        else:
            return False, latest_date, data_file
            
    except Exception as e:
        logger.warning(f"⚠️ 解析日期失败: {str(e)}")
        return True, latest_date, data_file

def get_historical_data():
    """获取板块历史数据"""
    logger.info("📊 获取板块历史数据模式")
    
    print("请选择历史数据获取选项:")
    print("1. 获取所有行业板块历史数据（指定天数）")
    print("2. 获取所有概念板块历史数据（指定天数）") 
    print("3. 获取单个板块历史数据（指定天数）")
    print("4. 获取所有行业板块全部历史数据")
    print("5. 获取所有概念板块全部历史数据")
    print("6. 获取单个板块全部历史数据")
    print("7. 🔄 增量更新所有行业板块数据（从最新日期开始）")
    print("8. 🔄 增量更新所有概念板块数据（从最新日期开始）")
    print("9. 🔄 增量更新单个板块数据（从最新日期开始）")
    print("10. 🧠 智能增量更新（自动检测需要更新的板块）")
    print("11. 返回主菜单")
    
    choice = input("请选择 (1-11): ").strip()
    
    if choice == "11":
        return
    
    # 新增智能增量更新功能
    if choice == "10":
        smart_incremental_update()
        return
    
    # 判断是否获取全部历史数据
    get_all_history = choice in ["4", "5", "6"]
    
    # 判断是否为增量更新
    is_incremental = choice in ["7", "8", "9"]
    
    if get_all_history:
        print("\n📅 将获取板块的全部历史数据（可能需要较长时间）")
        trading_days = 2000  # 设置一个大数值获取尽可能多的数据
    elif is_incremental:
        print("\n🔄 增量更新模式：将从最新数据日期开始获取到今天")
        trading_days = None  # 增量模式不需要设置天数
    else:
        # 设置交易日数量
        print("\n设置获取数据量:")
        try:
            trading_days = int(input("请输入要获取最近多少个交易日的数据 (回车默认30个): ").strip() or "30")
            if trading_days <= 0 or trading_days > 1000:
                trading_days = 30
                print("数量无效，使用默认30个交易日")
        except ValueError:
            trading_days = 30
            print("输入无效，使用默认30个交易日")
        
        print(f"将获取最近 {trading_days} 个交易日的日线数据")
    
    s_f_1 = Spider_func()
    
    if choice in ["1", "4", "7"]:
        # 获取所有行业板块历史数据
        if choice == "7":  # 增量更新
            data_type = "增量更新"
            save_dir = "data/datas_sector_historical/行业板块"
            logger.info(f"开始增量更新所有行业板块数据...")
        else:
            data_type = "全部" if get_all_history else f"最近{trading_days}个交易日"
            save_dir = "data/datas_sector_historical/行业板块_全部历史" if get_all_history else "data/datas_sector_historical/行业板块"
            logger.info(f"开始获取所有行业板块{data_type}历史数据...")
        
        all_data = s_f_1.get_all_sectors_historical_data(
            sector_type='industry',
            trading_days=trading_days,
            period='daily',
            save_dir=save_dir,
            is_incremental=(choice == "7")
        )
        logger.info(f"✅ 成功获取 {len(all_data)} 个行业板块的{data_type}历史数据")
        
    elif choice in ["2", "5", "8"]:
        # 获取所有概念板块历史数据
        if choice == "8":  # 增量更新
            data_type = "增量更新"
            save_dir = "data/datas_sector_historical/概念板块"
            logger.info(f"开始增量更新所有概念板块数据...")
        else:
            data_type = "全部" if get_all_history else f"最近{trading_days}个交易日"
            save_dir = "data/datas_sector_historical/概念板块_全部历史" if get_all_history else "data/datas_sector_historical/概念板块"
            logger.info(f"开始获取所有概念板块{data_type}历史数据...")
        
        all_data = s_f_1.get_all_sectors_historical_data(
            sector_type='concept',
            trading_days=trading_days,
            period='daily',
            save_dir=save_dir,
            is_incremental=(choice == "8")
        )
        logger.info(f"✅ 成功获取 {len(all_data)} 个概念板块的{data_type}历史数据")
        
    elif choice in ["3", "6", "9"]:
        # 获取单个板块历史数据
        print("\n输入板块信息:")
        sector_code = input("板块代码: ").strip()
        sector_type = input("板块类型 (industry/concept): ").strip()
        
        if not sector_code or sector_type not in ['industry', 'concept']:
            logger.error("❌ 板块信息输入错误")
            return
        
        if choice == "9":  # 增量更新
            data_type = "增量更新"
            logger.info(f"开始增量更新板块 {sector_code} 数据...")
            
            historical_df = s_f_1.get_historical_sector_data_from_eastmoney(
                sector_code=sector_code,
                sector_type=sector_type,
                trading_days=None,
                period='daily',
                is_incremental=True
            )
        else:
            data_type = "全部" if get_all_history else f"最近{trading_days}个交易日"
            logger.info(f"开始获取板块 {sector_code} 的{data_type}历史数据...")
            
            historical_df = s_f_1.get_historical_sector_data_from_eastmoney(
                sector_code=sector_code,
                sector_type=sector_type,
                trading_days=trading_days,
                period='daily'
            )
        
        if not historical_df.empty:
            # 保存数据
            if choice == "9":
                suffix = "增量更新"
                save_dir = "data/datas_sector_historical"
            else:
                suffix = "全部历史" if get_all_history else f"最近{trading_days}天"
                save_dir = "data/datas_sector"
            
            filename = f"板块{sector_code}_daily_{suffix}_数据.csv"
            filepath = os.path.join(save_dir, filename)
            os.makedirs(save_dir, exist_ok=True)
            historical_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"✅ {data_type}历史数据已保存: {filepath} ({len(historical_df)}条)")
        else:
            logger.warning(f"⚠️ 无法获取板块 {sector_code} 的{data_type}历史数据")

def smart_incremental_update():
    """智能增量更新功能"""
    logger.info("🧠 启动智能增量更新模式")
    
    s_f_1 = Spider_func()
    
    # 1. 检查并更新行业板块数据
    logger.info("🔍 检查行业板块数据...")
    try:
        industry_df = s_f_1.get_industry_data_from_eastmoney()
        if not industry_df.empty:
            update_count = 0
            skip_count = 0
            
            for index, row in industry_df.iterrows():
                sector_code = row['行业代码']
                sector_name = row['行业名称']
                
                needs_update, latest_date, data_file = check_sector_data_update_needed(
                    sector_name, sector_code, 'industry'
                )
                
                if needs_update:
                    logger.info(f"🔄 更新行业板块: {sector_name}({sector_code})")
                    if latest_date:
                        logger.info(f"  📅 从 {latest_date} 开始增量更新")
                    
                    # 执行增量更新
                    historical_df = s_f_1.get_historical_sector_data_from_eastmoney(
                        sector_code=sector_code,
                        sector_type='industry',
                        trading_days=None,
                        period='daily',
                        is_incremental=True
                    )
                    
                    if not historical_df.empty:
                        # 保存或合并数据
                        save_dir = "data/datas_sector_historical/行业板块_全部历史"
                        os.makedirs(save_dir, exist_ok=True)
                        filename = f"{sector_name}({sector_code})_daily_历史数据.csv"
                        filepath = os.path.join(save_dir, filename)
                        
                        if data_file and os.path.exists(data_file):
                            # 合并现有数据
                            existing_df = pd.read_csv(data_file, encoding='utf-8-sig')
                            combined_df = pd.concat([existing_df, historical_df], ignore_index=True)
                            # 去重并按日期排序
                            combined_df = combined_df.drop_duplicates(subset=['日期']).sort_values('日期')
                            combined_df.to_csv(filepath, index=False, encoding='utf-8-sig')
                        else:
                            historical_df.to_csv(filepath, index=False, encoding='utf-8-sig')
                        
                        update_count += 1
                        logger.info(f"  ✅ 已更新: {len(historical_df)} 条新数据")
                    else:
                        logger.warning(f"  ⚠️ 无新数据")
                else:
                    skip_count += 1
                    if index < 5:  # 只显示前几个跳过的
                        logger.info(f"  ⏭️ 跳过 {sector_name}: 数据已是最新")
                
                # 添加延迟避免频繁请求
                time.sleep(0.2)
            
            logger.info(f"📊 行业板块检查完成: 更新{update_count}个, 跳过{skip_count}个")
            
    except Exception as e:
        logger.error(f"❌ 检查行业板块数据失败: {str(e)}")
    
    # 2. 检查并更新概念板块数据
    logger.info("🔍 检查概念板块数据...")
    try:
        concept_df = s_f_1.get_concept_data_from_eastmoney()
        if not concept_df.empty:
            update_count = 0
            skip_count = 0
            
            # 由于概念板块较多，只检查前100个（可配置）
            max_concepts = 100
            logger.info(f"📝 概念板块较多({len(concept_df)}个)，检查前{max_concepts}个")
            
            for index, row in concept_df.head(max_concepts).iterrows():
                sector_code = row['概念代码']
                sector_name = row['概念名称']
                
                needs_update, latest_date, data_file = check_sector_data_update_needed(
                    sector_name, sector_code, 'concept'
                )
                
                if needs_update:
                    logger.info(f"🔄 更新概念板块: {sector_name}({sector_code})")
                    if latest_date:
                        logger.info(f"  📅 从 {latest_date} 开始增量更新")
                    
                    # 执行增量更新
                    historical_df = s_f_1.get_historical_sector_data_from_eastmoney(
                        sector_code=sector_code,
                        sector_type='concept',
                        trading_days=None,
                        period='daily',
                        is_incremental=True
                    )
                    
                    if not historical_df.empty:
                        # 保存或合并数据
                        save_dir = "data/datas_sector_historical/概念板块_全部历史"
                        os.makedirs(save_dir, exist_ok=True)
                        filename = f"{sector_name}({sector_code})_daily_历史数据.csv"
                        filepath = os.path.join(save_dir, filename)
                        
                        if data_file and os.path.exists(data_file):
                            # 合并现有数据
                            existing_df = pd.read_csv(data_file, encoding='utf-8-sig')
                            combined_df = pd.concat([existing_df, historical_df], ignore_index=True)
                            # 去重并按日期排序
                            combined_df = combined_df.drop_duplicates(subset=['日期']).sort_values('日期')
                            combined_df.to_csv(filepath, index=False, encoding='utf-8-sig')
                        else:
                            historical_df.to_csv(filepath, index=False, encoding='utf-8-sig')
                        
                        update_count += 1
                        logger.info(f"  ✅ 已更新: {len(historical_df)} 条新数据")
                    else:
                        logger.warning(f"  ⚠️ 无新数据")
                else:
                    skip_count += 1
                    if index < 3:  # 只显示前几个跳过的
                        logger.info(f"  ⏭️ 跳过 {sector_name}: 数据已是最新")
                
                # 添加延迟避免频繁请求
                time.sleep(0.2)
            
            logger.info(f"📊 概念板块检查完成: 更新{update_count}个, 跳过{skip_count}个")
            
    except Exception as e:
        logger.error(f"❌ 检查概念板块数据失败: {str(e)}")
    
    # 3. 检查并更新股票板块映射
    logger.info("🔍 检查股票板块映射...")
    try:
        needs_update, last_date, message = check_stock_mapping_update_needed()
        if needs_update:
            logger.info(f"🔄 更新股票板块映射: {message}")
            get_stock_sector_mapping(max_stocks=None, incremental=True)
        else:
            logger.info(f"⏭️ 跳过股票映射更新: {message}")
    except Exception as e:
        logger.error(f"❌ 检查股票映射失败: {str(e)}")
    
    logger.info("🎉 智能增量更新完成！")

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("🚀 AI股市预测系统 - 板块数据获取工具")
    logger.info("=" * 60)
    
    # 主菜单
    print("请选择功能:")
    print("1. 获取当前板块数据和股票板块映射")
    print("2. 获取板块历史数据")
    print("3. 🧠 智能增量更新（推荐）")
    print("4. 🔄 股票映射增量更新")
    print("5. 退出")
    
    try:
        main_choice = input("请选择 (1-5): ").strip()
        
        if main_choice == "2":
            get_historical_data()
            return
        elif main_choice == "3":
            smart_incremental_update()
            return
        elif main_choice == "4":
            logger.info("🔄 启动股票映射增量更新...")
            get_stock_sector_mapping(max_stocks=None, incremental=True)
            return
        elif main_choice == "5":
            logger.info("程序退出")
            return
        elif main_choice != "1":
            logger.info("使用默认模式: 获取当前板块数据")
    except KeyboardInterrupt:
        logger.info("用户取消操作")
        return
    
    # 原有的股票板块映射功能
    print("\n请选择股票数据获取模式:")
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
            batch_size=50,
            incremental=False
        )
        
        # 3. 创建数据摘要
        logger.info("第三步: 创建数据摘要...")
        create_sector_summary()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("🎉 所有板块数据获取完成！")
        logger.info(f"⏱️ 总耗时: {duration}")
        logger.info("📁 板块数据保存在 data/datas_sector/ 目录中")
        logger.info("📁 股票映射表保存在 data/datas_sector_historical/ 目录中")
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

def check_for_incomplete_tasks():
    """
    检查未完成的任务
    
    Returns:
        dict: 包含各种未完成任务的状态信息
    """
    status = {
        'stock_mapping_incomplete': False,
        'temp_file': None,
        'processed_count': 0,
        'sector_data_incomplete': [],
        'recommendations': []
    }
    
    # 1. 检查股票映射临时文件
    temp_file = "data/datas_sector_historical/股票板块映射_临时.csv"
    if os.path.exists(temp_file):
        try:
            df_temp = pd.read_csv(temp_file, encoding='utf-8-sig')
            status['stock_mapping_incomplete'] = True
            status['temp_file'] = temp_file
            status['processed_count'] = len(df_temp)
            status['recommendations'].append(f"发现未完成的股票映射任务，已处理{len(df_temp)}只股票")
        except Exception as e:
            logger.warning(f"⚠️ 读取临时文件失败: {str(e)}")
    
    # 2. 检查板块数据完整性
    data_dirs = [
        ("data/datas_sector_historical/行业板块_全部历史", "行业板块"),
        ("data/datas_sector_historical/概念板块_全部历史", "概念板块")
    ]
    
    for data_dir, data_type in data_dirs:
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            outdated_files = []
            
            for file in files[:10]:  # 只检查前10个文件，避免过慢
                file_path = os.path.join(data_dir, file)
                latest_date = get_latest_data_date(file_path)
                if latest_date:
                    try:
                        latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
                        days_diff = (datetime.now() - latest_dt).days
                        if days_diff >= 2:  # 超过2天未更新
                            outdated_files.append((file, latest_date, days_diff))
                    except:
                        pass
            
            if outdated_files:
                status['sector_data_incomplete'].append({
                    'type': data_type,
                    'outdated_count': len(outdated_files),
                    'examples': outdated_files[:3]
                })
                status['recommendations'].append(f"{data_type}有{len(outdated_files)}个文件需要更新")
    
    return status

def smart_resume_dialog():
    """智能恢复对话"""
    logger.info("🔍 检查未完成的任务...")
    
    status = check_for_incomplete_tasks()
    
    if not status['stock_mapping_incomplete'] and not status['sector_data_incomplete']:
        logger.info("✅ 所有任务都是最新的，无需恢复")
        return False
    
    print("\n" + "=" * 50)
    print("🔄 发现未完成的任务")
    print("=" * 50)
    
    if status['stock_mapping_incomplete']:
        print(f"📊 股票板块映射: 已处理{status['processed_count']}只股票，有待继续")
    
    for incomplete in status['sector_data_incomplete']:
        print(f"📈 {incomplete['type']}: {incomplete['outdated_count']}个文件需要更新")
        for file, date, days in incomplete['examples'][:2]:
            print(f"  - {file}: 最新数据{date} ({days}天前)")
    
    print("\n建议操作:")
    for i, recommendation in enumerate(status['recommendations'], 1):
        print(f"{i}. {recommendation}")
    
    print("\n选择操作:")
    print("1. 🧠 智能增量更新（推荐）")
    print("2. 🔄 继续股票映射任务")
    print("3. ⏭️ 跳过，执行新任务")
    print("4. 🚪 退出")
    
    choice = input("请选择 (1-4): ").strip()
    
    if choice == "1":
        smart_incremental_update()
        return True
    elif choice == "2" and status['stock_mapping_incomplete']:
        continue_from_interruption()
        return True
    elif choice == "3":
        return False
    elif choice == "4":
        logger.info("用户选择退出")
        return True
    else:
        logger.info("无效选择，继续执行新任务")
        return False

def continue_from_interruption():
    """从中断处继续获取数据"""
    logger.info("🔄 继续从中断处获取数据...")
    
    # 检查临时文件
    temp_file = "data/datas_sector_historical/股票板块映射_临时.csv"
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
            batch_size=50,
            incremental=False
        )
        
        create_sector_summary()
        logger.info("🎉 继续处理完成！")
        
    except Exception as e:
        logger.error(f"❌ 继续处理失败: {str(e)}")

if __name__ == "__main__":
    # 启动智能任务检查和恢复
    if smart_resume_dialog():
        # 如果智能恢复处理了任务，就不再执行主程序
        pass
    else:
        # 执行正常的主程序
        main()

