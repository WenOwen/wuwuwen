# -*- coding: utf-8 -*-
"""
增量追加股票历史数据（东方财富数据源）
优化版本：只获取最新数据后的增量数据，提高效率
"""

import time
import os
import pandas as pd
import datetime
from tqdm import tqdm
from samequant_functions_new import get_spider_client, OptimizedDownloadStocksList

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 6000)  # 最多显示数据的行数
pd.set_option('display.float_format', lambda x: '%.2f' % x)  # 禁用科学计数法


def get_latest_date_from_file(file_path):
    """
    从CSV文件中获取最新的交易日期
    """
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if not df.empty and '交易日期' in df.columns:
                latest_date = df['交易日期'].iloc[-1]
                return pd.to_datetime(latest_date).strftime('%Y-%m-%d')
        return None
    except Exception as e:
        print(f"读取文件最新日期失败 {file_path}: {e}")
        return None


def get_next_trading_date(date_str):
    """
    获取指定日期的下一个交易日
    """
    try:
        date_obj = pd.to_datetime(date_str)
        next_date = date_obj + datetime.timedelta(days=1)
        return next_date.strftime('%Y-%m-%d')
    except:
        return datetime.datetime.now().strftime('%Y-%m-%d')


def get_stock_list():
    """
    获取股票列表
    """
    try:
        downloader = OptimizedDownloadStocksList()
        stock_list_path = downloader.all_stocklist_path
        
        if not os.path.exists(stock_list_path):
            print("❌ 股票列表文件不存在，请先运行股票列表下载")
            return pd.DataFrame()
        
        df_stocks = pd.read_csv(stock_list_path, dtype={'股票代码': str}, encoding='utf-8')
        return df_stocks
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return pd.DataFrame()


def add_to_stocks_history_data():
    """
    增量追加股票历史数据
    """
    print("📈 开始增量追加股票历史数据...")
    start_time = time.time()
    
    try:
        # 获取爬虫客户端实例
        spider = get_spider_client()
        
        # 获取股票列表
        df_stocks = get_stock_list()
        if df_stocks.empty:
            print("❌ 无法获取股票列表")
            return False
        
        total_stocks = len(df_stocks)
        print(f"📊 共需检查 {total_stocks} 只股票的数据")
        print("="*60)
        
        success_count = 0
        error_count = 0
        skip_count = 0
        
        # 创建进度条
        pbar = tqdm(total=total_stocks, 
                   desc="增量更新进度", 
                   unit="只",
                   bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] 成功:{postfix}",
                   ncols=80,
                   leave=True,
                   dynamic_ncols=False,
                   mininterval=1.0,
                   maxinterval=10.0)
        
        try:
            for i, row in df_stocks.iterrows():
                code = row['股票代码']
                
                try:
                    # 构建文件路径
                    his_path = os.path.join(spider.stock_hisdata_dir, f'{code}.csv')
                    
                    if os.path.exists(his_path):
                        # 获取文件中的最新日期
                        latest_date = get_latest_date_from_file(his_path)
                        
                        if latest_date:
                            # 检查是否需要更新（最新日期是否为今天之前）
                            today = datetime.datetime.now().strftime('%Y-%m-%d')
                            if latest_date >= today:
                                skip_count += 1
                                pbar.update(1)
                                if i % 100 == 0:  # 每100个股票更新一次显示
                                    pbar.set_postfix_str(f"成功:{success_count} 跳过:{skip_count}")
                                continue
                            
                            # 获取增量历史数据（从最新日期的下一天开始）
                            df_new = spider.get_stock_history_data_with_real_market_cap(stock_code=code)
                            
                            if not df_new.empty:
                                # 读取现有数据
                                df_existing = pd.read_csv(his_path)
                                
                                # 过滤出新数据（日期大于最新日期的数据）
                                df_new['交易日期'] = pd.to_datetime(df_new['交易日期'])
                                latest_date_obj = pd.to_datetime(latest_date)
                                df_new_filtered = df_new[df_new['交易日期'] > latest_date_obj]
                                
                                if not df_new_filtered.empty:
                                    # 将日期转换回字符串格式
                                    df_new_filtered['交易日期'] = df_new_filtered['交易日期'].dt.strftime('%Y-%m-%d')
                                    
                                    # 合并数据
                                    df_combined = pd.concat([df_existing, df_new_filtered], ignore_index=True)
                                    
                                    # 保存数据
                                    df_combined.to_csv(his_path, index=False, encoding='utf-8')
                                    success_count += 1
                                    print(f"\n✅ {code} 新增 {len(df_new_filtered)} 条数据")
                                else:
                                    skip_count += 1
                            else:
                                skip_count += 1
                        else:
                            # 文件存在但无法读取最新日期，重新获取完整数据
                            df_complete = spider.get_stock_history_data_with_real_market_cap(stock_code=code)
                            if not df_complete.empty:
                                df_complete.to_csv(his_path, index=False, encoding='utf-8')
                                success_count += 1
                                print(f"\n✅ {code} 重新获取完整数据")
                            else:
                                error_count += 1
                    else:
                        # 文件不存在，获取完整历史数据
                        df_complete = spider.get_stock_history_data_with_real_market_cap(stock_code=code)
                        if not df_complete.empty:
                            # 创建目录
                            os.makedirs(os.path.dirname(his_path), exist_ok=True)
                            df_complete.to_csv(his_path, index=False, encoding='utf-8')
                            success_count += 1
                            print(f"\n✅ {code} 创建完整历史数据文件")
                        else:
                            error_count += 1
                    
                    # 添加延迟避免请求过频
                    time.sleep(0.3)
                        
                except Exception as e:
                    error_count += 1
                    print(f"\n❌ {code} 处理出错: {e}")
                
                # 更新进度条
                if i % 100 != 0:  # 避免重复更新
                    pbar.update(1)
                if i % 100 == 0 or success_count > 0:  # 每100个或有成功更新时才更新显示
                    pbar.set_postfix_str(f"成功:{success_count} 跳过:{skip_count}")
        
        finally:
            pbar.close()
        
        # 计算总耗时
        end_time = time.time()
        total_time = end_time - start_time
        
        # 显示最终统计结果
        print("\n" + "="*60)
        print("📊 增量更新完成统计:")
        print(f"   ✅ 成功更新: {success_count} 只股票")
        print(f"   ⏭️  跳过更新: {skip_count} 只股票")
        print(f"   ❌ 更新失败: {error_count} 只股票")
        print(f"   ⏱️  总耗时: {format_time(total_time)}")
        print(f"   📁 数据保存目录: {spider.stock_hisdata_dir}")
        
        if success_count > 0:
            print(f"   📈 平均每只股票耗时: {total_time/total_stocks:.2f} 秒")
            
        print("="*60)
        
        # 更新板块数据
        print("\n=== 更新板块数据 ===")
        update_sector_data(spider)
        
        return success_count > 0
        
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n❌ 增量更新过程中出现严重错误: {e}")
        print(f"⏱️  运行时间: {format_time(total_time)}")
        print("\n📋 详细错误信息:")
        import traceback
        traceback.print_exc()
        return False


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes} 分 {remaining_seconds:.1f} 秒"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours} 小时 {remaining_minutes} 分 {remaining_seconds:.1f} 秒"


def update_sector_data(spider):
    """
    更新板块数据
    """
    file_full_dir = os.path.dirname(os.path.abspath(__file__))
    # 修改为data/datas_em目录
    project_root = os.path.dirname(file_full_dir)
    datas_em_dir = os.path.join(project_root, 'data', 'datas_em')
    
    # 获取行业板块数据
    df_industry = spider.get_industry_data(sort_field='f3')
    if not df_industry.empty:
        industry_path = os.path.join(datas_em_dir, '行业板块数据.csv')
        df_industry.to_csv(industry_path, index=False, encoding='utf-8')
        print(f"✅ 行业板块数据已更新: {industry_path}")

    # 获取概念板块数据
    df_concept = spider.get_concept_data(sort_field='f3')
    if not df_concept.empty:
        concept_path = os.path.join(datas_em_dir, '概念板块数据.csv')
        df_concept.to_csv(concept_path, index=False, encoding='utf-8')
        print(f"✅ 概念板块数据已更新: {concept_path}")


def main():
    """主函数：增量追加股票历史数据"""
    success = add_to_stocks_history_data()
    
    if success:
        print("🎉 增量追加股票历史数据任务完成！")
    else:
        print("⚠️  增量追加任务未能成功完成，请检查网络连接或股票列表文件")


if __name__ == '__main__':
    # 交易日盘后追加历史行情数据(增量更新模式)
    main()

    # 实盘时，每日15:31定时运行
    import schedule
    schedule.every().day.at('15:31').do(add_to_stocks_history_data)