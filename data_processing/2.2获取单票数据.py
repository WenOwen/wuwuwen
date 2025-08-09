# -*- coding: utf-8 -*-
"""
获取全部A股历史数据（东方财富数据源）
优化版本：使用统一的API客户端和数据处理方法，带进度条显示
"""

import time
import os
import pandas as pd
from tqdm import tqdm
from samequant_functions_new import get_spider_client

def download_all_stocks_with_progress():
    """带进度条的股票数据下载函数"""
    print("📈 开始下载指定股票历史数据...")
    start_time = time.time()
    
    try:
        # 获取爬虫客户端实例
        spider = get_spider_client()
        
        # 指定要获取的股票代码列表
        stock_codes = ["sh600000",'sh600001', 'sh600002', 'sh600003', 'sh600005']
        
        # 创建股票列表DataFrame
        df_stocks = pd.DataFrame({'股票代码': stock_codes})
        total_stocks = len(df_stocks)
        
        print(f"📊 共需下载 {total_stocks} 只股票的历史数据")
        print("="*60)
        
        success_count = 0
        error_count = 0
        
        # 创建简洁的总进度条
        pbar = tqdm(total=total_stocks, 
                   desc="下载进度", 
                   unit="只",
                   bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] 成功:{postfix}",
                   ncols=80,
                   leave=True)
        
        try:
            for i, row in df_stocks.iterrows():
                code = row['股票代码']
                
                try:
                    # 获取历史数据（包含真实流通市值）
                    df_code = spider.get_stock_history_data_with_real_market_cap(stock_code=code)
                    
                    if not df_code.empty:
                        # 静默保存数据（不打印信息）
                        save_path = os.path.join(spider.stock_hisdata_dir, f'{code}.csv')
                        try:
                            # 创建目录（如果不存在）
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            # 保存文件（静默）
                            df_code.to_csv(save_path, index=False, encoding='utf-8')
                            success_count += 1
                        except Exception:
                            error_count += 1
                        
                        # 添加延迟避免请求过频
                        time.sleep(0.5)
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    # 只在出错时简单记录，不打印详细信息
                    pass
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix_str(f"{success_count}")
        
        finally:
            pbar.close()
        
        # 计算总耗时
        end_time = time.time()
        total_time = end_time - start_time
        
        # 显示最终统计结果
        print("\n" + "="*60)
        print("📊 下载完成统计:")
        print(f"   ✅ 成功下载: {success_count} 只股票")
        print(f"   ❌ 下载失败: {error_count} 只股票")
        print(f"   ⏱️  总耗时: {format_time(total_time)}")
        print(f"   📁 数据保存目录: {spider.stock_hisdata_dir}")
        
        if success_count > 0:
            print(f"   📈 平均每只股票耗时: {total_time/total_stocks:.2f} 秒")
            
        print("="*60)
        
        return success_count > 0
        
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n❌ 数据下载过程中出现严重错误: {e}")
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

def main():
    """主函数：下载指定股票历史数据"""
    success = download_all_stocks_with_progress()
    
    if success:
        print("🎉 指定股票历史数据下载任务完成！")
    else:
        print("⚠️  下载任务未能成功完成，请检查网络连接")

if __name__ == '__main__':
    main()