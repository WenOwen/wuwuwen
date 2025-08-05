# -*- coding: utf-8 -*-
# @老师微信:samequant
# @网站:打板哥网 www.dabange.com
# @更多源码下载地址: https://dabange.com/download
# @有偿服务：量化课程、量化数据、策略代写、实盘对接...

import pandas as pd
import os
import time
from samequant_functions import Spider_func

def check_missing_stocks():
    """
    检测哪些股票没有爬取到数据
    
    Returns:
        list: 缺失的股票代码列表
    """
    
    print("=== 开始检测缺失的股票数据 ===")
    
    # 1. 读取股票列表
    stock_list_path = 'stockcode_list/all_stock_list.csv'
    df_all = pd.read_csv(stock_list_path, dtype={'股票代码': str}, encoding='utf-8')
    all_stocks = set(df_all['股票代码'].tolist())
    print(f"📊 股票列表总数: {len(all_stocks)}")
    
    # 2. 获取已爬取的股票文件
    data_dir = 'datas_em'
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return list(all_stocks)
    
    # 获取已存在的股票文件
    existing_files = os.listdir(data_dir)
    existing_stocks = set()
    
    for file in existing_files:
        if file.endswith('.csv'):
            stock_code = file.replace('.csv', '')
            existing_stocks.add(stock_code)
    
    print(f"✅ 已爬取股票数量: {len(existing_stocks)}")
    
    # 3. 找出缺失的股票
    missing_stocks = all_stocks - existing_stocks
    print(f"❌ 缺失股票数量: {len(missing_stocks)}")
    
    # 4. 分析缺失的股票类型
    if missing_stocks:
        missing_list = list(missing_stocks)
        missing_list.sort()
        
        # 按股票类型分类
        sh_stocks = [s for s in missing_list if s.startswith('sh')]
        sz_stocks = [s for s in missing_list if s.startswith('sz')]
        bj_stocks = [s for s in missing_list if s.startswith('bj')]
        
        print(f"   - 上海股票缺失: {len(sh_stocks)}")
        print(f"   - 深圳股票缺失: {len(sz_stocks)}")
        print(f"   - 北京股票缺失: {len(bj_stocks)}")
        
        # 显示前20个缺失的股票
        print("\\n前20个缺失的股票:")
        for i, stock in enumerate(missing_list[:20]):
            stock_name = df_all[df_all['股票代码'] == stock]['名称'].iloc[0] if len(df_all[df_all['股票代码'] == stock]) > 0 else "未知"
            print(f"   {i+1:2d}. {stock} - {stock_name}")
        
        if len(missing_list) > 20:
            print(f"   ... 还有 {len(missing_list) - 20} 只股票")
    
    return list(missing_stocks)

def get_stock_data(s_f_1, stock_code):
    """
    获取股票历史数据
    """
    # 获取股票历史数据
    df = s_f_1.get_stock_history_data_from_eastmoney(stock_code=stock_code)
    return df

def crawl_missing_stocks(missing_stocks, batch_size=50):
    """
    爬取缺失的股票数据
    
    Args:
        missing_stocks: 缺失股票代码列表
        batch_size: 每批处理的股票数量
    """
    
    if not missing_stocks:
        print("✅ 没有缺失的股票，无需补爬！")
        return
    
    print(f"\\n=== 开始补爬 {len(missing_stocks)} 只缺失股票 ===")
    
    # 初始化爬虫
    s_f_1 = Spider_func()
    
    # 创建保存目录
    save_dir = 'datas_em'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    success_count = 0
    failed_stocks = []
    
    for i, stock_code in enumerate(missing_stocks):
        try:
            print(f"\\n📈 [{i+1}/{len(missing_stocks)}] 正在爬取: {stock_code}")
            
            # 获取股票历史数据
            df = get_stock_data(s_f_1, stock_code)
            
            if not df.empty:
                # 保存数据
                save_path = f'{save_dir}/{stock_code}.csv'
                # 确保数据格式正确，使用正确的换行符
                df.to_csv(save_path, index=False, encoding='utf-8', lineterminator='\n')
                
                print(f"   ✅ 成功保存: {save_path} (数据行数: {len(df)})")
                

                
                success_count += 1
                
                # 显示最新数据
                if len(df) > 0:
                    latest = df.iloc[-1]
                    print(f"   📊 最新数据: {latest['交易日期']} 收盘价: {latest['收盘价']}")
                
            else:
                print(f"   ❌ 获取数据失败: {stock_code}")
                failed_stocks.append(stock_code)
            
            # 每批次暂停
            if (i + 1) % batch_size == 0:
                print(f"\\n⏸️  已处理 {i+1} 只股票，暂停2秒...")
                time.sleep(2)
            else:
                time.sleep(0.5)  # 短暂延迟避免请求过快
                
        except Exception as e:
            print(f"   ❌ 爬取出错: {stock_code} - {str(e)}")
            failed_stocks.append(stock_code)
            continue
    
    # 总结报告
    print(f"\\n=== 补爬完成 ===")
    print(f"✅ 成功爬取: {success_count} 只股票")
    print(f"❌ 失败股票: {len(failed_stocks)} 只股票")
    
    if failed_stocks:
        print("\\n失败的股票列表:")
        for stock in failed_stocks[:10]:  # 只显示前10个
            print(f"   - {stock}")
        if len(failed_stocks) > 10:
            print(f"   ... 还有 {len(failed_stocks) - 10} 只")
        
        # 保存失败列表到文件
        failed_df = pd.DataFrame({'股票代码': failed_stocks})
        failed_df.to_csv('failed_stocks.csv', index=False, encoding='utf-8')
        print(f"\\n📝 失败股票列表已保存到: failed_stocks.csv")

def main():
    """
    主函数：检测并补爬缺失股票
    """
    
    # 1. 检测缺失股票
    missing_stocks = check_missing_stocks()
    
    if not missing_stocks:
        print("\\n🎉 恭喜！所有股票数据都已完整！")
        return
    
    # 2. 询问是否开始补爬
    print(f"\\n发现 {len(missing_stocks)} 只股票数据缺失")
    
    # 自动开始补爬（生产环境可以改为手动确认）
    auto_crawl = True
    
    if auto_crawl:
        print("开始自动补爬...")
        crawl_missing_stocks(missing_stocks, batch_size=30)
    else:
        user_input = input("是否开始补爬？(y/n): ")
        if user_input.lower() in ['y', 'yes', '是']:
            crawl_missing_stocks(missing_stocks, batch_size=30)
        else:
            print("取消补爬操作")

if __name__ == "__main__":
    main()