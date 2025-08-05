# -*- coding: utf-8 -*-
# @老师微信:samequant
# @网站:打板哥网 www.dabange.com
# @更多源码下载地址: https://dabange.com/download
# @有偿服务：量化课程、量化数据、策略代写、实盘对接...

import pandas as pd
import os
import time
from samequant_functions import Spider_func

def check_corrupted_files():
    """
    检查哪些文件数据格式有问题（所有数据在一行中）
    
    Returns:
        list: 有问题的文件列表
    """
    
    print("=== 开始检测数据格式有问题的文件 ===")
    
    data_dir = 'datas_em'
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return []
    
    corrupted_files = []
    
    # 获取所有csv文件
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        try:
            # 读取文件的前几行来检查格式
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
            
            # 如果第一行很长且包含很多逗号，说明数据格式有问题
            if len(first_line) > 1000 and first_line.count(',') > 50:
                print(f"❌ 发现格式问题文件: {file}")
                corrupted_files.append(file)
            elif not second_line:  # 如果只有一行数据
                print(f"❌ 发现单行数据文件: {file}")
                corrupted_files.append(file)
                
        except Exception as e:
            print(f"❌ 读取文件出错: {file} - {str(e)}")
            corrupted_files.append(file)
    
    print(f"📊 总共检查了 {len(files)} 个文件")
    print(f"❌ 发现 {len(corrupted_files)} 个有问题的文件")
    
    return corrupted_files

def fix_corrupted_file(stock_code, s_f_1):
    """
    修复单个有问题的文件
    
    Args:
        stock_code: 股票代码（不包含.csv后缀）
        s_f_1: Spider_func实例
    """
    
    print(f"🔧 正在修复: {stock_code}")
    
    try:
        # 重新获取股票数据
        df = s_f_1.get_stock_history_data_from_eastmoney(stock_code=stock_code)
        
        if not df.empty:
            # 保存修复后的数据
            save_path = f'datas_em/{stock_code}.csv'
            df.to_csv(save_path, index=False, encoding='utf-8', lineterminator='\n')
            
            print(f"   ✅ 修复成功: {save_path} (数据行数: {len(df)})")
            return True
        else:
            print(f"   ❌ 获取数据失败: {stock_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ 修复出错: {stock_code} - {str(e)}")
        return False

def main():
    """
    主函数：检测并修复有问题的文件
    """
    
    # 1. 检测有问题的文件
    corrupted_files = check_corrupted_files()
    
    if not corrupted_files:
        print("\n🎉 恭喜！所有文件数据格式都正常！")
        return
    
    # 2. 询问是否开始修复
    print(f"\n发现 {len(corrupted_files)} 个文件数据格式有问题")
    
    # 自动开始修复
    auto_fix = True
    
    if auto_fix:
        print("开始自动修复...")
        
        # 初始化爬虫
        s_f_1 = Spider_func()
        
        success_count = 0
        failed_files = []
        
        for i, file in enumerate(corrupted_files):
            try:
                # 提取股票代码（去掉.csv后缀）
                stock_code = file.replace('.csv', '')
                
                print(f"\n📈 [{i+1}/{len(corrupted_files)}] 正在修复: {stock_code}")
                
                # 修复文件
                if fix_corrupted_file(stock_code, s_f_1):
                    success_count += 1
                else:
                    failed_files.append(file)
                
                # 每批次暂停
                if (i + 1) % 10 == 0:
                    print(f"\n⏸️  已处理 {i+1} 个文件，暂停2秒...")
                    time.sleep(2)
                else:
                    time.sleep(0.5)  # 短暂延迟避免请求过快
                    
            except Exception as e:
                print(f"   ❌ 处理出错: {file} - {str(e)}")
                failed_files.append(file)
                continue
        
        # 总结报告
        print(f"\n=== 修复完成 ===")
        print(f"✅ 成功修复: {success_count} 个文件")
        print(f"❌ 失败文件: {len(failed_files)} 个")
        
        if failed_files:
            print("\n失败的文件列表:")
            for file in failed_files[:10]:  # 只显示前10个
                print(f"   - {file}")
            if len(failed_files) > 10:
                print(f"   ... 还有 {len(failed_files) - 10} 个")
            
            # 保存失败列表到文件
            failed_df = pd.DataFrame({'文件名': failed_files})
            failed_df.to_csv('failed_fix_files.csv', index=False, encoding='utf-8')
            print(f"\n📝 失败文件列表已保存到: failed_fix_files.csv")
    
    else:
        user_input = input("是否开始修复？(y/n): ")
        if user_input.lower() in ['y', 'yes', '是']:
            # 这里可以调用修复逻辑
            print("修复功能待实现...")
        else:
            print("取消修复操作")

if __name__ == "__main__":
    main() 