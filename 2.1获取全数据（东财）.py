import os
import pandas as pd
# 先导入本机samequant_functions功能类和函数

from samequant_functions import Spider_func
s_f_1 = Spider_func()

def get_stock_data(stock_code):
    """
    获取股票历史数据
    """
    # 获取股票历史数据
    df = s_f_1.get_stock_history_data_from_eastmoney(stock_code=stock_code)
    return df

# 下载单只个股
stock_code = 'sh600519'
df = get_stock_data(stock_code)
# 本.py文件所在目录
file_full_dir = os.path.dirname(os.path.abspath(__file__))

# 单支个股历史数据存储目录
path = file_full_dir + '/datas_em/{}.csv'.format(stock_code)
# 保存单只股票数据到datas_em目录
df.to_csv(path_or_buf=path, mode='w', index=False)
print(f"✅ 单只股票数据已保存到: {path}")
print(f"数据行数: {len(df)}")
print("数据预览:")
print(df.head())
# exit()

# 下载所有A股历史行情数据到datas_em目录
print("\n=== 开始下载所有A股历史数据到datas_em目录 ===")
s_f_1.download_all_stocks_history_kline_from_em()
print("✅ 所有A股历史数据下载完成！")

# 获取并保存板块数据
print("\n=== 开始获取板块数据 ===")
# 获取行业板块数据
df_industry = s_f_1.get_industry_data_from_eastmoney(sort_field='f3')
if not df_industry.empty:
    industry_path = file_full_dir + '/datas_em/行业板块数据.csv'
    df_industry.to_csv(industry_path, index=False, encoding='utf-8')
    print(f"✅ 行业板块数据已保存到: {industry_path}")

# 获取概念板块数据
df_concept = s_f_1.get_concept_data_from_eastmoney(sort_field='f3')
if not df_concept.empty:
    concept_path = file_full_dir + '/datas_em/概念板块数据.csv'
    df_concept.to_csv(concept_path, index=False, encoding='utf-8')
    print(f"✅ 概念板块数据已保存到: {concept_path}")

# 获取热门概念数据
df_hot_concepts = s_f_1.get_hot_concepts_from_eastmoney(limit=50)
if not df_hot_concepts.empty:
    hot_concepts_path = file_full_dir + '/datas_em/热门概念数据.csv'
    df_hot_concepts.to_csv(hot_concepts_path, index=False, encoding='utf-8')
    print(f"✅ 热门概念数据已保存到: {hot_concepts_path}")

print("✅ 所有数据下载完成！")
