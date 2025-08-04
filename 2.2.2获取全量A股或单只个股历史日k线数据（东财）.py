# -*- coding: utf-8 -*-   
# @老师微信:samequant
# @网站:打板哥网 www.dabange.com
# @更多源码下载地址: https://dabange.com/download
# @有偿服务：量化课程、量化数据、策略代写、实盘对接...
import os
# 先导入本机samequant_functions功能类和函数

from samequant_functions import Spider_func
s_f_1 = Spider_func()

# 下载单只个股
stock_code = 'sh600519'
df = s_f_1.get_stock_history_data_from_eastmoney(stock_code=stock_code)
# 本.py文件所在目录
file_full_dir = os.path.dirname(os.path.abspath(__file__))

# 单支个股历史数据存储目录
path = file_full_dir + '/stocks_historydata_em/{}.csv'.format(stock_code)
# df.to_csv(path_or_buf=path, mode='w', index=False)
# print(df)
# exit()

# 下载所有A股历史行情数据
s_f_1.download_all_stocks_history_kline_from_em()
