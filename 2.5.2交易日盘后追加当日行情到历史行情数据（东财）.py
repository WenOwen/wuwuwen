# -*- coding: utf-8 -*-
# @老师微信:samequant
# @网站:打板哥网 www.dabange.com
# @更多源码下载地址: https://dabange.com/download
# @有偿服务：量化课程、量化数据、策略代写、实盘对接...

from samequant_functions import *

# 方式一：合并东财
import pandas as pd

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 6000)  # 最多显示数据的行数
pd.set_option('display.float_format', lambda x: '%.2f' % x)  # 禁用科学计数法


def add_to_stocks_history_data():
    # 获取全量A股实时行情
    s_f_1 = Spider_func()
    df = s_f_1.get_recent_all_stock_kline_data_from_em()
    # print(df)
    # exit()
    # 盘后循环更新行情到历史数据
    for i in df.index:
        df_i = df[df.index == i]
        df_i = df_i[['交易日期', '股票代码', '股票名称', '开盘价', '收盘价', '最高价', '最低价', '前收盘价', '成交量', '成交额', '振幅', '涨跌额', '涨跌幅', '换手率', '总市值', '流通市值']]
        # print(df_i)
        trade_date = df.loc[i, '交易日期']
        # 读取历史行情数据
        stock_code = df.loc[i, '股票代码']
        # print(stock_code)
        his_path = 'stocks_historydata_em/{}.csv'.format(stock_code)
        if os.path.exists(his_path):
            print(stock_code)
            df_his = pd.read_csv(his_path)
            if not df_his.empty:
                # print(df_his.tail())
                df_his = df_his[['交易日期', '股票代码', '股票名称', '开盘价', '收盘价', '最高价', '最低价', '前收盘价', '成交量', '成交额', '振幅', '涨跌额', '涨跌幅', '换手率', '总市值', '流通市值']]

            tail1_date = df_his['交易日期'].iloc[-1]
            # print(trade_date, tail1_date)
            if trade_date > tail1_date:
                # print(df_i)
                df_all = pd.concat(objs=[df_his, df_i], ignore_index=False)
                # print(df_all)
                df_all.to_csv(his_path, mode='w', index=False)
        else:
            df_i.to_csv(his_path, mode='w', index=False)


if __name__ == '__main__':
    # 交易日盘后追加当时行情到历史行情数据(前提是之前以下载过所有个股完整的历史行情数据)
    add_to_stocks_history_data()

    # 实盘时，每日15:31定时运行
    import schedule
    schedule.every().day.at('15:31').do(add_to_stocks_history_data)