# 时间: 2022/3/11 22:14
# -*-coding:utf-8-*-
"""
沪深重要指数 http://quote.eastmoney.com/center/hszs.html
http://65.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=50&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=&fs=b:MK0010&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f11,f62,f128,f136,f115,f152
只取有用字段 http://65.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=50&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=&fs=b:MK0010&fields=f12,f13,f14
"""
import os
import time

import pandas as pd
import requests

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 10000)  # 最多显示数据的行数


class Download_index_data(object):
    def __init__(self):
        # .py文件所在目录
        self.file_full_dir = os.path.dirname(os.path.abspath(__file__))
        self.path_dir = self.file_full_dir + '/datas_index'

    def down_important_indexs_list(self):
        url = 'http://65.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=50&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=&fs=b:MK0010&fields=f12,f13,f14'

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }
        resp = requests.get(url=url, headers=headers).json()
        data = resp['data']['diff']
        df = pd.DataFrame(data)
        df.columns = ['指数代码', '市场', '指数名称']
        return df

    def down_index_history_data_from_eastmoney(self, index_code: str, period: str = '日', limit=10000):
        if period == '日':
            index_path = self.path_dir + '/zs{}.csv'.format(index_code)
        else:
            index_path = self.path_dir + '/{}_zs{}.csv'.format(period, index_code)

        if index_code[0] == '0':
            market = '1'
        elif index_code[0] == '3':
            market = '0'
        else:
            market = '0'
        secid = market + '.' + index_code

        if period == '周':
            klt = '102'
        elif period == '日':
            klt = '101'
        elif period == '5分钟':
            klt = '5'
        elif period == '30分钟':
            klt = '30'

        url = 'http://34.push2his.eastmoney.com/api/qt/stock/kline/get?secid={}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt={}&fqt=1&end=20500101&lmt={}'.format(
            secid, klt, limit)

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }
        resp = requests.get(url=url, headers=headers).json()
        # dktotal = resp['data']['dktotal']
        # print(resp['data']['klines'])
        data = resp['data']['klines']
        lst_data = []
        for i in data:
            # 字符串分割为列表
            lst_i = i.split(',')
            lst_data.append(lst_i)
        # print(data)
        df = pd.DataFrame(lst_data)
        df.columns = ['交易日期', '开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        df['代码'] = 'zs' + index_code
        df = df[['交易日期', '代码', '开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']]
        # 将数据列转为float格式
        df[['开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']] = df[
            ['开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']].astype(float)
        df['前收盘价'] = df['收盘价'].shift(1)
        df['开盘涨跌幅'] = df['开盘价'] / df['前收盘价'] - 1
        df['开盘涨跌幅'] = (df['开盘涨跌幅'] * 100).round(decimals=2)

        if not os.path.exists(self.path_dir):
            os.mkdir(self.path_dir)
        # print(df.tail())
        df.to_csv(path_or_buf=index_path, mode='w', encoding='utf-8', index=False)
        # print(f'指数{index_code},{period}数据更新完毕！')
        return df

    def download_important_index(self):
        # 根据重要指数列表，遍历下载重要指数历史数据
        important_indexs_lst = self.down_important_indexs_list()
        for index_code in important_indexs_lst['指数代码']:
            self.down_index_history_data_from_eastmoney(index_code, period='日')
            print('{}下载完毕！'.format(index_code))
            time.sleep(0.5)
        # print(index_lst)

    def cal_bias_10_mean_high(self, update=None) -> pd.DataFrame:
        """
        计算最高价与10日均线的乖离率，超过3.2%通常会有短线调整，连续超过3%要么是见底要么是见顶！
        :return:
        """
        if update == None:

            path = self.file_full_dir + '/datas_index/zs000001.csv'
            df = pd.read_csv(path, parse_dates=['交易日期'])
        else:
            df = self.down_index_history_data_from_eastmoney(index_code='000001', period='日')

        df['10日均线'] = df['收盘价'].rolling(10).mean()
        df['bias_10_mean_high'] = (df['最高价'] / df['10日均线'] - 1) * 100
        df['bias_10_mean_high'] = df['bias_10_mean_high'].round(2)

        df['当日收盘偏离最高价幅度'] = (df['收盘价'] / df['最高价'] - 1) * 100
        df['当日收盘偏离最高价幅度'] = df['当日收盘偏离最高价幅度'].round(2)
        df = df[['交易日期', '收盘价', '最高价', 'bias_10_mean_high', '涨跌幅', '当日收盘偏离最高价幅度']]

        return df


if __name__ == '__main__':
    # 下载单个指数数据
    d_i_1 = Download_index_data()

    # 上证指数 深圳成指 上证50 沪深300 中证500 中证1000 创业板指
    index_lst = ['000001', '399001', '000016', '000300', '000905', '000852', '399006']
    for index_code in index_lst:
        d_i_1.down_index_history_data_from_eastmoney(index_code=index_code, period='日')
        time.sleep(1)

    # 下载上证指数
    # df = d_i_1.down_index_history_data_from_eastmoney(index_code='000001', period='日')
    # print(df.tail(100))

    # 下载中证1000指数
    # df = d_i_1.down_index_history_data_from_eastmoney(index_code='000852', period='日')
    # print(df.tail(100))


    # 计算并查看最高价与10日均线的乖离率
    # df = d_i_1.cal_bias_10_mean_high()
    # print(df)
    pass
