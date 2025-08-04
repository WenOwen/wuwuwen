# -*- coding: utf-8 -*-
# @微信:samequant
# @网站:涨停客量化zhangtingke.com
# @更多源码下载地址: https://zhangtingke.com/download
# @有偿服务：量化课程、量化数据、策略代写、实盘对接...

import os
import pandas as pd
import requests  # pip install requests
import time
import random
import datetime
import json
from get_opening_calendar_from_szse.get_opening_calendar_from_szse import Get_stock_opening_calendar

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 6000)  # 最多显示数据的行数


class Download_stocks_list(object):
    def __init__(self):
        # 获取该class类所在.py文件的绝对目录
        self.file_full_dir = os.path.dirname(os.path.abspath(__file__))
        # 设置从沪深交易所下载的源文件和拼接后的所有A股列表csv存储目录
        self.path_dir = self.file_full_dir + '/stockcode_list'
        if not os.path.exists(self.path_dir):
            os.mkdir(self.path_dir)
        # 设置错误日志的存储目录
        self.error_dir = self.file_full_dir + '/error_txt'
        if not os.path.exists(self.error_dir):
            os.mkdir(self.error_dir)

        self.all_stocklist_path = self.path_dir + '/all_stock_list.csv'

    # 循环尝试从网页上抓取数据
    def get_response_from_SSE(self, url, ex_params, max_try_num=10, sleep_time=10):
        """
        :param url: 要抓取数据的网址
        :param max_try_num: 最多尝试抓取次数
        :param sleep_time: 抓取失败后停顿的时间
        :return: 返回抓取到的网页内容
        """
        get_success = False  # 是否成功抓取到内容
        response = None
        # 抓取内容 , 注意上交所有反爬机制，需要在headers里添加'X-Requested-With': 'XMLHttpRequest','Referer': 'http://www.sse.com.cn/assortment/stock/list/share/'
        for i in range(max_try_num):
            try:
                headers = {'X-Requested-With': 'XMLHttpRequest',
                           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) ' 'Chrome/56.0.2924.87 Safari/537.36',
                           'Referer': 'http://www.sse.com.cn/assortment/stock/list/share/'
                           }
                # 获取网页数据
                response = requests.get(url=url, headers=headers, timeout=10, params=ex_params)
                get_success = True  # 成功抓取到内容
                # time.sleep(1) # 休息1秒
                break
            except Exception as e:
                print('抓取数据报错，次数：', i + 1, '报错内容：', e)
                time.sleep(sleep_time)

        # 判断是否成功抓取内容
        if get_success:
            return response
        else:
            raise ValueError('get_response_from_internet抓取网页数据报错达到尝试上限，停止程序，请尽快检查问题所在')

    def get_sh_stocklist_normal_from_SSE(self, path_dir=None):
        """
        从上交所官网下载正常交易股票列表.xls存为csv
        源地址 http://www.sse.com.cn/assortment/stock/list/share/
        http://query.sse.com.cn/security/stock/downloadStockListFile.do?csrcCode=&stockCode=&areaName=&stockType=1
        :param path_dir:
        :return:
        """
        if path_dir == None:
            path_dir = self.path_dir
        url = 'http://query.sse.com.cn//sseQuery/commonExcelDd.do?sqlId=COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L&type=inParams&CSRC_CODE=&STOCK_CODE=&REG_PROVINCE=&STOCK_TYPE=1&COMPANY_STATUS=2,4,5,7,8'
        ex_params = {}
        path = path_dir + '/' + '上交所主板A股.xls'
        try:
            # 调用get_response_from_internet()函数，循环爬取目标网址
            response = self.get_response_from_SSE(url, ex_params, max_try_num=10, sleep_time=10)

            # 存储csv文件到指定目录
            with open(path, 'wb') as f:
                for chunk in response.iter_content(
                        chunk_size=10000):  # iter_content()边下载边存硬盘, chunk_size 可以自由调整为可以更好地适合您的用例的数字
                    if chunk:
                        f.write(chunk)

            df = pd.read_excel(path, dtype={'A股代码': str, '上市日期': str})
            # 重命名列名
            rename_dict = {'A股代码': '股票代码', '证券简称': '名称'}
            df.rename(columns=rename_dict, inplace=True)
            # 重命名股票代码，加上sz，sh前缀
            df['股票代码'] = 'sh' + df['股票代码']
            df['上市日期'] = df['上市日期'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
            df['上市状态'] = '正常交易'
            df['终止/暂停上市日期'] = None
            df = df[['股票代码', '名称', '上市日期', '上市状态', '终止/暂停上市日期']]
            path_csv = path_dir + '/' + '上交所主板A股.csv'
            df.to_csv(path_csv, index=False, mode='w', encoding='utf-8')
            print('存储 %s 成功！' % path_csv)
        except Exception as e:
            print('错误位置:循环爬取存储数据失败', e)  # 把exception输出出来

    def get_sh_stocklist_kechuang_from_SSE(self, path_dir=None):
        """
        从上交所官网下载科创板正常交易股票列表.xls存为csv
        源地址 http://www.sse.com.cn/assortment/stock/list/share/
        http://query.sse.com.cn/security/stock/downloadStockListFile.do?csrcCode=&stockCode=&areaName=&stockType=8
        :param path_dir:
        :return:
        """
        if path_dir == None:
            path_dir = self.path_dir
        url_new = 'http://query.sse.com.cn//sseQuery/commonExcelDd.do?sqlId=COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L&type=inParams&CSRC_CODE=&STOCK_CODE=&REG_PROVINCE=&STOCK_TYPE=8&COMPANY_STATUS=2,4,5,7,8'
        kcb_params = {}
        path = path_dir + '/' + '上交所科创板.xls'
        try:
            # 调用get_response_from_internet()函数，循环爬取目标网址
            response = self.get_response_from_SSE(url_new, kcb_params, max_try_num=10, sleep_time=10)

            # 存储xls文件到指定目录
            with open(path, 'wb') as f:
                for chunk in response.iter_content(
                        chunk_size=10000):  # iter_content()边下载边存硬盘, chunk_size 可以自由调整为可以更好地适合您的用例的数字
                    if chunk:
                        f.write(chunk)
            df6 = pd.read_excel(path, dtype={'A股代码': str, '上市日期': str})
            # 重命名列名
            rename_dict6 = {'A股代码': '股票代码', '证券简称': '名称'}
            df6.rename(columns=rename_dict6, inplace=True)
            # 重命名股票代码，加上sz，sh前缀
            df6['股票代码'] = 'sh' + df6['股票代码']
            df6['上市日期'] = df6['上市日期'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
            df6['上市状态'] = '正常交易'
            df6['终止/暂停上市日期'] = None
            df6 = df6[['股票代码', '名称', '上市日期', '上市状态', '终止/暂停上市日期']]

            path_csv = path_dir + '/' + '上交所科创板.csv'
            df6.to_csv(path_csv, index=False, mode='w', encoding='utf-8')
            print('存储 %s 成功！' % path_csv)
        except Exception as e:
            print('错误位置:循环爬取存储数据失败', e)  # 把exception输出出来

    def get_sh_stocklist_zanting_from_SSE(self, path_dir=None):
        """
        从上交所官网下载暂停上市列表.xls存为csv
        http://query.sse.com.cn/security/stock/downloadStockListFile.do?csrcCode=&stockCode=&areaName=&stockType=4
        :param path_dir:
        :return:
        """
        if path_dir == None:
            path_dir = self.path_dir
        url = 'http://query.sse.com.cn//sseQuery/commonExcelDd.do?sqlId=COMMON_SSE_CP_GPJCTPZ_GPLB_ZTGP_L&type=inParams&CSRC_CODE=&STOCK_CODE=&REG_PROVINCE=&STOCK_TYPE=1,2&COMPANY_STATUS=5'
        ex_params = {}
        path = path_dir + '/' + '上交所暂停上市公司.xls'

        try:
            # 调用get_response_from_internet()函数，循环爬取目标网址
            response = self.get_response_from_SSE(url, ex_params, max_try_num=10, sleep_time=10)

            # 存储csv文件到指定目录
            with open(path, 'wb') as f:
                for chunk in response.iter_content(
                        chunk_size=10000):  # iter_content()边下载边存硬盘, chunk_size 可以自由调整为可以更好地适合您的用例的数字
                    if chunk:
                        f.write(chunk)
            path_csv = path_dir + '/' + '上交所暂停上市公司.csv'
            df7 = pd.read_excel(path, dtype={'公司代码': str, '暂停上市日期': str})
            if not df7.empty:
                # 重命名列名
                rename_dict7 = {'公司代码': '股票代码', '公司简称': '名称', '暂停上市日期': '终止/暂停上市日期'}
                df7.rename(columns=rename_dict7, inplace=True)
                # 重命名股票代码，加上sz，sh前缀
                df7['股票代码'] = 'sh' + df7['股票代码']
                df7['上市状态'] = '暂停上市'
                df7['终止/暂停上市日期'] = df7['终止/暂停上市日期'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
                df7 = df7[['股票代码', '名称', '上市日期', '上市状态', '终止/暂停上市日期']]
                df7.to_csv(path_csv, index=False, mode='w', encoding='utf-8')
                print('存储 %s 成功！' % path_csv)
            else:
                df7 = pd.DataFrame(columns=['股票代码', '名称', '上市日期', '上市状态', '终止/暂停上市日期'])
                df7.to_csv(path_csv, index=False, mode='w', encoding='utf-8')
                print('上交所暂停上市列表为空！')

        except Exception as e:
            print('错误位置:循环爬取存储数据失败', e)  # 把exception输出出来

    def get_sh_stocklist_zhongzhi_from_SSE(self, path_dir=None):
        """
        从上交所官网下载终止上市列表.xls存为csv
        源地址 http://query.sse.com.cn/security/stock/downloadStockListFile.do?csrcCode=&stockCode=&areaName=&stockType=5
        :param path_dir:
        :return:
        """
        if path_dir == None:
            path_dir = self.path_dir
        url = 'http://query.sse.com.cn//sseQuery/commonExcelDd.do?sqlId=COMMON_SSE_CP_GPJCTPZ_GPLB_ZZGP_L&type=inParams&CSRC_CODE=&STOCK_CODE=&REG_PROVINCE=&STOCK_TYPE=1,2&COMPANY_STATUS=3'
        ex_params = {}
        path = path_dir + '/' + '上交所终止上市公司.xls'

        try:
            # 调用get_response_from_internet()函数，循环爬取目标网址
            response = self.get_response_from_SSE(url, ex_params, max_try_num=10, sleep_time=10)

            # 存储csv文件到指定目录
            with open(path, 'wb') as f:
                for chunk in response.iter_content(
                        chunk_size=10000):  # iter_content()边下载边存硬盘, chunk_size 可以自由调整为可以更好地适合您的用例的数字
                    if chunk:
                        f.write(chunk)

            df5 = pd.read_excel(path, dtype={'原公司代码': str, '上市日期': str, '终止上市日期': str})
            # print(df5)
            # 重命名列名
            rename_dict5 = {'原公司代码': '股票代码', '原公司简称': '名称', '终止上市日期': '终止/暂停上市日期'}
            df5.rename(columns=rename_dict5, inplace=True)
            # 重命名股票代码，加上sz，sh前缀
            df5['股票代码'] = 'sh' + df5['股票代码']
            df5['上市日期'] = df5['上市日期'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
            df5['终止/暂停上市日期'] = df5['终止/暂停上市日期'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
            df5['上市状态'] = '终止上市'
            df5 = df5[['股票代码', '名称', '上市日期', '上市状态', '终止/暂停上市日期']]
            # 过滤掉9开头的代码，只保留6开头的代码
            df5 = df5[df5['股票代码'].str[0:3] == 'sh6']

            path_csv = path_dir + '/' + '上交所终止上市公司.csv'
            df5.to_csv(path_csv, index=False, mode='w', encoding='utf-8')
            print('存储 %s 成功！' % path_csv)
        except Exception as e:
            print('错误位置:循环爬取存储数据失败', e)  # 把exception输出出来

    # 深交所
    # 循环尝试从网页上抓取数据
    def get_response_from_SZSE(self, url, ex_params, max_try_num=10, sleep_time=10):
        """
        :param url: 要抓取数据的网址
        :param max_try_num: 最多尝试抓取次数
        :param sleep_time: 抓取失败后停顿的时间
        :return: 返回抓取到的网页内容
        """
        get_success = False  # 是否成功抓取到内容
        response = None
        # 抓取内容
        for i in range(max_try_num):
            try:
                headers = {
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'}
                # 获取网页数据
                response = requests.get(url=url, headers=headers, timeout=10, params=ex_params)
                get_success = True  # 成功抓取到内容
                # time.sleep(1) # 休息1秒
                break
            except Exception as e:
                print('抓取数据报错，次数：', i + 1, '报错内容：', e)
                time.sleep(sleep_time)

        # 判断是否成功抓取内容
        if get_success:
            return response
        else:
            raise ValueError('get_response_from_internet抓取网页数据报错达到尝试上限，停止程序，请尽快检查问题所在')

    def get_sz_stocklist_normal_from_SZSE(self, path_dir=None):
        """
        从深交所官网下载正常交易股票列表.xlsx
        源地址 http://www.szse.cn/market/product/stock/list/index.html
        :param path_dir:
        :return:
        """
        if path_dir == None:
            path_dir = self.path_dir

        url = 'http://www.szse.cn/api/report/ShowReport'
        ex_params = {'SHOWTYPE': 'xlsx', 'CATALOGID': '1110', 'TABKEY': 'tab1', 'random': random.random()}
        path = path_dir + '/' + '深交所A股列表.xlsx'
        try:
            # 调用get_response_from_internet()函数，循环爬取目标网址
            response = self.get_response_from_SZSE(url, ex_params, max_try_num=10, sleep_time=10)

            # 存储csv文件到指定目录
            with open(path, 'wb') as f:
                for chunk in response.iter_content(
                        chunk_size=10000):  # iter_content()边下载边存硬盘, chunk_size 可以自由调整为可以更好地适合您的用例的数字
                    if chunk:
                        f.write(chunk)
            # 深交所正常上市A股列表
            df = pd.read_excel(path, sheet_name='A股列表', dtype={'A股代码': str}, parse_dates=['A股上市日期'])
            # 重命名列名
            rename_dict1 = {'A股简称': '名称', 'A股代码': '股票代码', 'A股上市日期': '上市日期'}
            df.rename(columns=rename_dict1, inplace=True)
            df = df[['股票代码', '名称', '上市日期']]
            # 重命名股票代码，加上sz，sh前缀
            df['股票代码'] = 'sz' + df['股票代码']
            df['上市状态'] = '正常交易'  # 1代表正常交易
            df['终止/暂停上市日期'] = None
            df = df[['股票代码', '名称', '上市日期', '上市状态', '终止/暂停上市日期']]
            df['上市日期'] = pd.to_datetime(df['上市日期'], format='%Y%m%d')
            path_csv = path_dir + '/' + '深交所A股列表.csv'
            df.to_csv(path_csv, mode='w', index=False)
            # print(df)
            print('存储 %s 成功！' % path_csv)
        except Exception as e:
            print('错误位置:循环爬取存储数据失败', e)  # 把exception输出出来

    def get_sz_stocklist_zanting_from_SZSE(self, path_dir=None):
        """
        从深交所官网下载暂停上市列表.xlsx
        源地址 http://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=1793_ssgs&TABKEY=tab1&random=0.8850545947086021
        :param path_dir:
        :return:
        """
        if path_dir == None:
            path_dir = self.path_dir
        url = 'http://www.szse.cn/api/report/ShowReport'
        ex_params = {'SHOWTYPE': 'xlsx', 'CATALOGID': '1793_ssgs', 'TABKEY': 'tab1', 'random': random.random()}
        path = path_dir + '/' + '深交所暂停上市.xlsx'
        try:
            # 调用get_response_from_internet()函数，循环爬取目标网址
            response = self.get_response_from_SZSE(url, ex_params, max_try_num=10, sleep_time=10)

            # 存储csv文件到指定目录
            with open(path, 'wb') as f:
                for chunk in response.iter_content(
                        chunk_size=10000):  # iter_content()边下载边存硬盘, chunk_size 可以自由调整为可以更好地适合您的用例的数字
                    if chunk:
                        f.write(chunk)
            df = pd.read_excel(path, dtype={'证券代码': str}, parse_dates=['暂停上市日期'])
            # 重命名列名
            rename_dict2 = {'证券简称': '名称', '证券代码': '股票代码', '暂停上市日期': '终止/暂停上市日期'}
            df.rename(columns=rename_dict2, inplace=True)
            # 重命名股票代码，加上sz，sh前缀
            df['股票代码'] = 'sz' + df['股票代码']
            df['上市状态'] = '暂停上市'  # 1代表正常交易
            df = df[['股票代码', '名称', '上市日期', '上市状态', '终止/暂停上市日期']]
            path_csv = path_dir + '/' + '深交所暂停上市.csv'
            df.to_csv(path_csv, mode='w', index=False)
            print('存储 %s 成功！' % path_csv)
        except Exception as e:
            print('错误位置:循环爬取存储数据失败', e)  # 把exception输出出来

    def get_sz_stocklist_zhongzhi_from_SZSE(self, path_dir=None):
        """
        从深交所官网下载终止上市列表.xlsx
        源地址 'http://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=1793_ssgs&TABKEY=tab2&random=0.10206064982308538'
        :param path_dir:
        :return:
        """
        if path_dir == None:
            path_dir = self.path_dir
        url = 'http://www.szse.cn/api/report/ShowReport'

        ex_params = {'SHOWTYPE': 'xlsx', 'CATALOGID': '1793_ssgs', 'TABKEY': 'tab2', 'random': random.random()}
        path = path_dir + '/' + '深交所终止上市.xlsx'
        try:
            # 调用get_response_from_internet()函数，循环爬取目标网址
            response = self.get_response_from_SZSE(url, ex_params, max_try_num=10, sleep_time=10)

            # 存储csv文件到指定目录
            with open(path, 'wb') as f:
                for chunk in response.iter_content(
                        chunk_size=10000):  # iter_content()边下载边存硬盘, chunk_size 可以自由调整为可以更好地适合您的用例的数字
                    if chunk:
                        f.write(chunk)
            df = pd.read_excel(path, dtype={'证券代码': str, '终止上市日期': str}, parse_dates=['上市日期'])
            # 重命名列名
            rename_dict3 = {'证券简称': '名称', '证券代码': '股票代码', '终止上市日期': '终止/暂停上市日期'}
            df.rename(columns=rename_dict3, inplace=True)
            # 剔除掉退市的B股
            df = df[df['股票代码'].str[0:2] != '20']
            # 重命名股票代码，加上sz，sh前缀
            df['股票代码'] = 'sz' + df['股票代码']
            df['终止/暂停上市日期'] = df['终止/暂停上市日期'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
            df['上市状态'] = '终止上市'
            df = df[['股票代码', '名称', '上市日期', '上市状态', '终止/暂停上市日期']]
            path_csv = path_dir + '/' + '深交所终止上市.csv'
            df.to_csv(path_csv, mode='w', index=False)
            print('存储 %s 成功！' % path_csv)
        except Exception as e:
            print('错误位置:循环爬取存储数据失败', e)  # 把exception输出出来

    # 拼接沪深股票列表含退市all_stock_list.csv
    def get_stock_list(self, path_dir=None):
        """
        :param path_dir:本机xlsx源文件存储目录,上交所需手工转为xlsx格式
        :param return:返回所有股票列表
        深交所市场数据/股票数据/暂停/终止上市 http://www.szse.cn/market/stock/suspend/index.html
        深交所A股列表 http://www.szse.cn/market/product/stock/list/index.html
        上交所A股列表 http://www.sse.com.cn/assortment/stock/list/share/
        上交所终止上市  http://www.sse.com.cn/assortment/stock/list/delisting/
        """
        if path_dir == None:
            path_dir = self.path_dir

        all_df = pd.DataFrame()
        # 深交所正常上市A股列表
        df1 = pd.read_csv(path_dir + '/深交所A股列表.csv', parse_dates=['上市日期'])
        all_df = pd.concat([all_df, df1], ignore_index=False)

        # 深交所暂停上市列表
        df2 = pd.read_csv(path_dir + '/深交所暂停上市.csv', parse_dates=['上市日期'])
        all_df = pd.concat([all_df, df2], ignore_index=False)

        # 深交所终止上市列表
        df3 = pd.read_csv(path_dir + '/深交所终止上市.csv', parse_dates=['上市日期'])
        all_df = pd.concat([all_df, df3], ignore_index=False)

        # 上交所主板A股列表
        df4 = pd.read_csv(path_dir + '/上交所主板A股.csv', dtype={'A股代码': str}, parse_dates=['上市日期'], encoding='utf-8')
        all_df = pd.concat([all_df, df4], ignore_index=False)

        # 上交所终止上市列表
        df5 = pd.read_csv(path_dir + '/上交所终止上市公司.csv', dtype={'原公司代码': str}, parse_dates=['上市日期'], encoding='utf-8')
        all_df = pd.concat([all_df, df5], ignore_index=False)

        # 上交所科创板正常交易
        df6 = pd.read_csv(path_dir + '/上交所科创板.csv', dtype={'A股代码': str}, date_parser=['上市日期'], encoding='utf-8')
        all_df = pd.concat([all_df, df6], ignore_index=False)

        # 上交所暂停上市列表
        df7 = pd.read_csv(path_dir + '/上交所暂停上市公司.csv', dtype={'公司代码': str}, date_parser=['上市日期'], encoding='utf-8')
        if not df7.empty:
            all_df = all_df.append(df7, ignore_index=False)

        # 先排序后去重， 升序，升序
        all_df['上市日期'] = pd.to_datetime(all_df['上市日期'])
        # 去掉两个深交所暂停上市的重复数据
        all_df.drop_duplicates(subset=['股票代码', '终止/暂停上市日期'], keep='last', inplace=True)
        all_df = all_df.sort_values(by=['股票代码', '上市日期'], ascending=[1, 1])
        # 重置索引
        all_df.reset_index(inplace=True, drop=True)
        all_df.to_csv(path_dir + '/all_stock_list.csv', encoding='utf-8', index=False)
        print('合并并保存all_stock_list.csv成功！')
        # print(all_df)
        return all_df

    def main(self):
        # 运行从上交所下载xls存为csv文件函数
        self.get_sh_stocklist_normal_from_SSE(path_dir=self.path_dir)
        time.sleep(1)
        self.get_sh_stocklist_zanting_from_SSE(path_dir=self.path_dir)
        time.sleep(1)
        self.get_sh_stocklist_zhongzhi_from_SSE(path_dir=self.path_dir)
        time.sleep(1)
        self.get_sh_stocklist_kechuang_from_SSE(path_dir=self.path_dir)
        time.sleep(1)

        # 运行从深交所下载xlsx文件函数
        self.get_sz_stocklist_normal_from_SZSE(path_dir=self.path_dir)
        time.sleep(1.5)
        self.get_sz_stocklist_zanting_from_SZSE(path_dir=self.path_dir)
        time.sleep(1.5)
        self.get_sz_stocklist_zhongzhi_from_SZSE(path_dir=self.path_dir)
        time.sleep(0.5)

        # 运行拼接主函数
        self.get_stock_list(path_dir=self.path_dir)

        # # 查看按交易状态分组统计数据
        # path = self.path_dir + '/all_stock_list.csv'
        # df = pd.read_csv(path, encoding='utf-8')
        # df1 = df.groupby(by='上市状态')['股票代码'].size()
        # print(df1)


class Spider_func():
    def __init__(self):
        # 获取该class类所在.py文件的绝对目录
        self.file_full_dir = os.path.dirname(os.path.abspath(__file__))
        # 设置网易日k历史数据存储目录
        self.stock_hisdata_dir = self.file_full_dir + '/datas_em'
        if not os.path.exists(self.stock_hisdata_dir):
            os.mkdir(self.stock_hisdata_dir)
        pass

    def symbol_to_stock_code(self, symbol):
        """
        将symbol转为带有交易所标识的股票代码
        :param symbol:
        :return:
        """

        if len(str(symbol)) < 6:
            # 转为字符串，并且在左边自动补齐0，使长度为6
            symbol = str(symbol).rjust(6, '0')
        elif len(str(symbol)) == 6:
            symbol = str(symbol)
        else:
            stock_code = symbol
            return stock_code

        if symbol.startswith('6'):
            stock_code = 'sh' + symbol
        elif symbol.startswith('0') or symbol.startswith('3'):
            stock_code = 'sz' + symbol
        elif symbol.startswith('4') or symbol.startswith('8') or symbol.startswith('9'):
            stock_code = 'bj' + symbol
        else:
            stock_code = symbol

        return stock_code

    # 循环尝试从网页上抓取数据
    def get_response_from_internet(self, url, max_try_num=10, sleep_time=10):
        """
        :param url: 要抓取数据的网址
        :param max_try_num: 最多尝试抓取次数
        :param sleep_time: 抓取失败后停顿的时间
        :return: 返回抓取到的网页内容
        """
        get_success = False  # 是否成功抓取到内容
        response = None
        # 抓取内容
        for i in range(max_try_num):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'}
                # 获取网页数据
                response = requests.get(url=url, headers=headers, timeout=10)
                get_success = True  # 成功抓取到内容
                # time.sleep(1)  # 休息1秒
                break
            except Exception as e:
                print('抓取数据报错，次数：', i + 1, '报错内容：', e)
                time.sleep(sleep_time)

        # 判断是否成功抓取内容
        if get_success:
            return response
        else:
            raise ValueError('get_response_from_internet抓取网页数据报错达到尝试上限，停止程序，请尽快检查问题所在')

    # 获取某只股票的上市日期
    def get_stock_start_date_from_csv(self, code, path):
        """
        :param code: 股票代码 600519 、002001或者 sz002001 sh600519
        :param path: 从csv源文件中读取个股对应上市日期，必填参数源文件目录
        """
        if code[0] == 's':
            stock_code = code
        else:
            if code[0] == 6 or code[0] == '6':
                stock_code = 'sh' + code
            else:
                stock_code = 'sz' + code

        df = pd.read_csv(path, encoding='utf-8',
                         parse_dates=['上市日期'])
        try:
            start_date = df[df['股票代码'] == stock_code]['上市日期'].iloc[0]
            # 格式化为 str ， 20210919
            start_date = start_date.strftime('%Y%m%d')
            return start_date
        except Exception as e:
            print('未找到该股对应上市日期数据，跳过！', e)
            return '19901201'

    # 从网易财经下载股票历史数据
    def download_stock_hisdata_from_163(self, code, start_date, end_date, path_dir):
        """
        源地址 http://quotes.money.163.com/service/chddata.html?code=0600519&start=20010827&end=20210918&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP
        :param code: 股票代码 例 600000
        :param start_date:下载的数据开始日期
        :param end_date: 下载的数据结束日期
        :param path_dir: 下载后的csv文件默认存在目录
        :return: 返回抓取到的网页内容
        """
        # 判断code是否为 600000
        if code[0] == 's':
            code = code[2:]
        else:
            code = code
        # 定义前缀、市场代码
        if code[0] == '6':
            market = '0'
            code_ex = 'sh'
        else:
            market = '1'
            code_ex = 'sz'

        # csv存取目录地址
        path = path_dir + '/' + code_ex + code + '.csv'
        # print(path)
        # 拼接股票历史行情数据下载地址
        download_url = "http://quotes.money.163.com/service/chddata.html?code=" + market + code + "&start=" + start_date + "&end=" + end_date + "&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP"
        # print(download_url)
        try:
            # 调用get_response_from_internet()函数，循环爬取目标网址
            response = self.get_response_from_internet(url=download_url, max_try_num=10, sleep_time=10)

            # 存储csv文件到指定目录
            with open(path, 'wb') as f:
                for chunk in response.iter_content(
                        chunk_size=10000):  # iter_content()边下载边存硬盘, chunk_size 可以自由调整为可以更好地适合您的用例的数字
                    if chunk:
                        f.write(chunk)
            print('股票：%s历史数据下载完成' % (code_ex + code))
        except Exception as e:
            print('错误位置:循环爬取存储数据失败', e)  # 把exception输出出来
            return pd.DataFrame()

        try:
            df = pd.read_csv(path, dtype={'股票代码': str}, sep=',', encoding='gbk')
            if len(df) >= 1:
                # 把csv里的 None 替换为 0，缺失的替换为 0，以便后续使对应数字列转为float类型
                df = df.replace('None', 0)
                df = df.fillna(value=0)
                # 重命名列名
                rename_dict = {'日期': '交易日期', '前收盘': '前收盘价', '成交金额': '成交额'}
                df.rename(columns=rename_dict, inplace=True)
                # 去掉字符串首位的空格
                df['股票代码'] = df['股票代码'].str.strip()
                # 拼接股票代码列
                df['股票代码'] = code_ex + df['股票代码'].str[1:]
                # 指定数字列数据类型为float
                df[['收盘价', '最高价', '最低价', '开盘价', '前收盘价', '涨跌额', '涨跌幅', '换手率', '成交量', '成交额', '总市值', '流通市值']] = df[
                    ['收盘价', '最高价', '最低价', '开盘价', '前收盘价', '涨跌额', '涨跌幅', '换手率', '成交量', '成交额', '总市值', '流通市值']].astype(
                    float)
                # 换一种写法：指定数字列数据类型为float，第4列开始
                # df.iloc[:, 3:] = df.iloc[:, 3:].astype(float)

                # 将日期列从srt格式转为日期格式
                df['交易日期'] = pd.to_datetime(df['交易日期'])
                # by参数指定按照什么进行排序，acsending参数指定是顺序还是逆序，1顺序，0逆序 ;别忘了赋值给df
                df = df.sort_values(by=['交易日期'], ascending=1)
                # 重置索引
                df.reset_index(drop=True, inplace=True)
                # # 重新处理排序后保存覆盖原csv
                # df.to_csv(path, index=False, encoding='gbk')
                # print('恭喜：%s处理后存储完成' % path)
                return df
            else:
                print('下载的数据条数为0，跳过')
                return df
        except Exception as e:
            print('错误位置:清洗整理原始数据', e)  # 把exception输出出来

    def download_all_stocks_history_kline_from_wangyi163(self):
        d_s_l_1 = Download_stocks_list()
        path = d_s_l_1.all_stocklist_path
        df = pd.read_csv(path, dtype={'股票代码': str}, encoding='utf-8')

        for i in range(len(df)):
            code = df.at[i, '股票代码']

            # 从本机csv文件中获取股票上市日期,str格式 19900918
            start_date = self.get_stock_start_date_from_csv(code, path=path)
            # 获取当前日期作为结尾日期，str格式 20210918
            end_date = datetime.date.today().strftime("%Y%m%d")
            # print(end_date)
            # 设置下载股票历史数据的csv文件存储目录
            path_dir = self.stock_hisdata_dir

            # 开始循环下载和存储
            df_code = self.download_stock_hisdata_from_163(code, start_date, end_date, path_dir=path_dir)
            if len(df_code) >= 1:
                save_path = path_dir + '/' + code + '.csv'
                df_code.to_csv(save_path, mode='w', index=False, encoding='gbk')
                print('恭喜：%s处理后存储完成' % path)
                time.sleep(1)
            else:
                print('未爬取到任何有效数据，记录并跳出循环!')
                path_lose = 'lose_stockdata_' + end_date + '.txt'
                with open(path_lose, mode='a') as f:
                    f.write('注意：%s未爬取%s到%s日起有效数据，跳出循环\n' % (datetime.datetime.now(), code, start_date))

    def get_stock_history_data_from_eastmoney(self, stock_code: str, period: str = '日', fqt='0') -> pd.DataFrame:
        """
        http://quote.eastmoney.com/sh600519.html
        默认：前复权 fqt=1 不复权fqt=0  后复权fqt=2
        1分钟数据 http://40.push2.eastmoney.com/api/qt/stock/trends2/sse?fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f17&fields2=f51,f52,f53,f54,f55,f56,f57,f58&mpi=1000&ut=fa5fd1943c7b386f172d6893dbfba10b&secid=0.300059&ndays=1&iscr=0&iscca=0&wbp2u=|0|0|0|web
        """
        if len(stock_code) == 6:
            symbol = stock_code
        elif len(stock_code) == 8:
            symbol = stock_code[2:]

        if symbol[0] == '6':
            market = '1'
        elif symbol[0] == '0' or symbol[0] == '3':
            market = '0'
        else:
            market = '0'
        secid = market + '.' + symbol

        if period == '周':
            klt = '102'
        elif period == '日':
            klt = '101'
        elif period == '5分钟':
            klt = '5'
        elif period == '30分钟':
            klt = '30'
        elif period == '1分钟':
            klt = '1'

        # url = 'http://push2his.eastmoney.com/api/qt/stock/kline/get?fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&ut=7eea3edcaed734bea9cbfc24409ed989&klt={}&fqt=1&secid={}&beg=0&end=20500000'.format(
        #     klt, secid)
        url0 = 'http://11.push2his.eastmoney.com/api/qt/stock/kline/get?cb=jQuery35108340375194521352_1715095486213&secid=1.600519&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=1&beg=0&end=20500101&smplmt=460&lmt=1000000&_=1715095486234'
        # 不复权
        url = f'http://22.push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt={klt}&fqt={fqt}&end=20500101&lmt=20000&_=1713025010737'
        # print(url)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }
        try:
            resp = requests.get(url=url, headers=headers, timeout=30)
            content = resp.content.decode('utf-8')
            if 'data' in content:
                # code = resp.json()['code']
                # name = resp.json()['name']
                data = resp.json()['data']['klines']
                lst_data = []

                for i in data:
                    # 字符串分割为列表
                    lst_i = i.split(',')
                    lst_data.append(lst_i)
                # print(data)
                df = pd.DataFrame(lst_data)
                # print(df)
                df.columns = ['交易日期', '开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                # 指定数字列数据类型为float
                df[['开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']] = df[
                    ['开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']].astype(
                    float)
                return df
            else:
                print(f'获取历史行情出错！股票代码: {stock_code}')
                return pd.DataFrame()
        except Exception as e:
            print(f'获取历史行情异常！股票代码: {stock_code}, 错误: {str(e)}')
            return pd.DataFrame()

    def download_all_stocks_history_kline_from_em(self):
        d_s_l_1 = Download_stocks_list()
        path = d_s_l_1.all_stocklist_path
        df = pd.read_csv(path, dtype={'股票代码': str}, encoding='utf-8')

        for i in df.index:
            try:
                code = df.loc[i, '股票代码']
                print(f"正在处理: {code}")
                # 从本机csv文件中获取股票上市日期,str格式 19900918
                start_date = self.get_stock_start_date_from_csv(code, path=path)
                # 获取当前日期作为结尾日期，str格式 20210918
                end_date = datetime.date.today().strftime("%Y%m%d")
                # print(end_date)
                # 设置下载股票历史数据的csv文件存储目录
                path_dir = self.stock_hisdata_dir

                # 开始循环下载和存储（包含板块信息）
                df_code = self.get_stock_history_data_from_eastmoney(stock_code=code)
                if not df_code.empty:
                    # 获取股票板块信息
                    stock_info = self.get_stock_industry_info_from_eastmoney(stock_code=code)
                    
                    if stock_info:
                        # 添加板块信息到DataFrame
                        df_code['所属行业'] = stock_info.get('所属行业', '')
                        df_code['概念板块'] = stock_info.get('概念板块', '')
                        df_code['地区'] = stock_info.get('地区', '')
                        df_code['总股本'] = stock_info.get('总股本', 0)
                        df_code['流通股'] = stock_info.get('流通股', 0)
                        df_code['每股收益'] = stock_info.get('每股收益', 0)
                        df_code['每股净资产'] = stock_info.get('每股净资产', 0)
                    
                    save_path = path_dir + '/' + code + '.csv'
                    df_code.to_csv(save_path, mode='w', index=False, encoding='utf-8')
                    print(f'✅ {code} 数据保存完成 (包含板块信息)')
                    time.sleep(1)
                else:
                    print(f'❌ {code} 未爬取到任何有效数据，记录并跳出循环!')
                    continue

            except Exception as e:
                print(f'❌ {code} 出错！{e}')
                continue

    def get_recent_all_stock_kline_data_from_em_old(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }
        # 获取13位当前时间戳
        now = time.time() * 1000
        now13 = int(now)
        url = f'http://21.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=20000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_={now13}'

        response = requests.get(url=url, headers=headers)

        resp_json = response.json()
        # total = resp_json['data']['total']
        df = pd.DataFrame(resp_json['data']['diff'])
        df = df[
            ['f12', 'f14', 'f2', 'f15', 'f16', 'f17', 'f18', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f10', 'f20', 'f21',
             'f9', 'f23', 'f11']]
        rename_dict = {'f12': '股票代码', 'f14': '股票名称', 'f2': '收盘价', 'f15': '最高价', 'f16': '最低价', 'f17': '开盘价',
                       'f18': '前收盘价', 'f10': '量比',
                       'f20': '总市值', 'f21': '流通市值', 'f3': '涨跌幅', 'f4': '涨跌额', 'f5': '成交量', 'f6': '成交额', 'f7': '振幅',
                       'f8': '换手率', 'f9': '动态市盈率', 'f23': '市净率', 'f11': '涨速'}
        df.rename(columns=rename_dict, inplace=True)

        g_o_1 = Get_stock_opening_calendar()
        recent_trade_date = g_o_1.get_recent_trade_date()
        df['交易日期'] = pd.to_datetime(recent_trade_date).strftime("%Y-%m-%d")
        # 取需要的列
        df = df[
            ['交易日期', '股票代码', '股票名称', '收盘价', '最高价', '最低价', '开盘价', '前收盘价', '振幅', '涨跌额', '涨跌幅', '换手率', '成交量', '成交额', '总市值',
             '流通市值', '涨速']]
        df['股票代码'] = df['股票代码'].apply(self.symbol_to_stock_code)
        # 去除北交所股票
        df = df[df['股票代码'].str[0:2] != 'bj']
        # 去掉或替换异常数据'-'为None
        # df.replace('-', None, inplace=True)
        df = df[df['最低价'] != '-']
        # 多列转为float格式
        df[['收盘价', '最高价', '最低价', '开盘价', '前收盘价', '涨跌额', '涨跌幅', '换手率', '成交量', '成交额', '总市值', '流通市值', '涨速']] = df[
            ['收盘价', '最高价', '最低价', '开盘价', '前收盘价', '涨跌额', '涨跌幅', '换手率', '成交量', '成交额', '总市值', '流通市值', '涨速']].astype(float)
        df['成交量'] = df['成交量'] * 100
        df.reset_index(drop=True, inplace=True)

        return df

    def get_recent_all_stock_kline_data_from_em(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }
        # 生成13位时间戳（精确到毫秒）
        timestamp13 = int(time.time() * 1000)
        all_df = pd.DataFrame()
        try:
            total = 7000
            pz = 200
            end_pz = int(total / pz)
            for i in range(1, end_pz, 1):
                # print(i)
                fields = 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152'
                url = f'http://99.push2.eastmoney.com/api/qt/clist/get?pn={i}&pz={pz}&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048&fields={fields}&_={timestamp13}'
                response = requests.get(url=url, headers=headers)
                if response.status_code != 200:
                    url_2 = f'http://69.push2.eastmoney.com/api/qt/clist/get?&pn=1&pz=20000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&wbp2u=|0|0|0|web&fid=f3&fs=m:1+t:2,m:1+t:23&fields={fields}&_={timestamp13}'
                    response = requests.get(url=url_2, headers=headers)

                resp_json = response.json()
                # total = resp_json['data']['total']
                # print(resp_json['data'])
                if resp_json['data'] == None:
                    break
                df = pd.DataFrame(resp_json['data']['diff'])

                df = df[
                    ['f12', 'f14', 'f2', 'f15', 'f16', 'f17', 'f18', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f10', 'f20',
                     'f21',
                     'f9', 'f23', 'f11']]
                rename_dict = {'f12': '股票代码', 'f14': '股票名称', 'f2': '收盘价', 'f15': '最高价', 'f16': '最低价', 'f17': '开盘价',
                               'f18': '前收盘价', 'f10': '量比',
                               'f20': '总市值', 'f21': '流通市值', 'f3': '涨跌幅', 'f4': '涨跌额', 'f5': '成交量', 'f6': '成交额',
                               'f7': '振幅',
                               'f8': '换手率', 'f9': '动态市盈率', 'f23': '市净率', 'f11': '涨速'}
                df.rename(columns=rename_dict, inplace=True)
                g_o_1 = Get_stock_opening_calendar()
                recent_trade_date = g_o_1.get_recent_trade_date()
                df['交易日期'] = pd.to_datetime(recent_trade_date).strftime("%Y-%m-%d")
                # 取需要的列
                # 取需要的列
                df = df[
                    ['交易日期', '股票代码', '股票名称', '收盘价', '最高价', '最低价', '开盘价', '前收盘价', '振幅', '涨跌额', '涨跌幅', '换手率', '成交量',
                     '成交额', '总市值',
                     '流通市值', '涨速']]
                df['股票代码'] = df['股票代码'].apply(self.symbol_to_stock_code)
                # 去除北交所股票
                df = df[df['股票代码'].str[0:2] != 'bj']
                # 去掉或替换异常数据'-'为None
                # df.replace('-', None, inplace=True)
                df = df[df['最低价'] != '-']
                # 多列转为float格式
                df[['收盘价', '最高价', '最低价', '开盘价', '前收盘价', '涨跌额', '涨跌幅', '换手率', '成交量', '成交额', '总市值', '流通市值', '涨速']] = df[
                    ['收盘价', '最高价', '最低价', '开盘价', '前收盘价', '涨跌额', '涨跌幅', '换手率', '成交量', '成交额', '总市值', '流通市值',
                     '涨速']].astype(float)
                df['成交量'] = df['成交量'] * 100
                all_df = pd.concat(objs=[all_df, df], ignore_index=True)
                time.sleep(0.1)
            all_df.drop_duplicates(subset=['股票代码', '交易日期'], inplace=True)
            all_df.reset_index(drop=True, inplace=True)
        except Exception as e:
            print(f'get_recent_1day_all_stock_data_from_dc_1time_200出错！ {e}')
        return all_df

    def get_stock_money_flow_from_eastmoney(self, stock_code: str, period: str = '当日') -> pd.DataFrame:
        """
        获取个股资金流向数据（净流入）
        :param stock_code: 股票代码，如 'sh600519' 或 '600519'
        :param period: 时间周期，'当日' 或 '历史'
        :return: 包含资金流向数据的DataFrame
        """
        if len(stock_code) == 6:
            symbol = stock_code
        elif len(stock_code) == 8:
            symbol = stock_code[2:]
        else:
            return pd.DataFrame()

        if symbol[0] == '6':
            market = '1'
        elif symbol[0] == '0' or symbol[0] == '3':
            market = '0'
        else:
            market = '0'
        secid = market + '.' + symbol

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }

        timestamp = int(time.time() * 1000)
        
        if period == '当日':
            # 当日分时资金流向 - 使用简化的API
            url = f'http://push2.eastmoney.com/api/qt/stock/fflow/get?lmt=0&klt=1&secid={secid}&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63&ut=b2884a393a59ad64002292a3e90d46a5&_={timestamp}'
        else:
            # 历史每日资金流向 - 使用简化的API
            url = f'http://push2his.eastmoney.com/api/qt/stock/fflow/get?lmt=500&klt=101&secid={secid}&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63&ut=b2884a393a59ad64002292a3e90d46a5&_={timestamp}'

        try:
            resp = requests.get(url=url, headers=headers)
            content = resp.content.decode('utf-8')
            
            # 移除jsonp回调函数包装
            if 'jQuery' in content and '(' in content:
                start = content.find('(') + 1
                end = content.rfind(')')
                json_str = content[start:end]
            else:
                json_str = content
            
            # 修复JSON解析警告，使用json.loads代替pd.read_json
            data = json.loads(json_str)
            
            if 'data' in data and data['data'] is not None:
                if period == '当日':
                    # 当日分时数据
                    klines = data['data']['klines']
                    if klines:
                        df_list = []
                        for line in klines:
                            parts = line.split(',')
                            df_list.append(parts)
                        
                        df = pd.DataFrame(df_list)
                        df.columns = ['时间', '主力净流入', '小单净流入', '中单净流入', '大单净流入', '超大单净流入', '主力净流入占比', '小单净流入占比', '中单净流入占比', '大单净流入占比', '超大单净流入占比', '收盘价', '涨跌幅']
                        
                        # 转换数据类型
                        numeric_cols = ['主力净流入', '小单净流入', '中单净流入', '大单净流入', '超大单净流入', '主力净流入占比', '小单净流入占比', '中单净流入占比', '大单净流入占比', '超大单净流入占比', '收盘价', '涨跌幅']
                        df[numeric_cols] = df[numeric_cols].astype(float)
                        
                        return df
                else:
                    # 历史每日数据
                    klines = data['data']['klines']
                    if klines:
                        df_list = []
                        for line in klines:
                            parts = line.split(',')
                            df_list.append(parts)
                        
                        df = pd.DataFrame(df_list)
                        df.columns = ['日期', '主力净流入', '小单净流入', '中单净流入', '大单净流入', '超大单净流入', '主力净流入占比', '小单净流入占比', '中单净流入占比', '大单净流入占比', '超大单净流入占比', '收盘价', '涨跌幅']
                        
                        # 转换数据类型
                        numeric_cols = ['主力净流入', '小单净流入', '中单净流入', '大单净流入', '超大单净流入', '主力净流入占比', '小单净流入占比', '中单净流入占比', '大单净流入占比', '超大单净流入占比', '收盘价', '涨跌幅']
                        df[numeric_cols] = df[numeric_cols].astype(float)
                        
                        return df
            
            print(f'获取{stock_code}资金流向数据失败')
            return pd.DataFrame()
            
        except Exception as e:
            print(f'获取{stock_code}资金流向出错：{e}')
            return pd.DataFrame()

    def get_all_stocks_money_flow_from_eastmoney(self, flow_type: str = '个股') -> pd.DataFrame:
        """
        获取全市场资金流向排行数据
        :param flow_type: 流向类型，'个股'、'行业'、'概念'
        :return: 包含资金流向排行的DataFrame
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }
        
        timestamp = int(time.time() * 1000)
        
        if flow_type == '个股':
            # 个股资金流向排行
            url = f'http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f62&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048&fields=f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f204,f205,f124&_={timestamp}'
        elif flow_type == '行业':
            # 行业资金流向排行
            url = f'http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=500&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f62&fs=m:90+t:2+f:!50&fields=f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f204,f205,f124&_={timestamp}'
        elif flow_type == '概念':
            # 概念资金流向排行
            url = f'http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=500&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f62&fs=m:90+t:3+f:!50&fields=f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f204,f205,f124&_={timestamp}'
        else:
            return pd.DataFrame()

        try:
            resp = requests.get(url=url, headers=headers)
            resp_json = resp.json()
            
            if 'data' in resp_json and resp_json['data'] is not None:
                df = pd.DataFrame(resp_json['data']['diff'])
                
                if not df.empty:
                    # 选择需要的列并重命名
                    df = df[['f12', 'f14', 'f2', 'f3', 'f62', 'f184', 'f66', 'f69', 'f72', 'f75', 'f78']]
                    
                    rename_dict = {
                        'f12': '代码', 
                        'f14': '名称', 
                        'f2': '最新价', 
                        'f3': '涨跌幅',
                        'f62': '主力净流入', 
                        'f184': '主力净流入占比',
                        'f66': '超大单净流入', 
                        'f69': '超大单净流入占比',
                        'f72': '大单净流入', 
                        'f75': '大单净流入占比',
                        'f78': '中单净流入'
                    }
                    
                    df.rename(columns=rename_dict, inplace=True)
                    
                    # 转换数据类型
                    numeric_cols = ['最新价', '涨跌幅', '主力净流入', '主力净流入占比', '超大单净流入', '超大单净流入占比', '大单净流入', '大单净流入占比', '中单净流入']
                    df[numeric_cols] = df[numeric_cols].astype(float)
                    
                    # 添加股票代码格式
                    if flow_type == '个股':
                        df['代码'] = df['代码'].apply(self.symbol_to_stock_code)
                    
                    return df
            
            print(f'获取{flow_type}资金流向排行数据失败')
            return pd.DataFrame()
            
        except Exception as e:
            print(f'获取{flow_type}资金流向排行出错：{e}')
            return pd.DataFrame()

    def get_stock_technical_indicators_from_eastmoney(self, stock_code: str, indicator: str = 'MACD', period: str = '日') -> pd.DataFrame:
        """
        获取个股技术指标数据
        :param stock_code: 股票代码，如 'sh600519' 或 '600519'
        :param indicator: 技术指标类型，'MACD', 'RSI', 'KDJ', 'BOLL', 'MA', 'EXPMA', 'CCI', 'WR'
        :param period: 时间周期，'日', '周', '月', '5分钟', '15分钟', '30分钟', '60分钟'
        :return: 包含技术指标数据的DataFrame
        """
        if len(stock_code) == 6:
            symbol = stock_code
        elif len(stock_code) == 8:
            symbol = stock_code[2:]
        else:
            return pd.DataFrame()

        if symbol[0] == '6':
            market = '1'
        elif symbol[0] == '0' or symbol[0] == '3':
            market = '0'
        else:
            market = '0'
        secid = market + '.' + symbol

        # 设置K线周期
        if period == '周':
            klt = '102'
        elif period == '日':
            klt = '101'
        elif period == '月':
            klt = '103'
        elif period == '5分钟':
            klt = '5'
        elif period == '15分钟':
            klt = '15'
        elif period == '30分钟':
            klt = '30'
        elif period == '60分钟':
            klt = '60'
        else:
            klt = '101'

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }

        timestamp = int(time.time() * 1000)
        
        # 构建技术指标API URL
        if indicator == 'MACD':
            # MACD指标
            url = f'http://push2his.eastmoney.com/api/qt/stock/trends2/get?fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13&fields2=f51,f52,f53,f54,f55,f56,f57,f58&secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&ndays=1&iscr=0&iscca=0&wbp2u=|0|0|0|web&_={timestamp}'
        elif indicator == 'RSI':
            # RSI指标 
            url = f'http://push2his.eastmoney.com/api/qt/stock/trends2/get?fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13&fields2=f51,f52,f53,f54,f55,f56,f57,f58&secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&ndays=1&iscr=0&iscca=0&wbp2u=|0|0|0|web&_={timestamp}'
        else:
            # 通用技术指标URL
            url = f'http://push2his.eastmoney.com/api/qt/stock/trends2/get?fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13&fields2=f51,f52,f53,f54,f55,f56,f57,f58&secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&ndays=1&iscr=0&iscca=0&wbp2u=|0|0|0|web&_={timestamp}'

        try:
            resp = requests.get(url=url, headers=headers)
            resp_json = resp.json()
            
            if 'data' in resp_json and resp_json['data'] is not None:
                trends = resp_json['data']['trends']
                if trends:
                    df_list = []
                    for line in trends:
                        parts = line.split(',')
                        df_list.append(parts)
                    
                    df = pd.DataFrame(df_list)
                    df.columns = ['时间', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅']
                    
                    # 转换数据类型
                    numeric_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅']
                    df[numeric_cols] = df[numeric_cols].astype(float)
                    
                    # 基于K线数据计算技术指标
                    if indicator == 'MACD':
                        df = self._calculate_macd(df)
                    elif indicator == 'RSI':
                        df = self._calculate_rsi(df)
                    elif indicator == 'KDJ':
                        df = self._calculate_kdj(df)
                    elif indicator == 'BOLL':
                        df = self._calculate_boll(df)
                    elif indicator == 'MA':
                        df = self._calculate_ma(df)
                    
                    return df
            
            print(f'获取{stock_code}技术指标数据失败')
            return pd.DataFrame()
            
        except Exception as e:
            print(f'获取{stock_code}技术指标出错：{e}')
            return pd.DataFrame()

    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """计算MACD指标"""
        close = df['收盘']
        
        # 计算EMA
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        
        # 计算DIF和DEA
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal).mean()
        
        # 计算MACD柱状图
        macd = 2 * (dif - dea)
        
        df['DIF'] = dif
        df['DEA'] = dea
        df['MACD'] = macd
        
        return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算RSI指标"""
        close = df['收盘']
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        df['RSI'] = rsi
        
        return df

    def _calculate_kdj(self, df: pd.DataFrame, period: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
        """计算KDJ指标"""
        high = df['最高']
        low = df['最低']
        close = df['收盘']
        
        # 计算RSV
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        
        # 计算K值
        k = rsv.ewm(span=k_smooth).mean()
        
        # 计算D值
        d = k.ewm(span=d_smooth).mean()
        
        # 计算J值
        j = 3 * k - 2 * d
        
        df['K'] = k
        df['D'] = d
        df['J'] = j
        
        return df

    def _calculate_boll(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """计算布林带指标"""
        close = df['收盘']
        
        # 计算中轨（移动平均线）
        middle = close.rolling(window=period).mean()
        
        # 计算标准差
        std = close.rolling(window=period).std()
        
        # 计算上轨和下轨
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        df['BOLL_UPPER'] = upper
        df['BOLL_MIDDLE'] = middle
        df['BOLL_LOWER'] = lower
        
        return df

    def _calculate_ma(self, df: pd.DataFrame, periods: list = [5, 10, 20, 60]) -> pd.DataFrame:
        """计算移动平均线"""
        close = df['收盘']
        
        for period in periods:
            df[f'MA{period}'] = close.rolling(window=period).mean()
        
        return df

    def get_stock_realtime_data_from_eastmoney(self, stock_code: str) -> pd.DataFrame:
        """
        获取个股实时行情数据（包含更多字段）
        :param stock_code: 股票代码，如 'sh600519' 或 '600519'
        :return: 包含实时行情数据的DataFrame
        """
        if len(stock_code) == 6:
            symbol = stock_code
        elif len(stock_code) == 8:
            symbol = stock_code[2:]
        else:
            return pd.DataFrame()

        if symbol[0] == '6':
            market = '1'
        elif symbol[0] == '0' or symbol[0] == '3':
            market = '0'
        else:
            market = '0'
        secid = market + '.' + symbol

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }

        timestamp = int(time.time() * 1000)
        
        # 获取更详细的实时数据
        fields = 'f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f57,f58,f59,f60,f61,f62,f63,f64,f65,f66,f71,f72,f73,f74,f75,f76,f77,f78,f79,f80,f81,f82,f83,f84,f85'
        url = f'http://push2.eastmoney.com/api/qt/stock/get?secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields={fields}&_={timestamp}'

        try:
            resp = requests.get(url=url, headers=headers)
            resp_json = resp.json()
            
            if 'data' in resp_json and resp_json['data'] is not None:
                data = resp_json['data']
                
                # 构建实时数据DataFrame
                realtime_data = {
                    '股票代码': stock_code,
                    '股票名称': data.get('f58', ''),
                    '最新价': data.get('f43', 0),
                    '涨跌额': data.get('f44', 0),
                    '涨跌幅': data.get('f45', 0),
                    '成交量': data.get('f47', 0),
                    '成交额': data.get('f48', 0),
                    '振幅': data.get('f49', 0),
                    '最高价': data.get('f50', 0),
                    '最低价': data.get('f51', 0),
                    '开盘价': data.get('f52', 0),
                    '昨收价': data.get('f60', 0),
                    '换手率': data.get('f71', 0),
                    '市盈率': data.get('f72', 0),
                    '市净率': data.get('f73', 0),
                    '总市值': data.get('f74', 0),
                    '流通市值': data.get('f75', 0),
                    '涨停价': data.get('f76', 0),
                    '跌停价': data.get('f77', 0),
                    '委比': data.get('f78', 0),
                    '委差': data.get('f79', 0),
                    '量比': data.get('f80', 0),
                    '外盘': data.get('f81', 0),
                    '内盘': data.get('f82', 0)
                }
                
                df = pd.DataFrame([realtime_data])
                return df
            
            print(f'获取{stock_code}实时数据失败')
            return pd.DataFrame()
            
        except Exception as e:
            print(f'获取{stock_code}实时数据出错：{e}')
            return pd.DataFrame()

    def get_stock_opening_auction_data_from_eastmoney(self, stock_code: str) -> pd.DataFrame:
        """
        获取个股开盘集合竞价数据（9:15-9:25）
        :param stock_code: 股票代码，如 'sh600519' 或 '600519'
        :return: 包含开盘集合竞价数据的DataFrame
        """
        if len(stock_code) == 6:
            symbol = stock_code
        elif len(stock_code) == 8:
            symbol = stock_code[2:]
        else:
            return pd.DataFrame()

        if symbol[0] == '6':
            market = '1'
        elif symbol[0] == '0' or symbol[0] == '3':
            market = '0'
        else:
            market = '0'
        secid = market + '.' + symbol

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }

        # 获取当前时间，判断是否在集合竞价时间段
        import datetime
        now = datetime.datetime.now()
        current_time = now.strftime("%H:%M")
        
        print(f"当前时间: {current_time}")
        
        # 开盘集合竞价时间段：9:15-9:25
        if "09:15" <= current_time <= "09:25":
            print("✅ 当前处于开盘集合竞价时间段")
        else:
            print("⚠️ 当前不在开盘集合竞价时间段，获取历史集合竞价数据")

        timestamp = int(time.time() * 1000)
        
        # 开盘集合竞价专用API
        url = f'http://push2.eastmoney.com/api/qt/stock/details/get?fields1=f1,f2,f3,f4&fields2=f51,f52,f53,f54,f55&secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&_={timestamp}'

        try:
            resp = requests.get(url=url, headers=headers)
            resp_json = resp.json()
            
            if 'data' in resp_json and resp_json['data'] is not None:
                details = resp_json['data']['details']
                if details:
                    df_list = []
                    auction_data = []  # 专门存储集合竞价时间段的数据
                    
                    for detail in details:
                        parts = detail.split(',')
                        if len(parts) >= 5:
                            time_str = parts[0]
                            # 筛选开盘集合竞价时间段的数据 (09:15-09:25)
                            if time_str.startswith('09:1') or time_str.startswith('09:2'):
                                if '09:15' <= time_str <= '09:25':
                                    auction_data.append(parts)
                            df_list.append(parts)
                    
                    # 优先返回集合竞价时间段的数据
                    data_to_use = auction_data if auction_data else df_list
                    
                    if data_to_use:
                        df = pd.DataFrame(data_to_use)
                        # 动态设置列名，根据实际数据列数
                        if len(df.columns) == 5:
                            df.columns = ['时间', '价格', '成交量', '成交额', '性质']
                            numeric_cols = ['价格', '成交量', '成交额']
                        elif len(df.columns) == 6:
                            df.columns = ['时间', '价格', '涨跌', '成交量', '成交额', '性质']
                            numeric_cols = ['价格', '涨跌', '成交量', '成交额']
                        else:
                            # 如果列数不匹配，使用通用列名
                            df.columns = [f'col_{i}' for i in range(len(df.columns))]
                            print(f"警告：开盘集合竞价数据列数为 {len(df.columns)}，使用通用列名")
                            return df
                        
                        # 转换数据类型
                        try:
                            df[numeric_cols] = df[numeric_cols].astype(float)
                        except:
                            print("警告：数据类型转换失败，返回原始数据")
                        
                        # 添加说明列
                        if auction_data:
                            df['数据类型'] = '开盘集合竞价'
                        else:
                            df['数据类型'] = '普通交易'
                            
                        return df
            
            print(f'获取{stock_code}开盘集合竞价数据失败')
            return pd.DataFrame()
            
        except Exception as e:
            print(f'获取{stock_code}开盘集合竞价数据出错：{e}')
            return pd.DataFrame()

    def get_stock_auction_data_from_eastmoney(self, stock_code: str, auction_type: str = '开盘集合竞价') -> pd.DataFrame:
        """
        获取个股竞价数据
        :param stock_code: 股票代码，如 'sh600519' 或 '600519'
        :param auction_type: 竞价类型，'开盘集合竞价', '收盘集合竞价', '分时成交'
        :return: 包含竞价数据的DataFrame
        """
        if len(stock_code) == 6:
            symbol = stock_code
        elif len(stock_code) == 8:
            symbol = stock_code[2:]
        else:
            return pd.DataFrame()

        if symbol[0] == '6':
            market = '1'
        elif symbol[0] == '0' or symbol[0] == '3':
            market = '0'
        else:
            market = '0'
        secid = market + '.' + symbol

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }

        timestamp = int(time.time() * 1000)

        try:
            if auction_type == '开盘集合竞价' or auction_type == '集合竞价':
                # 调用专门的开盘集合竞价方法
                return self.get_stock_opening_auction_data_from_eastmoney(stock_code)
            
            elif auction_type == '收盘集合竞价':
                # 获取收盘集合竞价数据 (14:57-15:00)
                url = f'http://push2.eastmoney.com/api/qt/stock/details/get?fields1=f1,f2,f3,f4&fields2=f51,f52,f53,f54,f55&secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&_={timestamp}'
                
                resp = requests.get(url=url, headers=headers)
                resp_json = resp.json()
                
                if 'data' in resp_json and resp_json['data'] is not None:
                    details = resp_json['data']['details']
                    if details:
                        df_list = []
                        for detail in details:
                            parts = detail.split(',')
                            if len(parts) >= 5:
                                df_list.append(parts)
                        
                        if df_list:
                            df = pd.DataFrame(df_list)
                            # 动态设置列名，根据实际数据列数
                            if len(df.columns) == 5:
                                df.columns = ['时间', '价格', '涨跌', '成交量', '成交额']
                                numeric_cols = ['价格', '涨跌', '成交量', '成交额']
                            elif len(df.columns) == 6:
                                df.columns = ['时间', '价格', '涨跌', '成交量', '成交额', '性质']
                                numeric_cols = ['价格', '涨跌', '成交量', '成交额']
                            else:
                                # 如果列数不匹配，使用通用列名
                                df.columns = [f'col_{i}' for i in range(len(df.columns))]
                                print(f"警告：集合竞价数据列数为 {len(df.columns)}，使用通用列名")
                                return df
                            
                            # 转换数据类型
                            try:
                                df[numeric_cols] = df[numeric_cols].astype(float)
                            except:
                                print("警告：数据类型转换失败，返回原始数据")
                            
                            return df
                            
            elif auction_type == '分时成交':
                # 获取分时成交明细
                url = f'http://push2.eastmoney.com/api/qt/stock/details/get?fields1=f1,f2,f3,f4&fields2=f51,f52,f53,f54,f55&secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&_={timestamp}'
                
                resp = requests.get(url=url, headers=headers)
                resp_json = resp.json()
                
                if 'data' in resp_json and resp_json['data'] is not None:
                    details = resp_json['data']['details']
                    if details:
                        df_list = []
                        for detail in details:
                            parts = detail.split(',')
                            if len(parts) >= 5:
                                df_list.append(parts)
                        
                        if df_list:
                            df = pd.DataFrame(df_list)
                            # 动态设置列名，根据实际数据列数
                            if len(df.columns) == 5:
                                df.columns = ['时间', '价格', '涨跌', '成交量', '成交额']
                                numeric_cols = ['价格', '涨跌', '成交量', '成交额']
                            elif len(df.columns) == 6:
                                df.columns = ['时间', '价格', '涨跌', '成交量', '成交额', '性质']
                                numeric_cols = ['价格', '涨跌', '成交量', '成交额']
                            else:
                                # 如果列数不匹配，使用通用列名
                                df.columns = [f'col_{i}' for i in range(len(df.columns))]
                                print(f"警告：分时成交数据列数为 {len(df.columns)}，使用通用列名")
                                return df
                            
                            # 转换数据类型
                            try:
                                df[numeric_cols] = df[numeric_cols].astype(float)
                            except:
                                print("警告：数据类型转换失败，返回原始数据")
                            
                            return df
                            
            print(f'获取{stock_code}竞价数据失败')
            return pd.DataFrame()
            
        except Exception as e:
            print(f'获取{stock_code}竞价数据出错：{e}')
            return pd.DataFrame()

    def get_stock_bid_ask_data_from_eastmoney(self, stock_code: str) -> pd.DataFrame:
        """
        获取个股五档买卖盘口数据
        :param stock_code: 股票代码，如 'sh600519' 或 '600519'
        :return: 包含五档买卖数据的DataFrame
        """
        if len(stock_code) == 6:
            symbol = stock_code
        elif len(stock_code) == 8:
            symbol = stock_code[2:]
        else:
            return pd.DataFrame()

        if symbol[0] == '6':
            market = '1'
        elif symbol[0] == '0' or symbol[0] == '3':
            market = '0'
        else:
            market = '0'
        secid = market + '.' + symbol

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }

        timestamp = int(time.time() * 1000)
        
        # 获取五档买卖盘口数据
        fields = 'f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28'
        url = f'http://push2.eastmoney.com/api/qt/stock/get?secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields={fields}&_={timestamp}'

        try:
            resp = requests.get(url=url, headers=headers)
            resp_json = resp.json()
            
            if 'data' in resp_json and resp_json['data'] is not None:
                data = resp_json['data']
                
                # 构建五档买卖数据
                bid_ask_data = {
                    '股票代码': stock_code,
                    '卖五价': data.get('f31', 0),
                    '卖五量': data.get('f32', 0),
                    '卖四价': data.get('f33', 0),
                    '卖四量': data.get('f34', 0),
                    '卖三价': data.get('f35', 0),
                    '卖三量': data.get('f36', 0),
                    '卖二价': data.get('f37', 0),
                    '卖二量': data.get('f38', 0),
                    '卖一价': data.get('f39', 0),
                    '卖一量': data.get('f40', 0),
                    '买一价': data.get('f19', 0),
                    '买一量': data.get('f20', 0),
                    '买二价': data.get('f21', 0),
                    '买二量': data.get('f22', 0),
                    '买三价': data.get('f23', 0),
                    '买三量': data.get('f24', 0),
                    '买四价': data.get('f25', 0),
                    '买四量': data.get('f26', 0),
                    '买五价': data.get('f27', 0),
                    '买五量': data.get('f28', 0)
                }
                
                df = pd.DataFrame([bid_ask_data])
                return df
            
            print(f'获取{stock_code}五档买卖数据失败')
            return pd.DataFrame()
            
        except Exception as e:
            print(f'获取{stock_code}五档买卖数据出错：{e}')
            return pd.DataFrame()

    def get_stock_minute_data_from_eastmoney(self, stock_code: str, ndays: int = 1) -> pd.DataFrame:
        """
        获取个股分时数据
        :param stock_code: 股票代码，如 'sh600519' 或 '600519'
        :param ndays: 获取天数，1=当日，5=5天
        :return: 包含分时数据的DataFrame
        """
        if len(stock_code) == 6:
            symbol = stock_code
        elif len(stock_code) == 8:
            symbol = stock_code[2:]
        else:
            return pd.DataFrame()

        if symbol[0] == '6':
            market = '1'
        elif symbol[0] == '0' or symbol[0] == '3':
            market = '0'
        else:
            market = '0'
        secid = market + '.' + symbol

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }

        timestamp = int(time.time() * 1000)
        
        # 获取分时数据
        url = f'http://push2his.eastmoney.com/api/qt/stock/trends2/get?fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f17&fields2=f51,f52,f53,f54,f55,f56,f57,f58&secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&ndays={ndays}&iscr=0&iscca=0&_={timestamp}'

        try:
            resp = requests.get(url=url, headers=headers)
            resp_json = resp.json()
            
            if 'data' in resp_json and resp_json['data'] is not None:
                trends = resp_json['data']['trends']
                if trends:
                    df_list = []
                    for trend in trends:
                        parts = trend.split(',')
                        if len(parts) >= 8:
                            df_list.append(parts)
                    
                    if df_list:
                        df = pd.DataFrame(df_list)
                        df.columns = ['时间', '价格', '成交量', '成交额', '均价', '买入', '卖出', '最新量']
                        
                        # 转换数据类型
                        numeric_cols = ['价格', '成交量', '成交额', '均价', '买入', '卖出', '最新量']
                        df[numeric_cols] = df[numeric_cols].astype(float)
                        
                        return df
            
            print(f'获取{stock_code}分时数据失败')
            return pd.DataFrame()
            
        except Exception as e:
            print(f'获取{stock_code}分时数据出错：{e}')
            return pd.DataFrame()

    def get_industry_data_from_eastmoney(self, sort_field: str = 'f3') -> pd.DataFrame:
        """
        获取行业板块数据
        :param sort_field: 排序字段，f3=涨跌幅，f62=主力净流入，f84=总市值
        :return: 包含行业板块数据的DataFrame
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }
        
        timestamp = int(time.time() * 1000)
        
        # 行业板块数据API
        fields = 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f62,f184,f66,f69,f72,f75,f78,f81,f82,f84,f85,f86,f87,f124'
        url = f'http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=500&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid={sort_field}&fs=m:90+t:2+f:!50&fields={fields}&_={timestamp}'

        try:
            resp = requests.get(url=url, headers=headers)
            resp_json = resp.json()
            
            if 'data' in resp_json and resp_json['data'] is not None:
                df = pd.DataFrame(resp_json['data']['diff'])
                
                if not df.empty:
                    # 选择并重命名列
                    df = df[['f12', 'f14', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f15', 'f16', 'f17', 'f18', 'f20', 'f21', 'f62', 'f184', 'f84', 'f124']]
                    
                    rename_dict = {
                        'f12': '行业代码',
                        'f14': '行业名称', 
                        'f2': '最新价',
                        'f3': '涨跌幅',
                        'f4': '涨跌额',
                        'f5': '成交量',
                        'f6': '成交额',
                        'f7': '振幅',
                        'f15': '最高价',
                        'f16': '最低价',
                        'f17': '开盘价',
                        'f18': '昨收价',
                        'f20': '总市值',
                        'f21': '流通市值',
                        'f62': '主力净流入',
                        'f184': '主力净流入占比',
                        'f84': '总股本',
                        'f124': '更新时间'
                    }
                    
                    df.rename(columns=rename_dict, inplace=True)
                    
                    # 转换数据类型
                    numeric_cols = ['最新价', '涨跌幅', '涨跌额', '成交量', '成交额', '振幅', '最高价', '最低价', '开盘价', '昨收价', '总市值', '流通市值', '主力净流入', '主力净流入占比', '总股本']
                    df[numeric_cols] = df[numeric_cols].astype(float)
                    
                    return df
            
            print('获取行业板块数据失败')
            return pd.DataFrame()
            
        except Exception as e:
            print(f'获取行业板块数据出错：{e}')
            return pd.DataFrame()

    def get_concept_data_from_eastmoney(self, sort_field: str = 'f3') -> pd.DataFrame:
        """
        获取概念板块数据
        :param sort_field: 排序字段，f3=涨跌幅，f62=主力净流入，f84=总市值
        :return: 包含概念板块数据的DataFrame
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }
        
        timestamp = int(time.time() * 1000)
        
        # 概念板块数据API
        fields = 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f62,f184,f66,f69,f72,f75,f78,f81,f82,f84,f85,f86,f87,f124'
        url = f'http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid={sort_field}&fs=m:90+t:3+f:!50&fields={fields}&_={timestamp}'

        try:
            resp = requests.get(url=url, headers=headers)
            resp_json = resp.json()
            
            if 'data' in resp_json and resp_json['data'] is not None:
                df = pd.DataFrame(resp_json['data']['diff'])
                
                if not df.empty:
                    # 选择并重命名列
                    df = df[['f12', 'f14', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f15', 'f16', 'f17', 'f18', 'f20', 'f21', 'f62', 'f184', 'f84', 'f124']]
                    
                    rename_dict = {
                        'f12': '概念代码',
                        'f14': '概念名称', 
                        'f2': '最新价',
                        'f3': '涨跌幅',
                        'f4': '涨跌额',
                        'f5': '成交量',
                        'f6': '成交额',
                        'f7': '振幅',
                        'f15': '最高价',
                        'f16': '最低价',
                        'f17': '开盘价',
                        'f18': '昨收价',
                        'f20': '总市值',
                        'f21': '流通市值',
                        'f62': '主力净流入',
                        'f184': '主力净流入占比',
                        'f84': '总股本',
                        'f124': '更新时间'
                    }
                    
                    df.rename(columns=rename_dict, inplace=True)
                    
                    # 转换数据类型
                    numeric_cols = ['最新价', '涨跌幅', '涨跌额', '成交量', '成交额', '振幅', '最高价', '最低价', '开盘价', '昨收价', '总市值', '流通市值', '主力净流入', '主力净流入占比', '总股本']
                    df[numeric_cols] = df[numeric_cols].astype(float)
                    
                    return df
            
            print('获取概念板块数据失败')
            return pd.DataFrame()
            
        except Exception as e:
            print(f'获取概念板块数据出错：{e}')
            return pd.DataFrame()

    def get_stock_industry_info_from_eastmoney(self, stock_code: str) -> dict:
        """
        获取个股所属行业和概念信息
        :param stock_code: 股票代码，如 'sh600519' 或 '600519'
        :return: 包含行业和概念信息的字典
        """
        if len(stock_code) == 6:
            symbol = stock_code
        elif len(stock_code) == 8:
            symbol = stock_code[2:]
        else:
            return {}

        if symbol[0] == '6':
            market = '1'
        elif symbol[0] == '0' or symbol[0] == '3':
            market = '0'
        else:
            market = '0'
        secid = market + '.' + symbol

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }

        timestamp = int(time.time() * 1000)
        
        # 获取股票详细信息，包含行业和概念
        url = f'http://push2.eastmoney.com/api/qt/stock/get?secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields=f57,f58,f84,f85,f86,f87,f127,f116,f117&_={timestamp}'

        try:
            resp = requests.get(url=url, headers=headers)
            resp_json = resp.json()
            
            if 'data' in resp_json and resp_json['data'] is not None:
                data = resp_json['data']
                
                stock_info = {
                    '股票代码': stock_code,
                    '股票名称': data.get('f58', ''),
                    '所属行业': data.get('f127', ''),
                    '概念板块': data.get('f116', ''),
                    '地区': data.get('f117', ''),
                    '总股本': data.get('f84', 0),
                    '流通股': data.get('f85', 0),
                    '每股收益': data.get('f86', 0),
                    '每股净资产': data.get('f87', 0)
                }
                
                return stock_info
            
            print(f'获取{stock_code}行业概念信息失败')
            return {}
            
        except Exception as e:
            print(f'获取{stock_code}行业概念信息出错：{e}')
            return {}

    def get_hot_concepts_from_eastmoney(self, limit: int = 50) -> pd.DataFrame:
        """
        获取热门概念板块排行
        :param limit: 返回数量限制
        :return: 包含热门概念的DataFrame
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }
        
        timestamp = int(time.time() * 1000)
        
        # 热门概念API（按涨跌幅排序）
        fields = 'f12,f14,f2,f3,f4,f5,f6,f7,f62,f184,f104,f105,f140,f141,f136'
        url = f'http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz={limit}&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:90+t:3+f:!50&fields={fields}&_={timestamp}'

        try:
            resp = requests.get(url=url, headers=headers)
            resp_json = resp.json()
            
            if 'data' in resp_json and resp_json['data'] is not None:
                df = pd.DataFrame(resp_json['data']['diff'])
                
                if not df.empty:
                    # 选择并重命名列
                    df = df[['f12', 'f14', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f62', 'f184', 'f104', 'f105']]
                    
                    rename_dict = {
                        'f12': '概念代码',
                        'f14': '概念名称', 
                        'f2': '最新指数',
                        'f3': '涨跌幅',
                        'f4': '涨跌额',
                        'f5': '成交量',
                        'f6': '成交额',
                        'f7': '振幅',
                        'f62': '主力净流入',
                        'f184': '主力净流入占比',
                        'f104': '上涨家数',
                        'f105': '下跌家数'
                    }
                    
                    df.rename(columns=rename_dict, inplace=True)
                    
                    # 转换数据类型
                    numeric_cols = ['最新指数', '涨跌幅', '涨跌额', '成交量', '成交额', '振幅', '主力净流入', '主力净流入占比', '上涨家数', '下跌家数']
                    df[numeric_cols] = df[numeric_cols].astype(float)
                    
                    return df
            
            print('获取热门概念数据失败')
            return pd.DataFrame()
            
        except Exception as e:
            print(f'获取热门概念数据出错：{e}')
            return pd.DataFrame()

    def get_stocks_by_concept_from_eastmoney(self, concept_code: str) -> pd.DataFrame:
        """
        获取指定概念板块的成分股
        :param concept_code: 概念板块代码
        :return: 包含成分股数据的DataFrame
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
        }
        
        timestamp = int(time.time() * 1000)
        
        # 概念成分股API
        fields = 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152'
        url = f'http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=2000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=b:{concept_code}+f:!50&fields={fields}&_={timestamp}'

        try:
            resp = requests.get(url=url, headers=headers)
            resp_json = resp.json()
            
            if 'data' in resp_json and resp_json['data'] is not None:
                df = pd.DataFrame(resp_json['data']['diff'])
                
                if not df.empty:
                    # 选择并重命名列
                    df = df[['f12', 'f14', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f15', 'f16', 'f17', 'f18', 'f20', 'f21', 'f62']]
                    
                    rename_dict = {
                        'f12': '股票代码',
                        'f14': '股票名称', 
                        'f2': '最新价',
                        'f3': '涨跌幅',
                        'f4': '涨跌额',
                        'f5': '成交量',
                        'f6': '成交额',
                        'f7': '振幅',
                        'f15': '最高价',
                        'f16': '最低价',
                        'f17': '开盘价',
                        'f18': '昨收价',
                        'f20': '总市值',
                        'f21': '流通市值',
                        'f62': '主力净流入'
                    }
                    
                    df.rename(columns=rename_dict, inplace=True)
                    
                    # 添加交易所前缀
                    df['股票代码'] = df['股票代码'].apply(self.symbol_to_stock_code)
                    
                    # 转换数据类型
                    numeric_cols = ['最新价', '涨跌幅', '涨跌额', '成交量', '成交额', '振幅', '最高价', '最低价', '开盘价', '昨收价', '总市值', '流通市值', '主力净流入']
                    df[numeric_cols] = df[numeric_cols].astype(float)
                    
                    return df
            
            print(f'获取概念{concept_code}成分股数据失败')
            return pd.DataFrame()
            
        except Exception as e:
            print(f'获取概念{concept_code}成分股数据出错：{e}')
            return pd.DataFrame()


class Common_functions():
    def __init__(self):
        pass

    # 导入某文件夹下所有文件的文件名
    @staticmethod
    def get_file_list_in_one_dir(path, filetype='.csv'):
        """
        从指定文件夹下，导入所有csv文件的文件名
        :param path: 文件目录
        :param filetype: 文件类型,默认.csv 比如 .csv , .xlsx
        :return: 返回所有文件名列表
        """
        file_list = []

        # 系统自带函数os.walk，用于遍历文件夹中的所有文件
        for root, dirs, files in os.walk(path):
            if files:  # 当files不为空的时候
                for f in files:
                    if f.endswith(filetype):
                        file_list.append(f[:8])

        return sorted(file_list)


if __name__ == '__main__':
    # 下载A股列表
    # D_stock_lst_1 = Download_stocks_list()
    # D_stock_lst_1.main()

    # 下载股票历史行情数据
    s_f_1 = Spider_func()
    # df = s_f_1.get_stock_history_data_from_eastmoney(stock_code='001212')
    # print(df)
    # s_f_1.download_all_stocks_history_kline_from_em()

    # 从通达信获取实时行情盘口数据
    # tdx_1 = Tdx_datas()
    # df = tdx_1.get_stocks_quote(stock_lst=['600519', '000001'])
    # print(df)

    # 获取全量A股实时行情
    s_f_1 = Spider_func()
    df = s_f_1.get_recent_all_stock_kline_data_from_em()
    print(df)

    # 从通达信获取历史财报数据
    # tdx_f_1 = Tdx_datas()
    # 输入报告期参数下载更新并存储某个财报期的所有个股财报，不填入参则循环下载所有历史财报
    # tdx_f_1.get_history_financial_reports_to_csv()

    pass
