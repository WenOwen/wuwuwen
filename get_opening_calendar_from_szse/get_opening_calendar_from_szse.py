# -*-coding:utf-8-*-
"""
获取A股开始和休市安排
深交所官网 http://www.szse.cn/aboutus/calendar/
api：http://www.szse.cn/api/report/exchange/onepersistenthour/monthList?month=2022-04&random=0.35667885796333776
"""
import time
import os
import pandas as pd
import requests


class Get_stock_opening_calendar():
    def __init__(self):
        # 获取该class类所在.py文件的绝对目录
        self.file_full_dir = os.path.dirname(os.path.abspath(__file__))
        # 交易日历文件路径
        self.stock_trade_calendar_path = self.file_full_dir + '\china_stock_trade_calendar.csv'

        # 当前日期
        self.today = time.strftime('%Y%m%d')
        # 检查新的年份交易日历是否已更新
        self.get_opening_calendar()
        pass

    def get_opening_calendar(self, year_num=None):
        month_lst = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        # 获取当前年份
        if year_num == None:
            year = time.localtime().tm_year
        else:
            year = year_num
        if os.path.exists(self.stock_trade_calendar_path):
            df = pd.read_csv(self.stock_trade_calendar_path, parse_dates=['交易日期'])
            year_lst = df['交易日期'].dt.year.to_list()
            year_lst = list(set(year_lst))
            if year in year_lst:
                # print('无需更新!')
                return
            else:
                all_df = pd.DataFrame()
                for month in month_lst:
                    print('{}年{}月'.format(year, month))
                    url = 'http://www.szse.cn/api/report/exchange/onepersistenthour/monthList?month={}-{}&random=0.35667885796333776'.format(
                        year, month)
                    headers = {
                        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'}
                    resp = requests.get(url=url, headers=headers)
                    if resp.status_code == 200:
                        resp_json = resp.json()
                        df = pd.DataFrame(resp_json['data'])
                        df.rename(columns={'jyrq': '交易日期', 'zrxh': '星期', 'jybz': '是否开市'}, inplace=True)
                        df = df[['交易日期', '是否开市', '星期']]
                        all_df = pd.concat([all_df, df], ignore_index=True)
                    time.sleep(1)
                all_df.to_csv(self.stock_trade_calendar_path, mode='a', header=False, encoding='utf-8', index=False)
                print('获取并追加成功！')
        else:
            all_df = pd.DataFrame()
            for month in month_lst:
                print('{}年{}月'.format(year, month))
                url = 'http://www.szse.cn/api/report/exchange/onepersistenthour/monthList?month={}-{}&random=0.35667885796333776'.format(
                    year, month)
                headers = {
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'}
                resp = requests.get(url=url, headers=headers)
                if resp.status_code == 200:
                    resp_json = resp.json()
                    df = pd.DataFrame(resp_json['data'])
                    df.rename(columns={'jyrq': '交易日期', 'zrxh': '星期', 'jybz': '是否开市'}, inplace=True)
                    df = df[['交易日期', '是否开市', '星期']]
                    all_df = pd.concat([all_df, df], ignore_index=True)
                time.sleep(1)
            all_df.to_csv(self.stock_trade_calendar_path, mode='a', header=True, encoding='utf-8', index=False)
            print('获取并追加成功！')
            return all_df

    def get_trade_date_after_n(self, date_start, n: 0):
        """
        获取某个交易日后n天后的交易日期
        """
        df_trade_calendar = pd.read_csv(self.stock_trade_calendar_path, encoding='utf-8', parse_dates=['交易日期'])
        df_trade_calendar = df_trade_calendar[df_trade_calendar['是否开市'] == 1]
        df_trade_calendar.reset_index(drop=True, inplace=True)
        i = df_trade_calendar[df_trade_calendar['交易日期'] == pd.to_datetime(date_start)].index
        i_n = i + n
        # print(i, i_n)
        date_afer_n = df_trade_calendar.loc[i_n, '交易日期'].iloc[0]
        # 转为字符串格式日期
        date_afer_n = date_afer_n.strftime('%Y%m%d')
        return date_afer_n

    def get_trade_date_forward_n(self, date_start, n: 0):
        """
        获取某个交易日前n天的交易日期
        """
        df_trade_calendar = pd.read_csv(self.stock_trade_calendar_path, encoding='utf-8', parse_dates=['交易日期'])
        df_trade_calendar = df_trade_calendar[df_trade_calendar['是否开市'] == 1]
        df_trade_calendar.reset_index(drop=True, inplace=True)
        i = df_trade_calendar[df_trade_calendar['交易日期'] == pd.to_datetime(date_start)].index
        i_n = i - n
        # print(i, i_n)
        date_forward_n = df_trade_calendar.loc[i_n, '交易日期'].iloc[0]
        # 转为字符串格式日期
        date_forward_n = date_forward_n.strftime('%Y%m%d')
        return date_forward_n

    def get_recent_trade_date(self, date=None):
        """
        根据深交所交易日历，获取最近的开市日期
        """
        if date == None:
            today = time.strftime('%Y%m%d')
            df_trade_calendar = pd.read_csv(self.stock_trade_calendar_path, encoding='utf-8', parse_dates=['交易日期'])
            # 计算当年的第一个交易日
            # first_trade_day = df_trade_calendar[(df_trade_calendar['是否开市'] == 1)]['交易日期'].min().strftime('%Y%m%d')
            if (pd.to_datetime(time.strftime('%H:%M:%S')) >= pd.to_datetime('09:15:00')):
                df_trade_calendar = df_trade_calendar[
                    (df_trade_calendar['是否开市'] == 1) & (df_trade_calendar['交易日期'] <= pd.to_datetime(today))]
            elif (pd.to_datetime(time.strftime('%H:%M:%S')) < pd.to_datetime('09:15:00')):
                df_trade_calendar = df_trade_calendar[
                    (df_trade_calendar['是否开市'] == 1) & (df_trade_calendar['交易日期'] < pd.to_datetime(today))]
            df_trade_calendar.reset_index(drop=True, inplace=True)

            if df_trade_calendar.empty == False:
                recent_trade_date = df_trade_calendar['交易日期'].max().strftime('%Y%m%d')
                return recent_trade_date
        else:
            df_trade_calendar = pd.read_csv(self.stock_trade_calendar_path, encoding='utf-8', parse_dates=['交易日期'])
            df_trade_calendar = df_trade_calendar[
                (df_trade_calendar['是否开市'] == 1) & (df_trade_calendar['交易日期'] <= pd.to_datetime(date))]
            df_trade_calendar.reset_index(drop=True, inplace=True)
            recent_trade_date = df_trade_calendar['交易日期'].max().strftime('%Y%m%d')
            return recent_trade_date

    def is_tradable_day(self, date=None):
        if date == None:
            date = time.strftime('%Y%m%d')
        else:
            date = pd.to_datetime(date).strftime('%Y%m%d')
        df_trade_calendar = pd.read_csv(self.stock_trade_calendar_path, encoding='utf-8', parse_dates=['交易日期'])
        tradable = df_trade_calendar[df_trade_calendar['交易日期'] == pd.to_datetime(date)]['是否开市'].iloc[0]
        return tradable

    def get_trade_date_lst(self):
        # 获取交易日期表
        df_trade_canlerdar = pd.read_csv(self.stock_trade_calendar_path)
        df_trade_canlerdar = df_trade_canlerdar[df_trade_canlerdar['是否开市'] == 1]
        # 将日期格式2022-01-05改为20220105
        df_trade_canlerdar['交易日期'] = df_trade_canlerdar['交易日期'].str.replace('-', '')
        trade_canlerdar_lst = df_trade_canlerdar['交易日期'].tolist()
        return trade_canlerdar_lst

    # 获取当前交易日向前n个交易日的列表
    def get_rencent_n_trade_days(self, n=30):
        trade_canlerdar_lst = self.get_trade_date_lst()
        # print(trade_canlerdar_lst)
        rencent_date = self.get_recent_trade_date()
        trade_date_forward_n = self.get_trade_date_forward_n(date_start=rencent_date, n=n)
        # 列表推导式 获取当前交易日向前n个交易日的列表
        new_lst = [date for date in trade_canlerdar_lst if date >= trade_date_forward_n and date <= rencent_date]
        print(new_lst)
        return new_lst

    def how_n_days_ago_by_date(self, start_date='20230101'):
        trade_canlerdar_lst = self.get_trade_date_lst()
        # print(trade_canlerdar_lst)
        rencent_date = self.get_recent_trade_date()
        new_lst = [date for date in trade_canlerdar_lst if date<=rencent_date and date>=start_date]
        # print(new_lst)
        n_days_ago = len(new_lst)
        return n_days_ago

if __name__ == '__main__':
    g_calendar = Get_stock_opening_calendar()
    # 获取当前交易日向前n个交易日的列表
    # g_calendar.get_rencent_n_trade_days()

    # 根据日期 计算该日期距离最近交易日的天数
    # n_days_ago = g_calendar.how_n_days_ago_by_date()
    # print(n_days_ago)

    # 当日或某日是否为交易日
    a = g_calendar.is_tradable_day()
    print(a)
    # 获取A股交易日历并存储为csv
    # df_cal = g_calendar.get_opening_calendar(year_num='2025')
    # print(df_cal)

    # df_trade_date_lst = g_calendar.get_trade_date_lst()
    # print(df_trade_date_lst)
    # a = g_calendar.get_recent_trade_date(date='20230101')
    # print(a)
    # d_r = g_calendar.get_recent_trade_date()
    # print(d_r)
    pass
