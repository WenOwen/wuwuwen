# -*- coding: utf-8 -*-
# @老师微信:samequant
# @网站:打板哥网 www.dabange.com
# @更多源码下载地址: https://dabange.com/download
# @有偿服务：量化课程、量化数据、策略代写、实盘对接...

from tdx import Tdx_datas
# 从通达信获取历史财报数据
tdx_f_1 = Tdx_datas()
# 输入报告期参数下载更新并存储某个财报期的所有个股财报，不填入参则循环下载所有历史财报
tdx_f_1.get_history_financial_reports_to_csv()
# exit()

# 获取某个股最新财务报告
df_f = tdx_f_1.get_financial_report('600000')
# print(df_f)

# 根据财务报告期读取所有个股的财务报表
df_a = tdx_f_1.read_all_stocks_financial_report_by_period(report_date='20240331')
print(df_a.head(10))


if __name__ == '__main__':
    # 同样，这也是需要定时程序，每日17:00点运行更新
    import schedule
    schedule.every().day.at('17:00').do(tdx_f_1.get_history_financial_reports_to_csv)
    pass