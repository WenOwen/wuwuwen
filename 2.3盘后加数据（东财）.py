from samequant_functions import *

# 方式一：合并东财
import pandas as pd

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 6000)  # 最多显示数据的行数
pd.set_option('display.float_format', lambda x: '%.2f' % x)  # 禁用科学计数法


def get_stock_data(s_f_1, stock_code):
    """
    获取股票历史数据
    """
    # 获取股票历史数据
    df = s_f_1.get_stock_history_data_from_eastmoney(stock_code=stock_code)
    return df

def add_to_stocks_history_data():
    # 获取全量A股实时行情
    s_f_1 = Spider_func()
    df = s_f_1.get_recent_all_stock_kline_data_from_em()
    # print(df)
    # exit()
    # 盘后循环更新行情到历史数据
    for i in df.index:
        df_i = df[df.index == i]
        # 确保包含所有必要的列
        required_columns = ['交易日期', '股票代码', '股票名称', '开盘价', '收盘价', '最高价', '最低价', '前收盘价', '成交量', '成交额', '振幅', '涨跌额', '涨跌幅', '换手率', '总市值', '流通市值']
        available_columns = [col for col in required_columns if col in df_i.columns]
        df_i = df_i[available_columns]
        
        # print(df_i)
        trade_date = df.loc[i, '交易日期']
        # 读取历史行情数据
        stock_code = df.loc[i, '股票代码']
        # print(stock_code)
        his_path = 'datas_em/{}.csv'.format(stock_code)  # 修改为datas_em目录
        if os.path.exists(his_path):
            print(stock_code)
            df_his = pd.read_csv(his_path)
            if not df_his.empty:
                # 确保历史数据包含相同的列
                df_his = df_his[available_columns]

            tail1_date = df_his['交易日期'].iloc[-1]
            # print(trade_date, tail1_date)
            if trade_date > tail1_date:
                # print(df_i)
                df_all = pd.concat(objs=[df_his, df_i], ignore_index=False)
                # print(df_all)
                df_all.to_csv(his_path, mode='w', index=False)
        else:
            # 如果文件不存在，获取完整的历史数据
            print(f"文件不存在，获取完整历史数据: {stock_code}")
            df_complete = get_stock_data(s_f_1, stock_code)
            if not df_complete.empty:
                df_complete.to_csv(his_path, mode='w', index=False)
                print(f"✅ 已创建完整历史数据文件: {his_path}")
    
    # 更新板块数据
    print("\n=== 更新板块数据 ===")
    update_sector_data(s_f_1)

def update_sector_data(s_f_1):
    """
    更新板块数据
    """
    file_full_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 获取行业板块数据
    df_industry = s_f_1.get_industry_data_from_eastmoney(sort_field='f3')
    if not df_industry.empty:
        industry_path = file_full_dir + '/datas_em/行业板块数据.csv'
        df_industry.to_csv(industry_path, index=False, encoding='utf-8')
        print(f"✅ 行业板块数据已更新: {industry_path}")

    # 获取概念板块数据
    df_concept = s_f_1.get_concept_data_from_eastmoney(sort_field='f3')
    if not df_concept.empty:
        concept_path = file_full_dir + '/datas_em/概念板块数据.csv'
        df_concept.to_csv(concept_path, index=False, encoding='utf-8')
        print(f"✅ 概念板块数据已更新: {concept_path}")

    # 获取热门概念数据
    df_hot_concepts = s_f_1.get_hot_concepts_from_eastmoney(limit=50)
    if not df_hot_concepts.empty:
        hot_concepts_path = file_full_dir + '/datas_em/热门概念数据.csv'
        df_hot_concepts.to_csv(hot_concepts_path, index=False, encoding='utf-8')
        print(f"✅ 热门概念数据已更新: {hot_concepts_path}")


if __name__ == '__main__':
    # 交易日盘后追加当时行情到历史行情数据(前提是之前以下载过所有个股完整的历史行情数据)
    add_to_stocks_history_data()

    # 实盘时，每日15:31定时运行
    import schedule
    schedule.every().day.at('15:31').do(add_to_stocks_history_data)