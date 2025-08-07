# -*- coding: utf-8 -*-
# @站长微|信:samequant
# @网站:打板哥网 www.dabange.com
# @更多源码下载地址: https://dabange.com/download
# @有偿服务：量化交易训练营入门课程、量化源码、量化数据、策略代写、实盘对接...
# 导入通达信行情源
# pip install pytdx==1.72
import pandas as pd
import os
import time
from pytdx.hq import TdxHq_API
from pytdx.config.hosts import hq_hosts
from pytdx.crawler.history_financial_crawler import HistoryFinancialListCrawler, HistoryFinancialCrawler
from pytdx.crawler.base_crawler import demo_reporthook

class Tdx_datas():
    def __init__(self):
        # 获取该class类所在.py文件的绝对目录
        self.file_full_dir = os.path.dirname(os.path.abspath(__file__))
        df = pd.DataFrame(hq_hosts)
        # 设置数据标题栏列名
        df.columns = ['行情源名', 'IP', '端口号']
        # 设置行情源为：招商证券深圳行情,119.147.212.81,7709
        self.hq_name = df.iat[25, 0]  # 行情源名称
        self.hq_ip = df.iat[25, 1]  # 行情源ip地址
        self.hq_port = df.iat[25, 2]  # 行情源端口号
        print("当前行情源信息：", self.hq_name, self.hq_ip, self.hq_port)
        self.api = TdxHq_API()

        # 本机通达信的安装根目录
        self.tdx_root = "D:/new_dongb_v6"

        # 板块相关参数
        BLOCK_SZ = "block_zs.dat"
        BLOCK_FG = "block_fg.dat"
        BLOCK_GN = "block_gn.dat"
        BLOCK_DEFAULT = "block.dat"

        # 查看各列数据类型
        # print(df.dtypes)

    def return_market(self, stock_code: str):
        if stock_code[0] == '6':
            market = 1
        else:
            market = 0
        return market

    def get_stock_code_quote(self, stock_symbol: str):
        """
        根据股票代码获取股票实时行情
        """
        # 连接行情源
        with self.api.connect(ip=self.hq_ip, port=self.hq_port):
            lst = []
            if stock_symbol[0] == '6':
                lst.append((1, stock_symbol))
            elif stock_symbol[0] == '0' or stock_symbol[0] == '3':
                lst.append((0, stock_symbol))
            elif stock_symbol[0] == '4' or stock_symbol[0] == '8':
                lst.append((2, stock_symbol))
            print(lst)
            # 获取多只股票的最新行情信息 0 代表深交所 1 上交所
            data = self.api.get_security_quotes(lst)
            df = self.api.to_df(data)
            # print(df.columns)
            df.rename(columns={'market': '市场', 'code': '简码', 'active1': 'active1', 'price': '最新价', 'last_close': '昨收价',
                               'open': '开盘价', 'high': '最高价', 'low': '最低价', 'reversed_bytes0': 'reversed_bytes0',
                               'reversed_bytes1': 'reversed_bytes1', 'vol': '成交量', 'cur_vol': '现量', 'amount': '成交额',
                               's_vol': '主卖量', 'b_vol': '主买量',
                               'reversed_bytes2': 'reversed_bytes2', 'reversed_bytes3': 'reversed_bytes3',
                               'bid1': '买一价', 'ask1': '卖一价',
                               'bid_vol1': '买一量', 'ask_vol1': '卖一量', 'bid2': '买二价', 'ask2': '卖二价', 'bid_vol2': '买二量',
                               'ask_vol2': '卖二量', 'bid3': '买三价', 'ask3': '卖三价', 'bid_vol3': '买三量', 'ask_vol3': '卖三量',
                               'bid4': '买四价', 'ask4': '卖四价',
                               'bid_vol4': '买四量', 'ask_vol4': '卖四量', 'bid5': '买五价', 'ask5': '卖五价', 'bid_vol5': '买五量',
                               'ask_vol5': '卖五量',
                               'reversed_bytes4': 'reversed_bytes4', 'reversed_bytes5': 'reversed_bytes5',
                               'reversed_bytes6': 'reversed_bytes6',
                               'reversed_bytes7': 'reversed_bytes7', 'reversed_bytes8': 'reversed_bytes8',
                               'reversed_bytes9': '涨速',
                               'active2': 'active2'}, inplace=True)
            df['当前时间'] = time.strftime('%Y%m%d %H:%M:%S')
            df = df[['市场', '简码', '最新价', '昨收价', '开盘价', '最高价', '最低价',
                     '成交量', '现量', '成交额', '主卖量', '主买量',
                     '买一价', '卖一价', '买一量', '卖一量', '买二价',
                     '卖二价', '买二量', '卖二量', '买三价', '卖三价', '买三量', '卖三量', '买四价', '卖四价', '买四量',
                     '卖四量', '买五价', '卖五价', '买五量', '卖五量', '涨速', '当前时间']]

        return df

    def get_stocks_quote(self, stock_lst: list):
        """
        根据股票代码获取股票实时行情
        """
        # 连接行情源
        with self.api.connect(ip=self.hq_ip, port=self.hq_port):
            lst = []
            for stock_symbol in stock_lst:
                if stock_symbol[0] == '6':
                    lst.append((1, stock_symbol))
                elif stock_symbol[0] == '0' or stock_symbol[0] == '3':
                    lst.append((0, stock_symbol))
                elif stock_symbol[0] == '4' or stock_symbol[0] == '8':
                    lst.append((2, stock_symbol))
            # print(lst)
            # 获取多只股票的最新行情信息 0 代表深交所 1 上交所
            data = self.api.get_security_quotes(lst)
            df = self.api.to_df(data)
            # print(df.columns)
            df.rename(columns={'market': '市场', 'code': '简码', 'active1': 'active1', 'price': '最新价', 'last_close': '昨收价',
                               'open': '开盘价', 'high': '最高价', 'low': '最低价', 'reversed_bytes0': 'reversed_bytes0',
                               'reversed_bytes1': 'reversed_bytes1', 'vol': '成交量', 'cur_vol': '现量', 'amount': '成交额',
                               's_vol': '主卖量', 'b_vol': '主买量',
                               'reversed_bytes2': 'reversed_bytes2', 'reversed_bytes3': 'reversed_bytes3',
                               'bid1': '买一价', 'ask1': '卖一价',
                               'bid_vol1': '买一量', 'ask_vol1': '卖一量', 'bid2': '买二价', 'ask2': '卖二价', 'bid_vol2': '买二量',
                               'ask_vol2': '卖二量', 'bid3': '买三价', 'ask3': '卖三价', 'bid_vol3': '买三量', 'ask_vol3': '卖三量',
                               'bid4': '买四价', 'ask4': '卖四价',
                               'bid_vol4': '买四量', 'ask_vol4': '卖四量', 'bid5': '买五价', 'ask5': '卖五价', 'bid_vol5': '买五量',
                               'ask_vol5': '卖五量',
                               'reversed_bytes4': 'reversed_bytes4', 'reversed_bytes5': 'reversed_bytes5',
                               'reversed_bytes6': 'reversed_bytes6',
                               'reversed_bytes7': 'reversed_bytes7', 'reversed_bytes8': 'reversed_bytes8',
                               'reversed_bytes9': '涨速',
                               'active2': 'active2'}, inplace=True)
            df['当前时间'] = time.strftime('%Y%m%d %H:%M:%S')
            df = df[['市场', '简码', '最新价', '昨收价', '开盘价', '最高价', '最低价',
                     '成交量', '现量', '成交额', '主卖量', '主买量',
                     '买一价', '卖一价', '买一量', '卖一量', '买二价',
                     '卖二价', '买二量', '卖二量', '买三价', '卖三价', '买三量', '卖三量', '买四价', '卖四价', '买四量',
                     '卖四量', '买五价', '卖五价', '买五量', '卖五量', '涨速', '当前时间']]

        return df

    def float_to_date(self, f):
        """
        将5位或6位的带小数点的日期，转为标准年月日格式
        """
        y_m_d = ''
        i = int(f)
        s = str(i)
        if len(s) == 5:
            y_m_d = '200' + s
        elif len(s) == 6:
            if int(s[0:2]) >= 88:
                y_m_d = '19' + s
            else:
                y_m_d = '20' + s
        return y_m_d

    def symbol_to_stock_code(self, symbol):
        """
        将symbol转为带有交易所标识的股票代码
        :param symbol:
        :return:
        """
        if str(symbol).startswith('6'):
            stock_code = 'sh' + str(symbol)
        elif str(symbol).startswith('0') or str(symbol).startswith('3'):
            stock_code = 'sz' + str(symbol)
        elif str(symbol).startswith('4') or str(symbol).startswith('8') or str(symbol).startswith('9'):
            stock_code = 'bj' + str(symbol)
        else:
            stock_code = str(symbol)
        return stock_code

    def get_history_financial_reports_to_csv(self, report_date=None):
        crawler = HistoryFinancialListCrawler()
        list_data = crawler.fetch_and_parse()
        df_lst = pd.DataFrame(data=list_data)
        print('财务报告源文件列表：{}'.format(df_lst))

        if report_date != None:
            filename_in = 'gpcw{}.zip'.format(str(report_date))
            df_lst = df_lst[df_lst['filename'] == filename_in]
        else:
            pass

        # 遍历下载zip文件和读取zip，最后并存储为csv
        datacrawler = HistoryFinancialCrawler()
        tmp_dir = self.file_full_dir + "/tmp"
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        financial_csv_dir = self.file_full_dir + '/financial_csv'
        if not os.path.exists(financial_csv_dir):
            os.mkdir(financial_csv_dir)

        for i in df_lst.index:
            filename = df_lst.loc[i, 'filename']
            filesize = df_lst.loc[i, "filesize"]
            c_path = financial_csv_dir + '/{}.csv'.format(filename[:-4])
            if i > 4 and os.path.exists(c_path):
                continue

            # 下载.zip财报文件
            path_to_download = tmp_dir + "/{}".format(filename)
            datacrawler.fetch_and_parse(reporthook=demo_reporthook, filename=filename, filesize=filesize,
                                        path_to_download=path_to_download)

            try:
                # 读取财报文件
                with open(path_to_download, "rb") as fp:
                    result = datacrawler.parse(download_file=fp)
                    # print(result)

                def to_df(data):
                    if len(data) == 0:
                        return pd.DataFrame()

                    total_lengh = len(data[0])
                    col = ['code', 'report_date']

                    length = total_lengh - 2
                    for i in range(0, length):
                        col.append("col" + str(i + 1))

                    df = pd.DataFrame(data=data, columns=col)
                    df.set_index('code', inplace=True)
                    return df

                # 转为df
                df_c = to_df(data=result)

                # print(df_c.columns)

                if (len(df_c) >= 1):
                    df_c.reset_index(inplace=True)
                    # 通过打印发现 3个日期列'col313' 'col314' 'col315'格式为float，且位数不足，需特殊处理
                    df_c['col313'] = df_c['col313'].apply(self.float_to_date)
                    df_c['col314'] = df_c['col314'].apply(self.float_to_date)
                    df_c['col315'] = df_c['col315'].apply(self.float_to_date)
                    # 给代码加上交易所前缀
                    df_c['code'] = df_c['code'].apply(self.symbol_to_stock_code)

                    rename_dict = {'code': '代码', 'report_date': '报告期', 'col1': '基本每股收益', 'col2': '扣除非经常性损益每股收益',
                                   'col3': '每股未分配利润',
                                   'col4': '每股净资产', 'col5': '每股资本公积金',
                                   'col6': '净资产收益率', 'col7': '每股经营现金流量', 'col8': '货币资金', 'col9': '交易性金融资产',
                                   'col10': '应收票据',
                                   'col11': '应收账款', 'col12': '预付款项', 'col13': '其他应收款', 'col14': '应收关联公司款',
                                   'col15': '应收利息',
                                   'col16': '应收股利', 'col17': '存货', 'col18': '其中：消耗性生物资产', 'col19': '一年内到期的非流动资产',
                                   'col20': '其他流动资产',
                                   'col21': '流动资产合计', 'col22': '可供出售金融资产', 'col23': '持有至到期投资', 'col24': '长期应收款',
                                   'col25': '长期股权投资',
                                   'col26': '投资性房地产', 'col27': '固定资产', 'col28': '在建工程', 'col29': '工程物资',
                                   'col30': '固定资产清理',
                                   'col31': '生产性生物资产', 'col32': '油气资产', 'col33': '无形资产', 'col34': '开发支出',
                                   'col35': '商誉',
                                   'col36': '长期待摊费用', 'col37': '递延所得税资产', 'col38': '其他非流动资产', 'col39': '非流动资产合计',
                                   'col40': '资产总计',
                                   'col41': '短期借款', 'col42': '交易性金融负债', 'col43': '应付票据', 'col44': '应付账款',
                                   'col45': '预收款项',
                                   'col46': '应付职工薪酬', 'col47': '应交税费', 'col48': '应付利息', 'col49': '应付股利',
                                   'col50': '其他应付款',
                                   'col51': '应付关联公司款', 'col52': '一年内到期的非流动负债', 'col53': '其他流动负债', 'col54': '流动负债合计',
                                   'col55': '长期借款',
                                   'col56': '应付债券', 'col57': '长期应付款', 'col58': '专项应付款', 'col59': '预计负债',
                                   'col60': '递延所得税负债',
                                   'col61': '其他非流动负债', 'col62': '非流动负债合计', 'col63': '负债合计', 'col64': '实收资本（或股本）',
                                   'col65': '资本公积',
                                   'col66': '盈余公积', 'col67': '减：库存股', 'col68': '未分配利润', 'col69': '少数股东权益',
                                   'col70': '外币报表折算价差',
                                   'col71': '非正常经营项目收益调整', 'col72': '所有者权益（或股东权益）合计', 'col73': '负债和所有者（或股东权益）合计',
                                   'col74': '其中：营业收入',
                                   'col75': '其中：营业成本', 'col76': '营业税金及附加', 'col77': '销售费用', 'col78': '管理费用',
                                   'col79': '堪探费用',
                                   'col80': '财务费用', 'col81': '资产减值损失', 'col82': '加：公允价值变动净收益', 'col83': '投资收益',
                                   'col84': '其中：对联营企业和合营企业的投资收益', 'col85': '影响营业利润的其他科目', 'col86': '三、营业利润',
                                   'col87': '加：补贴收入',
                                   'col88': '营业外收入', 'col89': '减：营业外支出', 'col90': '其中：非流动资产处置净损失',
                                   'col91': '加：影响利润总额的其他科目',
                                   'col92': '四、利润总额', 'col93': '减：所得税', 'col94': '加：影响净利润的其他科目', 'col95': '五、净利润',
                                   'col96': '归属于母公司所有者的净利润', 'col97': '少数股东损益', 'col98': '销售商品、提供劳务收到的现金',
                                   'col99': '收到的税费返还',
                                   'col100': '收到其他与经营活动有关的现金', 'col101': '经营活动现金流入小计', 'col102': '购买商品、接受劳务支付的现金',
                                   'col103': '支付给职工以及为职工支付的现金', 'col104': '支付的各项税费', 'col105': '支付其他与经营活动有关的现金',
                                   'col106': '经营活动现金流出小计',
                                   'col107': '经营活动产生的现金流量净额', 'col108': '收回投资收到的现金', 'col109': '取得投资收益收到的现金',
                                   'col110': '处置固定资产、无形资产和其他长期资产收回的现金净额', 'col111': '处置子公司及其他营业单位收到的现金净额',
                                   'col112': '收到其他与投资活动有关的现金',
                                   'col113': '投资活动现金流入小计', 'col114': '购建固定资产、无形资产和其他长期资产支付的现金', 'col115': '投资支付的现金',
                                   'col116': '取得子公司及其他营业单位支付的现金净额', 'col117': '支付其他与投资活动有关的现金',
                                   'col118': '投资活动现金流出小计',
                                   'col119': '投资活动产生的现金流量净额', 'col120': '吸收投资收到的现金', 'col121': '取得借款收到的现金',
                                   'col122': '收到其他与筹资活动有关的现金',
                                   'col123': '筹资活动现金流入小计', 'col124': '偿还债务支付的现金', 'col125': '分配股利、利润或偿付利息支付的现金',
                                   'col126': '支付其他与筹资活动有关的现金', 'col127': '筹资活动现金流出小计', 'col128': '筹资活动产生的现金流量净额',
                                   'col129': '四、汇率变动对现金的影响', 'col130': '四(2)、其他原因对现金的影响',
                                   'col131': '五、现金及现金等价物净增加额',
                                   'col132': '期初现金及现金等价物余额', 'col133': '期末现金及现金等价物余额', 'col134': '净利润',
                                   'col135': '资产减值准备',
                                   'col136': '固定资产折旧、油气资产折耗、生产性生物资产折旧', 'col137': '无形资产摊销', 'col138': '长期待摊费用摊销',
                                   'col139': '处置固定资产、无形资产和其他长期资产的损失', 'col140': '固定资产报废损失', 'col141': '公允价值变动损失',
                                   'col142': '财务费用',
                                   'col143': '投资损失', 'col144': '递延所得税资产减少', 'col145': '递延所得税负债增加',
                                   'col146': '存货的减少',
                                   'col147': '经营性应收项目的减少', 'col148': '经营性应付项目的增加', 'col149': '其他',
                                   'col150': '经营活动产生的现金流量净额2',
                                   'col151': '债务转为资本', 'col152': '一年内到期的可转换公司债券', 'col153': '融资租入固定资产',
                                   'col154': '现金的期末余额',
                                   'col155': '现金的期初余额', 'col156': '现金等价物的期末余额', 'col157': '现金等价物的期初余额',
                                   'col158': '现金及现金等价物净增加额',
                                   'col159': '流动比率', 'col160': '速动比率', 'col161': '现金比率(%)', 'col162': '利息保障倍数',
                                   'col163': '非流动负债比率(%)',
                                   'col164': '流动负债比率(%)', 'col165': '现金到期债务比率(%)', 'col166': '有形资产净值债务率(%)',
                                   'col167': '权益乘数(%)',
                                   'col168': '股东的权益/负债合计(%)', 'col169': '有形资产/负债合计(%)',
                                   'col170': '经营活动产生的现金流量净额/负债合计(%)',
                                   'col171': 'EBITDA/负债合计(%)', 'col172': '应收帐款周转率', 'col173': '存货周转率',
                                   'col174': '运营资金周转率',
                                   'col175': '总资产周转率', 'col176': '固定资产周转率', 'col177': '应收帐款周转天数',
                                   'col178': '存货周转天数',
                                   'col179': '流动资产周转率', 'col180': '流动资产周转天数', 'col181': '总资产周转天数',
                                   'col182': '股东权益周转率',
                                   'col183': '营业收入增长率(%)', 'col184': '净利润增长率(%)', 'col185': '净资产增长率(%)',
                                   'col186': '固定资产增长率(%)',
                                   'col187': '总资产增长率(%)', 'col188': '投资收益增长率(%)', 'col189': '营业利润增长率(%)',
                                   'col190': '暂无',
                                   'col191': '暂无', 'col192': '暂无', 'col193': '成本费用利润率(%)', 'col194': '营业利润率',
                                   'col195': '营业税金率',
                                   'col196': '营业成本率', 'col197': '净资产收益率', 'col198': '投资收益率', 'col199': '销售净利率(%)',
                                   'col200': '总资产报酬率',
                                   'col201': '净利润率', 'col202': '销售毛利率(%)', 'col203': '三费比重', 'col204': '管理费用率',
                                   'col205': '财务费用率',
                                   'col206': '扣除非经常性损益后的净利润', 'col207': '息税前利润(EBIT)',
                                   'col208': '息税折旧摊销前利润(EBITDA)',
                                   'col209': 'EBITDA/营业总收入(%)', 'col210': '资产负债率(%)', 'col211': '流动资产比率',
                                   'col212': '货币资金比率',
                                   'col213': '存货比率', 'col214': '固定资产比率', 'col215': '负债结构比',
                                   'col216': '归属于母公司股东权益/全部投入资本(%)',
                                   'col217': '股东的权益/带息债务(%)', 'col218': '有形资产/净债务(%)', 'col219': '每股经营性现金流(元)',
                                   'col220': '营业收入现金含量(%)',
                                   'col221': '经营活动产生的现金流量净额/经营活动净收益(%)', 'col222': '销售商品提供劳务收到的现金/营业收入(%)',
                                   'col223': '经营活动产生的现金流量净额/营业收入', 'col224': '资本支出/折旧和摊销', 'col225': '每股现金流量净额(元)',
                                   'col226': '经营净现金比率（短期债务）', 'col227': '经营净现金比率（全部债务）',
                                   'col228': '经营活动现金净流量与净利润比率',
                                   'col229': '全部资产现金回收率', 'col230': '营业收入', 'col231': '营业利润',
                                   'col232': '归属于母公司所有者的净利润',
                                   'col233': '扣除非经常性损益后的净利润', 'col234': '经营活动产生的现金流量净额', 'col235': '投资活动产生的现金流量净额',
                                   'col236': '筹资活动产生的现金流量净额', 'col237': '现金及现金等价物净增加额', 'col238': '总股本',
                                   'col239': '已上市流通A股',
                                   'col240': '已上市流通B股', 'col241': '已上市流通H股', 'col242': '股东人数(户)',
                                   'col243': '第一大股东的持股数量',
                                   'col244': '十大流通股东持股数量合计(股)', 'col245': '十大股东持股数量合计(股)', 'col246': '机构总量（家）',
                                   'col247': '机构持股总量(股)',
                                   'col248': 'QFII机构数', 'col249': 'QFII持股量', 'col250': '券商机构数', 'col251': '券商持股量',
                                   'col252': '保险机构数',
                                   'col253': '保险持股量', 'col254': '基金机构数', 'col255': '基金持股量', 'col256': '社保机构数',
                                   'col257': '社保持股量',
                                   'col258': '私募机构数', 'col259': '私募持股量', 'col260': '财务公司机构数', 'col261': '财务公司持股量',
                                   'col262': '年金机构数',
                                   'col263': '年金持股量', 'col264': '十大流通股东中持有A股合计(股)', 'col265': '第一大流通股东持股量(股)',
                                   'col266': '自由流通股(股)',
                                   'col267': '受限流通A股(股)', 'col268': '一般风险准备(金融类)', 'col269': '其他综合收益(利润表)',
                                   'col270': '综合收益总额(利润表)',
                                   'col271': '归属于母公司股东权益(资产负债表)', 'col272': '银行机构数(家)(机构持股)',
                                   'col273': '银行持股量(股)(机构持股)',
                                   'col274': '一般法人机构数(家)(机构持股)', 'col275': '一般法人持股量(股)(机构持股)',
                                   'col276': '近一年净利润(元)',
                                   'col277': '信托机构数(家)(机构持股)', 'col278': '信托持股量(股)(机构持股)',
                                   'col279': '特殊法人机构数(家)(机构持股)',
                                   'col280': '特殊法人持股量(股)(机构持股)', 'col281': '加权净资产收益率(每股指标)',
                                   'col282': '扣非每股收益(单季度财务指标)',
                                   'col283': '最近一年营业收入()', 'col284': '国家队持股数量(万股)', 'col285': '业绩预告-本期净利润同比增幅下限%',
                                   'col286': '业绩预告-本期净利润同比增幅上限%', 'col287': '业绩快报-归母净利润', 'col288': '业绩快报-扣非净利润',
                                   'col289': '业绩快报-总资产',
                                   'col290': '业绩快报-净资产', 'col291': '业绩快报-每股收益', 'col292': '业绩快报-摊薄净资产收益率',
                                   'col293': '业绩快报-加权净资产收益率',
                                   'col294': '业绩快报-每股净资产', 'col295': '应付票据及应付账款(资产负债表)',
                                   'col296': '应收票据及应收账款(资产负债表)',
                                   'col297': '递延收益(资产负债表)', 'col298': '其他综合收益(资产负债表)', 'col299': '其他权益工具(资产负债表)',
                                   'col300': '其他收益(利润表)',
                                   'col301': '资产处置收益(利润表)', 'col302': '持续经营净利润(利润表)', 'col303': '终止经营净利润(利润表)',
                                   'col304': '研发费用(利润表)',
                                   'col305': '其中:利息费用(利润表-财务费用)', 'col306': '其中:利息收入(利润表-财务费用)',
                                   'col307': '近一年经营活动现金流净额',
                                   'col308': '近一年归母净利润', 'col309': '近一年扣非净利润', 'col310': '近一年现金净流量',
                                   'col311': '基本每股收益(单季度)',
                                   'col312': '营业总收入(单季度)', 'col313': '业绩预告公告日期', 'col314': '财报公告日期',
                                   'col315': '业绩快报公告日期',
                                   'col316': '近一年投资活动现金流净额', 'col317': '业绩预告-本期净利润下限', 'col318': '业绩预告-本期净利润上限',
                                   'col319': '营业总收入TTM',
                                   'col320': '员工总数(人)', 'col321': '每股企业自由现金流', 'col322': '每股股东自由现金流',
                                   'col323': '备用323',
                                   'col324': '备用324', 'col325': '备用325', 'col326': '备用326', 'col327': '备用327',
                                   'col328': '备用328',
                                   'col329': '备用329', 'col330': '备用330', 'col331': '备用331', 'col332': '备用332',
                                   'col333': '备用333',
                                   'col334': '备用334', 'col335': '备用335', 'col336': '备用336', 'col337': '备用337',
                                   'col338': '备用338',
                                   'col339': '备用339', 'col340': '备用340', 'col341': '备用341', 'col342': '备用342',
                                   'col343': '备用343',
                                   'col344': '备用344', 'col345': '备用345', 'col346': '备用346', 'col347': '备用347',
                                   'col348': '备用348',
                                   'col349': '备用349', 'col350': '备用350', 'col351': '备用351', 'col352': '备用352',
                                   'col353': '备用353',
                                   'col354': '备用354', 'col355': '备用355', 'col356': '备用356', 'col357': '备用357',
                                   'col358': '备用358',
                                   'col359': '备用359', 'col360': '备用360', 'col361': '备用361', 'col362': '备用362',
                                   'col363': '备用363',
                                   'col364': '备用364', 'col365': '备用365', 'col366': '备用366', 'col367': '备用367',
                                   'col368': '备用368',
                                   'col369': '备用369', 'col370': '备用370', 'col371': '备用371', 'col372': '备用372',
                                   'col373': '备用373',
                                   'col374': '备用374', 'col375': '备用375', 'col376': '备用376', 'col377': '备用377',
                                   'col378': '备用378',
                                   'col379': '备用379', 'col380': '备用380', 'col381': '备用381', 'col382': '备用382',
                                   'col383': '备用383',
                                   'col384': '备用384', 'col385': '备用385', 'col386': '备用386', 'col387': '备用387',
                                   'col388': '备用388',
                                   'col389': '备用389', 'col390': '备用390', 'col391': '备用391', 'col392': '备用392',
                                   'col393': '备用393',
                                   'col394': '备用394', 'col395': '备用395', 'col396': '备用396', 'col397': '备用397',
                                   'col398': '备用398',
                                   'col399': '备用399', 'col400': '备用400', 'col401': '专项储备', 'col402': '结算备付金',
                                   'col403': '拆出资金',
                                   'col404': '发放贷款及垫款', 'col405': '衍生金融资产', 'col406': '应收保费', 'col407': '应收分保账款',
                                   'col408': '应收分保合同准备金',
                                   'col409': '买入返售金融资产', 'col410': '划分为持有待售的资产', 'col411': '发放贷款及垫款',
                                   'col412': '向中央银行借款',
                                   'col413': '吸收存款及同业存放', 'col414': '拆入资金', 'col415': '衍生金融负债',
                                   'col416': '卖出回购金融资产款',
                                   'col417': '应付手续费及佣金', 'col418': '应付分保账款', 'col419': '保险合同准备金',
                                   'col420': '代理买卖证券款',
                                   'col421': '代理承销证券款', 'col422': '划分为持有待售的负债', 'col423': '预计负债', 'col424': '递延收益',
                                   'col425': '其中:优先股',
                                   'col426': '永续债非流动负债科目', 'col427': '长期应付职工薪酬', 'col428': '其中:优先股',
                                   'col429': '永续债所有者权益科目',
                                   'col430': '债权投资', 'col431': '其他债权投资', 'col432': '其他权益工具投资',
                                   'col433': '其他非流动金融资产',
                                   'col434': '合同负债',
                                   'col435': '合同资产', 'col436': '其他资产', 'col437': '应收款项融资', 'col438': '使用权资产',
                                   'col439': '租赁负债',
                                   'col440': '备用440', 'col441': '备用441', 'col442': '备用442', 'col443': '备用443',
                                   'col444': '备用444',
                                   'col445': '备用445', 'col446': '备用446', 'col447': '备用447', 'col448': '备用448',
                                   'col449': '备用449',
                                   'col450': '备用450', 'col451': '备用451', 'col452': '备用452', 'col453': '备用453',
                                   'col454': '备用454',
                                   'col455': '备用455', 'col456': '备用456', 'col457': '备用457', 'col458': '备用458',
                                   'col459': '备用459',
                                   'col460': '备用460', 'col461': '备用461', 'col462': '备用462', 'col463': '备用463',
                                   'col464': '备用464',
                                   'col465': '备用465', 'col466': '备用466', 'col467': '备用467', 'col468': '备用468',
                                   'col469': '备用469',
                                   'col470': '备用470', 'col471': '备用471', 'col472': '备用472', 'col473': '备用473',
                                   'col474': '备用474',
                                   'col475': '备用475', 'col476': '备用476', 'col477': '备用477', 'col478': '备用478',
                                   'col479': '备用479',
                                   'col480': '备用480', 'col481': '备用481', 'col482': '备用482', 'col483': '备用483',
                                   'col484': '备用484',
                                   'col485': '备用485', 'col486': '备用486', 'col487': '备用487', 'col488': '备用488',
                                   'col489': '备用489',
                                   'col490': '备用490', 'col491': '备用491', 'col492': '备用492', 'col493': '备用493',
                                   'col494': '备用494',
                                   'col495': '备用495', 'col496': '备用496', 'col497': '备用497', 'col498': '备用498',
                                   'col499': '备用499',
                                   'col500': '备用500', 'col501': '稀释每股收益', 'col502': '营业总收入', 'col503': '汇兑收益',
                                   'col504': '其中:归属于母公司综合收益', 'col505': '其中:归属于少数股东综合收益', 'col506': '利息收入',
                                   'col507': '已赚保费',
                                   'col508': '手续费及佣金收入', 'col509': '利息支出', 'col510': '手续费及佣金支出', 'col511': '退保金',
                                   'col512': '赔付支出净额',
                                   'col513': '提取保险合同准备金净额', 'col514': '保单红利支出', 'col515': '分保费用',
                                   'col516': '其中:非流动资产处置利得',
                                   'col517': '信用减值损失', 'col518': '净敞口套期收益', 'col519': '营业总成本', 'col520': '信用减值损失',
                                   'col521': '资产减值损失',
                                   'col522': '备用522', 'col523': '备用523', 'col524': '备用524', 'col525': '备用525',
                                   'col526': '备用526',
                                   'col527': '备用527', 'col528': '备用528', 'col529': '备用529', 'col530': '备用530',
                                   'col531': '备用531',
                                   'col532': '备用532', 'col533': '备用533', 'col534': '备用534', 'col535': '备用535',
                                   'col536': '备用536',
                                   'col537': '备用537', 'col538': '备用538', 'col539': '备用539', 'col540': '备用540',
                                   'col541': '备用541',
                                   'col542': '备用542', 'col543': '备用543', 'col544': '备用544', 'col545': '备用545',
                                   'col546': '备用546',
                                   'col547': '备用547', 'col548': '备用548', 'col549': '备用549', 'col550': '备用550',
                                   'col551': '备用551',
                                   'col552': '备用552', 'col553': '备用553', 'col554': '备用554', 'col555': '备用555',
                                   'col556': '备用556',
                                   'col557': '备用557', 'col558': '备用558', 'col559': '备用559', 'col560': '备用560',
                                   'col561': ':其他原因对现金的影响2',
                                   'col562': '客户存款和同业存放款项净增加额', 'col563': '向中央银行借款净增加额',
                                   'col564': '向其他金融机构拆入资金净增加额',
                                   'col565': '收到原保险合同保费取得的现金', 'col566': '收到再保险业务现金净额', 'col567': '保户储金及投资款净增加额',
                                   'col568': '处置以公允价值计量且其变动计入当期损益的金融资产净增加额', 'col569': '收取利息、手续费及佣金的现金',
                                   'col570': '拆入资金净增加额',
                                   'col571': '回购业务资金净增加额', 'col572': '客户贷款及垫款净增加额', 'col573': '存放中央银行和同业款项净增加额',
                                   'col574': '支付原保险合同赔付款项的现金', 'col575': '支付利息、手续费及佣金的现金', 'col576': '支付保单红利的现金',
                                   'col577': '其中:子公司吸收少数股东投资收到的现金', 'col578': '其中:子公司支付给少数股东的股利利润',
                                   'col579': '投资性房地产的折旧及摊销',
                                   'col580': '信用减值损失'}
                    df_c.rename(columns=rename_dict, inplace=True)

                    # 去掉包含"备用"的无用列名
                    fillter_columns = []
                    for i in df_c.columns:
                        if '备用' in i:
                            continue
                        else:
                            fillter_columns.append(i)
                    df_c = df_c[fillter_columns]

                    c_path = financial_csv_dir + '/{}.csv'.format(filename[:-4])
                    df_c.to_csv(c_path, encoding='utf-8', mode='w', index=False)
                    print('{}.csv存储成功！'.format(filename[:-4]))

                else:
                    print('{}数据为None，跳过！'.format(filename))

            except Exception as e:
                print(e)
            time.sleep(2)
        return df_c

    def add_stock_financial_reports_to_csv(self, report_date):
        try:
            path = os.getcwd() + '/financial_csv/gpcw{}.csv'.format(report_date)
            df = pd.read_csv(path, encoding='utf-8')
            # df.sort_values(by=['报告期'], ascending=True, inplace=True)
        except:
            print('该报告暂无源数据文件，跳出！')
            return
        stock_financial_csv_dir = self.file_full_dir + '/stock_financial_csv'
        if not os.path.exists(stock_financial_csv_dir):
            os.mkdir(stock_financial_csv_dir)
        for stock_code in df['代码']:
            print(stock_code)
            df_i = df[df['代码'] == stock_code]
            path_stock = stock_financial_csv_dir + '/{}.csv'.format(stock_code)

            if os.path.exists(path_stock):
                try:
                    df_history = pd.read_csv(path_stock, encoding='utf-8', parse_dates=['报告期'])
                except:
                    df_history = pd.read_csv(path_stock, encoding='utf-8', parse_dates=['报告期'])
                # 判断该报告期数据是否已存在，存在则跳过
                pd_report_date = pd.to_datetime(report_date)
                if pd_report_date not in df_history['报告期'].tolist():
                    df_i.to_csv(path_stock, mode='a', header=False, encoding='utf-8', index=False)
                    print('{}报告期{}财务报告追加成功！'.format(stock_code, report_date))
                else:
                    pass
            else:
                df_i.to_csv(path_stock, mode='a', encoding='utf-8', index=False)

    def get_financial_report(self, stock_code):
        """
        根据个股代码返回个股财报
        :param stock_code:
        :return:
        """
        market = self.return_market(stock_code=stock_code)
        with self.api.connect(ip=self.hq_ip, port=self.hq_port):
            # 获取最新财务数据
            financial_statement = self.api.get_finance_info(market, stock_code)
            financial_statement = str(financial_statement)[12:-1]
            # print(type(financial_statement), financial_statement)
            # eval函数将字符串转python格式，然后转为df
            df = pd.DataFrame(eval(financial_statement)).T

            # 将第一行设为索引
            df.columns = df.loc[0, :]
            # 删除第一行
            df = df[1:]
            df.reset_index(drop=True, inplace=True)

            rename_dict = {'market': '市场', 'code': '股票代码', 'liutongguben': '流通股本', 'province': '省份编号', 'industry': '行业',
                           'updated_date': '更新日期', 'ipo_date': '上市日期', 'zongguben': '总股本', 'guojiagu': '国家股',
                           'faqirenfarengu': '发起人法人股',
                           'farengu': '法人股', 'bgu': 'B股', 'hgu': '港股', 'zhigonggu': '职工股', 'zongzichan': '总资产',
                           'liudongzichan': '流通资产',
                           'gudingzichan': '固定资产', 'wuxingzichan': '无形资产', 'gudongrenshu': '股东人数',
                           'liudongfuzhai': '流动负债',
                           'changqifuzhai': '长期负债', 'zibengongjijin': '资本公积金', 'jingzichan': '净资产',
                           'zhuyingshouru': '主营业务收入',
                           'zhuyinglirun': '主营业务利润', 'yingshouzhangkuan': '应收账款', 'yingyelirun': '营业利润',
                           'touzishouyu': '投资收益',
                           'jingyingxianjinliu': '经营现金流', 'zongxianjinliu': '总现金流', 'cunhuo': '存货',
                           'lirunzonghe': '利润总额',
                           'shuihoulirun': '税后利润', 'jinglirun': '净利润', 'weifenpeilirun': '未分配利润',
                           'meigujingzichan': '每股净资产',
                           'baoliu2': '保留2'}
            df.rename(columns=rename_dict, inplace=True)
            # print(df.columns)
            df = df[['市场', '股票代码', '流通股本', '省份编号', '行业', '更新日期', '上市日期',
                     '总股本', '国家股', '发起人法人股', '法人股', 'B股', '港股', '职工股',
                     '总资产', '流通资产', '固定资产', '无形资产', '股东人数', '流动负债',
                     '长期负债', '资本公积金', '净资产', '主营业务收入', '主营业务利润',
                     '应收账款', '营业利润', '投资收益', '经营现金流', '总现金流', '存货',
                     '利润总额', '税后利润', '净利润', '未分配利润', '每股净资产']]

        return df

    def get_company_information_content(self, stock_code):
        """
        查询公司信息目录  和 公司概况
        """
        market = self.return_market(stock_code=stock_code)

        with self.api.connect(ip=self.hq_ip, port=self.hq_port):
            data = self.api.get_company_info_category(market, stock_code)
            df = pd.DataFrame(data)
            print(df)

            # 读取公司概况
            df_c = df.copy()
            df_c = df_c[df_c['name'] == '公司概况'].iloc[0]
            filename = df_c['filename']
            start = df_c['start']
            length = df_c['length']
            data1 = self.api.get_company_info_content(market, stock_code, filename, start, length)
            print(data1)

    def read_financial_report_by_period(self, report_date=None, sign: str = None):
        """
        # 读取某报告期财报数据
        :param report_date:
        :param sign: 财报预告及该报告期下个报告期的财务预告；
        :return:
        """
        path = self.file_full_dir + '/financial_csv/gpcw{}.csv'.format(report_date)
        df_r = pd.read_csv(path, encoding='utf-8', parse_dates=['报告期', '业绩预告公告日期', '财报公告日期', '业绩快报公告日期'])
        # 自定义用到的列名
        use_cols = ['代码', '报告期', '五、净利润', '净利润增长率(%)', '经营活动现金净流量与净利润比率', '近一年归母净利润', '归属于母公司股东权益(资产负债表)', '基本每股收益',
                    '扣除非经常性损益每股收益', '每股未分配利润', '每股净资产',
                    '每股资本公积金', '营业总收入TTM', '每股经营现金流量',
                    '净资产收益率', '加权净资产收益率(每股指标)', '净利润率',
                    '销售毛利率(%)', '机构总量（家）', '商誉', '业绩预告公告日期', '财报公告日期', '业绩快报公告日期', '业绩预告-本期净利润同比增幅下限%',
                    '业绩预告-本期净利润同比增幅上限%', '业绩预告-本期净利润下限', '业绩预告-本期净利润上限', '总股本', '已上市流通A股']
        df_r = df_r[use_cols]
        df_r.rename(columns={'代码': '股票代码'}, inplace=True)

        if sign == None:
            return df_r
        elif sign == '财报预告':
            df_r = df_r[df_r['业绩预告公告日期'].notnull()]
            df_r.sort_values(by=['业绩预告公告日期'], ascending=0, inplace=True)
            df_r.reset_index(drop=True, inplace=True)

        return df_r

    def read_all_stocks_financial_report_by_period(self, report_date=None):
        """
        # 读取某报告期财报数据
        :param report_date:
        :param sign: 财报预告及该报告期下个报告期的财务预告；
        :return:
        """
        path = self.file_full_dir + '/financial_csv/gpcw{}.csv'.format(report_date)
        df_r = pd.read_csv(path, encoding='utf-8', parse_dates=['报告期', '业绩预告公告日期', '财报公告日期', '业绩快报公告日期'])

        return df_r

if __name__ == '__main__':
    get_stock_code_quote()