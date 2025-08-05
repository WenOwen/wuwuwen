from samequant_functions import Spider_func

# 初始化Spider_func实例
s_f_1 = Spider_func()

print("=== 获取行业板块数据（按涨跌幅排序） ===")
df_industry = s_f_1.get_industry_data_from_eastmoney(sort_field='f3')
if not df_industry.empty:
    print(f"行业板块数据行数: {len(df_industry)}")
    print("涨幅排行前10的行业:")
    print(df_industry[['行业名称', '涨跌幅', '主力净流入', '成交额', '总市值']].head(10))
    print()

print("=== 获取概念板块数据（按涨跌幅排序） ===")
df_concept = s_f_1.get_concept_data_from_eastmoney(sort_field='f3')
if not df_concept.empty:
    print(f"概念板块数据行数: {len(df_concept)}")
    print("涨幅排行前10的概念:")
    print(df_concept[['概念名称', '涨跌幅', '主力净流入', '成交额', '总市值', '流通市值']].head(10))
    print()

print("=== 获取热门概念板块排行 ===")
df_hot_concepts = s_f_1.get_hot_concepts_from_eastmoney(limit=20)
if not df_hot_concepts.empty:
    print(f"热门概念数据行数: {len(df_hot_concepts)}")
    print("热门概念排行前20:")
    print(df_hot_concepts[['概念名称', '涨跌幅', '主力净流入', '上涨家数', '下跌家数']])
    print()

print("=== 获取个股行业概念信息 ===")
stock_codes = ['sh600519', 'sz000001', 'sz000002']
for stock_code in stock_codes:
    stock_info = s_f_1.get_stock_industry_info_from_eastmoney(stock_code=stock_code)
    if stock_info:
        print(f"{stock_info['股票名称']}({stock_info['股票代码']}):")
        print(f"  所属行业: {stock_info['所属行业']}")
        print(f"  概念板块: {stock_info['概念板块']}")
        print(f"  地区: {stock_info['地区']}")
        print(f"  总股本: {stock_info['总股本']}")
        print(f"  流通股: {stock_info['流通股']}")
        print()

# 获取一个热门概念的成分股
if not df_hot_concepts.empty:
    top_concept_code = df_hot_concepts.iloc[0]['概念代码']
    top_concept_name = df_hot_concepts.iloc[0]['概念名称']
    
    print(f"=== 获取{top_concept_name}概念的成分股 ===")
    df_concept_stocks = s_f_1.get_stocks_by_concept_from_eastmoney(concept_code=top_concept_code)
    if not df_concept_stocks.empty:
        print(f"{top_concept_name}概念成分股数量: {len(df_concept_stocks)}")
        print("涨幅排行前10的成分股:")
        print(df_concept_stocks[['股票代码', '股票名称', '最新价', '涨跌幅', '主力净流入', '总市值']].head(10))
        print()

print("=== 获取行业板块数据（按主力净流入排序） ===")
df_industry_flow = s_f_1.get_industry_data_from_eastmoney(sort_field='f62')
if not df_industry_flow.empty:
    print("主力净流入排行前10的行业:")
    print(df_industry_flow[['行业名称', '涨跌幅', '主力净流入', '主力净流入占比', '成交额']].head(10))