# -*- coding: utf-8 -*-
"""
samequant_functions_optimized.py 使用示例
展示如何使用优化后的各个模块
"""

import pandas as pd
import time
from samequant_functions_optimized import (
    get_stock_downloader, 
    get_spider_client, 
    get_common_functions,
    Config,
    EastmoneyAPIClient,
    BaseDataProcessor
)

def demo_stock_downloader():
    """演示股票列表下载功能"""
    print("\n" + "="*50)
    print("📈 股票列表下载示例")
    print("="*50)
    
    downloader = get_stock_downloader()
    
    print("正在下载股票列表...")
    all_stocks = downloader.main()
    
    if not all_stocks.empty:
        print(f"✅ 成功获取 {len(all_stocks)} 只股票")
        print(f"📁 保存路径: {downloader.all_stocklist_path}")
        
        # 显示统计信息
        status_counts = all_stocks['上市状态'].value_counts()
        print("\n📊 股票状态统计:")
        for status, count in status_counts.items():
            print(f"   {status}: {count}只")
        
        # 显示前5条记录
        print("\n📋 前5条记录:")
        print(all_stocks.head().to_string(index=False))
    else:
        print("❌ 股票列表下载失败")

def demo_realtime_market_data():
    """演示实时市场数据获取"""
    print("\n" + "="*50)
    print("📊 实时市场数据获取示例")
    print("="*50)
    
    spider = get_spider_client()
    
    print("正在获取实时市场数据...")
    market_data = spider.get_realtime_market_data()
    
    if not market_data.empty:
        print(f"✅ 成功获取 {len(market_data)} 只股票的实时数据")
        
        # 显示统计信息
        print(f"\n📈 涨停股票数量: {len(market_data[market_data['涨跌幅'] >= 9.9])}")
        print(f"📉 跌停股票数量: {len(market_data[market_data['涨跌幅'] <= -9.9])}")
        print(f"💰 成交额前10名:")
        
        top_volume = market_data.nlargest(10, '成交额')[['股票代码', '股票名称', '成交额', '涨跌幅']]
        print(top_volume.to_string(index=False))
        
        # 保存数据
        save_path = f"实时行情_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        BaseDataProcessor.clean_and_save_dataframe(market_data, save_path)
        
    else:
        print("❌ 实时市场数据获取失败")

def demo_sector_data():
    """演示板块数据获取"""
    print("\n" + "="*50)
    print("🏢 板块数据获取示例")
    print("="*50)
    
    spider = get_spider_client()
    
    # 获取行业数据
    print("正在获取行业板块数据...")
    industry_data = spider.get_industry_data(sort_field='f3')  # 按涨跌幅排序
    
    if not industry_data.empty:
        print(f"✅ 成功获取 {len(industry_data)} 个行业板块数据")
        
        # 显示涨幅前10的行业
        top_industries = industry_data.head(10)[['行业名称', '涨跌幅', '主力净流入', '总市值']]
        print("\n📈 涨幅前10行业:")
        print(top_industries.to_string(index=False))
    
    # 获取概念数据
    print("\n正在获取概念板块数据...")
    concept_data = spider.get_concept_data(sort_field='f3')  # 按涨跌幅排序
    
    if not concept_data.empty:
        print(f"✅ 成功获取 {len(concept_data)} 个概念板块数据")
        
        # 显示涨幅前10的概念
        top_concepts = concept_data.head(10)[['概念名称', '涨跌幅', '主力净流入', '总市值']]
        print("\n🚀 涨幅前10概念:")
        print(top_concepts.to_string(index=False))
        
        # 保存数据
        BaseDataProcessor.clean_and_save_dataframe(
            industry_data, f"行业数据_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        BaseDataProcessor.clean_and_save_dataframe(
            concept_data, f"概念数据_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )

def demo_eastmoney_api_client():
    """演示东方财富API客户端的高级用法"""
    print("\n" + "="*50)
    print("🔌 东方财富API客户端高级示例")
    print("="*50)
    
    client = EastmoneyAPIClient()
    
    # 获取特定股票的实时数据
    print("正在获取茅台(600519)的实时数据...")
    stock_data = client.get_stock_realtime_data(['600519'])
    
    if not stock_data.empty:
        print("✅ 茅台实时数据:")
        print(stock_data[['f12', 'f14', 'f2', 'f3', 'f4']].to_string(index=False))
    
    # 自定义字段获取数据
    print("\n正在获取自定义字段的市场数据...")
    custom_fields = 'f12,f14,f2,f3,f4,f5,f6'  # 只获取基础字段
    custom_data = client.get_stock_realtime_data([], custom_fields)
    
    if not custom_data.empty:
        print(f"✅ 获取到 {len(custom_data)} 条自定义字段数据")

def demo_common_functions():
    """演示通用功能"""
    print("\n" + "="*50)
    print("🛠️ 通用功能示例")
    print("="*50)
    
    common = get_common_functions()
    
    # 获取当前目录下的CSV文件列表
    current_dir = "."
    csv_files = common.get_file_list_in_directory(current_dir, '.csv')
    
    print(f"📁 当前目录下的CSV文件 ({len(csv_files)} 个):")
    for file in csv_files[:10]:  # 只显示前10个
        print(f"   {file}")
    
    if len(csv_files) > 10:
        print(f"   ... 还有 {len(csv_files) - 10} 个文件")

def demo_data_processor():
    """演示数据处理功能"""
    print("\n" + "="*50)
    print("⚙️ 数据处理功能示例")
    print("="*50)
    
    # 创建示例数据
    sample_data = pd.DataFrame({
        'f12': ['600519', '000001', '000002'],
        'f14': ['贵州茅台', '平安银行', '万科A'],
        'f2': ['1800.50', '15.20', '18.30'],
        'f3': ['2.35', '-1.20', '0.80']
    })
    
    print("📋 原始数据:")
    print(sample_data.to_string(index=False))
    
    # 应用字段映射
    processor = BaseDataProcessor()
    
    # 重命名列
    field_mapping = Config.FIELD_MAPPINGS['stock_basic']
    processed_data = processor.rename_dataframe_columns(sample_data, field_mapping)
    
    print("\n🔄 重命名后:")
    print(processed_data.to_string(index=False))
    
    # 转换数值列
    numeric_cols = ['最新价', '涨跌幅']
    processed_data = processor.convert_numeric_columns(processed_data, numeric_cols)
    
    print("\n🔢 数值转换后:")
    print(processed_data.dtypes)
    
    # 格式化股票代码
    processed_data['股票代码'] = processed_data['股票代码'].apply(processor.symbol_to_stock_code)
    
    print("\n📈 最终处理结果:")
    print(processed_data.to_string(index=False))

def demo_configuration():
    """演示配置管理"""
    print("\n" + "="*50)
    print("⚙️ 配置管理示例")
    print("="*50)
    
    print("🌐 网络配置:")
    print(f"   默认超时时间: {Config.DEFAULT_TIMEOUT}秒")
    print(f"   默认重试次数: {Config.DEFAULT_MAX_RETRIES}次")
    print(f"   可用用户代理数量: {len(Config.USER_AGENTS)}个")
    
    print("\n🔗 API基础URL:")
    for name, url in Config.EASTMONEY_BASE_URLS.items():
        print(f"   {name}: {url}")
    
    print("\n📋 字段映射示例 (股票基础字段):")
    basic_mapping = Config.FIELD_MAPPINGS['stock_basic']
    for field, name in list(basic_mapping.items())[:5]:
        print(f"   {field} -> {name}")
    print("   ...")

def main():
    """主函数，运行所有示例"""
    print("🚀 samequant_functions_optimized.py 完整使用示例")
    print("=" * 60)
    
    try:
        # 演示配置管理
        demo_configuration()
        
        # 演示数据处理
        demo_data_processor()
        
        # 演示通用功能
        demo_common_functions()
        
        # 演示实时市场数据（快速）
        demo_realtime_market_data()
        
        # 演示板块数据
        demo_sector_data()
        
        # 演示API客户端高级用法
        demo_eastmoney_api_client()
        
        # 演示股票下载（时间较长，可选）
        # demo_stock_downloader()
        
        print("\n" + "="*60)
        print("✅ 所有示例运行完成！")
        print("📝 相关文件已保存到当前目录")
        print("📖 详细说明请参考: 优化说明.md")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()