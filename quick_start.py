#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI股市预测系统 - 快速启动脚本
立即检查环境并开始数据收集
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """检查环境"""
    logger.info("🔍 检查系统环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        logger.error("❌ Python版本过低，需要3.8+")
        return False
    
    logger.info(f"✅ Python版本: {sys.version.split()[0]}")
    
    # 检查必要目录
    directories = ['datas_em', 'logs', 'models', 'backup']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ 目录检查: {directory}")
    
    return True

def install_dependencies():
    """安装依赖"""
    logger.info("📦 检查并安装依赖...")
    
    try:
        # 检查核心依赖
        required_packages = ['pandas', 'numpy']
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} 已安装")
            except ImportError:
                logger.info(f"🔄 安装 {package}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                             check=True, capture_output=True)
        
        return True
    except Exception as e:
        logger.error(f"❌ 依赖安装失败: {e}")
        return False

def check_existing_data():
    """检查现有数据"""
    logger.info("📊 检查现有股票数据...")
    
    data_dir = Path('datas_em')
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        logger.warning("⚠️ 没有发现股票数据文件")
        logger.info("📝 请先运行: python 2.1获取全数据（东财）.py")
        return False, 0
    
    # 快速检查数据质量
    valid_files = 0
    total_records = 0
    
    for file in csv_files[:10]:  # 只检查前10个文件
        try:
            df = pd.read_csv(file)
            if len(df) > 50:  # 至少50条记录
                valid_files += 1
                total_records += len(df)
        except:
            continue
    
    logger.info(f"✅ 发现 {len(csv_files)} 个数据文件")
    logger.info(f"✅ 有效文件: {valid_files}, 总记录数: {total_records}")
    
    return len(csv_files) > 0, len(csv_files)

def run_data_collection():
    """运行数据收集"""
    logger.info("🚀 开始数据收集...")
    
    scripts_to_run = [
        '2.1获取全数据（东财）.py',
        '2.7获取资金流向数据.py', 
        '2.10获取板块数据.py'
    ]
    
    for script in scripts_to_run:
        if os.path.exists(script):
            try:
                logger.info(f"🔄 运行 {script}...")
                result = subprocess.run([sys.executable, script], 
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    logger.info(f"✅ {script} 执行成功")
                else:
                    logger.warning(f"⚠️ {script} 执行有警告: {result.stderr[:200]}")
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️ {script} 执行超时")
            except Exception as e:
                logger.error(f"❌ {script} 执行失败: {e}")
        else:
            logger.warning(f"⚠️ 脚本不存在: {script}")

def create_test_config():
    """创建测试配置"""
    logger.info("⚙️ 创建初始配置...")
    
    # 创建股票池配置
    config_content = '''# 测试股票池
TEST_STOCKS = [
    "sh600519",  # 贵州茅台
    "sz000001",  # 平安银行
    "sz000002",  # 万科A
    "sh600036",  # 招商银行
    "sz000858"   # 五粮液
]

# 系统配置
SYSTEM_CONFIG = {
    "sequence_length": 60,
    "prediction_days": [1, 3, 5],
    "min_data_points": 100,
    "batch_size": 32
}
'''
    
    with open('config/test_config.py', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    logger.info("✅ 配置文件创建完成")

def test_core_modules():
    """测试核心模块"""
    logger.info("🧪 测试核心模块...")
    
    try:
        # 测试特征工程
        from feature_engineering import FeatureEngineering
        fe = FeatureEngineering()
        logger.info("✅ 特征工程模块加载成功")
        
        # 如果有数据，进行简单测试
        data_dir = Path('datas_em')
        csv_files = list(data_dir.glob('*.csv'))
        
        if csv_files:
            test_file = csv_files[0]
            df = pd.read_csv(test_file)
            
            if len(df) > 100:
                logger.info(f"🔄 使用 {test_file.name} 测试特征工程...")
                df_features = fe.create_all_features(df)
                logger.info(f"✅ 特征工程测试成功，生成 {df_features.shape[1]} 个特征")
                return True
        
        logger.info("✅ 模块加载测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 模块测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 AI股市预测系统 - 快速启动检查")
    print("=" * 50)
    
    # 1. 环境检查
    if not check_environment():
        print("❌ 环境检查失败，请检查Python版本")
        return
    
    # 2. 依赖检查
    if not install_dependencies():
        print("❌ 依赖安装失败")
        return
    
    # 3. 数据检查
    has_data, file_count = check_existing_data()
    
    # 4. 如果没有数据，尝试收集
    if not has_data:
        print("\n📥 开始收集股票数据...")
        run_data_collection()
        
        # 重新检查
        has_data, file_count = check_existing_data()
    
    # 5. 创建配置
    Path('config').mkdir(exist_ok=True)
    create_test_config()
    
    # 6. 测试核心模块
    modules_ok = test_core_modules()
    
    # 总结报告
    print("\n" + "=" * 50)
    print("📋 快速启动检查报告:")
    print(f"   📁 数据文件数量: {file_count}")
    print(f"   🧪 模块测试: {'✅ 通过' if modules_ok else '❌ 失败'}")
    
    if has_data and modules_ok:
        print("\n🎉 系统基础环境准备完成！")
        print("\n📝 下一步建议:")
        print("   1. 运行完整数据收集: python 2.1获取全数据（东财）.py")
        print("   2. 测试特征工程: python tests/test_feature_engineering.py") 
        print("   3. 开始模型训练: python initial_training.py")
        print("   4. 启动Web界面: streamlit run streamlit_app.py")
    else:
        print("\n⚠️ 需要手动处理以下问题:")
        if not has_data:
            print("   - 数据收集失败，请检查网络和数据源")
        if not modules_ok:
            print("   - 模块加载失败，请检查依赖安装")

if __name__ == "__main__":
    main()