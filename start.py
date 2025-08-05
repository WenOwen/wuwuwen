#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI股市预测系统 - 一键启动脚本
功能：检查环境、初始化数据、启动所有服务
"""

import os
import sys
import subprocess
import time
import logging
import argparse
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        logger.error("Python版本必须≥3.8，当前版本: %s", sys.version)
        return False
    logger.info("Python版本检查通过: %s", sys.version.split()[0])
    return True


def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'tensorflow', 
        'xgboost', 'streamlit', 'fastapi', 'uvicorn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info("✓ %s 已安装", package)
        except ImportError:
            missing_packages.append(package)
            logger.warning("✗ %s 未安装", package)
    
    if missing_packages:
        logger.error("缺少依赖包: %s", ', '.join(missing_packages))
        logger.info("请运行: pip install -r requirements.txt")
        return False
    
    logger.info("所有依赖包检查通过")
    return True


def check_directories():
    """检查并创建必要的目录"""
    directories = [
        'datas_em',
        'models', 
        'logs',
        'stockcode_list',
        'backup'
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info("创建目录: %s", directory)
        else:
            logger.info("✓ 目录存在: %s", directory)
    
    return True


def check_data_files():
    """检查数据文件"""
    data_dir = Path('datas_em')
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        logger.warning("datas_em目录中没有股票数据文件")
        logger.info("请先运行数据获取脚本: python 2.1获取全数据（东财）.py")
        return False
    
    logger.info("发现 %d 个股票数据文件", len(csv_files))
    return True


def create_stock_list():
    """创建默认股票列表"""
    stock_list_file = Path('stockcode_list/all_stock_list.csv')
    
    if not stock_list_file.exists():
        logger.info("创建默认股票列表...")
        
        default_stocks = [
            ('sh600519', '贵州茅台'),
            ('sz000001', '平安银行'),
            ('sz000002', '万科A'),
            ('sh600036', '招商银行'),
            ('sz000858', '五粮液'),
            ('sh600000', '浦发银行'),
            ('sz000858', '五粮液'),
            ('sh601318', '中国平安'),
            ('sz002415', '海康威视'),
            ('sh600276', '恒瑞医药')
        ]
        
        with open(stock_list_file, 'w', encoding='utf-8') as f:
            f.write('股票代码,股票名称\n')
            for code, name in default_stocks:
                f.write(f'{code},{name}\n')
        
        logger.info("默认股票列表创建完成")
    
    return True


def check_redis():
    """检查Redis服务"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        logger.info("✓ Redis连接正常")
        return True
    except Exception as e:
        logger.warning("Redis连接失败: %s", str(e))
        logger.info("将使用内存缓存替代Redis")
        return False


def train_initial_models():
    """训练初始模型"""
    logger.info("检查是否存在训练好的模型...")
    
    models_dir = Path('models')
    model_folders = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if not model_folders:
        logger.info("未发现训练好的模型，开始训练初始模型...")
        logger.warning("这可能需要较长时间，请耐心等待...")
        
        try:
            # 这里应该调用训练脚本
            logger.info("由于首次训练耗时较长，请手动运行:")
            logger.info("python training_pipeline.py")
            return False
        except Exception as e:
            logger.error("模型训练失败: %s", str(e))
            return False
    else:
        logger.info("发现 %d 个训练好的模型", len(model_folders))
        return True


def start_api_service():
    """启动API服务"""
    logger.info("启动API服务...")
    
    try:
        # 启动FastAPI服务
        cmd = [
            sys.executable, '-m', 'uvicorn',
            'prediction_service:app',
            '--host', '0.0.0.0',
            '--port', '8000',
            '--reload'
        ]
        
        logger.info("执行命令: %s", ' '.join(cmd))
        process = subprocess.Popen(cmd)
        
        # 等待服务启动
        time.sleep(3)
        
        # 检查服务是否正常
        import requests
        try:
            response = requests.get('http://localhost:8000/', timeout=5)
            if response.status_code == 200:
                logger.info("✓ API服务启动成功")
                return process
            else:
                logger.error("API服务响应异常: %s", response.status_code)
                return None
        except requests.exceptions.RequestException as e:
            logger.error("API服务连接失败: %s", str(e))
            return None
            
    except Exception as e:
        logger.error("启动API服务失败: %s", str(e))
        return None


def start_web_interface():
    """启动Web界面"""
    logger.info("启动Streamlit Web界面...")
    
    try:
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            'streamlit_app.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0'
        ]
        
        logger.info("执行命令: %s", ' '.join(cmd))
        process = subprocess.Popen(cmd)
        
        # 等待服务启动
        time.sleep(5)
        
        logger.info("✓ Web界面启动成功")
        logger.info("🌐 访问地址: http://localhost:8501")
        
        return process
        
    except Exception as e:
        logger.error("启动Web界面失败: %s", str(e))
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI股市预测系统启动脚本')
    parser.add_argument('--skip-checks', action='store_true', help='跳过环境检查')
    parser.add_argument('--api-only', action='store_true', help='只启动API服务')
    parser.add_argument('--web-only', action='store_true', help='只启动Web界面')
    parser.add_argument('--no-models', action='store_true', help='跳过模型检查')
    
    args = parser.parse_args()
    
    logger.info("🚀 开始启动AI股市预测系统...")
    
    # 环境检查
    if not args.skip_checks:
        logger.info("📋 进行环境检查...")
        
        if not check_python_version():
            sys.exit(1)
        
        if not check_dependencies():
            logger.error("❌ 依赖检查失败")
            sys.exit(1)
        
        if not check_directories():
            logger.error("❌ 目录检查失败")
            sys.exit(1)
        
        create_stock_list()
        
        if not check_data_files():
            logger.warning("⚠️ 数据文件检查失败，某些功能可能不可用")
        
        check_redis()
        
        if not args.no_models and not train_initial_models():
            logger.warning("⚠️ 模型检查失败，预测功能可能不可用")
    
    # 启动服务
    processes = []
    
    if not args.web_only:
        logger.info("🔧 启动API服务...")
        api_process = start_api_service()
        if api_process:
            processes.append(api_process)
        else:
            logger.error("❌ API服务启动失败")
            if not args.api_only:
                sys.exit(1)
    
    if not args.api_only:
        logger.info("🌐 启动Web界面...")
        web_process = start_web_interface()
        if web_process:
            processes.append(web_process)
        else:
            logger.error("❌ Web界面启动失败")
            sys.exit(1)
    
    # 显示启动信息
    logger.info("🎉 系统启动完成！")
    logger.info("📊 API文档: http://localhost:8000/docs")
    logger.info("🌐 Web界面: http://localhost:8501")
    logger.info("📝 按Ctrl+C停止所有服务")
    
    # 等待用户中断
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 正在停止服务...")
        
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        
        logger.info("✅ 所有服务已停止")


if __name__ == "__main__":
    main()