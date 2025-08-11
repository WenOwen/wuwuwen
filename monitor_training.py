#!/usr/bin/env python3
"""
训练监控脚本
"""
import time
import subprocess
import os
from pathlib import Path

def check_training_progress():
    """检查训练进度"""
    print("🔍 检查LightGBM训练进度...")
    
    # 检查进程
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    lightgbm_processes = [line for line in result.stdout.split('\n') if 'lightgbm' in line and 'python' in line]
    
    print(f"📊 当前运行的训练进程: {len(lightgbm_processes)}")
    for proc in lightgbm_processes:
        if 'ultra_robust' in proc:
            print(f"  🎯 激进配置: {proc.split()[-1]}")
        elif 'robust' in proc:
            print(f"  🔧 标准配置: {proc.split()[-1]}")
        else:
            print(f"  📈 原始配置: {proc.split()[-1]}")
    
    # 检查结果目录
    results_dirs = [
        'results/lightgbm_direction',
        'results/lightgbm_direction_robust', 
        'results/lightgbm_direction_ultra_robust'
    ]
    
    print("\n📁 结果目录状态:")
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
            print(f"  {results_dir}: {len(subdirs)} 个训练结果")
            for subdir in subdirs:
                metrics_file = os.path.join(results_dir, subdir, 'metrics.json')
                if os.path.exists(metrics_file):
                    print(f"    ✅ {subdir}: 训练完成")
                else:
                    print(f"    🔄 {subdir}: 训练中...")
        else:
            print(f"  {results_dir}: 不存在")

if __name__ == "__main__":
    check_training_progress()