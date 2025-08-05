#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据收集脚本 - 优先确保有可用数据
"""

import subprocess
import sys
import os
import time

def run_data_scripts():
    """运行数据收集脚本"""
    print("📥 开始数据收集...")
    
    scripts = [
        ('2.1获取全数据（东财）.py', '获取基础K线数据'),
        ('2.7获取资金流向数据.py', '获取资金流向数据'),
        ('2.10获取板块数据.py', '获取板块数据')
    ]
    
    success_count = 0
    
    for script, description in scripts:
        if os.path.exists(script):
            print(f"\n🔄 {description}...")
            print(f"运行: {script}")
            
            try:
                start_time = time.time()
                result = subprocess.run([sys.executable, script], 
                                      capture_output=True, text=True, timeout=600)
                
                elapsed = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"✅ {description} 成功 (耗时: {elapsed:.1f}秒)")
                    success_count += 1
                    
                    # 显示部分输出
                    if result.stdout:
                        lines = result.stdout.split('\n')[:5]
                        for line in lines:
                            if line.strip():
                                print(f"  📄 {line.strip()}")
                else:
                    print(f"⚠️ {description} 有警告:")
                    if result.stderr:
                        error_lines = result.stderr.split('\n')[:3]
                        for line in error_lines:
                            if line.strip():
                                print(f"  ⚠️ {line.strip()}")
                                
            except subprocess.TimeoutExpired:
                print(f"⏰ {description} 超时 (10分钟)")
            except Exception as e:
                print(f"❌ {description} 失败: {e}")
        else:
            print(f"⚠️ 脚本不存在: {script}")
    
    print(f"\n📊 数据收集结果: {success_count}/{len(scripts)} 成功")
    
    # 检查数据文件
    data_dir = 'datas_em'
    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        print(f"📁 数据文件数量: {len(csv_files)}")
        
        if len(csv_files) > 10:
            print("✅ 数据收集成功，可以开始使用系统！")
            return True
        else:
            print("⚠️ 数据文件较少，建议检查网络连接")
            return False
    else:
        print("❌ 数据目录不存在")
        return False

if __name__ == "__main__":
    print("🚀 股票数据收集工具")
    print("=" * 40)
    
    success = run_data_scripts()
    
    if success:
        print("\n🎯 下一步:")
        print("1. 运行最小化系统: python quick_start_minimal.py")
        print("2. 测试预测功能: python simple_predictor.py")
    else:
        print("\n🆘 如果数据收集失败:")
        print("1. 检查网络连接")
        print("2. 检查API是否可用")
        print("3. 稍后重试")