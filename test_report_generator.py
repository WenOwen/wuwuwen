# -*- coding: utf-8 -*-
"""
测试训练报告自动生成功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.training_report_generator import TrainingReportGenerator
import json
from datetime import datetime

def test_report_generator():
    """测试报告生成器"""
    
    print("🧪 测试训练报告自动生成功能...")
    
    # 模拟训练数据
    mock_model_save_path = "models/test_model"
    os.makedirs(mock_model_save_path, exist_ok=True)
    
    # 创建模拟的性能摘要文件
    performance_data = {
        "model_name": "test_model_1d",
        "ensemble_accuracy": 0.5567,
        "cv_accuracy_mean": 0.5423,
        "cv_accuracy_std": 0.0098,
        "feature_count": 170,
        "training_samples": 40000,
        "test_samples": 10000,
        "total_samples": 50000,
        "training_time": "20250806_200000",
        "gpu_optimized": True
    }
    
    with open(os.path.join(mock_model_save_path, "performance_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, ensure_ascii=False, indent=2)
    
    # 模拟训练信息
    training_info = {
        'stock_codes': ['sh600000', 'sz000001', 'sz300750'] * 100,  # 300只股票
        'prediction_days': 1,
        'training_samples': 40000,
        'test_samples': 10000,
        'training_time': datetime.now().isoformat(),
        'gpu_config': {'gpu_strategy': 'MirroredStrategy'}
    }
    
    # 模拟测试结果
    results = {
        'LightGBM': {'accuracy': 0.5678, 'model_type': 'LightGBM'},
        'LSTM': {'accuracy': 0.5234, 'model_type': 'LSTM'},
        'Transformer': {'accuracy': 0.5156, 'model_type': 'Transformer'},
        'CNN-LSTM': {'accuracy': 0.5289, 'model_type': 'CNN-LSTM'},
        'Ensemble': {'accuracy': 0.5567, 'model_type': 'Ensemble'}
    }
    
    # 模拟交叉验证结果
    cv_results = {
        'accuracy': 0.5423,
        'accuracy_std': 0.0098,
        'precision': 0.4567,
        'recall': 0.3456,
        'f1': 0.3956,
        'auc': 0.5678
    }
    
    # 模拟特征名称
    feature_names = [
        '开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额',
        'price_change', 'SMA_5', 'EMA_10', 'RSI_14', 'MACD',
        'sector_industry_return', 'sector_concept_return', 'sector_hot_rank',
        'sector_region_code', 'sector_relative_strength'
    ] + [f'feature_{i}' for i in range(16, 171)]  # 总共170个特征
    
    # 创建报告生成器
    generator = TrainingReportGenerator()
    
    try:
        # 生成报告
        report_path = generator.generate_training_report(
            model_save_path=mock_model_save_path,
            training_info=training_info,
            results=results,
            cv_results=cv_results,
            feature_names=feature_names,
            stock_codes=training_info['stock_codes'],
            prediction_days=1
        )
        
        if os.path.exists(report_path):
            print(f"✅ 报告生成成功: {report_path}")
            
            # 读取并显示报告的前几行
            with open(report_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"\n📄 报告内容预览（前10行）:")
                for i, line in enumerate(lines[:10], 1):
                    print(f"{i:2d}: {line.rstrip()}")
                
            print(f"\n📊 报告统计:")
            print(f"   总行数: {len(lines)}")
            print(f"   文件大小: {os.path.getsize(report_path) / 1024:.1f} KB")
            
        else:
            print("❌ 报告生成失败")
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        import shutil
        if os.path.exists(mock_model_save_path):
            shutil.rmtree(mock_model_save_path)
        print("\n🧹 测试文件已清理")

if __name__ == "__main__":
    test_report_generator()