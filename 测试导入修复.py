#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试所有模块的导入是否正常工作
验证文件重新组织后的导入路径修复
"""

import sys
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_core_imports():
    """测试core目录中的模块导入"""
    logger.info("🧪 测试core模块导入...")
    
    # 添加路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'core'))
    
    test_results = {}
    
    # 测试各个模块的导入
    modules_to_test = [
        ('stock_sector_mapping', 'StockSectorMapping'),
        ('feature_engineering', 'FeatureEngineering'),
        ('ai_models', 'EnsembleModel'),
        ('enhanced_ai_models', 'create_enhanced_ensemble_model'),
        ('training_pipeline', 'ModelTrainingPipeline'),
        ('prediction_service', 'PredictionService'),
        ('prediction_service_no_redis', 'PredictionServiceNoRedis'),
    ]
    
    for module_name, class_name in modules_to_test:
        try:
            # 动态导入模块
            if module_name.startswith('core.'):
                module = __import__(module_name, fromlist=[class_name])
            else:
                # 从core目录导入
                full_module_name = f'core.{module_name}'
                module = __import__(full_module_name, fromlist=[class_name])
            
            # 检查类是否存在
            if hasattr(module, class_name):
                test_results[module_name] = "✅ 成功"
                logger.info(f"✅ {module_name}.{class_name} 导入成功")
            else:
                test_results[module_name] = f"⚠️ 模块导入成功但缺少 {class_name}"
                logger.warning(f"⚠️ {module_name} 导入成功但缺少 {class_name}")
                
        except ImportError as e:
            test_results[module_name] = f"❌ 导入失败: {str(e)}"
            logger.error(f"❌ {module_name} 导入失败: {str(e)}")
        except Exception as e:
            test_results[module_name] = f"❌ 其他错误: {str(e)}"
            logger.error(f"❌ {module_name} 其他错误: {str(e)}")
    
    return test_results

def test_direct_script_execution():
    """测试直接运行脚本的情况"""
    logger.info("🧪 测试直接脚本执行...")
    
    scripts_to_test = [
        'core/training_pipeline.py',
        'core/test_stock_sector_features.py',
        'data_processing/获取板块数据并保存CSV.py'
    ]
    
    test_results = {}
    
    for script in scripts_to_test:
        if os.path.exists(script):
            try:
                # 尝试编译脚本检查语法
                with open(script, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                compile(source_code, script, 'exec')
                test_results[script] = "✅ 语法正确"
                logger.info(f"✅ {script} 语法检查通过")
                
            except SyntaxError as e:
                test_results[script] = f"❌ 语法错误: {str(e)}"
                logger.error(f"❌ {script} 语法错误: {str(e)}")
            except Exception as e:
                test_results[script] = f"❌ 其他错误: {str(e)}"
                logger.error(f"❌ {script} 其他错误: {str(e)}")
        else:
            test_results[script] = "❌ 文件不存在"
            logger.error(f"❌ {script} 文件不存在")
    
    return test_results

def test_specific_imports():
    """测试具体的重要导入"""
    logger.info("🧪 测试具体重要导入...")
    
    test_results = {}
    
    # 测试训练流水线的关键导入
    try:
        from core.training_pipeline import ModelTrainingPipeline
        pipeline = ModelTrainingPipeline()
        test_results['训练流水线初始化'] = "✅ 成功"
        logger.info("✅ 训练流水线初始化成功")
    except Exception as e:
        test_results['训练流水线初始化'] = f"❌ 失败: {str(e)}"
        logger.error(f"❌ 训练流水线初始化失败: {str(e)}")
    
    # 测试板块映射的关键导入
    try:
        from core.stock_sector_mapping import StockSectorMapping
        mapping = StockSectorMapping()
        test_results['板块映射初始化'] = "✅ 成功"
        logger.info("✅ 板块映射初始化成功")
    except Exception as e:
        test_results['板块映射初始化'] = f"❌ 失败: {str(e)}"
        logger.error(f"❌ 板块映射初始化失败: {str(e)}")
    
    # 测试特征工程的关键导入
    try:
        from core.feature_engineering import FeatureEngineering
        fe = FeatureEngineering()
        test_results['特征工程初始化'] = "✅ 成功"
        logger.info("✅ 特征工程初始化成功")
    except Exception as e:
        test_results['特征工程初始化'] = f"❌ 失败: {str(e)}"
        logger.error(f"❌ 特征工程初始化失败: {str(e)}")
    
    return test_results

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("🔧 AI股市预测系统 - 导入修复验证")
    logger.info("=" * 60)
    
    all_results = {}
    
    # 1. 测试core模块导入
    core_results = test_core_imports()
    all_results.update(core_results)
    
    # 2. 测试直接脚本执行
    script_results = test_direct_script_execution()
    all_results.update(script_results)
    
    # 3. 测试具体重要导入
    specific_results = test_specific_imports()
    all_results.update(specific_results)
    
    # 汇总结果
    logger.info("\n" + "=" * 60)
    logger.info("📊 测试结果汇总:")
    logger.info("=" * 60)
    
    success_count = 0
    total_count = len(all_results)
    
    for test_name, result in all_results.items():
        print(f"{test_name}: {result}")
        if result.startswith("✅"):
            success_count += 1
    
    logger.info(f"\n📈 成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        logger.info("🎉 所有导入测试通过！系统已准备就绪")
    else:
        logger.warning(f"⚠️ 有 {total_count - success_count} 个导入问题需要解决")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()