# -*- coding: utf-8 -*-
"""
完整股票训练脚本 - 使用所有可用股票进行完整训练
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.training_pipeline import ModelTrainingPipeline
from memory_monitor import MemoryMonitor, monitor_memory_during_training
import gc

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'complete_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """完整训练主函数"""
    logger.info("🚀 开始完整股票训练流程")
    logger.info("=" * 80)
    
    # 初始化内存监控
    memory_monitor = MemoryMonitor(warning_threshold=0.75, critical_threshold=0.85)
    memory_monitor.log_memory_status("训练开始前")
    
    # 初始化训练管道
    pipeline = ModelTrainingPipeline(
        data_dir="data/datas_em",  # 明确指定使用data/datas_em目录
        enable_batch_cache=False,  # 彻底禁用批量缓存
        cache_workers=1  # 缓存已禁用，工作进程数无效
    )
    
    # 获取所有可用股票
    logger.info("📊 扫描所有可用股票...")
    all_stocks = pipeline.get_available_stocks()  # 不限制数量，获取所有股票
    
    if not all_stocks:
        logger.error("❌ 未找到任何有效的股票数据")
        return
    
    logger.info(f"✅ 发现 {len(all_stocks)} 只有效股票")
    logger.info(f"   前10只股票: {all_stocks[:10]}")
    
    # 现在有了真正的分批处理，可以处理所有股票
    logger.info(f"🚀 使用分批处理技术，可以安全处理所有 {len(all_stocks)} 只股票")
    logger.info("   - 每批处理50只股票，避免内存溢出")
    logger.info("   - 每只股票最多500个样本")
    logger.info("   - 总样本数限制50,000个")
    logger.info("   - 回望窗口减少到30天")
    
    memory_monitor.log_memory_status("股票扫描完成")
    
    # 显示训练配置
    logger.info("\n📋 训练配置:")
    logger.info(f"   训练轮次: {pipeline.config['training_params']['epochs']}")
    logger.info(f"   批次大小: {pipeline.config['training_params']['batch_size']}")
    logger.info(f"   早停轮次: {pipeline.config['training_params']['early_stopping_patience']}")
    logger.info(f"   交叉验证折数: {pipeline.config['training_params']['cv_folds']}")
    logger.info(f"   LightGBM树数量: {pipeline.config['lightgbm_config']['n_estimators']}")
    logger.info(f"   预测天数: {pipeline.config['prediction_days']}")
    logger.info(f"   最小样本数: {pipeline.config['min_samples']}")
    
    # 跳过缓存预热（已禁用缓存）
    logger.info("\n⚠️ 缓存已禁用，跳过预热步骤")
    warmup_time = timedelta(0)
    
    # 批量训练所有预测天数的模型
    logger.info("\n🎯 步骤2: 开始批量模型训练...")
    start_training = datetime.now()
    
    try:
        # 内存监控下的模型训练
        memory_monitor.log_memory_status("开始模型训练")
        
        # 只训练一个预测天数的模型以节省内存
        prediction_days_list = pipeline.config['prediction_days'][:1]  # 只取第一个预测天数
        logger.info(f"⚠️ 为节省内存，只训练 {prediction_days_list} 天预测模型")
        
        models = {}
        for prediction_days in prediction_days_list:
            logger.info(f"\n🎯 训练 {prediction_days} 天预测模型...")
            memory_monitor.log_memory_status(f"训练{prediction_days}天模型前")
            
            model = pipeline.train_model(
                stock_codes=all_stocks,
                prediction_days=prediction_days,
                use_hyperparameter_optimization=False,  # 关闭超参数优化以节省时间和内存
                save_model=True,
                clear_cache=False
            )
            models[prediction_days] = model
            
            # 强制垃圾回收
            gc.collect()
            memory_monitor.log_memory_status(f"训练{prediction_days}天模型后")
            
            # 检查内存状态
            status = memory_monitor.check_memory_status()
            if status == "CRITICAL":
                logger.error("❌ 内存使用率过高，停止训练")
                break
        
        total_training_time = datetime.now() - start_training
        
        # 训练结果汇总
        logger.info("\n" + "=" * 80)
        logger.info("🎉 完整训练流程完成！")
        logger.info("=" * 80)
        logger.info(f"📊 训练统计:")
        logger.info(f"   处理股票数: {len(all_stocks)}")
        logger.info(f"   训练模型数: {len(models)}/{len(pipeline.config['prediction_days'])}")
        logger.info(f"   模型训练时间: {total_training_time}")
        logger.info(f"   总耗时: {total_training_time}（无缓存预热）")
        
        # 显示各模型性能
        logger.info(f"\n📈 模型性能汇总:")
        for prediction_days, model in models.items():
            logger.info(f"   {prediction_days}天预测模型: 已保存")
        
        # 缓存已禁用，无需显示缓存统计
        logger.info(f"\n📊 缓存状态: 已禁用")
        
        # 性能历史
        summary = pipeline.get_performance_summary()
        if not summary.empty:
            logger.info(f"\n📋 性能历史摘要:")
            print(summary)
        
        logger.info(f"\n💾 模型文件保存在: {pipeline.model_dir}")
        logger.info(f"📝 训练完成报告已自动生成在各模型目录中")
        
        # 显示内存使用摘要
        memory_summary = memory_monitor.get_memory_summary()
        logger.info(f"\n📊 内存使用摘要:")
        logger.info(f"   初始内存: {memory_summary['initial_mb']:.1f} MB")
        logger.info(f"   当前内存: {memory_summary['current_mb']:.1f} MB")
        logger.info(f"   峰值内存: {memory_summary['peak_mb']:.1f} MB")
        logger.info(f"   内存增长: {memory_summary['growth_mb']:.1f} MB")
        logger.info(f"   系统使用率: {memory_summary['system_usage_percent']:.1f}%")
        
        # 优化建议
        suggestions = memory_monitor.suggest_optimizations()
        if suggestions:
            logger.info(f"\n💡 内存优化建议:")
            for suggestion in suggestions:
                logger.info(f"   - {suggestion}")
        
        logger.info(f"✅ 完整训练流程成功完成！")
        
    except Exception as e:
        logger.error(f"❌ 训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 完整训练成功完成！")
    else:
        print("\n❌ 训练过程中出现错误，请查看日志")
        sys.exit(1)