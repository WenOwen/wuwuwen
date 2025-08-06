# -*- coding: utf-8 -*-
"""
内存优化的股票训练脚本 - 解决大量股票训练时的内存溢出问题
"""

import os
import sys
import logging
import gc
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# 导入LightGBM（需要在导入模型之前）
try:
    import lightgbm as lgb
except ImportError:
    print("⚠️ 警告: LightGBM未安装，请安装: pip install lightgbm")
    sys.exit(1)

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.training_pipeline import ModelTrainingPipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'memory_optimized_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MemoryOptimizedPipeline(ModelTrainingPipeline):
    """内存优化的训练管道"""
    
    def __init__(self, *args, **kwargs):
        # 强制禁用所有缓存
        kwargs['enable_batch_cache'] = False
        kwargs['cache_workers'] = 1
        super().__init__(*args, **kwargs)
        
        # 内存优化配置
        self.batch_size = 50  # 每批处理的股票数量
        self.max_samples_per_stock = 500  # 每只股票最大样本数
        self.feature_reduction_ratio = 0.7  # 特征降维比例
    
    def memory_optimized_prepare_data(self, stock_codes: List[str], 
                                    prediction_days: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """
        内存优化的数据准备方法 - 分批处理 + 数据压缩
        """
        logger.info(f"🚀 开始内存优化数据准备，股票数量: {len(stock_codes)}")
        logger.info(f"   批处理大小: {self.batch_size}")
        logger.info(f"   最大样本数/股票: {self.max_samples_per_stock}")
        
        # 分批处理股票
        all_batches = []
        feature_names = None
        feature_info = None
        total_samples = 0
        processed_stocks = 0
        
        for i in range(0, len(stock_codes), self.batch_size):
            batch_stocks = stock_codes[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(stock_codes) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"\n📦 处理批次 {batch_num}/{total_batches} ({len(batch_stocks)} 只股票)")
            
            # 处理当前批次
            try:
                batch_X, batch_y, batch_feature_names, batch_feature_info = self._process_stock_batch(
                    batch_stocks, prediction_days
                )
                
                if batch_X is not None and len(batch_X) > 0:
                    # 限制每批的样本数量以控制内存
                    if len(batch_X) > self.max_samples_per_stock * len(batch_stocks):
                        max_samples = self.max_samples_per_stock * len(batch_stocks)
                        indices = np.random.choice(len(batch_X), max_samples, replace=False)
                        batch_X = batch_X[indices]
                        batch_y = batch_y[indices]
                    
                    all_batches.append((batch_X, batch_y))
                    total_samples += len(batch_X)
                    processed_stocks += len(batch_stocks)
                    
                    if feature_names is None:
                        feature_names = batch_feature_names
                        feature_info = batch_feature_info
                    
                    logger.info(f"   ✅ 批次完成: {len(batch_X)} 个样本")
                    
                    # 强制垃圾回收
                    del batch_X, batch_y
                    gc.collect()
                else:
                    logger.warning(f"   ❌ 批次无有效数据")
                    
            except Exception as e:
                logger.error(f"   ❌ 批次处理失败: {str(e)}")
                continue
        
        if not all_batches:
            raise ValueError("没有成功处理的批次数据")
        
        logger.info(f"\n🔄 合并所有批次数据...")
        logger.info(f"   总批次数: {len(all_batches)}")
        logger.info(f"   总样本数: {total_samples}")
        logger.info(f"   处理股票数: {processed_stocks}/{len(stock_codes)}")
        
        # 逐步合并批次数据以节省内存
        X_combined = None
        y_combined = None
        
        for i, (batch_X, batch_y) in enumerate(all_batches):
            if X_combined is None:
                X_combined = batch_X.copy()
                y_combined = batch_y.copy()
            else:
                X_combined = np.vstack([X_combined, batch_X])
                y_combined = np.hstack([y_combined, batch_y])
            
            # 删除已处理的批次数据
            del batch_X, batch_y
            
            if (i + 1) % 10 == 0:  # 每10个批次强制垃圾回收
                gc.collect()
                logger.info(f"   已合并 {i+1}/{len(all_batches)} 个批次")
        
        # 清理批次列表
        del all_batches
        gc.collect()
        
        # 最终数据压缩和采样
        if len(X_combined) > 50000:  # 如果样本数过多，进行采样
            logger.info(f"🔽 样本数过多({len(X_combined)})，进行随机采样...")
            indices = np.random.choice(len(X_combined), 50000, replace=False)
            X_combined = X_combined[indices]
            y_combined = y_combined[indices]
            logger.info(f"   采样后样本数: {len(X_combined)}")
        
        logger.info(f"\n✅ 内存优化数据准备完成:")
        logger.info(f"   最终样本数: {len(X_combined)}")
        logger.info(f"   特征数: {len(feature_names)}")
        logger.info(f"   正样本比例: {y_combined.mean():.3f}")
        logger.info(f"   内存占用估计: {X_combined.nbytes / 1024 / 1024:.1f} MB")
        
        return X_combined, y_combined, feature_names, feature_info
    
    def _process_stock_batch(self, stock_codes: List[str], 
                           prediction_days: int) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """处理单个股票批次"""
        batch_X, batch_y = [], []
        feature_names = None
        feature_info = None
        
        for stock_code in stock_codes:
            try:
                # 加载股票数据
                df = self.load_stock_data(stock_code)
                
                if len(df) < self.config['min_samples']:
                    continue
                
                # 数据预处理 - 限制数据量
                if len(df) > 2000:  # 如果数据过多，只取最近的数据
                    df = df.tail(2000)
                
                # 特征工程（简化版）
                df_features = self.feature_engineer.create_all_features(df, stock_code)
                
                # 准备模型数据
                X, y, feature_names_temp, feature_info_temp = self.feature_engineer.prepare_model_data(
                    df_features, 
                    prediction_days=prediction_days,
                    lookback_window=min(30, self.config['sequence_length'])  # 减少回望窗口
                )
                
                if len(X) > 0:
                    # 限制每只股票的样本数
                    if len(X) > self.max_samples_per_stock:
                        indices = np.random.choice(len(X), self.max_samples_per_stock, replace=False)
                        X = X[indices]
                        y = y[indices]
                    
                    batch_X.append(X)
                    batch_y.append(y)
                    
                    if feature_names is None:
                        feature_names = feature_names_temp
                        feature_info = feature_info_temp
                
                # 及时清理内存
                del df, df_features, X, y
                gc.collect()
                
            except Exception as e:
                logger.warning(f"处理股票 {stock_code} 失败: {str(e)}")
                continue
        
        if not batch_X:
            return None, None, None, None
        
        # 合并批次内的数据
        X_batch = np.vstack(batch_X)
        y_batch = np.hstack(batch_y)
        
        # 清理临时数据
        del batch_X, batch_y
        gc.collect()
        
        return X_batch, y_batch, feature_names, feature_info
    
    def memory_optimized_train_model(self, stock_codes: List[str], 
                                   prediction_days: int = 1) -> object:
        """内存优化的模型训练"""
        logger.info(f"🚀 开始内存优化模型训练，预测 {prediction_days} 天")
        
        # 使用优化的数据准备方法
        X, y, feature_names, feature_info = self.memory_optimized_prepare_data(stock_codes, prediction_days)
        
        # 数据分割
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # 创建轻量级模型（主要使用LightGBM）
        from core.ai_models import LightGBMModel
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        model = LightGBMModel()
        
        # 训练模型
        logger.info("🎯 开始模型训练...")
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train, 
                 X_test.reshape(X_test.shape[0], -1), y_test)
        
        # 手动评估模型（因为LightGBMModel没有evaluate方法）
        logger.info("📊 评估模型性能...")
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        y_pred = model.predict(X_test_2d)
        y_proba = model.predict_proba(X_test_2d)[:, 1]
        
        # 计算各种指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        logger.info(f"✅ 模型训练完成:")
        logger.info(f"   准确率: {accuracy:.4f}")
        logger.info(f"   精确率: {precision:.4f}")
        logger.info(f"   召回率: {recall:.4f}")
        logger.info(f"   F1分数: {f1:.4f}")
        
        return model

def main():
    """内存优化的主函数"""
    logger.info("🚀 开始内存优化股票训练流程")
    logger.info("=" * 80)
    
    # 初始化优化的训练管道
    pipeline = MemoryOptimizedPipeline(
        data_dir="data/datas_em",
        enable_batch_cache=False,
        cache_workers=1
    )
    
    # 获取所有可用股票
    logger.info("📊 扫描所有可用股票...")
    all_stocks = pipeline.get_available_stocks()
    
    if not all_stocks:
        logger.error("❌ 未找到任何有效的股票数据")
        return False
    
    logger.info(f"✅ 发现 {len(all_stocks)} 只有效股票")
    
    # 为了避免内存问题，可以选择处理部分股票
    if len(all_stocks) > 1000:
        logger.info(f"⚠️ 股票数量过多({len(all_stocks)})，随机选择1000只进行训练")
        import random
        all_stocks = random.sample(all_stocks, 1000)
    
    logger.info(f"🎯 实际训练股票数: {len(all_stocks)}")
    
    try:
        # 训练模型（只训练一个预测天数以节省内存）
        prediction_days = 1
        logger.info(f"\n🎯 开始训练 {prediction_days} 天预测模型...")
        
        model = pipeline.memory_optimized_train_model(all_stocks, prediction_days)
        
        # 保存模型
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"models/memory_optimized_model_{prediction_days}d_{timestamp}"
        os.makedirs(model_path, exist_ok=True)
        
        if hasattr(model, 'save_model'):
            model.save_model(os.path.join(model_path, 'model.pkl'))
        else:
            import joblib
            joblib.dump(model, os.path.join(model_path, 'model.pkl'))
        
        logger.info(f"✅ 模型已保存到: {model_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 内存优化训练流程完成！")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 内存优化训练成功完成！")
    else:
        print("\n❌ 训练过程中出现错误，请查看日志")
        sys.exit(1)