#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为训练脚本添加涨跌方向预测模式的修改脚本
"""

def modify_training_script():
    """修改训练脚本添加方向预测支持"""
    print("🔧 开始添加涨跌方向预测功能...")
    
    # 读取原文件
    with open('lightgbm_stock_train.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"   原文件行数: {len(lines)}")
    
    # 新的目标变量处理代码
    new_target_processing = '''            # 检查预测模式
            prediction_mode = direct_training.get('prediction_mode', 'regression')
            self.prediction_mode = prediction_mode  # 保存预测模式
            
            # 设置目标变量
            raw_targets = full_data['next_day_target']
            
            if prediction_mode == 'direction':
                # 🎯 方向预测模式：涨跌幅 > 0 为看多(1)，<= 0 为看空(0)
                self.y = (raw_targets > 0).astype(int)
                actual_target_column = 'next_day_direction'
                self.logger.info(f"   🎯 预测模式: 涨跌方向预测（二分类）")
                self.logger.info(f"   📊 看多样本: {(self.y == 1).sum():,} ({(self.y == 1).mean()*100:.1f}%)")
                self.logger.info(f"   📊 看空样本: {(self.y == 0).sum():,} ({(self.y == 0).mean()*100:.1f}%)")
            else:
                # 📈 回归预测模式：预测具体涨跌幅
                self.y = raw_targets
                actual_target_column = 'next_day_target'
                self.logger.info(f"   📈 预测模式: 涨跌幅预测（回归）")
                self.logger.info(f"   📊 目标值范围: [{self.y.min():.4f}, {self.y.max():.4f}]")
'''
    
    # 新的评估方法代码
    new_evaluate_method = '''    def evaluate_model(self) -> Dict:
        """评估模型"""
        try:
            self.logger.info("📊 评估模型性能...")
            
            # 获取预测结果
            if hasattr(self, 'prediction_mode') and self.prediction_mode == 'direction':
                # 二分类：获取概率预测
                y_train_pred_proba = self.model.predict(self.X_train)
                y_val_pred_proba = self.model.predict(self.X_val)
                y_test_pred_proba = self.model.predict(self.X_test)
                
                # 转换为类别预测（概率 > 0.5 为看多）
                y_train_pred = (y_train_pred_proba > 0.5).astype(int)
                y_val_pred = (y_val_pred_proba > 0.5).astype(int)
                y_test_pred = (y_test_pred_proba > 0.5).astype(int)
            else:
                # 回归：直接预测数值
                y_train_pred = self.model.predict(self.X_train)
                y_val_pred = self.model.predict(self.X_val)
                y_test_pred = self.model.predict(self.X_test)
                y_train_pred_proba = y_val_pred_proba = y_test_pred_proba = None
            
            # 计算评估指标
            eval_config = self.config.get('evaluation', {})
            metrics_list = eval_config.get('metrics', ['rmse', 'mae', 'r2_score'])
            
            results = {}
            
            for split, y_true, y_pred, y_pred_proba in [
                ('train', self.y_train, y_train_pred, y_train_pred_proba),
                ('val', self.y_val, y_val_pred, y_val_pred_proba),
                ('test', self.y_test, y_test_pred, y_test_pred_proba)
            ]:
                split_metrics = {}
                
                for metric in metrics_list:
                    try:
                        if hasattr(self, 'prediction_mode') and self.prediction_mode == 'direction':
                            # 🎯 分类指标
                            if metric == 'accuracy':
                                value = np.mean(y_true == y_pred) * 100
                            elif metric == 'auc' and y_pred_proba is not None:
                                from sklearn.metrics import roc_auc_score
                                value = roc_auc_score(y_true, y_pred_proba)
                            elif metric == 'precision':
                                from sklearn.metrics import precision_score
                                value = precision_score(y_true, y_pred, zero_division=0)
                            elif metric == 'recall':
                                from sklearn.metrics import recall_score
                                value = recall_score(y_true, y_pred, zero_division=0)
                            elif metric == 'f1_score':
                                from sklearn.metrics import f1_score
                                value = f1_score(y_true, y_pred, zero_division=0)
                            elif metric == 'log_loss' and y_pred_proba is not None:
                                from sklearn.metrics import log_loss
                                # 处理概率边界问题
                                y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1-1e-15)
                                value = log_loss(y_true, y_pred_proba_clipped)
                            else:
                                continue
                        else:
                            # 📈 回归指标
                            if metric == 'rmse':
                                value = np.sqrt(mean_squared_error(y_true, y_pred))
                            elif metric == 'mae':
                                value = mean_absolute_error(y_true, y_pred)
                            elif metric == 'mape':
                                value = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
                            elif metric == 'r2_score':
                                value = r2_score(y_true, y_pred)
                            elif metric == 'explained_variance':
                                value = explained_variance_score(y_true, y_pred)
                            elif metric == 'directional_accuracy':
                                # 方向准确率（股票预测特有指标）
                                direction_true = np.sign(y_true)
                                direction_pred = np.sign(y_pred)
                                value = np.mean(direction_true == direction_pred) * 100
                            else:
                                continue
                        
                        split_metrics[metric] = float(value)
                        
                    except Exception as e:
                        self.logger.warning(f"   计算指标 {metric} 失败: {e}")
                        continue
                
                results[split] = split_metrics
            
            # 输出结果
            prediction_type = "方向预测" if hasattr(self, 'prediction_mode') and self.prediction_mode == 'direction' else "回归预测"
            self.logger.info(f"   📈 评估结果 ({prediction_type}):")
            for split, metrics in results.items():
                self.logger.info(f"     {split.upper()}:")
                for metric, value in metrics.items():
                    if metric in ['mape', 'directional_accuracy', 'accuracy']:
                        self.logger.info(f"       {metric}: {value:.2f}%")
                    else:
                        self.logger.info(f"       {metric}: {value:.6f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 模型评估失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
'''
    
    # 构建新文件
    new_lines = []
    i = 0
    
    while i < len(lines):
        # 替换目标变量处理部分（391-393行）
        if i == 390 and lines[i].strip() == "# 设置目标变量（明天的涨跌幅）":
            # 跳过原有的3行
            i += 3
            # 插入新的目标变量处理代码
            new_lines.append(new_target_processing)
            new_lines.append('\n')
        # 替换evaluate_model方法
        elif i < len(lines) and lines[i].strip().startswith("def evaluate_model(self) -> Dict:"):
            # 找到方法结束位置
            indent_level = len(lines[i]) - len(lines[i].lstrip())
            i += 1
            while i < len(lines):
                current_line = lines[i]
                if (current_line.strip() and 
                    not current_line.startswith(' ' * (indent_level + 1)) and
                    not current_line.strip().startswith('"""') and
                    not current_line.strip().startswith('try:') and
                    current_line.strip() != '' and
                    (current_line.startswith('    def ') or current_line.startswith('def '))):
                    break
                i += 1
            
            # 插入新的评估方法
            new_lines.append(new_evaluate_method)
            new_lines.append('\n')
            continue
        else:
            new_lines.append(lines[i])
            i += 1
    
    # 写入新文件
    with open('lightgbm_stock_train.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"   修改后行数: {len(new_lines)}")
    print("   ✅ 修改完成！")
    print("   🎯 添加功能:")
    print("   - 支持涨跌方向预测模式")
    print("   - 二分类目标变量转换")
    print("   - 分类评估指标（AUC、准确率等）")
    print("   - 自动模式检测和切换")

if __name__ == "__main__":
    modify_training_script()