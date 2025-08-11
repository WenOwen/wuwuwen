#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为LightGBM训练脚本添加可视化功能
包括学习曲线、混淆矩阵、特征重要性等图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 强制设置matplotlib后端和字体
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 清除matplotlib字体缓存
import matplotlib.font_manager as fm
try:
    # 清理并重建字体缓存
    fm._rebuild()
    # 尝试添加系统中的中文字体
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    ]
    for font_path in font_paths:
        try:
            import os
            if os.path.exists(font_path):
                fm.fontManager.addfont(font_path)
        except:
            continue
    fm._rebuild()
except:
    pass

# 确保中文字体加载成功
def ensure_chinese_font():
    """确保中文字体正确加载"""
    try:
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'AR PL UMing CN', 'AR PL UKai CN']
        
        selected_font = 'DejaVu Sans'  # 默认字体
        for font in chinese_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 静默模式，不打印字体信息
        return True
    except:
        return False

# 初始化字体
ensure_chinese_font()

sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

class LightGBMVisualizer:
    """LightGBM训练可视化器"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.training_history = {'train': [], 'val': []}
        
    def record_callback(self):
        """创建记录训练历史的回调函数"""
        def _callback(env):
            # 记录训练和验证损失
            if env.evaluation_result_list:
                for eval_result in env.evaluation_result_list:
                    dataset_name = eval_result[0]
                    metric_name = eval_result[1]
                    metric_value = eval_result[2]
                    
                    if dataset_name in self.training_history:
                        if len(self.training_history[dataset_name]) == 0:
                            self.training_history[dataset_name] = []
                        self.training_history[dataset_name].append({
                            'iteration': env.iteration,
                            'metric': metric_name,
                            'value': metric_value
                        })
        return _callback
    
    def plot_learning_curves(self, model=None):
        """绘制学习曲线"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('🚀 LightGBM 训练学习曲线', fontsize=16, fontweight='bold')
            
            # 1. 训练历史曲线
            if self.training_history and len(self.training_history.get('train', [])) > 0:
                ax1 = axes[0, 0]
                
                # 提取AUC数据
                train_auc = [x['value'] for x in self.training_history['train'] if x['metric'] == 'auc']
                val_auc = [x['value'] for x in self.training_history['val'] if x['metric'] == 'auc']
                iterations = list(range(1, len(train_auc) + 1))
                
                if train_auc and val_auc:
                    ax1.plot(iterations, train_auc, 'b-', label='训练集 AUC', linewidth=2)
                    ax1.plot(iterations, val_auc, 'r-', label='验证集 AUC', linewidth=2)
                    ax1.set_xlabel('迭代次数')
                    ax1.set_ylabel('AUC')
                    ax1.set_title('AUC学习曲线')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                else:
                    ax1.text(0.5, 0.5, '无AUC历史数据', ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('AUC学习曲线')
            else:
                axes[0, 0].text(0.5, 0.5, '无训练历史数据', ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('学习曲线')
            
            # 2. 特征重要性Top20
            if model:
                ax2 = axes[0, 1]
                try:
                    importance = model.feature_importance(importance_type='gain')
                    feature_names = [f'特征_{i}' for i in range(len(importance))]  # 简化特征名
                    
                    # 取前20个重要特征
                    top_indices = np.argsort(importance)[-20:]
                    top_importance = importance[top_indices]
                    top_features = [feature_names[i] for i in top_indices]
                    
                    y_pos = np.arange(len(top_features))
                    bars = ax2.barh(y_pos, top_importance, color='skyblue', alpha=0.8)
                    ax2.set_yticks(y_pos)
                    ax2.set_yticklabels([f'特征{i}' for i in range(len(top_features))], fontsize=8)
                    ax2.set_xlabel('重要性')
                    ax2.set_title('🔥 Top20 特征重要性')
                    ax2.grid(True, alpha=0.3)
                    
                    # 添加数值标签
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax2.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                                f'{width:.0f}', ha='left', va='center', fontsize=8)
                except Exception as e:
                    ax2.text(0.5, 0.5, f'特征重要性绘制失败: {str(e)}', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('🔥 特征重要性')
            else:
                axes[0, 1].text(0.5, 0.5, '无模型数据', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('🔥 特征重要性')
            
            # 3. 过拟合检测图
            ax3 = axes[1, 0]
            if self.training_history and len(self.training_history.get('train', [])) > 0:
                train_auc = [x['value'] for x in self.training_history['train'] if x['metric'] == 'auc']
                val_auc = [x['value'] for x in self.training_history['val'] if x['metric'] == 'auc']
                
                if train_auc and val_auc:
                    # 计算过拟合程度
                    iterations = list(range(1, len(train_auc) + 1))
                    overfitting_gap = [t - v for t, v in zip(train_auc, val_auc)]
                    
                    ax3.plot(iterations, overfitting_gap, 'purple', linewidth=2, label='过拟合差距')
                    ax3.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='理想状态')
                    ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='轻微过拟合')
                    ax3.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='严重过拟合')
                    ax3.fill_between(iterations, 0, overfitting_gap, alpha=0.3, color='purple')
                    ax3.set_xlabel('迭代次数')
                    ax3.set_ylabel('AUC差距 (训练-验证)')
                    ax3.set_title('⚠️ 过拟合检测')
                    ax3.legend(fontsize=8)
                    ax3.grid(True, alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, '无过拟合数据', ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('⚠️ 过拟合检测')
            else:
                ax3.text(0.5, 0.5, '无训练历史数据', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('⚠️ 过拟合检测')
            
            # 4. 收敛性分析
            ax4 = axes[1, 1]
            if self.training_history and len(self.training_history.get('val', [])) > 0:
                val_auc = [x['value'] for x in self.training_history['val'] if x['metric'] == 'auc']
                if val_auc and len(val_auc) > 10:
                    # 计算验证集AUC的滑动平均和标准差
                    window_size = min(10, len(val_auc) // 4)
                    val_auc_smooth = pd.Series(val_auc).rolling(window=window_size).mean()
                    val_auc_std = pd.Series(val_auc).rolling(window=window_size).std()
                    
                    iterations = list(range(1, len(val_auc) + 1))
                    ax4.plot(iterations, val_auc, 'lightblue', alpha=0.5, label='原始验证AUC')
                    ax4.plot(iterations, val_auc_smooth, 'darkblue', linewidth=2, label='平滑验证AUC')
                    ax4.fill_between(iterations, 
                                   val_auc_smooth - val_auc_std, 
                                   val_auc_smooth + val_auc_std, 
                                   alpha=0.2, color='blue', label='±1标准差')
                    ax4.set_xlabel('迭代次数')
                    ax4.set_ylabel('AUC')
                    ax4.set_title('验证集收敛性分析')
                    ax4.legend(fontsize=8)
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, '数据不足，无法分析收敛性', ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('收敛性分析')
            else:
                ax4.text(0.5, 0.5, '无收敛性数据', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('收敛性分析')
            
            plt.tight_layout()
            
            # 保存图片
            learning_curve_path = self.results_dir / "learning_curves.png"
            plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"   ✅ 学习曲线图已保存: {learning_curve_path}")
            return learning_curve_path
            
        except Exception as e:
            print(f"   ❌ 学习曲线绘制失败: {e}")
            plt.close()
            return None
    
    def plot_confusion_matrix_and_metrics(self, y_true, y_pred_proba, y_pred_binary, dataset_name="Test"):
        """绘制混淆矩阵和性能指标"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{dataset_name} 集分类性能分析', fontsize=16, fontweight='bold')
            
            # 1. 混淆矩阵
            ax1 = axes[0, 0]
            cm = confusion_matrix(y_true, y_pred_binary)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                       xticklabels=['看空(0)', '看多(1)'], 
                       yticklabels=['看空(0)', '看多(1)'])
            ax1.set_title('混淆矩阵')
            ax1.set_xlabel('预测标签')
            ax1.set_ylabel('真实标签')
            
            # 2. ROC曲线
            ax2 = axes[0, 1]
            try:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                ax2.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC曲线 (AUC = {roc_auc:.4f})')
                ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
                ax2.set_xlim([0.0, 1.0])
                ax2.set_ylim([0.0, 1.05])
                ax2.set_xlabel('假正率 (FPR)')
                ax2.set_ylabel('真正率 (TPR)')
                ax2.set_title('ROC曲线')
                ax2.legend(loc="lower right")
                ax2.grid(True, alpha=0.3)
            except Exception as e:
                ax2.text(0.5, 0.5, f'ROC曲线绘制失败: {str(e)}', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('ROC曲线')
            
            # 3. 预测概率分布
            ax3 = axes[1, 0]
            try:
                # 按真实标签分组绘制概率分布
                prob_pos = y_pred_proba[y_true == 1]  # 真正类的预测概率
                prob_neg = y_pred_proba[y_true == 0]  # 真负类的预测概率
                
                ax3.hist(prob_neg, bins=30, alpha=0.7, label='真实看空', color='red', density=True)
                ax3.hist(prob_pos, bins=30, alpha=0.7, label='真实看多', color='green', density=True)
                ax3.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, label='分类阈值')
                ax3.set_xlabel('预测概率')
                ax3.set_ylabel('密度')
                ax3.set_title('预测概率分布')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            except Exception as e:
                ax3.text(0.5, 0.5, f'概率分布绘制失败: {str(e)}', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('预测概率分布')
            
            # 4. 详细指标表格
            ax4 = axes[1, 1]
            ax4.axis('off')
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                # 计算详细指标
                accuracy = accuracy_score(y_true, y_pred_binary)
                precision = precision_score(y_true, y_pred_binary, zero_division=0)
                recall = recall_score(y_true, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true, y_pred_binary, zero_division=0)
                
                # 计算混淆矩阵指标
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # 创建指标表格
                metrics_data = [
                    ['指标', '数值', '解释'],
                    ['准确率 (Accuracy)', f'{accuracy:.4f}', '正确预测的比例'],
                    ['精确率 (Precision)', f'{precision:.4f}', '预测看多中真正看多的比例'],
                    ['召回率 (Recall)', f'{recall:.4f}', '真正看多中被预测为看多的比例'],
                    ['F1分数', f'{f1:.4f}', '精确率和召回率的调和平均'],
                    ['特异性 (Specificity)', f'{specificity:.4f}', '真正看空中被预测为看空的比例'],
                    ['AUC', f'{roc_auc:.4f}', 'ROC曲线下面积'],
                    ['', '', ''],
                    ['样本统计', '', ''],
                    ['真正例 (TP)', f'{tp}', '预测看多且实际看多'],
                    ['假正例 (FP)', f'{fp}', '预测看多但实际看空'],
                    ['真负例 (TN)', f'{tn}', '预测看空且实际看空'],
                    ['假负例 (FN)', f'{fn}', '预测看空但实际看多'],
                ]
                
                # 绘制表格
                table = ax4.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                                 cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2)
                
                # 设置表格样式
                for i in range(len(metrics_data)):
                    for j in range(3):
                        cell = table[(i, j)]
                        if i == 0:  # 头部
                            cell.set_facecolor('#4CAF50')
                            cell.set_text_props(weight='bold', color='white')
                        elif i == 8:  # 分割行
                            cell.set_facecolor('#E8F5E8')
                            cell.set_text_props(weight='bold')
                        else:
                            cell.set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')
                
                ax4.set_title('详细性能指标', pad=20, fontweight='bold')
                
            except Exception as e:
                ax4.text(0.5, 0.5, f'指标表格生成失败: {str(e)}', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('性能指标')
            
            plt.tight_layout()
            
            # 保存图片
            confusion_matrix_path = self.results_dir / f"confusion_matrix_{dataset_name.lower()}.png"
            plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"   ✅ {dataset_name}集混淆矩阵图已保存: {confusion_matrix_path}")
            return confusion_matrix_path
            
        except Exception as e:
            print(f"   ❌ 混淆矩阵绘制失败: {e}")
            plt.close()
            return None
    
    def plot_prediction_analysis(self, y_true, y_pred_proba, y_pred_binary):
        """绘制预测分析图"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('🔍 预测结果详细分析', fontsize=16, fontweight='bold')
            
            # 1. 预测置信度分析
            ax1 = axes[0, 0]
            confidence = np.abs(y_pred_proba - 0.5) * 2  # 将概率转换为置信度[0,1]
            
            # 按置信度分桶分析准确率
            bins = np.linspace(0, 1, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            accuracies = []
            counts = []
            
            for i in range(len(bins)-1):
                mask = (confidence >= bins[i]) & (confidence < bins[i+1])
                if np.sum(mask) > 0:
                    bin_accuracy = np.mean(y_true[mask] == y_pred_binary[mask])
                    accuracies.append(bin_accuracy)
                    counts.append(np.sum(mask))
                else:
                    accuracies.append(0)
                    counts.append(0)
            
            bars = ax1.bar(bin_centers, accuracies, width=0.08, alpha=0.7, color='skyblue')
            ax1.set_xlabel('预测置信度')
            ax1.set_ylabel('准确率')
            ax1.set_title('置信度 vs 准确率')
            ax1.set_ylim([0, 1])
            ax1.grid(True, alpha=0.3)
            
            # 添加样本数量标签
            for i, (bar, count) in enumerate(zip(bars, counts)):
                if count > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'n={count}', ha='center', va='bottom', fontsize=8)
            
            # 2. 错误分析
            ax2 = axes[0, 1]
            
            # 分析不同类型的错误
            correct_pred = (y_true == y_pred_binary)
            error_types = []
            error_counts = []
            colors = []
            
            # 真正例（正确预测看多）
            tp_mask = (y_true == 1) & (y_pred_binary == 1)
            error_types.append('正确看多')
            error_counts.append(np.sum(tp_mask))
            colors.append('green')
            
            # 真负例（正确预测看空）
            tn_mask = (y_true == 0) & (y_pred_binary == 0)
            error_types.append('正确看空')
            error_counts.append(np.sum(tn_mask))
            colors.append('blue')
            
            # 假正例（错误预测看多）
            fp_mask = (y_true == 0) & (y_pred_binary == 1)
            error_types.append('误判看多')
            error_counts.append(np.sum(fp_mask))
            colors.append('orange')
            
            # 假负例（错误预测看空）
            fn_mask = (y_true == 1) & (y_pred_binary == 0)
            error_types.append('误判看空')
            error_counts.append(np.sum(fn_mask))
            colors.append('red')
            
            wedges, texts, autotexts = ax2.pie(error_counts, labels=error_types, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
            ax2.set_title('🔍 预测结果分布')
            
            # 3. 阈值敏感性分析
            ax3 = axes[1, 0]
            thresholds = np.linspace(0.1, 0.9, 17)
            precision_scores = []
            recall_scores = []
            f1_scores = []
            
            for threshold in thresholds:
                y_pred_thresh = (y_pred_proba >= threshold).astype(int)
                
                from sklearn.metrics import precision_score, recall_score, f1_score
                prec = precision_score(y_true, y_pred_thresh, zero_division=0)
                rec = recall_score(y_true, y_pred_thresh, zero_division=0)
                f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
                
                precision_scores.append(prec)
                recall_scores.append(rec)
                f1_scores.append(f1)
            
            ax3.plot(thresholds, precision_scores, 'b-', label='精确率', linewidth=2)
            ax3.plot(thresholds, recall_scores, 'r-', label='召回率', linewidth=2)
            ax3.plot(thresholds, f1_scores, 'g-', label='F1分数', linewidth=2)
            ax3.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='默认阈值')
            ax3.set_xlabel('分类阈值')
            ax3.set_ylabel('指标值')
            ax3.set_title('⚖️ 阈值敏感性分析')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 概率校准分析
            ax4 = axes[1, 1]
            
            # 将预测概率分桶，计算每桶的实际正例比例
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            observed_freqs = []
            predicted_freqs = []
            bin_counts = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    observed_freq = y_true[in_bin].mean()
                    predicted_freq = y_pred_proba[in_bin].mean()
                    bin_count = in_bin.sum()
                else:
                    observed_freq = 0
                    predicted_freq = bin_lower + (bin_upper - bin_lower) / 2
                    bin_count = 0
                
                observed_freqs.append(observed_freq)
                predicted_freqs.append(predicted_freq)
                bin_counts.append(bin_count)
            
            # 绘制校准曲线
            ax4.plot([0, 1], [0, 1], 'k--', label='完美校准')
            ax4.plot(predicted_freqs, observed_freqs, 'o-', label='模型校准', linewidth=2, markersize=6)
            
            # 添加置信区间（基于样本数量）
            for i, (pred_freq, obs_freq, count) in enumerate(zip(predicted_freqs, observed_freqs, bin_counts)):
                if count > 0:
                    # 简单的二项式置信区间
                    std_err = np.sqrt(obs_freq * (1 - obs_freq) / count) if count > 0 else 0
                    ax4.errorbar(pred_freq, obs_freq, yerr=1.96*std_err, fmt='none', alpha=0.5)
            
            ax4.set_xlabel('平均预测概率')
            ax4.set_ylabel('实际正例比例')
            ax4.set_title('📏 概率校准曲线')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim([0, 1])
            ax4.set_ylim([0, 1])
            
            plt.tight_layout()
            
            # 保存图片
            prediction_analysis_path = self.results_dir / "prediction_analysis.png"
            plt.savefig(prediction_analysis_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"   🔍 预测分析图已保存: {prediction_analysis_path}")
            return prediction_analysis_path
            
        except Exception as e:
            print(f"   ❌ 预测分析图绘制失败: {e}")
            plt.close()
            return None
    
    def generate_all_visualizations(self, model, y_train, y_val, y_test, 
                                  y_train_pred_proba, y_val_pred_proba, y_test_pred_proba):
        """生成所有可视化图表"""
        print("🎨 生成可视化图表...")
        
        results = {}
        
        # 1. 学习曲线
        learning_curve_path = self.plot_learning_curves(model)
        if learning_curve_path:
            results['learning_curves'] = str(learning_curve_path)
        
        # 2. 转换为二分类预测
        y_train_pred_binary = (y_train_pred_proba > 0.5).astype(int)
        y_val_pred_binary = (y_val_pred_proba > 0.5).astype(int)
        y_test_pred_binary = (y_test_pred_proba > 0.5).astype(int)
        
        # 3. 各数据集的混淆矩阵
        for dataset_name, y_true, y_pred_proba, y_pred_binary in [
            ('Train', y_train, y_train_pred_proba, y_train_pred_binary),
            ('Val', y_val, y_val_pred_proba, y_val_pred_binary),
            ('Test', y_test, y_test_pred_proba, y_test_pred_binary)
        ]:
            cm_path = self.plot_confusion_matrix_and_metrics(y_true, y_pred_proba, y_pred_binary, dataset_name)
            if cm_path:
                results[f'confusion_matrix_{dataset_name.lower()}'] = str(cm_path)
        
        # 4. 测试集预测分析
        pred_analysis_path = self.plot_prediction_analysis(y_test, y_test_pred_proba, y_test_pred_binary)
        if pred_analysis_path:
            results['prediction_analysis'] = str(pred_analysis_path)
        
        # 5. 保存可视化结果摘要
        viz_summary_path = self.results_dir / "visualization_summary.json"
        with open(viz_summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 可视化完成! 生成 {len(results)} 个图表")
        print(f"📁 图表保存位置: {self.results_dir}")
        
        return results


# 使用示例的代码
def add_visualization_to_training_script():
    """
    将可视化功能集成到训练脚本的示例代码
    """
    example_code = '''
# 在 lightgbm_stock_train.py 中的修改示例：

# 1. 在类初始化中添加
def __init__(self, config_path: str):
    # ... 现有代码 ...
    self.visualizer = None  # 可视化器

# 2. 在训练方法中添加记录回调
def train_model(self) -> bool:
    try:
        # ... 现有代码 ...
        
        # 创建可视化器
        from visualization_extension import LightGBMVisualizer
        self.visualizer = LightGBMVisualizer(self.results_save_dir)
        
        # 修改callbacks，添加记录回调
        callbacks = [
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(verbose),
            self.visualizer.record_callback()  # 添加这行
        ]
        
        # ... 其余训练代码 ...
        
# 3. 在save_results方法中添加可视化生成
def save_results(self, evaluation_results: Dict) -> bool:
    try:
        # ... 现有保存代码 ...
        
        # 生成可视化图表
        if hasattr(self, 'visualizer') and self.visualizer:
            # 获取预测概率
            if hasattr(self, 'prediction_mode') and self.prediction_mode == 'direction':
                y_train_pred_proba = self.model.predict(self.X_train)
                y_val_pred_proba = self.model.predict(self.X_val)
                y_test_pred_proba = self.model.predict(self.X_test)
            else:
                # 回归模式，不生成分类相关图表
                return True
            
            # 生成所有可视化图表
            viz_results = self.visualizer.generate_all_visualizations(
                self.model,
                self.y_train, self.y_val, self.y_test,
                y_train_pred_proba, y_val_pred_proba, y_test_pred_proba
            )
            
            self.logger.info(f"   🎨 可视化图表已生成: {len(viz_results)} 个")
        
        return True
    except Exception as e:
        # ... 错误处理 ...
    '''
    
    return example_code

if __name__ == "__main__":
    print("LightGBM可视化扩展模块")
    print("=" * 50)
    print("功能：")
    print("1. 学习曲线图（训练vs验证）")
    print("2. 特征重要性图")
    print("3. 过拟合检测图")
    print("4. 收敛性分析图")
    print("5. 混淆矩阵图")
    print("6. ROC曲线图")
    print("7. 预测概率分布图")
    print("8. 预测结果详细分析")
    print("9. 概率校准曲线")
    print("10. 阈值敏感性分析")
    print("\n使用方法：")
    print("from visualization_extension import LightGBMVisualizer")
    print("# 在训练脚本中集成即可")