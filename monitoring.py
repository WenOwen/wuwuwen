"""
监控和可视化模块
包含注意力权重可视化、特征重要性分析、模型诊断等功能
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号


class AttentionVisualizer:
    """注意力权重可视化器"""
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or self._create_default_feature_names()
        
    def _create_default_feature_names(self) -> List[str]:
        """创建默认特征名称"""
        names = []
        # 个股特征 (15维)
        stock_features = ['股价开盘', '股价最高', '股价最低', '股价收盘', '成交量',
                         'RSI', 'MACD', 'KDJ_K', 'KDJ_D', 'KDJ_J',
                         '布林上轨', '布林中轨', '布林下轨', 'ATR', '换手率']
        names.extend(stock_features)
        
        # 板块特征 (5维)
        sector_features = ['板块指数', '板块热度', '涨停板块强度', '连板板块强度', '板块资金流']
        names.extend(sector_features)
        
        # 指数特征 (5维)
        index_features = ['上证指数', '深证成指', '创业板指', '市场波动率', 'VIX指数']
        names.extend(index_features)
        
        # 情绪特征 (3维)
        sentiment_features = ['涨停板块情绪', '连板板块情绪', '市场情绪综合']
        names.extend(sentiment_features)
        
        return names
        
    def plot_attention_heatmap(self, attention_weights: np.ndarray, 
                              dates: Optional[List[str]] = None,
                              save_path: Optional[str] = None,
                              title: str = "注意力权重热力图") -> None:
        """
        绘制注意力权重热力图
        
        Args:
            attention_weights: 注意力权重 (n_samples, seq_len)
            dates: 日期列表
            save_path: 保存路径
            title: 图表标题
        """
        plt.figure(figsize=(15, 8))
        
        # 如果样本太多，只显示最近的部分
        if attention_weights.shape[0] > 100:
            attention_weights = attention_weights[-100:]
            if dates:
                dates = dates[-100:]
                
        # 创建热力图
        sns.heatmap(
            attention_weights.T,  # 转置使时间步在y轴
            cmap='YlOrRd',
            cbar_kws={'label': '注意力权重'},
            xticklabels=dates[::max(1, len(dates)//10)] if dates else False,
            yticklabels=[f'T-{i}' for i in range(attention_weights.shape[1]-1, -1, -1)]
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('样本（时间）', fontsize=12)
        plt.ylabel('时间步', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"注意力热力图已保存到: {save_path}")
            
        plt.show()
        
    def plot_attention_time_series(self, attention_weights: np.ndarray,
                                  dates: Optional[List[str]] = None,
                                  crisis_dates: Optional[List[str]] = None,
                                  save_path: Optional[str] = None) -> None:
        """
        绘制注意力权重时序图
        
        Args:
            attention_weights: 注意力权重 (n_samples, seq_len)
            dates: 日期列表
            crisis_dates: 危机日期列表
            save_path: 保存路径
        """
        # 计算每个时间步的平均注意力权重
        mean_attention = attention_weights.mean(axis=0)
        std_attention = attention_weights.std(axis=0)
        
        plt.figure(figsize=(12, 6))
        
        # 绘制平均注意力权重
        time_steps = np.arange(len(mean_attention))
        plt.plot(time_steps, mean_attention, 'b-', linewidth=2, label='平均注意力权重')
        plt.fill_between(time_steps, 
                        mean_attention - std_attention,
                        mean_attention + std_attention,
                        alpha=0.3, color='blue', label='±1标准差')
        
        plt.xlabel('时间步 (T-n天)', fontsize=12)
        plt.ylabel('注意力权重', fontsize=12)
        plt.title('时间步注意力权重分布', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 标记重要时间步
        max_idx = np.argmax(mean_attention)
        plt.axvline(x=max_idx, color='red', linestyle='--', alpha=0.7)
        plt.text(max_idx, max(mean_attention), f'T-{len(mean_attention)-max_idx-1}天\n最高权重', 
                ha='center', va='bottom', color='red', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"注意力时序图已保存到: {save_path}")
            
        plt.show()
        
    def plot_crisis_attention(self, attention_weights: np.ndarray,
                             dates: pd.DatetimeIndex,
                             crisis_events: Dict[str, str],
                             save_path: Optional[str] = None) -> None:
        """
        分析危机期间的注意力分布
        
        Args:
            attention_weights: 注意力权重 (n_samples, seq_len)
            dates: 日期索引
            crisis_events: 危机事件字典 {日期: 事件描述}
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        # 按特征类型分组分析注意力
        feature_groups = {
            '个股特征 (0-15)': (0, 16),
            '板块特征 (16-21)': (16, 22), 
            '指数特征 (22-26)': (22, 27),
            '情绪特征 (27-28)': (27, 29)
        }
        
        for idx, (group_name, (start, end)) in enumerate(feature_groups.items()):
            if idx >= len(axes):
                break
                
            # 计算该组特征的平均注意力权重
            group_attention = attention_weights.mean(axis=0)  # 简化版本
            
            axes[idx].plot(group_attention, label=group_name)
            axes[idx].set_title(f'{group_name}注意力权重', fontweight='bold')
            axes[idx].set_xlabel('时间步')
            axes[idx].set_ylabel('注意力权重')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend()
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"危机注意力分析图已保存到: {save_path}")
            
        plt.show()


class FeatureImportanceAnalyzer:
    """特征重要性分析器"""
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or self._create_default_feature_names()
        
    def _create_default_feature_names(self) -> List[str]:
        """创建默认特征名称"""
        names = []
        names.extend([f'个股特征_{i}' for i in range(15)])
        names.extend([f'板块特征_{i}' for i in range(5)])
        names.extend([f'指数特征_{i}' for i in range(5)])
        names.extend([f'情绪特征_{i}' for i in range(3)])
        return names
        
    def calculate_feature_importance(self, model: torch.nn.Module, 
                                   X: torch.Tensor,
                                   method: str = 'gradient') -> np.ndarray:
        """
        计算特征重要性
        
        Args:
            model: 训练好的模型
            X: 输入数据 (n_samples, seq_len, n_features)
            method: 计算方法 ('gradient', 'permutation')
            
        Returns:
            importance: 特征重要性 (n_features,)
        """
        if method == 'gradient':
            return self._gradient_importance(model, X)
        elif method == 'permutation':
            return self._permutation_importance(model, X)
        else:
            raise ValueError(f"不支持的方法: {method}")
            
    def _gradient_importance(self, model: torch.nn.Module, 
                           X: torch.Tensor) -> np.ndarray:
        """基于梯度的特征重要性"""
        model.eval()
        X.requires_grad_(True)
        
        output, _ = model(X)
        
        # 计算输出对输入的梯度
        grad = torch.autograd.grad(
            outputs=output.sum(),
            inputs=X,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # 计算特征重要性（梯度的平均绝对值）
        importance = grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()
        
        return importance
        
    def _permutation_importance(self, model: torch.nn.Module,
                              X: torch.Tensor) -> np.ndarray:
        """基于排列的特征重要性"""
        model.eval()
        
        with torch.no_grad():
            # 原始预测
            baseline_output, _ = model(X)
            baseline_loss = torch.nn.functional.mse_loss(
                baseline_output, torch.zeros_like(baseline_output)
            ).item()
            
        importance = np.zeros(X.shape[-1])
        
        for feature_idx in range(X.shape[-1]):
            # 复制数据并打乱指定特征
            X_permuted = X.clone()
            perm_indices = torch.randperm(X.shape[0])
            X_permuted[:, :, feature_idx] = X_permuted[perm_indices, :, feature_idx]
            
            with torch.no_grad():
                permuted_output, _ = model(X_permuted)
                permuted_loss = torch.nn.functional.mse_loss(
                    permuted_output, torch.zeros_like(permuted_output)
                ).item()
                
            # 重要性 = 性能下降程度
            importance[feature_idx] = permuted_loss - baseline_loss
            
        return importance
        
    def plot_feature_importance(self, importance: np.ndarray,
                               top_k: int = 20,
                               save_path: Optional[str] = None) -> None:
        """绘制特征重要性图"""
        # 获取top_k重要特征
        top_indices = np.argsort(importance)[::-1][:top_k]
        top_importance = importance[top_indices]
        top_names = [self.feature_names[i] for i in top_indices]
        
        # 创建颜色映射（按特征类型）
        colors = []
        for idx in top_indices:
            if idx < 15:
                colors.append('skyblue')  # 个股
            elif idx < 20:
                colors.append('lightgreen')  # 板块
            elif idx < 25:
                colors.append('salmon')  # 指数
            else:
                colors.append('gold')  # 情绪
                
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_names)), top_importance, color=colors)
        
        plt.yticks(range(len(top_names)), top_names)
        plt.xlabel('特征重要性', fontsize=12)
        plt.title(f'Top {top_k} 特征重要性', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # 添加图例
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='skyblue', label='个股特征'),
            plt.Rectangle((0,0),1,1, facecolor='lightgreen', label='板块特征'),
            plt.Rectangle((0,0),1,1, facecolor='salmon', label='指数特征'),
            plt.Rectangle((0,0),1,1, facecolor='gold', label='情绪特征')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        # 添加数值标签
        for bar, value in zip(bars, top_importance):
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f'{value:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征重要性图已保存到: {save_path}")
            
        plt.show()


class ModelDiagnostics:
    """模型诊断工具"""
    
    def __init__(self):
        self.diagnostics = {}
        
    def analyze_predictions(self, predictions: np.ndarray, 
                          targets: np.ndarray,
                          dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """分析预测结果"""
        results = {}
        
        # 基础统计
        results['prediction_stats'] = {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'skewness': self._calculate_skewness(predictions),
            'kurtosis': self._calculate_kurtosis(predictions)
        }
        
        results['target_stats'] = {
            'mean': np.mean(targets),
            'std': np.std(targets),
            'min': np.min(targets),
            'max': np.max(targets),
            'skewness': self._calculate_skewness(targets),
            'kurtosis': self._calculate_kurtosis(targets)
        }
        
        # 相关性分析
        results['correlation'] = np.corrcoef(predictions, targets)[0, 1]
        
        # 残差分析
        residuals = predictions - targets
        results['residual_stats'] = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'mae': np.mean(np.abs(residuals)),
            'rmse': np.sqrt(np.mean(residuals**2))
        }
        
        # 分位数分析
        results['quantile_analysis'] = self._quantile_analysis(predictions, targets)
        
        return results
        
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
        
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
        
    def _quantile_analysis(self, predictions: np.ndarray, 
                          targets: np.ndarray) -> Dict[str, float]:
        """分位数分析"""
        # 按预测值分组
        n_quantiles = 5
        quantile_edges = np.quantile(predictions, np.linspace(0, 1, n_quantiles + 1))
        
        quantile_returns = {}
        for i in range(n_quantiles):
            mask = (predictions >= quantile_edges[i]) & (predictions < quantile_edges[i + 1])
            if i == n_quantiles - 1:  # 最后一个分位数包含边界
                mask = (predictions >= quantile_edges[i]) & (predictions <= quantile_edges[i + 1])
                
            if np.sum(mask) > 0:
                quantile_returns[f'Q{i+1}'] = np.mean(targets[mask])
            else:
                quantile_returns[f'Q{i+1}'] = 0.0
                
        return quantile_returns
        
    def plot_prediction_analysis(self, predictions: np.ndarray,
                               targets: np.ndarray,
                               dates: Optional[pd.DatetimeIndex] = None,
                               save_path: Optional[str] = None) -> None:
        """绘制预测分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 预测vs实际散点图
        axes[0, 0].scatter(targets, predictions, alpha=0.6, s=10)
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('实际收益率')
        axes[0, 0].set_ylabel('预测收益率')
        axes[0, 0].set_title('预测vs实际')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 残差图
        residuals = predictions - targets
        axes[0, 1].scatter(predictions, residuals, alpha=0.6, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 1].set_xlabel('预测收益率')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title('残差分析')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 预测分布
        axes[1, 0].hist(predictions, bins=50, alpha=0.7, label='预测', color='blue')
        axes[1, 0].hist(targets, bins=50, alpha=0.7, label='实际', color='red')
        axes[1, 0].set_xlabel('收益率')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('分布对比')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 时序图（如果有日期）
        if dates is not None:
            # 只显示最近的数据点
            n_show = min(500, len(dates))
            show_dates = dates[-n_show:]
            show_pred = predictions[-n_show:]
            show_target = targets[-n_show:]
            
            axes[1, 1].plot(show_dates, show_pred, label='预测', alpha=0.8)
            axes[1, 1].plot(show_dates, show_target, label='实际', alpha=0.8)
            axes[1, 1].set_xlabel('日期')
            axes[1, 1].set_ylabel('收益率')
            axes[1, 1].set_title('时序对比（最近500天）')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        else:
            # QQ图
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('残差Q-Q图')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测分析图已保存到: {save_path}")
            
        plt.show()


class InteractiveVisualizer:
    """交互式可视化器（使用Plotly）"""
    
    def __init__(self):
        pass
        
    def create_interactive_attention_plot(self, attention_weights: np.ndarray,
                                        dates: List[str],
                                        save_path: Optional[str] = None) -> None:
        """创建交互式注意力权重图"""
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights.T,
            x=dates,
            y=[f'T-{i}' for i in range(attention_weights.shape[1]-1, -1, -1)],
            colorscale='YlOrRd',
            colorbar=dict(title="注意力权重")
        ))
        
        fig.update_layout(
            title='交互式注意力权重热力图',
            xaxis_title='日期',
            yaxis_title='时间步',
            width=1200,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"交互式注意力图已保存到: {save_path}")
        else:
            fig.show()
            
    def create_feature_importance_dashboard(self, importance_data: Dict[str, np.ndarray],
                                          feature_names: List[str],
                                          save_path: Optional[str] = None) -> None:
        """创建特征重要性仪表板"""
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=list(importance_data.keys()),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for idx, (method, importance) in enumerate(importance_data.items()):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            # 获取top 15特征
            top_indices = np.argsort(importance)[::-1][:15]
            top_importance = importance[top_indices]
            top_names = [feature_names[i] for i in top_indices]
            
            fig.add_trace(
                go.Bar(x=top_importance, y=top_names, 
                      orientation='h', name=method,
                      marker_color=colors[idx % len(colors)]),
                row=row, col=col
            )
            
        fig.update_layout(
            title_text="特征重要性对比仪表板",
            showlegend=False,
            height=800,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"特征重要性仪表板已保存到: {save_path}")
        else:
            fig.show()


def create_monitoring_report(model: torch.nn.Module,
                           X: torch.Tensor, y: torch.Tensor,
                           attention_weights: np.ndarray,
                           dates: pd.DatetimeIndex,
                           save_dir: str = './monitoring_report') -> None:
    """创建完整的监控报告"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("生成监控报告...")
    
    # 1. 注意力可视化
    print("1. 生成注意力可视化...")
    att_visualizer = AttentionVisualizer()
    att_visualizer.plot_attention_heatmap(
        attention_weights, 
        dates=[d.strftime('%Y-%m-%d') for d in dates[-len(attention_weights):]],
        save_path=os.path.join(save_dir, 'attention_heatmap.png')
    )
    
    att_visualizer.plot_attention_time_series(
        attention_weights,
        save_path=os.path.join(save_dir, 'attention_timeseries.png')
    )
    
    # 2. 特征重要性分析
    print("2. 分析特征重要性...")
    importance_analyzer = FeatureImportanceAnalyzer()
    
    # 梯度重要性
    grad_importance = importance_analyzer.calculate_feature_importance(
        model, X, method='gradient'
    )
    importance_analyzer.plot_feature_importance(
        grad_importance,
        save_path=os.path.join(save_dir, 'feature_importance.png')
    )
    
    # 3. 预测分析
    print("3. 分析预测结果...")
    model.eval()
    with torch.no_grad():
        predictions, _ = model(X)
        predictions = predictions.squeeze().cpu().numpy()
        
    diagnostics = ModelDiagnostics()
    analysis_results = diagnostics.analyze_predictions(
        predictions, y.cpu().numpy(), dates[-len(predictions):]
    )
    
    diagnostics.plot_prediction_analysis(
        predictions, y.cpu().numpy(), dates[-len(predictions):],
        save_path=os.path.join(save_dir, 'prediction_analysis.png')
    )
    
    # 4. 保存分析结果
    print("4. 保存分析结果...")
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'parameters': sum(p.numel() for p in model.parameters()),
            'device': str(next(model.parameters()).device)
        },
        'analysis_results': analysis_results,
        'feature_importance': {
            'gradient': grad_importance.tolist()
        }
    }
    
    with open(os.path.join(save_dir, 'analysis_report.json'), 'w', encoding='utf-8') as f:
        import json
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"监控报告已生成并保存到: {save_dir}")


if __name__ == "__main__":
    # 测试监控功能
    print("测试监控和可视化功能...")
    
    # 创建示例数据
    n_samples, seq_len, n_features = 100, 30, 28
    attention_weights = np.random.softmax(np.random.randn(n_samples, seq_len), axis=1)
    
    # 创建日期
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    
    # 测试注意力可视化
    att_visualizer = AttentionVisualizer()
    att_visualizer.plot_attention_heatmap(attention_weights[:50])  # 只显示前50个样本
    att_visualizer.plot_attention_time_series(attention_weights)
    
    # 测试特征重要性
    importance_analyzer = FeatureImportanceAnalyzer()
    fake_importance = np.random.exponential(0.1, n_features)  # 模拟重要性分布
    importance_analyzer.plot_feature_importance(fake_importance)
    
    # 测试预测分析
    predictions = np.random.randn(n_samples) * 0.05
    targets = predictions + np.random.randn(n_samples) * 0.02  # 添加噪声
    
    diagnostics = ModelDiagnostics()
    analysis_results = diagnostics.analyze_predictions(predictions, targets, dates)
    print("分析结果:", analysis_results)
    
    diagnostics.plot_prediction_analysis(predictions, targets, dates)
    
    print("监控功能测试完成！")