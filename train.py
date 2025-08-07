#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的训练脚本
使用新的27维特征数据进行训练
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

# 导入模型
from model import SiameseLSTMModel, ModelEvaluator


class ImprovedTrainer:
    """改进的训练器"""
    
    def __init__(self, data_dir: str = "./data/processed_v2"):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = ModelEvaluator()
        
        print(f"使用设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_data(self):
        """加载处理后的数据"""
        print("正在加载数据...")
        
        # 加载特征和目标数据
        X = np.load(os.path.join(self.data_dir, "X_features.npy"))
        y = np.load(os.path.join(self.data_dir, "y_targets.npy"))
        
        # 加载股票代码
        with open(os.path.join(self.data_dir, "stock_codes.json"), 'r', encoding='utf-8') as f:
            stock_codes = json.load(f)
        
        # 加载数据信息
        with open(os.path.join(self.data_dir, "data_info.json"), 'r', encoding='utf-8') as f:
            data_info = json.load(f)
        
        print(f"数据形状: X={X.shape}, y={y.shape}")
        print(f"特征维度: {data_info['feature_dims']}")
        print(f"涉及股票: {data_info['num_stocks']} 只")
        
        return X, y, stock_codes, data_info
    
    def create_data_loaders(self, X, y, train_ratio=0.8, batch_size=64):
        """创建数据加载器"""
        # 时序数据按时间顺序划分
        split_idx = int(len(X) * train_ratio)
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"训练集: {X_train.shape[0]} 样本")
        print(f"验证集: {X_val.shape[0]} 样本")
        
        # 转换为tensor
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def create_model(self, input_dim=27):
        """创建模型"""
        model = SiameseLSTMModel(
            input_dim=input_dim,
            hidden_dim=64,  # 增加隐藏层维度
            num_layers=2,
            dropout=0.3,
            output_dim=1,
            task_type='regression'
        ).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"模型参数总数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        return model
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        predictions = []
        targets = []
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 前向传播
            output, attention_weights = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(output.squeeze().detach().cpu().numpy())
            targets.extend(y_batch.cpu().numpy())
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        ic = self.evaluator.calculate_ic(np.array(predictions), np.array(targets))
        
        return {'loss': avg_loss, 'ic': ic}
    
    def validate_epoch(self, model, val_loader, criterion):
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        attention_weights_list = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output, attention_weights = model(X_batch)
                loss = criterion(output.squeeze(), y_batch)
                
                total_loss += loss.item()
                predictions.extend(output.squeeze().cpu().numpy())
                targets.extend(y_batch.cpu().numpy())
                attention_weights_list.append(attention_weights.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 计算评估指标
        ic = self.evaluator.calculate_ic(predictions, targets)
        rank_ic = self.evaluator.calculate_rank_ic(predictions, targets)
        
        # 计算其他指标
        mse = mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return {
            'loss': avg_loss, 'ic': ic, 'rank_ic': rank_ic, 
            'mse': mse, 'r2': r2,
            'attention_weights': np.concatenate(attention_weights_list, axis=0),
            'predictions': predictions,
            'targets': targets
        }
    
    def train_model(self, model, train_loader, val_loader, num_epochs=20):
        """训练模型"""
        print("开始训练模型...")
        
        # 优化器和学习率调度器
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        criterion = nn.MSELoss()
        
        # 早停
        best_ic = -float('inf')
        patience = 5
        patience_counter = 0
        best_model_state = None
        
        train_history = {'loss': [], 'ic': []}
        val_history = {'loss': [], 'ic': [], 'rank_ic': [], 'mse': [], 'r2': []}
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # 验证
            val_metrics = self.validate_epoch(model, val_loader, criterion)
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            train_history['loss'].append(train_metrics['loss'])
            train_history['ic'].append(train_metrics['ic'])
            val_history['loss'].append(val_metrics['loss'])
            val_history['ic'].append(val_metrics['ic'])
            val_history['rank_ic'].append(val_metrics['rank_ic'])
            val_history['mse'].append(val_metrics['mse'])
            val_history['r2'].append(val_metrics['r2'])
            
            # 打印结果
            print(f"训练 - Loss: {train_metrics['loss']:.6f}, IC: {train_metrics['ic']:.4f}")
            print(f"验证 - Loss: {val_metrics['loss']:.6f}, IC: {val_metrics['ic']:.4f}, "
                  f"Rank IC: {val_metrics['rank_ic']:.4f}")
            print(f"验证 - MSE: {val_metrics['mse']:.6f}, R²: {val_metrics['r2']:.4f}")
            print(f"学习率: {current_lr:.2e}")
            
            # 早停检查
            if val_metrics['ic'] > best_ic:
                best_ic = val_metrics['ic']
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"★ 新的最佳IC: {best_ic:.4f}")
            else:
                patience_counter += 1
                print(f"早停计数: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("早停触发！")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"恢复最佳模型，IC: {best_ic:.4f}")
        
        # 最终验证
        final_metrics = self.validate_epoch(model, val_loader, criterion)
        
        return {
            'model': model,
            'train_history': train_history,
            'val_history': val_history,
            'final_metrics': final_metrics,
            'best_ic': best_ic
        }
    
    def plot_training_history(self, train_history, val_history, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(train_history['loss'], label='训练损失', color='blue')
        axes[0, 0].plot(val_history['loss'], label='验证损失', color='red')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # IC曲线
        axes[0, 1].plot(train_history['ic'], label='训练IC', color='blue')
        axes[0, 1].plot(val_history['ic'], label='验证IC', color='red')
        axes[0, 1].set_title('IC曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rank IC曲线
        axes[1, 0].plot(val_history['rank_ic'], label='验证Rank IC', color='green')
        axes[1, 0].set_title('Rank IC曲线')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Rank IC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # R²曲线
        axes[1, 1].plot(val_history['r2'], label='验证R²', color='purple')
        axes[1, 1].set_title('R²曲线')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")
        
        plt.show()
    
    def analyze_predictions(self, predictions, targets, save_path=None):
        """分析预测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 预测vs实际散点图
        axes[0, 0].scatter(targets, predictions, alpha=0.6, s=10)
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('实际收益率 (%)')
        axes[0, 0].set_ylabel('预测收益率 (%)')
        axes[0, 0].set_title('预测vs实际')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 残差图
        residuals = predictions - targets
        axes[0, 1].scatter(predictions, residuals, alpha=0.6, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 1].set_xlabel('预测收益率 (%)')
        axes[0, 1].set_ylabel('残差 (%)')
        axes[0, 1].set_title('残差分析')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 预测分布
        axes[1, 0].hist(predictions, bins=50, alpha=0.7, label='预测', color='blue')
        axes[1, 0].hist(targets, bins=50, alpha=0.7, label='实际', color='red')
        axes[1, 0].set_xlabel('收益率 (%)')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('分布对比')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 时序对比（最近500个点）
        n_show = min(500, len(predictions))
        show_idx = np.arange(len(predictions) - n_show, len(predictions))
        axes[1, 1].plot(show_idx, predictions[-n_show:], label='预测', alpha=0.8)
        axes[1, 1].plot(show_idx, targets[-n_show:], label='实际', alpha=0.8)
        axes[1, 1].set_xlabel('样本索引')
        axes[1, 1].set_ylabel('收益率 (%)')
        axes[1, 1].set_title(f'时序对比（最近{n_show}个点）')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测分析图已保存到: {save_path}")
        
        plt.show()
    
    def save_results(self, results, save_dir="./results_improved"):
        """保存训练结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        torch.save(results['model'].state_dict(), 
                  os.path.join(save_dir, "best_model.pth"))
        
        # 保存训练历史
        history_df = pd.DataFrame({
            'epoch': range(len(results['train_history']['loss'])),
            'train_loss': results['train_history']['loss'],
            'train_ic': results['train_history']['ic'],
            'val_loss': results['val_history']['loss'],
            'val_ic': results['val_history']['ic'],
            'val_rank_ic': results['val_history']['rank_ic'],
            'val_mse': results['val_history']['mse'],
            'val_r2': results['val_history']['r2']
        })
        history_df.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)
        
        # 保存最终指标
        final_metrics = {
            'best_ic': results['best_ic'],
            'final_ic': results['final_metrics']['ic'],
            'final_rank_ic': results['final_metrics']['rank_ic'],
            'final_mse': results['final_metrics']['mse'],
            'final_r2': results['final_metrics']['r2'],
            'final_loss': results['final_metrics']['loss']
        }
        
        with open(os.path.join(save_dir, "final_metrics.json"), 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"训练结果已保存到: {save_dir}")


def main():
    """主训练流程"""
    # 创建训练器
    trainer = ImprovedTrainer()
    
    # 加载数据
    X, y, stock_codes, data_info = trainer.load_data()
    
    # 创建数据加载器
    train_loader, val_loader = trainer.create_data_loaders(X, y, batch_size=128)
    
    # 创建模型
    input_dim = data_info['feature_dims']['total']
    model = trainer.create_model(input_dim=input_dim)
    
    # 训练模型
    results = trainer.train_model(model, train_loader, val_loader, num_epochs=30)
    
    # 绘制训练历史
    trainer.plot_training_history(
        results['train_history'], 
        results['val_history'],
        save_path="./results_improved/training_history.png"
    )
    
    # 分析预测结果
    trainer.analyze_predictions(
        results['final_metrics']['predictions'],
        results['final_metrics']['targets'],
        save_path="./results_improved/prediction_analysis.png"
    )
    
    # 保存结果
    trainer.save_results(results)
    
    print("\n" + "="*60)
    print("训练完成！")
    print(f"最佳IC: {results['best_ic']:.4f}")
    print(f"最终IC: {results['final_metrics']['ic']:.4f}")
    print(f"最终Rank IC: {results['final_metrics']['rank_ic']:.4f}")
    print(f"最终R²: {results['final_metrics']['r2']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()