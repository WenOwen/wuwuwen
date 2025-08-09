"""
Siamese-LSTM + 时序注意力模型实现
用于多源时序数据的股票收益预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math
from torch.utils.checkpoint import checkpoint


class MultiHeadTimeAttention(nn.Module):
    """多头时间注意力机制"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadTimeAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 可学习的位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, hidden_dim) * 0.1)
        
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: LSTM输出 (batch_size, seq_len, hidden_dim)
            
        Returns:
            weighted_output: 加权后的输出 (batch_size, hidden_dim)
            attention_weights: 注意力权重 (batch_size, num_heads, seq_len)
        """
        batch_size, seq_len, _ = lstm_output.shape
        
        # 添加位置编码
        pos_enc = self.pos_encoding[:, :seq_len, :].expand(batch_size, -1, -1)
        lstm_output = lstm_output + pos_enc
        
        # 计算Q、K、V
        Q = self.query(lstm_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(lstm_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(lstm_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # 输出投影
        output = self.out_proj(attended)
        
        # 全局池化：对序列维度取平均
        weighted_output = output.mean(dim=1)  # (batch_size, hidden_dim)
        
        # 返回平均注意力权重
        avg_attention = attention_weights.mean(dim=1)  # (batch_size, seq_len)
        
        return weighted_output, avg_attention


class TimeAttention(nn.Module):
    """简单时间注意力机制（保持向后兼容）"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super(TimeAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: LSTM输出 (batch_size, seq_len, hidden_dim)
            
        Returns:
            weighted_output: 加权后的输出 (batch_size, hidden_dim)
            attention_weights: 注意力权重 (batch_size, seq_len)
        """
        # 计算注意力分数
        attention_scores = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_len)
        
        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        
        # 应用dropout
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
        weighted_output = (lstm_output * attention_weights_expanded).sum(dim=1)  # (batch_size, hidden_dim)
        
        return weighted_output, attention_weights


class SiameseLSTMModel(nn.Module):
    """Siamese-LSTM + 时序注意力模型"""
    
    def __init__(self, 
                 input_dim: int = 29,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 dropout: float = 0.2,
                 output_dim: int = 1,
                 task_type: str = 'regression',
                 use_multihead_attention: bool = True,
                 num_attention_heads: int = 8,
                 gradient_checkpointing: bool = False,
                 layer_norm: bool = False,
                 residual_connections: bool = False):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比例
            output_dim: 输出维度
            task_type: 任务类型 ('regression' 或 'classification')
            use_multihead_attention: 是否使用多头注意力
            num_attention_heads: 注意力头数
            gradient_checkpointing: 是否使用梯度检查点（节省显存）
            layer_norm: 是否添加层归一化
            residual_connections: 是否添加残差连接
        """
        super(SiameseLSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.task_type = task_type
        self.gradient_checkpointing = gradient_checkpointing
        self.layer_norm = layer_norm
        self.residual_connections = residual_connections
        
        # 共享LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 时间注意力机制
        if use_multihead_attention:
            self.time_attention = MultiHeadTimeAttention(hidden_dim, num_attention_heads, dropout)
        else:
            self.time_attention = TimeAttention(hidden_dim, dropout)
        
        # 层归一化
        self.lstm_layer_norm = nn.LayerNorm(hidden_dim) if layer_norm else None
        self.attention_layer_norm = nn.LayerNorm(hidden_dim) if layer_norm else None
        
        # 增强的全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name and param.dim() >= 2:
                    # LSTM权重使用Xavier初始化
                    nn.init.xavier_uniform_(param)
                elif ('norm' in name or 'BatchNorm' in name) and param.dim() >= 1:
                    # BatchNorm和LayerNorm权重初始化为1
                    nn.init.constant_(param, 1)
                elif param.dim() >= 2:
                    # 其他2D以上权重使用He初始化
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif param.dim() == 1:
                    # 1D权重使用正态分布初始化
                    nn.init.normal_(param, 0, 0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, seq_len, input_dim)
            
        Returns:
            output: 模型输出 (batch_size, output_dim)
            attention_weights: 注意力权重 (batch_size, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM前向传播（支持梯度检查点）
        if self.gradient_checkpointing and self.training:
            lstm_out, (h_n, c_n) = checkpoint(self.lstm, x)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        
        # 层归一化 + 残差连接（如果启用）
        if self.layer_norm:
            lstm_out_norm = self.lstm_layer_norm(lstm_out)
            if self.residual_connections and lstm_out.shape[-1] == x.shape[-1]:
                # 只有当维度匹配时才能添加残差连接
                lstm_out_norm = lstm_out_norm + x
        else:
            lstm_out_norm = lstm_out
        
        # 时间注意力（支持梯度检查点）
        if self.gradient_checkpointing and self.training:
            attended_output, attention_weights = checkpoint(
                self.time_attention, lstm_out_norm
            )
        else:
            attended_output, attention_weights = self.time_attention(lstm_out_norm)
        
        # 注意力层的层归一化 + 残差连接
        if self.layer_norm:
            attended_output_norm = self.attention_layer_norm(attended_output)
            if self.residual_connections:
                # 对于注意力输出，残差连接到LSTM输出的平均值
                lstm_mean = lstm_out_norm.mean(dim=1)  # (batch_size, hidden_dim)
                attended_output_norm = attended_output_norm + lstm_mean
        else:
            attended_output_norm = attended_output
        
        # 全连接层
        output = self.fc(attended_output_norm)
        
        # 根据任务类型处理输出
        if self.task_type == 'classification':
            output = torch.sigmoid(output)
            
        return output, attention_weights
        
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取特征重要性（基于梯度）
        
        Args:
            x: 输入数据 (batch_size, seq_len, input_dim)
            
        Returns:
            importance: 特征重要性 (input_dim,)
        """
        x.requires_grad_(True)
        output, _ = self.forward(x)
        
        # 计算输出对输入的梯度
        grad = torch.autograd.grad(
            outputs=output.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 计算特征重要性（梯度的平均绝对值）
        importance = grad.abs().mean(dim=(0, 1))
        
        return importance


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 模型预测 (batch_size, 1)
            targets: 真实标签 (batch_size,)
        """
        # 确保输入在[0,1]范围内
        inputs = torch.clamp(inputs.squeeze(), 1e-8, 1 - 1e-8)
        targets = targets.float()
        
        # 计算二元交叉熵
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # 计算pt
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        
        # 计算alpha权重
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # 计算focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedMSELoss(nn.Module):
    """加权均方误差损失"""
    
    def __init__(self, reduction: str = 'mean'):
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: 模型预测 (batch_size, 1)
            targets: 真实标签 (batch_size,)
            weights: 样本权重 (batch_size,)
        """
        inputs = inputs.squeeze()
        mse_loss = (inputs - targets) ** 2
        
        if weights is not None:
            mse_loss = mse_loss * weights
            
        if self.reduction == 'mean':
            return mse_loss.mean()
        elif self.reduction == 'sum':
            return mse_loss.sum()
        else:
            return mse_loss


class ModelEvaluator:
    """模型评估器"""
    
    @staticmethod
    def calculate_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算信息系数(IC)"""
        return np.corrcoef(predictions, targets)[0, 1]
        
    @staticmethod
    def calculate_rank_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算排序信息系数(Rank IC)"""
        from scipy.stats import spearmanr
        return spearmanr(predictions, targets)[0]
        
    @staticmethod
    def calculate_accuracy(predictions: np.ndarray, targets: np.ndarray, 
                         threshold: float = 0.5) -> float:
        """计算分类准确率"""
        pred_binary = (predictions > threshold).astype(int)
        target_binary = (targets > 0).astype(int)
        return (pred_binary == target_binary).mean()
        
    @staticmethod
    def calculate_auc(predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算AUC"""
        try:
            from sklearn.metrics import roc_auc_score
            target_binary = (targets > 0).astype(int)
            return roc_auc_score(target_binary, predictions)
        except ImportError:
            print("警告：sklearn未安装，无法计算AUC")
            return 0.0
            
    @staticmethod
    def calculate_sharpe_ratio(predictions: np.ndarray, targets: np.ndarray,
                             top_k: int = None) -> float:
        """计算基于预测的投资组合夏普比率"""
        if top_k is None:
            top_k = len(predictions) // 10  # 默认取前10%
            
        # 根据预测排序，选择top_k只股票
        sorted_indices = np.argsort(predictions)[::-1]
        top_returns = targets[sorted_indices[:top_k]]
        
        if len(top_returns) == 0 or top_returns.std() == 0:
            return 0.0
            
        return top_returns.mean() / top_returns.std() * np.sqrt(252)  # 年化夏普比率


# 注意：配置已迁移到config.py文件中统一管理
# 此函数保留用于向后兼容，但建议使用config.py中的配置
def create_model_config():
    """创建模型配置（已废弃，请使用config.py）"""
    import warnings
    warnings.warn(
        "create_model_config()已废弃，请使用config.py中的Config类",
        DeprecationWarning,
        stacklevel=2
    )
    
    # 返回基本配置以保持向后兼容
    return {
        'input_dim': 29,
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout': 0.2,
        'output_dim': 1,
        'task_type': 'regression',
        'use_multihead_attention': True,
        'num_attention_heads': 8,
    }


if __name__ == "__main__":
    # 测试模型 - 使用新的配置系统
    try:
        from config import Config
        config = Config.MODEL_CONFIG
    except ImportError:
        # 如果config.py不存在，使用默认配置
        config = create_model_config()
    
    # 创建模型
    model = SiameseLSTMModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        task_type=config['task_type'],
        use_multihead_attention=config['use_multihead_attention'],
        num_attention_heads=config['num_attention_heads']
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size, seq_len, input_dim = 32, 30, 29
    x = torch.randn(batch_size, seq_len, input_dim)
    
    with torch.no_grad():
        output, attention_weights = model(x)
        
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"注意力权重和: {attention_weights.sum(dim=1).mean():.4f}")
    
    # 测试损失函数
    targets = torch.randn(batch_size)
    
    # 回归损失
    mse_loss = WeightedMSELoss()
    loss_value = mse_loss(output, targets)
    print(f"MSE损失: {loss_value:.4f}")
    
    # 分类损失（将输出转为概率）
    focal_loss = FocalLoss()
    prob_output = torch.sigmoid(output)
    binary_targets = (targets > 0).float()
    focal_loss_value = focal_loss(prob_output, binary_targets)
    print(f"Focal损失: {focal_loss_value:.4f}")
    
    # 注意：以下是未训练模型在随机数据上的测试，仅用于验证模型结构
    print("=" * 60)
    print("注意：这是未训练模型的测试结果，仅用于验证模型结构！")
    print("真实性能需要在实际数据上训练后才能评估。")
    print("=" * 60)
    
    # 可选：测试评估指标功能（但结果无实际意义）
    if False:  # 默认关闭随机数据测试
        evaluator = ModelEvaluator()
        pred_np = output.squeeze().numpy()
        target_np = targets.numpy()
        
        ic = evaluator.calculate_ic(pred_np, target_np)
        accuracy = evaluator.calculate_accuracy(pred_np, target_np)
        
        print(f"随机数据IC: {ic:.4f}")
        print(f"随机数据准确率: {accuracy:.4f}")