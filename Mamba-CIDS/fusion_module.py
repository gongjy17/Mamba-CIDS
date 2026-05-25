import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from config import *

class LocalAttentionPooling(nn.Module):
    """局部注意力池化，用于增强局部特征"""
    def __init__(self, window_size=3, d_model=HIDDEN_DIM):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            enhanced_x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 滑动窗口处理
        # 填充以保持序列长度不变
        padding = self.window_size // 2
        x_padded = F.pad(x, (0, 0, padding, padding), mode="constant", value=0)
        
        # 提取窗口
        windows = x_padded.unfold(1, self.window_size, 1)  # (batch_size, seq_len, d_model, window_size)
        windows = windows.permute(0, 1, 3, 2)  # (batch_size, seq_len, window_size, d_model)
        
        # 计算自注意力
        Q = K = V = windows
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 应用注意力权重
        enhanced_windows = torch.matmul(attn_weights, V)
        
        # 取窗口中心作为输出
        enhanced_x = enhanced_windows[:, :, self.window_size // 2, :]
        
        return enhanced_x

class DynamicGatedFusion(nn.Module):
    """动态门控融合模块"""
    def __init__(self, d_model=HIDDEN_DIM):
        super().__init__()
        self.local_attn = LocalAttentionPooling()
        
        # 门控网络
        self.W_t = nn.Linear(d_model, d_model)
        self.W_l = nn.Linear(d_model, d_model)
        self.b_g = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, traffic_hidden, log_hidden):
        """
        Args:
            traffic_hidden: (batch_size, traffic_seq_len, d_model)
            log_hidden: (batch_size, log_seq_len, d_model)
        
        Returns:
            fused_features: (batch_size, d_model)
        """
        # 局部特征增强
        traffic_enhanced = self.local_attn(traffic_hidden)
        log_enhanced = self.local_attn(log_hidden)
        
        # 全局平均池化
        traffic_global = torch.mean(traffic_enhanced, dim=1)
        log_global = torch.mean(log_enhanced, dim=1)
        
        # 动态权重计算
        g = torch.sigmoid(self.W_t(traffic_global) + self.W_l(log_global) + self.b_g)
        
        # 门控融合
        fused_features = g * traffic_global + (1 - g) * log_global
        
        return fused_features

class LightweightBidirectionalMamba(nn.Module):
    """轻量化双向Mamba，用于深度跨模态融合"""
    def __init__(self, num_layers=FUSION_MAMBA_LAYERS, d_model=HIDDEN_DIM):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2,
                bidirectional=True  # 双向Mamba
            ) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            hidden_states: (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return x

class TwoStageFusion(nn.Module):
    """两阶段融合模块"""
    def __init__(self):
        super().__init__()
        self.dynamic_gate = DynamicGatedFusion()
        self.bi_mamba = LightweightBidirectionalMamba()
    
    def forward(self, traffic_hidden, log_hidden):
        """
        Args:
            traffic_hidden: (batch_size, traffic_seq_len, d_model)
            log_hidden: (batch_size, log_seq_len, d_model)
        
        Returns:
            deep_fused: (batch_size, seq_len, d_model)
        """
        # 第一阶段：动态门控融合
        gate_fused = self.dynamic_gate(traffic_hidden, log_hidden)
        
        # 扩展序列维度以适配双向Mamba
        # 论文中输出为(40, 768)，这里简化为(1, 768)扩展到(40, 768)
        gate_fused_expanded = gate_fused.unsqueeze(1).repeat(1, 40, 1)
        
        # 第二阶段：双向Mamba深度融合
        deep_fused = self.bi_mamba(gate_fused_expanded)
        
        return deep_fused