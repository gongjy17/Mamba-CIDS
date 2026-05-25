import torch
import torch.nn as nn
from mamba_encoder import MambaEncoder
from fusion_module import TwoStageFusion
from config import *

class MambaCIDS(nn.Module):
    """Mamba-CIDS整体模型"""
    def __init__(self):
        super().__init__()
        # 流量编码器
        self.traffic_encoder = MambaEncoder(VOCAB_SIZE_TRAFFIC)
        
        # 日志编码器
        self.log_encoder = MambaEncoder(VOCAB_SIZE_LOG)
        
        # 进程名嵌入
        self.process_embedding = nn.Embedding(1000, HIDDEN_DIM)  # 假设最多1000个进程
        
        # 两阶段融合模块
        self.fusion = TwoStageFusion()
        
        # 分类头
        self.classifier = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
    
    def forward(self, traffic_tokens, log_tokens, process_ids):
        """
        Args:
            traffic_tokens: (batch_size, num_packets, tokens_per_packet)
            log_tokens: (batch_size, num_logs, tokens_per_log)
            process_ids: (batch_size, num_logs)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = traffic_tokens.shape[0]
        
        # 流量编码：展平数据包维度
        traffic_tokens_flat = traffic_tokens.view(batch_size, -1)
        traffic_hidden = self.traffic_encoder(traffic_tokens_flat)
        
        # 日志编码：展平日志条目维度
        log_tokens_flat = log_tokens.view(batch_size, -1)
        log_hidden = self.log_encoder(log_tokens_flat)
        
        # 加入进程名通道特征
        process_emb = self.process_embedding(process_ids)
        process_emb_flat = process_emb.view(batch_size, -1, HIDDEN_DIM)
        log_hidden = log_hidden + process_emb_flat
        
        # 两阶段融合
        fused_features = self.fusion(traffic_hidden, log_hidden)
        
        # 全局平均池化
        pooled_features = torch.mean(fused_features, dim=1)
        
        # 分类
        logits = self.classifier(pooled_features)
        
        return logits