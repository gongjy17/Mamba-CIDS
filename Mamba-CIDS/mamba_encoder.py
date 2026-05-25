import torch
import torch.nn as nn
from mamba_ssm import Mamba
from config import *

class MambaEncoder(nn.Module):
    """单向Mamba编码器，用于流量和日志特征编码"""
    def __init__(self, vocab_size, num_layers=ENCODER_MAMBA_LAYERS, d_model=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(5000, d_model)  # 足够大的位置嵌入
        
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2,
                bidirectional=False  # 单向Mamba
            ) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch_size, seq_len)
        
        Returns:
            hidden_states: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.shape
        
        # 嵌入层
        x = self.embedding(input_ids)
        
        # 位置嵌入
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_embedding(positions)
        
        # Mamba层
        for layer in self.layers:
            x = layer(x)
        
        # 归一化
        x = self.norm(x)
        
        return x

class PretrainMamba(nn.Module):
    """用于预训练的Mamba模型，包含编码器和解码器"""
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = MambaEncoder(vocab_size)
        
        # 解码器：2层单向Mamba
        self.decoder_layers = nn.ModuleList([
            Mamba(
                d_model=HIDDEN_DIM,
                d_state=16,
                d_conv=4,
                expand=2,
                bidirectional=False
            ) for _ in range(2)
        ])
        
        self.decoder_norm = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Linear(HIDDEN_DIM, vocab_size)
    
    def forward(self, input_ids, mask_positions=None):
        """
        预训练前向传播：掩码-重建任务
        """
        # 编码器
        hidden_states = self.encoder(input_ids)
        
        # 解码器
        x = hidden_states
        for layer in self.decoder_layers:
            x = layer(x)
        
        x = self.decoder_norm(x)
        logits = self.head(x)
        
        return logits

def pretrain_encoder(encoder, dataloader, vocab_size, num_epochs=PRETRAIN_EPOCHS, save_path=None):
    """预训练Mamba编码器"""
    model = PretrainMamba(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            batch_size, seq_len = input_ids.shape
            
            # 随机掩码90%的token
            mask = torch.rand(batch_size, seq_len, device=DEVICE) < MASK_RATIO
            # 确保[CLS] token不被掩码
            mask[:, 0] = False
            
            masked_input = input_ids.clone()
            masked_input[mask] = vocab_size  # 使用特殊的掩码token
            
            # 前向传播
            logits = model(masked_input)
            
            # 只计算掩码位置的损失
            loss = criterion(logits[mask], model.encoder.embedding(input_ids[mask]))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Pretrain Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # 保存编码器权重
    if save_path:
        torch.save(model.encoder.state_dict(), save_path)
    
    # 将预训练好的权重加载到传入的encoder中
    encoder.load_state_dict(model.encoder.state_dict())
    
    return encoder