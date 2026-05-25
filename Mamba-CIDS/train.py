import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from data_preprocessing import get_data_loaders
from model import MambaCIDS
from config import *

def train_model():
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # 初始化模型
    model = MambaCIDS().to(DEVICE)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    best_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            traffic_tokens = batch["traffic_tokens"].to(DEVICE)
            log_tokens = batch["log_tokens"].to(DEVICE)
            process_ids = batch["process_ids"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            # 前向传播
            logits = model(traffic_tokens, log_tokens, process_ids)
            loss = criterion(logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                traffic_tokens = batch["traffic_tokens"].to(DEVICE)
                log_tokens = batch["log_tokens"].to(DEVICE)
                process_ids = batch["process_ids"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                
                logits = model(traffic_tokens, log_tokens, process_ids)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        macro_precision = precision_score(all_labels, all_preds, average="macro")
        macro_recall = recall_score(all_labels, all_preds, average="macro")
        macro_accuracy = accuracy_score(all_labels, all_preds)
        
        print(f"\nValidation Results:")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro Accuracy: {macro_accuracy:.4f}")
        
        # 保存最佳模型
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), "best_mamba_cids.pth")
            print(f"Best model saved with F1: {best_f1:.4f}")
    
    print(f"\nTraining completed. Best validation F1: {best_f1:.4f}")
    
    # 测试最佳模型
    model.load_state_dict(torch.load("best_mamba_cids.pth"))
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            traffic_tokens = batch["traffic_tokens"].to(DEVICE)
            log_tokens = batch["log_tokens"].to(DEVICE)
            process_ids = batch["process_ids"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            logits = model(traffic_tokens, log_tokens, process_ids)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算测试指标
    test_f1 = f1_score(all_labels, all_preds, average="macro")
    test_precision = precision_score(all_labels, all_preds, average="macro")
    test_recall = recall_score(all_labels, all_preds, average="macro")
    test_accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\nTest Results:")
    print(f"Macro F1: {test_f1:.4f}")
    print(f"Macro Precision: {test_precision:.4f}")
    print(f"Macro Recall: {test_recall:.4f}")
    print(f"Macro Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    train_model()