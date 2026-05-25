import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import get_data_loaders
from model import MambaCIDS
from config import *

def evaluate_model():
    # 获取数据加载器
    _, _, test_loader = get_data_loaders()
    
    # 加载模型
    model = MambaCIDS().to(DEVICE)
    model.load_state_dict(torch.load("best_mamba_cids.pth"))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # 推理时间测量
    total_time = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            traffic_tokens = batch["traffic_tokens"].to(DEVICE)
            log_tokens = batch["log_tokens"].to(DEVICE)
            process_ids = batch["process_ids"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            # 测量推理时间
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            logits = model(traffic_tokens, log_tokens, process_ids)
            end_event.record()
            
            torch.cuda.synchronize()
            inference_time = start_event.elapsed_time(end_event)  # 毫秒
            total_time += inference_time
            num_samples += traffic_tokens.shape[0]
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算平均推理时间
    avg_inference_time = total_time / num_samples
    print(f"Average inference time per sample: {avg_inference_time:.2f} ms")
    
    # 分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()