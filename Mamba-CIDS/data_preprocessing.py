import os
import re
import json
import torch
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP
from torch.utils.data import Dataset, DataLoader
from transformers import WordPieceTokenizer
from config import *

class TrafficLogDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "traffic_tokens": torch.tensor(item["traffic_tokens"], dtype=torch.long),
            "log_tokens": torch.tensor(item["log_tokens"], dtype=torch.long),
            "process_ids": torch.tensor(item["process_ids"], dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }

class DataPreprocessor:
    def __init__(self):
        self.log_tokenizer = self._load_or_train_log_tokenizer()
        self.process_vocab = self._build_process_vocab()
        
    def _load_or_train_log_tokenizer(self):
        """加载或训练日志专用WordPiece分词器"""
        tokenizer_path = f"{DATASET_ROOT}/log_tokenizer.json"
        if os.path.exists(tokenizer_path):
            return WordPieceTokenizer.from_file(tokenizer_path)
        
        # 训练新的分词器
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.trainers import WordPieceTrainer
        from tokenizers.pre_tokenizers import Whitespace
        
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = WordPieceTrainer(
            vocab_size=VOCAB_SIZE_LOG,
            min_frequency=2,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", 
                           "[USER]", "[TIME]", "[IP]", "[PORT]", "[PROCESS]"]
        )
        
        # 收集所有日志文本
        log_files = []
        for root, _, files in os.walk(MD1_DATASET_PATH):
            for file in files:
                if file.endswith(".log"):
                    log_files.append(os.path.join(root, file))
        
        tokenizer.train(log_files, trainer)
        tokenizer.save(tokenizer_path)
        return WordPieceTokenizer.from_file(tokenizer_path)
    
    def _build_process_vocab(self):
        """构建进程名词汇表"""
        process_vocab = {"[PAD]": 0, "[UNK]": 1}
        process_set = set()
        
        # 收集所有进程名
        for root, _, files in os.walk(MD1_DATASET_PATH):
            for file in files:
                if file.endswith(".log"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        for line in f:
                            match = re.search(r'process=(\w+)', line)
                            if match:
                                process_set.add(match.group(1))
        
        for i, process in enumerate(process_set, start=2):
            process_vocab[process] = i
        
        return process_vocab
    
    def variable_normalization(self, log_line):
        """日志变量归一化"""
        # 替换用户名
        log_line = re.sub(r'user \w+', r'user [USER]', log_line)
        # 替换时间戳
        log_line = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', r'[TIME]', log_line)
        # 替换IP地址
        log_line = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', r'[IP]', log_line)
        # 替换端口号
        log_line = re.sub(r':\d{1,5}\b', r':[PORT]', log_line)
        return log_line
    
    def extract_process_name(self, log_line):
        """从日志行提取进程名"""
        match = re.search(r'process=(\w+)', log_line)
        if match:
            return match.group(1)
        return "[UNK]"
    
    def preprocess_traffic(self, pcap_file):
        """流量预处理：流拆分、bi-gram编码、截断填充"""
        packets = rdpcap(pcap_file)
        flows = {}
        
        # 按五元组拆分流
        for pkt in packets:
            if IP not in pkt:
                continue
            
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto = pkt[IP].proto
            
            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
            else:
                continue
            
            flow_key = (src_ip, dst_ip, src_port, dst_port, proto)
            
            if flow_key not in flows:
                flows[flow_key] = []
            
            # 提取payload并进行bi-gram编码
            payload = bytes(pkt.payload)
            tokens = []
            for i in range(len(payload) - 1):
                token = f"{payload[i]:02x}{payload[i+1]:02x}"
                tokens.append(int(token, 16))  # 转换为0-65535的整数
            
            # 截断或填充到MAX_TOKENS_PER_PACKET
            if len(tokens) > MAX_TOKENS_PER_PACKET:
                tokens = tokens[:MAX_TOKENS_PER_PACKET]
            else:
                tokens += [0] * (MAX_TOKENS_PER_PACKET - len(tokens))  # 0对应"0000"
            
            flows[flow_key].append(tokens)
        
        # 每个流取前MAX_PACKETS_PER_FLOW个数据包
        all_flow_tokens = []
        for flow in flows.values():
            if len(flow) > MAX_PACKETS_PER_FLOW:
                flow = flow[:MAX_PACKETS_PER_FLOW]
            else:
                # 填充空数据包
                empty_pkt = [0] * MAX_TOKENS_PER_PACKET
                flow += [empty_pkt] * (MAX_PACKETS_PER_FLOW - len(flow))
            
            all_flow_tokens.extend(flow)
        
        # 最终流量tokens形状: (MAX_PACKETS_PER_FLOW * 流数, MAX_TOKENS_PER_PACKET)
        # 这里简化为取前MAX_PACKETS_PER_FLOW个数据包
        if len(all_flow_tokens) > MAX_PACKETS_PER_FLOW:
            all_flow_tokens = all_flow_tokens[:MAX_PACKETS_PER_FLOW]
        else:
            empty_pkt = [0] * MAX_TOKENS_PER_PACKET
            all_flow_tokens += [empty_pkt] * (MAX_PACKETS_PER_FLOW - len(all_flow_tokens))
        
        return np.array(all_flow_tokens)
    
    def preprocess_logs(self, log_file):
        """日志预处理：变量归一化、分词、多通道特征提取"""
        log_tokens = []
        process_ids = []
        
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 变量归一化
                normalized_line = self.variable_normalization(line)
                
                # 提取进程名
                process_name = self.extract_process_name(line)
                process_id = self.process_vocab.get(process_name, 1)  # 1是[UNK]
                
                # 分词
                encoding = self.log_tokenizer.encode(normalized_line)
                tokens = encoding.ids
                
                # 截断或填充
                if len(tokens) > LOG_MAX_SEQ_LENGTH:
                    tokens = tokens[:LOG_MAX_SEQ_LENGTH]
                else:
                    tokens += [0] * (LOG_MAX_SEQ_LENGTH - len(tokens))
                
                log_tokens.append(tokens)
                process_ids.append(process_id)
        
        # 取前LOG_MAX_SEQ_LENGTH条日志
        if len(log_tokens) > LOG_MAX_SEQ_LENGTH:
            log_tokens = log_tokens[:LOG_MAX_SEQ_LENGTH]
            process_ids = process_ids[:LOG_MAX_SEQ_LENGTH]
        else:
            empty_tokens = [0] * LOG_MAX_SEQ_LENGTH
            log_tokens += [empty_tokens] * (LOG_MAX_SEQ_LENGTH - len(log_tokens))
            process_ids += [0] * (LOG_MAX_SEQ_LENGTH - len(process_ids))
        
        return np.array(log_tokens), np.array(process_ids)
    
    def align_traffic_logs(self, traffic_file, log_file, time_window=60):
        """基于时序和实体的流量-日志对齐算法"""
        # 这里简化实现，实际应根据时间戳和IP地址对齐
        # 论文中：以网络流持续时间为窗口，匹配时间戳在窗口内且IP相关的日志
        traffic_tokens = self.preprocess_traffic(traffic_file)
        log_tokens, process_ids = self.preprocess_logs(log_file)
        
        return traffic_tokens, log_tokens, process_ids
    
    def load_dataset(self):
        """加载MD1数据集并预处理"""
        data_list = []
        label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
        
        for label_name in CLASS_NAMES:
            label_dir = f"{MD1_DATASET_PATH}/{label_name}"
            if not os.path.exists(label_dir):
                continue
            
            for sample_id in os.listdir(label_dir):
                sample_dir = f"{label_dir}/{sample_id}"
                if not os.path.isdir(sample_dir):
                    continue
                
                traffic_file = f"{sample_dir}/traffic.pcap"
                log_file = f"{sample_dir}/system.log"
                
                if not os.path.exists(traffic_file) or not os.path.exists(log_file):
                    continue
                
                try:
                    traffic_tokens, log_tokens, process_ids = self.align_traffic_logs(traffic_file, log_file)
                    
                    data_list.append({
                        "traffic_tokens": traffic_tokens,
                        "log_tokens": log_tokens,
                        "process_ids": process_ids,
                        "label": label_map[label_name]
                    })
                except Exception as e:
                    print(f"Error processing {sample_dir}: {e}")
                    continue
        
        # 划分训练集、验证集、测试集 (8:1:1)
        np.random.shuffle(data_list)
        total = len(data_list)
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)
        
        train_data = data_list[:train_size]
        val_data = data_list[train_size:train_size+val_size]
        test_data = data_list[train_size+val_size:]
        
        return (
            TrafficLogDataset(train_data),
            TrafficLogDataset(val_data),
            TrafficLogDataset(test_data)
        )

def get_data_loaders():
    """获取数据加载器"""
    preprocessor = DataPreprocessor()
    train_dataset, val_dataset, test_dataset = preprocessor.load_dataset()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader