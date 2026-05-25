import torch

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径
DATASET_ROOT = "/dataset"
MD1_DATASET_PATH = f"{DATASET_ROOT}/MD1"

# 数据预处理参数
MAX_PACKETS_PER_FLOW = 5
MAX_TOKENS_PER_PACKET = 96
PAD_TOKEN = "0000"
LOG_MAX_SEQ_LENGTH = 128

# 模型参数
HIDDEN_DIM = 768
ENCODER_MAMBA_LAYERS = 4
FUSION_MAMBA_LAYERS = 2
VOCAB_SIZE_TRAFFIC = 65536  # 2^16个bi-gram组合
VOCAB_SIZE_LOG = 30000      # 日志词汇表大小

# 训练参数
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 50
PRETRAIN_EPOCHS = 20
MASK_RATIO = 0.9

# 分类参数
NUM_CLASSES = 6  # normal, mirai, ransomware, disk-wipe, resource-hijacking, end-point-dos
CLASS_NAMES = [
    "normal", "mirai", "ransomware", "disk-wipe", 
    "resource-hijacking", "end-point-dos"
]