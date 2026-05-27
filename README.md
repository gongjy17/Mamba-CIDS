# Mamba-CIDS：基于双流Mamba编码和两阶段融合的多数据源入侵检测方法
```text
这是论文《基于双流Mamba编码和两阶段融合的多数据源入侵检测方法》的PyTorch实现。该方法首次将Mamba架构引入多数据源入侵检测领域，通过双流Mamba编码器分别处理流量和日志数据，并采用两阶段融合策略实现深度跨模态语义交互，在MD1数据集上取得了98.31%的宏平均F1分数。
```
# 环境要求
```text
- Python 3.8+
- PyTorch 1.11.0+
- CUDA 11.3+ (推荐使用GPU加速)
- 其他依赖见requirements.txt
```
# 数据集
```text
CIC-IDS-2018：https://www.unb.ca/cic/datasets/ids-2018.html
CREME：https://github.com/buihuukhoi/CREME
```
# 代码结构
```text
config.py: 全局配置参数，包括数据路径、模型参数和训练参数
data_preprocessing.py: 数据预处理模块，实现流量和日志的预处理及对齐
mamba_encoder.py: 单向 Mamba 编码器实现，包含预训练逻辑
fusion_module.py: 两阶段融合模块，实现动态门控融合和双向 Mamba 深度融合
model.py: Mamba-CIDS 整体模型定义
train.py: 训练脚本，包含完整的训练和验证流程
evaluate.py: 评估脚本，用于在测试集上评估模型性能
utils.py: 工具函数，包括随机种子设置、参数计数等
```
