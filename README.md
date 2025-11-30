# AI4FaultDiagnosis

基于神经网络的系统级故障诊断

## 项目简介

使用 BPNN（反向传播神经网络）实现 PMC 模型下的故障诊断。输入测试综合征，输出故障节点集合。

**参考论文**: *Comparison-Based System-Level Fault Diagnosis: A Neural Network Approach* (Elhadef & Nayak, 2012)

## 环境要求

- Python >= 3.10
- macOS / Linux / Windows

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/AI4FaultDiagnosis.git
cd AI4FaultDiagnosis
```

### 2. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 运行

```bash
python main.py
```

输出示例：
```
=== BPNN Fault Diagnosis for 4-D Hypercube ===
Nodes: 16, Max faults: 4

Generating 20000 samples and training...
Epoch [20/100] Train: 0.0732, Val: 0.0194
...
Epoch [100/100] Train: 0.0338, Val: 0.0064

Evaluating on 1000 test cases...

=== Results ===
Accuracy: 99.00%
Precision: 99.98%
Recall: 99.78%
```

## 项目结构

```
AI4FaultDiagnosis/
├── topologies/          # 网络拓扑
│   ├── base.py          # 抽象基类
│   └── hypercube.py     # 超立方体
├── models/              # 诊断模型
│   ├── base.py          # 抽象基类
│   └── bpnn.py          # BPNN
├── diagnosis/           # 诊断协议
│   └── pmc.py           # PMC 模型
├── main.py              # 主程序
└── requirements.txt     # 依赖
```

## 核心概念

### PMC 模型

| 测试者状态 | 被测者状态 | 测试结果 |
|-----------|-----------|---------|
| 无故障 | 无故障 | 0 |
| 无故障 | 故障 | 1 |
| 故障 | 任意 | 不可靠（0或1）|

### 超立方体拓扑

- N 维超立方体有 2^N 个节点
- 每个节点有 N 个邻居
- 节点编号的二进制表示只差一位即为邻居

## 参数配置

修改 `main.py` 中的参数：

```python
dimension = 4       # 超立方体维度（节点数 = 2^4 = 16）
max_faults = 4      # 最大故障数
n_samples = 20000   # 训练样本数
epochs = 100        # 训练轮数
```

## 扩展指南

### 添加新拓扑

```python
# topologies/torus.py
from .base import BaseTopology

class Torus(BaseTopology):
    def __init__(self, rows: int, cols: int):
        ...
    
    @property
    def n_nodes(self) -> int:
        ...
    
    def get_neighbors(self, node: int) -> list:
        ...
```

### 添加新模型

```python
# models/gnn.py
from .base import BaseModel

class GNN(BaseModel):
    def train(self, X, Y, epochs):
        ...
    
    def predict(self, x):
        ...
```

## License

MIT
