# AI4FaultDiagnosis

基于神经网络的系统级故障诊断

## 项目简介

使用 BPNN（反向传播神经网络）实现 PMC 模型下的故障诊断。输入测试综合征，输出故障节点集合。

**参考论文**: *Comparison-Based System-Level Fault Diagnosis: A Neural Network Approach* (Elhadef & Nayak, 2012)

## 环境要求

- Python >= 3.10
- macOS / Linux / Windows

## 快速开始

```bash
# 1. 克隆 & 进入项目
git clone https://github.com/your-repo/AI4FaultDiagnosis.git
cd AI4FaultDiagnosis

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行
python main.py
```

## 命令行参数

```bash
python main.py [OPTIONS]
```

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `-d, --dimension` | 超立方体维度（节点数 = 2^d） | 4 |
| `-f, --faults` | 故障数（整数）或故障率（小数） | 0.25 |
| `-n, --n_samples` | 总样本数 | 1000 |
| `-e, --epochs` | 训练轮数 | 100 |
| `--save NAME` | 保存数据集 | - |
| `--load NAME` | 加载数据集 | - |

**示例**:
```bash
# 5维超立方体，最多6个故障，5000样本
python main.py -d 5 -f 6 -n 5000

# 生成并保存数据集
python main.py -d 4 -n 2000 --save hypercube_4d

# 加载已有数据集训练（可复现）
python main.py --load hypercube_4d
```

## 项目结构

```
AI4FaultDiagnosis/
├── topologies/          # 网络拓扑
│   ├── base.py
│   └── hypercube.py
├── models/              # 诊断模型
│   ├── base.py
│   └── bpnn.py
├── diagnosis/           # 诊断协议
│   └── pmc.py
├── data/                # 数据生成与管理
│   ├── generator.py
│   └── dataset.py
├── evaluation/          # 评估模块
│   └── metrics.py
├── utils/               # 工具模块
│   └── logger.py
├── datasets/            # 保存的数据集
│   └── {name}/{timestamp}/
│       ├── metadata.json
│       ├── train/1.npz, 2.npz, ...
│       ├── val/1.npz, 2.npz, ...
│       └── test/1.npz, 2.npz, ...
├── main.py
└── requirements.txt
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

## 扩展指南

### 添加新拓扑

```python
# topologies/torus.py
from .base import BaseTopology

class Torus(BaseTopology):
    @property
    def n_nodes(self) -> int: ...
    def get_neighbors(self, node: int) -> list: ...
```

### 添加新模型

```python
# models/gnn.py
from .base import BaseModel

class GNN(BaseModel):
    def train(self, train_data, val_data, epochs): ...
    def predict(self, x): ...
```

## License

MIT
