"""
反向传播神经网络 (BPNN) 模型

BPNN 是最经典的神经网络之一，通过反向传播算法来训练。
核心思想：前向传播计算输出 → 计算误差 → 反向传播更新权重
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseModel
from utils import get_logger


class BPNN(BaseModel):
    """
    反向传播神经网络 (Backpropagation Neural Network)
    
    网络结构：
        输入层 (syndrome) 
            ↓
        隐藏层1 (Linear → ReLU → Dropout)
            ↓
        隐藏层2 (Linear → ReLU → Dropout)
            ↓
        输出层 (Linear → Sigmoid)  → 每个节点的故障概率 (0~1)
    
    为什么用这些组件：
    - Linear: 全连接层，学习特征之间的关系
    - ReLU: 激活函数，引入非线性，让网络能学习复杂模式
    - Dropout: 随机丢弃部分神经元，防止过拟合
    - Sigmoid: 将输出压缩到 (0,1)，表示故障概率
    """
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list = None):
        """
        初始化网络结构
        
        Args:
            input_size: 输入维度 = syndrome 大小 (测试结果数量)
            output_size: 输出维度 = 节点数量 (每个节点一个故障概率)
            hidden_sizes: 隐藏层大小列表，默认 [节点数*4, 节点数*2]
        """
        # 默认隐藏层大小
        if hidden_sizes is None:
            hidden_sizes = [output_size * 4, output_size * 2]
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for h_size in hidden_sizes:
            # 每个隐藏层包含: 全连接 → 激活 → Dropout
            layers.append(nn.Linear(prev_size, h_size))  # 全连接层
            layers.append(nn.ReLU())                      # 激活函数
            layers.append(nn.Dropout(0.2))                # 20% 的神经元随机丢弃
            prev_size = h_size
        
        # 输出层: 全连接 → Sigmoid
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())  # 输出范围 (0, 1)，表示故障概率
        
        # 将所有层组合成一个顺序模型
        self.network = nn.Sequential(*layers)
        
        # 优化器: Adam (自适应学习率，效果好且稳定)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        
        # 损失函数: 二元交叉熵 (Binary Cross Entropy)
        # 适合多标签分类问题 (每个节点独立判断是否故障)
        self.criterion = nn.BCELoss()
    
    def train(self, train_data: tuple, val_data: tuple, epochs: int = 100, batch_size: int = 64):
        """
        训练模型
        
        训练流程:
        1. 前向传播: 输入 syndrome → 网络 → 预测输出
        2. 计算损失: 比较预测值和真实标签的差距
        3. 反向传播: 计算每个参数对损失的贡献 (梯度)
        4. 更新参数: 按梯度方向调整权重，减小损失
        
        Args:
            train_data: (X_train, Y_train) 训练集
            val_data: (X_val, Y_val) 验证集
            epochs: 训练轮数 (遍历整个训练集的次数)
            batch_size: 批次大小 (每次更新使用的样本数)
        """
        logger = get_logger()
        
        # 转换为 PyTorch 张量
        X_train, Y_train = torch.FloatTensor(train_data[0]), torch.FloatTensor(train_data[1])
        X_val, Y_val = torch.FloatTensor(val_data[0]), torch.FloatTensor(val_data[1])
        
        # 数据加载器: 自动分批、打乱数据
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, Y_train),
            batch_size=batch_size,
            shuffle=True  # 每个 epoch 打乱顺序，防止模型记住顺序
        )
        
        # 训练循环
        for epoch in range(epochs):
            # ========== 训练阶段 ==========
            self.network.train()  # 开启训练模式 (启用 Dropout)
            train_loss = 0
            
            for bx, by in train_loader:
                # 1. 清除上一次的梯度 (PyTorch 会累积梯度)
                self.optimizer.zero_grad()
                
                # 2. 前向传播: 输入 → 预测
                predictions = self.network(bx)
                
                # 3. 计算损失
                loss = self.criterion(predictions, by)
                
                # 4. 反向传播: 计算梯度
                loss.backward()
                
                # 5. 更新参数
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # ========== 验证阶段 ==========
            self.network.eval()  # 开启评估模式 (关闭 Dropout)
            with torch.no_grad():  # 不计算梯度，节省内存
                val_predictions = self.network(X_val)
                val_loss = self.criterion(val_predictions, Y_val).item()
            
            # 每 20 轮打印一次
            if (epoch + 1) % 20 == 0:
                avg_train_loss = train_loss / len(train_loader)
                logger.info(f"Epoch [{epoch+1}/{epochs}] Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测节点故障概率
        
        Args:
            x: 单个 syndrome (测试结果向量)
            
        Returns:
            每个节点的故障概率，范围 (0, 1)
            概率 > 0.5 判定为故障
        """
        self.network.eval()  # 评估模式
        with torch.no_grad():
            inp = torch.FloatTensor(x)
            if inp.dim() == 1:
                inp = inp.unsqueeze(0)  # 添加 batch 维度: [n] → [1, n]
            return self.network(inp).squeeze().numpy()
