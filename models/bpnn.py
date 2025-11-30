"""反向传播神经网络 (BPNN) 模型"""

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
    输入层 -> 隐藏层1 (ReLU + Dropout) -> 隐藏层2 (ReLU + Dropout) -> 输出层 (Sigmoid)
    
    - 输入：测试综合征（所有测试结果）
    - 输出：每个节点是故障的概率（0~1之间）
    """
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list = None):
        """
        Args:
            input_size: 输入维度（综合征大小）
            output_size: 输出维度（节点数量）
            hidden_sizes: 隐藏层大小列表，默认 [output_size*4, output_size*2]
        """
        if hidden_sizes is None:
            hidden_sizes = [output_size * 4, output_size * 2]
        
        # 构建网络层
        layers = []
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h_size),  # 全连接层
                nn.ReLU(),                      # 激活函数
                nn.Dropout(0.2)                 # 防止过拟合
            ])
            prev_size = h_size
        
        # 输出层：Sigmoid 将输出压缩到 (0, 1)，表示故障概率
        layers.extend([nn.Linear(prev_size, output_size), nn.Sigmoid()])
        
        self.network = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()  # 二元交叉熵损失，适合多标签分类
    
    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 100, 
              batch_size: int = 64, val_ratio: float = 0.1):
        """
        训练模型
        
        Args:
            X: 训练数据
            Y: 标签
            epochs: 训练轮数
            batch_size: 批次大小
            val_ratio: 验证集比例（默认 10%）
        """
        logger = get_logger()
        
        # ===== 1. 划分训练集和验证集 =====
        n = len(X)
        idx = np.random.permutation(n)  # 随机打乱索引
        val_size = int(n * val_ratio)
        
        X_val = torch.FloatTensor(X[idx[:val_size]])
        Y_val = torch.FloatTensor(Y[idx[:val_size]])
        X_train = torch.FloatTensor(X[idx[val_size:]])
        Y_train = torch.FloatTensor(Y[idx[val_size:]])
        
        # ===== 2. 创建数据加载器 =====
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, Y_train),
            batch_size=batch_size, 
            shuffle=True  # 每个 epoch 打乱数据
        )
        
        # ===== 3. 训练循环 =====
        best_val_loss = float('inf')
        for epoch in range(epochs):
            # --- 训练阶段 ---
            self.network.train()  # 开启训练模式（启用 Dropout）
            train_loss = 0
            for bx, by in train_loader:
                self.optimizer.zero_grad()          # 清除梯度
                loss = self.criterion(self.network(bx), by)  # 计算损失
                loss.backward()                     # 反向传播
                self.optimizer.step()               # 更新权重
                train_loss += loss.item()
            
            # --- 验证阶段 ---
            self.network.eval()  # 开启评估模式（关闭 Dropout）
            with torch.no_grad():  # 不计算梯度，节省内存
                val_loss = self.criterion(self.network(X_val), Y_val).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # 每 20 轮打印一次
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] Train: {train_loss/len(train_loader):.4f}, Val: {val_loss:.4f}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测节点故障概率"""
        self.network.eval()
        with torch.no_grad():
            inp = torch.FloatTensor(x)
            if inp.dim() == 1:
                inp = inp.unsqueeze(0)  # 添加 batch 维度
            return self.network(inp).squeeze().numpy()
