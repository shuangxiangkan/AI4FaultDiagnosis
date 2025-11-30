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
    
    网络结构：输入层 -> 隐藏层 (ReLU + Dropout) -> 输出层 (Sigmoid)
    """
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list = None):
        if hidden_sizes is None:
            hidden_sizes = [output_size * 4, output_size * 2]
        
        layers = []
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.extend([nn.Linear(prev_size, h_size), nn.ReLU(), nn.Dropout(0.2)])
            prev_size = h_size
        layers.extend([nn.Linear(prev_size, output_size), nn.Sigmoid()])
        
        self.network = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
    
    def train(self, train_data: tuple, val_data: tuple, epochs: int = 100, batch_size: int = 64):
        """
        训练模型
        
        Args:
            train_data: (X_train, Y_train)
            val_data: (X_val, Y_val)
            epochs: 训练轮数
            batch_size: 批次大小
        """
        logger = get_logger()
        
        X_train, Y_train = torch.FloatTensor(train_data[0]), torch.FloatTensor(train_data[1])
        X_val, Y_val = torch.FloatTensor(val_data[0]), torch.FloatTensor(val_data[1])
        
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, Y_train),
            batch_size=batch_size, shuffle=True
        )
        
        for epoch in range(epochs):
            # Train
            self.network.train()
            train_loss = 0
            for bx, by in train_loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.network(bx), by)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validate
            self.network.eval()
            with torch.no_grad():
                val_loss = self.criterion(self.network(X_val), Y_val).item()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] Train: {train_loss/len(train_loader):.4f}, Val: {val_loss:.4f}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            inp = torch.FloatTensor(x)
            if inp.dim() == 1:
                inp = inp.unsqueeze(0)
            return self.network(inp).squeeze().numpy()
