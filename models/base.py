"""诊断模型的抽象基类"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    诊断模型的基类
    
    所有具体的模型（如 BPNN, GNN 等）都需要继承这个类，
    并实现 train 和 predict 方法。
    """
    
    @abstractmethod
    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int):
        """
        训练模型
        
        Args:
            X: 输入数据（综合征），shape = (样本数, 综合征大小)
            Y: 标签（节点状态），shape = (样本数, 节点数)
               每个元素是 0（无故障）或 1（有故障）
            epochs: 训练轮数
        """
        pass
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测节点故障概率
        
        Args:
            x: 单个综合征，shape = (综合征大小,)
            
        Returns:
            每个节点是故障的概率，shape = (节点数,)
        """
        pass
