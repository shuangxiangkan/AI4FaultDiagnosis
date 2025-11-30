"""N维超立方体网络拓扑"""

import numpy as np
import random
from .base import BaseTopology


class Hypercube(BaseTopology):
    """
    N维超立方体网络
    
    特点：
    - 节点数量 = 2^n（n 是维度）
    - 每个节点有 n 个邻居
    - 节点编号用二进制表示，相邻节点只有一个 bit 不同
    
    例如：3维超立方体有 8 个节点，每个节点有 3 个邻居
    节点 0 (000) 的邻居是 1 (001), 2 (010), 4 (100)
    """
    
    def __init__(self, dimension: int):
        self.dim = dimension
        self._n_nodes = 2 ** dimension
        self._adj = self._build_adjacency()
    
    @property
    def n_nodes(self) -> int:
        return self._n_nodes
    
    @property
    def syndrome_size(self) -> int:
        """PMC 模型下 syndrome 的大小"""
        return self._n_nodes * self.dim
    
    def _build_adjacency(self) -> dict:
        """构建邻接表：通过翻转二进制位来找邻居"""
        adj = {i: [] for i in range(self._n_nodes)}
        for node in range(self._n_nodes):
            for bit in range(self.dim):
                neighbor = node ^ (1 << bit)
                adj[node].append(neighbor)
        return adj
    
    def get_neighbors(self, node: int) -> list:
        return self._adj[node]
    
    def generate_PMC_syndrome(self, faulty_nodes: set) -> np.ndarray:
        """
        生成 PMC 模型下的测试综合征
        
        PMC 规则：
        - 无故障节点 u 测试邻居 v：v 无故障返回 0，v 故障返回 1
        - 故障节点 u 测试任何节点：结果不可靠（随机 0 或 1）
        
        Args:
            faulty_nodes: 故障节点集合
            
        Returns:
            syndrome 数组
        """
        syndrome = []
        for u in range(self._n_nodes):
            for v in self._adj[u]:
                if u not in faulty_nodes:
                    result = 1 if v in faulty_nodes else 0
                else:
                    result = random.randint(0, 1)
                syndrome.append(result)
        return np.array(syndrome, dtype=np.float32)
