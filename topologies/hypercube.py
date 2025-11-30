"""N维超立方体网络拓扑"""

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
        """
        Args:
            dimension: 超立方体的维度，节点数 = 2^dimension
        """
        self.dim = dimension
        self._n_nodes = 2 ** dimension
        self._adj = self._build_adjacency()
    
    @property
    def n_nodes(self) -> int:
        return self._n_nodes
    
    def _build_adjacency(self) -> dict:
        """
        构建邻接表
        
        通过翻转二进制位来找邻居：
        - 节点 5 (101) 翻转第 0 位 -> 4 (100)
        - 节点 5 (101) 翻转第 1 位 -> 7 (111)
        - 节点 5 (101) 翻转第 2 位 -> 1 (001)
        """
        adj = {i: [] for i in range(self._n_nodes)}
        for node in range(self._n_nodes):
            for bit in range(self.dim):
                # 异或操作翻转第 bit 位
                neighbor = node ^ (1 << bit)
                adj[node].append(neighbor)
        return adj
    
    def get_neighbors(self, node: int) -> list:
        return self._adj[node]
