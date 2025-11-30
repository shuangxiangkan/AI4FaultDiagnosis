"""PMC 诊断模型"""

import numpy as np
import random
from topologies.base import BaseTopology


class PMCDiagnosis:
    """
    PMC (Preparata, Metze, Chien) 诊断模型
    
    核心规则：
    1. 无故障节点 u 测试邻居 v：
       - 如果 v 无故障，结果 = 0
       - 如果 v 有故障，结果 = 1
    2. 有故障节点 u 测试任何节点：
       - 结果不可靠，可能是 0 或 1（随机）
    
    这是系统级故障诊断的经典模型之一。
    """
    
    def __init__(self, topology: BaseTopology):
        """
        Args:
            topology: 网络拓扑结构
        """
        self.topology = topology
    
    @property
    def syndrome_size(self) -> int:
        """综合征的大小（所有测试结果的数量）"""
        return len(self.topology.get_all_edges())
    
    def generate_syndrome(self, faulty_nodes: set) -> np.ndarray:
        """
        生成测试综合征（syndrome）
        
        综合征是所有测试结果的集合，用于诊断哪些节点有故障。
        
        Args:
            faulty_nodes: 故障节点的集合
            
        Returns:
            一维数组，包含所有测试结果（0 或 1）
        """
        syndrome = []
        for u, v in self.topology.get_all_edges():
            if u not in faulty_nodes:
                # 无故障测试者：结果可靠
                result = 1 if v in faulty_nodes else 0
            else:
                # 有故障测试者：结果不可靠（随机）
                result = random.randint(0, 1)
            syndrome.append(result)
        return np.array(syndrome, dtype=np.float32)
