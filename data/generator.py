"""数据生成模块"""

import numpy as np
import random
from diagnosis.pmc import PMCDiagnosis


def generate_data(pmc: PMCDiagnosis, n_nodes: int, max_faults: int, n_samples: int):
    """
    生成训练/测试数据
    
    Args:
        pmc: PMC 诊断模型
        n_nodes: 节点总数
        max_faults: 最大故障数
        n_samples: 样本数量
        
    Returns:
        X: 综合征数据，shape = (n_samples, syndrome_size)
        Y: 标签，shape = (n_samples, n_nodes)
    """
    X, Y = [], []
    for _ in range(n_samples):
        # 随机生成 1 到 max_faults 个故障
        n_faults = random.randint(1, max_faults)
        faulty = set(random.sample(range(n_nodes), n_faults))
        
        # 生成综合征和对应的标签
        X.append(pmc.generate_syndrome(faulty))
        Y.append(np.array([1.0 if i in faulty else 0.0 for i in range(n_nodes)], dtype=np.float32))
    
    return np.array(X), np.array(Y)

