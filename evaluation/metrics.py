"""模型评估模块"""

import numpy as np
import random
from models.base import BaseModel
from diagnosis.pmc import PMCDiagnosis


def evaluate(model: BaseModel, pmc: PMCDiagnosis, n_nodes: int, max_faults: int, n_tests: int = 1000):
    """
    评估模型性能
    
    指标说明：
    - Accuracy: 完全正确识别所有故障节点的比例
    - Precision: 预测为故障的节点中，真正故障的比例
    - Recall: 真正故障的节点中，被正确预测的比例
    """
    correct, total_prec, total_rec = 0, 0, 0
    
    for _ in range(n_tests):
        # 随机生成故障
        n_faults = random.randint(1, max_faults)
        actual = set(random.sample(range(n_nodes), n_faults))
        
        # 模型预测（概率 > 0.5 判定为故障）
        syndrome = pmc.generate_syndrome(actual)
        pred = set(np.where(model.predict(syndrome) > 0.5)[0])
        
        # 计算指标
        if pred == actual:
            correct += 1
        
        total_prec += len(pred & actual) / len(pred) if pred else 0
        total_rec += len(pred & actual) / len(actual) if actual else 1
    
    return {
        "accuracy": correct / n_tests,
        "precision": total_prec / n_tests,
        "recall": total_rec / n_tests
    }

