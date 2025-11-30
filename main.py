"""
基于 BPNN 的 PMC 模型故障诊断系统

这个程序实现了论文的核心思想：
使用神经网络从测试综合征中识别故障节点
"""

import numpy as np
import random
from topologies import Hypercube
from models import BPNN
from diagnosis import PMCDiagnosis


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


def evaluate(model: BPNN, pmc: PMCDiagnosis, n_nodes: int, max_faults: int, n_tests: int = 1000):
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


def main():
    # ===== 配置参数 =====
    dimension = 4       # 超立方体维度（节点数 = 2^4 = 16）
    max_faults = 4      # 最大故障数（t-diagnosable）
    n_samples = 20000   # 训练样本数
    epochs = 100        # 训练轮数
    
    print(f"=== BPNN Fault Diagnosis for {dimension}-D Hypercube ===")
    print(f"Nodes: {2**dimension}, Max faults: {max_faults}\n")
    
    # ===== 初始化 =====
    topo = Hypercube(dimension)           # 创建超立方体拓扑
    pmc = PMCDiagnosis(topo)              # 创建 PMC 诊断模型
    model = BPNN(pmc.syndrome_size, topo.n_nodes)  # 创建神经网络
    
    # ===== 训练 =====
    print(f"Generating {n_samples} samples and training...")
    X, Y = generate_data(pmc, topo.n_nodes, max_faults, n_samples)
    model.train(X, Y, epochs)
    
    # ===== 测试 =====
    print("\nEvaluating on 1000 test cases...")
    results = evaluate(model, pmc, topo.n_nodes, max_faults)
    
    print(f"\n=== Results ===")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Precision: {results['precision']*100:.2f}%")
    print(f"Recall: {results['recall']*100:.2f}%")


if __name__ == "__main__":
    main()
