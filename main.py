"""
基于 BPNN 的 PMC 模型故障诊断系统

这个程序实现了论文的核心思想：
使用神经网络从测试综合征中识别故障节点
"""

from topologies import Hypercube
from models import BPNN
from diagnosis import PMCDiagnosis
from data import generate_data
from evaluation import evaluate
from utils import setup_logger


def main():
    # ===== 初始化日志 =====
    logger = setup_logger()
    
    # ===== 配置参数 =====
    dimension = 4       # 超立方体维度（节点数 = 2^4 = 16）
    max_faults = 4      # 最大故障数（t-diagnosable）
    n_samples = 20000   # 训练样本数
    epochs = 100        # 训练轮数
    
    logger.info(f"=== BPNN Fault Diagnosis for {dimension}-D Hypercube ===")
    logger.info(f"Nodes: {2**dimension}, Max faults: {max_faults}")
    
    # ===== 初始化 =====
    topo = Hypercube(dimension)           # 创建超立方体拓扑
    pmc = PMCDiagnosis(topo)              # 创建 PMC 诊断模型
    model = BPNN(pmc.syndrome_size, topo.n_nodes)  # 创建神经网络
    
    # ===== 训练 =====
    logger.info(f"Generating {n_samples} samples and training...")
    X, Y = generate_data(pmc, topo.n_nodes, max_faults, n_samples)
    model.train(X, Y, epochs)
    
    # ===== 测试 =====
    logger.info("Evaluating on 1000 test cases...")
    results = evaluate(model, pmc, topo.n_nodes, max_faults)
    
    logger.info("=== Results ===")
    logger.info(f"Accuracy: {results['accuracy']*100:.2f}%")
    logger.info(f"Precision: {results['precision']*100:.2f}%")
    logger.info(f"Recall: {results['recall']*100:.2f}%")


if __name__ == "__main__":
    main()
