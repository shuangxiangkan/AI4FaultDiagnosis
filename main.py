"""
基于 BPNN 的 PMC 模型故障诊断系统
"""

import argparse
from topologies import Hypercube
from models import BPNN
from diagnosis import PMCDiagnosis
from data import generate_data, save_dataset, load_dataset
from evaluation import evaluate
from utils import setup_logger, visualize_syndrome


def parse_args():
    parser = argparse.ArgumentParser(description="BPNN-based Fault Diagnosis under PMC Model")
    parser.add_argument("-d", "--dimension", type=int, default=4,
                        help="超立方体维度，节点数 = 2^d (default: 4)")
    parser.add_argument("-f", "--faults", type=str, default="0.25",
                        help="故障数：整数表示具体数目，小数表示故障率 (default: 0.25)")
    parser.add_argument("-n", "--n_samples", type=int, default=1000,
                        help="总样本数 (default: 1000)")
    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help="训练轮数 (default: 100)")
    parser.add_argument("--load", type=str, default=None,
                        help="加载已有数据集（数据集名称）")
    parser.add_argument("--save", type=str, default=None,
                        help="保存数据集（数据集名称）")
    parser.add_argument("--visualize", type=str, default=None,
                        help="可视化单个 syndrome 文件路径")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()
    
    # 可视化模式
    if args.visualize:
        logger.info(f"Visualizing: {args.visualize}")
        output = visualize_syndrome(args.visualize, args.dimension)
        logger.info(f"Saved: {output}")
        return
    
    # 解析参数
    dimension = args.dimension
    n_nodes = 2 ** dimension
    fault_val = float(args.faults)
    max_faults = max(1, int(n_nodes * fault_val)) if fault_val < 1 else int(fault_val)
    
    logger.info(f"=== BPNN Fault Diagnosis for {dimension}-D Hypercube ===")
    
    # 初始化拓扑
    topo = Hypercube(dimension)
    pmc = PMCDiagnosis(topo)
    
    # 加载或生成数据
    if args.load:
        logger.info(f"Loading dataset: {args.load}")
        train_data, val_data, test_data, metadata = load_dataset(args.load)
        n_nodes = metadata.get("n_nodes", n_nodes)
        max_faults = metadata.get("max_faults", max_faults)
        logger.info(f"Loaded - Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[0])}")
    else:
        logger.info(f"Generating data (train/val/test = 80/10/10)...")
        train_data, val_data, test_data = generate_data(pmc, n_nodes, max_faults, args.n_samples)
        logger.info(f"Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[0])}")
        
        if args.save:
            metadata = {"n_nodes": n_nodes, "max_faults": max_faults, "dimension": dimension}
            path = save_dataset(train_data, val_data, test_data, args.save, metadata)
            logger.info(f"Dataset saved: {path}")
    
    logger.info(f"Nodes: {n_nodes}, Max faults: {max_faults}")
    
    # 训练
    model = BPNN(pmc.syndrome_size, n_nodes)
    logger.info("Training...")
    model.train(train_data, val_data, args.epochs)
    
    # 测试
    logger.info("Evaluating on test set...")
    results = evaluate(model, test_data)
    
    logger.info("=== Results ===")
    logger.info(f"Accuracy: {results['accuracy']*100:.2f}%")
    logger.info(f"Precision: {results['precision']*100:.2f}%")
    logger.info(f"Recall: {results['recall']*100:.2f}%")


if __name__ == "__main__":
    main()
