"""
基于 BPNN 的 PMC 模型故障诊断系统

整体流程：
1. 生成/加载数据：根据 PMC 模型生成测试综合征 (syndrome) 和故障标签
2. 训练模型：使用 BPNN 学习从 syndrome 到故障节点的映射
3. 评估模型：在测试集上评估诊断准确率

使用方法：
    python main.py                      # 默认参数运行
    python main.py -d 5 -f 6 -n 5000    # 自定义参数
    python main.py --save my_data       # 保存数据集
    python main.py --load my_data       # 加载数据集
    python main.py --visualize xxx.npz  # 可视化 syndrome
"""

import argparse
from topologies import Hypercube
from models import BPNN
from data import generate_data, save_dataset, load_dataset
from evaluation import evaluate
from utils import setup_logger, visualize_syndrome


def parse_args():
    """解析命令行参数"""
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
    """
    主函数：执行故障诊断的完整流程
    
    流程图：
        参数解析
            ↓
        [可视化模式?] → 是 → 可视化 syndrome → 结束
            ↓ 否
        创建超立方体拓扑
            ↓
        [加载数据?] → 是 → 从磁盘加载
            ↓ 否
        生成新数据 → [保存?] → 保存到磁盘
            ↓
        创建 BPNN 模型
            ↓
        训练模型 (train + validation)
            ↓
        评估模型 (test)
            ↓
        输出结果
    """
    # ==================== 初始化 ====================
    args = parse_args()
    logger = setup_logger()
    
    # ==================== 可视化模式 ====================
    # 如果指定了 --visualize，只进行可视化，不训练
    if args.visualize:
        logger.info(f"Visualizing: {args.visualize}")
        output = visualize_syndrome(args.visualize, args.dimension)
        logger.info(f"Saved: {output}")
        return
    
    # ==================== 解析参数 ====================
    dimension = args.dimension          # 超立方体维度
    n_nodes = 2 ** dimension            # 节点数 = 2^dimension
    
    # 解析故障数：小数表示故障率，整数表示具体数目
    # 例如：0.25 表示 25% 的节点可能故障，4 表示最多 4 个节点故障
    fault_val = float(args.faults)
    if fault_val < 1:
        max_faults = max(1, int(n_nodes * fault_val))  # 按比例计算
    else:
        max_faults = int(fault_val)  # 直接使用指定数目
    
    logger.info(f"=== BPNN Fault Diagnosis for {dimension}-D Hypercube ===")
    
    # ==================== 创建拓扑 ====================
    # 超立方体：每个节点用 n 位二进制表示，相邻节点只差一位
    # 例如：4维超立方体有 16 个节点，每个节点有 4 个邻居
    topo = Hypercube(dimension)
    
    # ==================== 数据准备 ====================
    if args.load:
        # 从磁盘加载已有数据集（用于复现实验）
        logger.info(f"Loading dataset: {args.load}")
        train_data, val_data, test_data, metadata = load_dataset(args.load)
        n_nodes = metadata.get("n_nodes", n_nodes)
        max_faults = metadata.get("max_faults", max_faults)
        logger.info(f"Loaded - Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[0])}")
    else:
        # 生成新数据：随机生成故障场景，按 PMC 模型生成 syndrome
        logger.info(f"Generating data (train/val/test = 80/10/10)...")
        train_data, val_data, test_data = generate_data(topo, max_faults, args.n_samples)
        logger.info(f"Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[0])}")
        
        # 可选：保存数据集到磁盘
        if args.save:
            metadata = {"n_nodes": n_nodes, "max_faults": max_faults, "dimension": dimension}
            path = save_dataset(train_data, val_data, test_data, args.save, metadata)
            logger.info(f"Dataset saved: {path}")
    
    logger.info(f"Nodes: {n_nodes}, Max faults: {max_faults}")
    
    # ==================== 训练模型 ====================
    # 创建 BPNN：输入是 syndrome，输出是每个节点的故障概率
    model = BPNN(
        input_size=topo.syndrome_size,  # syndrome 大小 = 节点数 × 维度
        output_size=n_nodes              # 每个节点一个输出
    )
    logger.info("Training...")
    model.train(train_data, val_data, args.epochs)
    
    # ==================== 评估模型 ====================
    # 在测试集上评估：计算准确率、精确率、召回率
    logger.info("Evaluating on test set...")
    results = evaluate(model, test_data)
    
    # 输出结果
    logger.info("=== Results ===")
    logger.info(f"Accuracy: {results['accuracy']*100:.2f}%")   # 完全正确识别的比例
    logger.info(f"Precision: {results['precision']*100:.2f}%") # 预测故障中真正故障的比例
    logger.info(f"Recall: {results['recall']*100:.2f}%")       # 真正故障中被找到的比例


if __name__ == "__main__":
    main()
