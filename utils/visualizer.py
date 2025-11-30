"""Syndrome 可视化模块"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def visualize_syndrome(syndrome_path: str, dimension: int = None):
    """
    可视化单个 syndrome 文件
    
    Args:
        syndrome_path: .npz 文件路径
        dimension: 超立方体维度（如果为 None，则从 metadata.json 读取）
    """
    # 加载数据
    data = np.load(syndrome_path)
    syndrome = data["syndrome"]
    label = data["label"]
    
    n_nodes = len(label)
    
    # 自动获取 dimension
    if dimension is None:
        # 尝试从 metadata.json 读取
        dataset_dir = os.path.dirname(os.path.dirname(syndrome_path))
        metadata_path = os.path.join(dataset_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                dimension = metadata.get("dimension")
        
        # 如果还是 None，从节点数推断
        if dimension is None:
            dimension = int(np.log2(n_nodes))
    
    # 构建超立方体图
    G = nx.hypercube_graph(dimension)
    G = nx.relabel_nodes(G, {n: int("".join(map(str, n)), 2) for n in G.nodes()})
    
    # 故障节点
    faulty_nodes = set(np.where(label == 1)[0])
    node_colors = ["#ff6b6b" if i in faulty_nodes else "#69db7c" for i in range(n_nodes)]
    
    # 构建 syndrome 字典：(u, v) -> test_result
    syndrome_dict = {}
    idx = 0
    for u in range(n_nodes):
        for bit in range(dimension):
            v = u ^ (1 << bit)
            syndrome_dict[(u, v)] = syndrome[idx]
            idx += 1
    
    # 对于无向边，取可靠的测试结果
    edge_colors = []
    edge_styles = []
    edges = []
    
    for u in range(n_nodes):
        for bit in range(dimension):
            v = u ^ (1 << bit)
            if u > v:
                continue
            edges.append((u, v))
            
            if u not in faulty_nodes:
                test_result = syndrome_dict[(u, v)]
            elif v not in faulty_nodes:
                test_result = syndrome_dict[(v, u)]
            else:
                test_result = -1
            
            if test_result == -1:
                edge_colors.append("#adb5bd")
                edge_styles.append("dotted")
            elif test_result == 1:
                edge_colors.append("#ff6b6b")
                edge_styles.append("dashed")
            else:
                edge_colors.append("#69db7c")
                edge_styles.append("solid")
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    for (u, v), color, style in zip(edges, edge_colors, edge_styles):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color,
                              style=style, width=2, ax=ax)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
    
    binary_labels = {i: format(i, f'0{dimension}b') for i in range(n_nodes)}
    nx.draw_networkx_labels(G, pos, labels=binary_labels, font_size=8, font_weight="bold", ax=ax)
    
    from matplotlib.lines import Line2D
    faulty_binary = {format(i, f'0{dimension}b') for i in faulty_nodes}
    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b', markersize=12, label='Faulty Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#69db7c', markersize=12, label='Healthy Node'),
        Line2D([0], [0], color='#ff6b6b', linestyle='dashed', linewidth=2, label='Test Result: 1'),
        Line2D([0], [0], color='#69db7c', linestyle='solid', linewidth=2, label='Test Result: 0'),
        Line2D([0], [0], color='#adb5bd', linestyle='dotted', linewidth=2, label='Unreliable (both faulty)'),
    ]
    ax.legend(handles=legend, loc='upper left')
    
    ax.set_title(f"Hypercube {dimension}D - Faulty: {faulty_binary}")
    plt.tight_layout()
    plt.savefig(syndrome_path.replace(".npz", ".png"), dpi=150)
    plt.show()
    
    return syndrome_path.replace(".npz", ".png")
