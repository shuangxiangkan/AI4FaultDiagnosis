"""数据集保存与加载模块"""

import os
import json
import numpy as np
from datetime import datetime

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")


def save_dataset(train_data: tuple, val_data: tuple, test_data: tuple,
                 name: str, metadata: dict = None) -> str:
    """
    保存数据集，每个 syndrome 单独保存
    
    目录结构:
        datasets/{name}/{timestamp}/
            metadata.json
            train/1.npz, 2.npz, ...
            val/1.npz, 2.npz, ...
            test/1.npz, 2.npz, ...
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = os.path.join(DATASETS_DIR, name, timestamp)
    
    # 保存各个子集
    for split_name, (X, Y) in [("train", train_data), ("val", val_data), ("test", test_data)]:
        split_dir = os.path.join(dataset_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for i, (x, y) in enumerate(zip(X, Y), 1):
            np.savez(os.path.join(split_dir, f"{i}.npz"), syndrome=x, label=y)
    
    # 保存元数据
    meta = metadata or {}
    meta.update({
        "timestamp": timestamp,
        "train_size": len(train_data[0]),
        "val_size": len(val_data[0]),
        "test_size": len(test_data[0])
    })
    with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    return dataset_dir


def load_dataset(name: str, timestamp: str = None) -> tuple:
    """
    加载数据集
    
    Args:
        name: 数据集名称
        timestamp: 时间戳，None 表示加载最新的
        
    Returns:
        (train_data, val_data, test_data, metadata)
    """
    name_dir = os.path.join(DATASETS_DIR, name)
    
    # 找最新的时间戳
    if timestamp is None:
        timestamps = sorted(os.listdir(name_dir))
        timestamp = timestamps[-1]
    
    dataset_dir = os.path.join(name_dir, timestamp)
    
    # 加载元数据
    with open(os.path.join(dataset_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # 加载各个子集
    def load_split(split_name):
        split_dir = os.path.join(dataset_dir, split_name)
        files = sorted(os.listdir(split_dir), key=lambda x: int(x.split(".")[0]))
        X, Y = [], []
        for fname in files:
            data = np.load(os.path.join(split_dir, fname))
            X.append(data["syndrome"])
            Y.append(data["label"])
        return np.array(X), np.array(Y)
    
    train_data = load_split("train")
    val_data = load_split("val")
    test_data = load_split("test")
    
    return train_data, val_data, test_data, metadata
