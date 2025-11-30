"""模型评估模块"""

import numpy as np
from models.base import BaseModel


def evaluate(model: BaseModel, test_data: tuple) -> dict:
    """
    在测试集上评估模型
    
    Args:
        model: 训练好的模型
        test_data: (X_test, Y_test)
        
    Returns:
        包含 accuracy, precision, recall 的字典
    """
    X_test, Y_test = test_data
    correct, total_prec, total_rec = 0, 0, 0
    
    for x, y in zip(X_test, Y_test):
        actual = set(np.where(y == 1)[0])
        pred = set(np.where(model.predict(x) > 0.5)[0])
        
        if pred == actual:
            correct += 1
        
        total_prec += len(pred & actual) / len(pred) if pred else 0
        total_rec += len(pred & actual) / len(actual) if actual else 1
    
    n = len(X_test)
    return {
        "accuracy": correct / n,
        "precision": total_prec / n,
        "recall": total_rec / n
    }
