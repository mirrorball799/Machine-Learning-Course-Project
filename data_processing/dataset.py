"""
PyTorch Dataset 类
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PowerDataset(Dataset):
    """家庭电力消耗时间序列数据集"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: (N, input_len, D) 输入特征
            y: (N, output_len) 目标序列
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和测试 DataLoader"""
    train_dataset = PowerDataset(X_train, y_train)
    test_dataset = PowerDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, test_loader
