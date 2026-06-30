"""
训练器 — 支持单卡 / DataParallel 多卡 / MPS
"""

import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def _unwrap(model: nn.Module) -> nn.Module:
    """获取裸模型 (绕过 DataParallel wrapper)"""
    return model.module if isinstance(model, nn.DataParallel) else model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    is_sca: bool = False,
    group_ids: list = None,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        if is_sca:
            pred = _unwrap(model)(X, group_ids=group_ids, target=y)
        else:
            pred = model(X, target=y)

        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    is_sca: bool = False,
    group_ids: list = None,
) -> tuple:
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n_samples = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        if is_sca:
            pred = _unwrap(model)(X, group_ids=group_ids, target=y)
        else:
            pred = model(X, target=y)

        mse_sum += ((pred - y) ** 2).sum().item()
        mae_sum += (pred - y).abs().sum().item()
        n_samples += y.numel()

    return mse_sum / n_samples, mae_sum / n_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str,
    output_window: int,
    device: str,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 30,
    is_sca: bool = False,
    group_ids: list = None,
    grad_clip: float = 1.0,
    output_dir: str = None,
) -> dict:
    """完整训练流程，每个 epoch 保存日志"""
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # 多 GPU 支持
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)

    best_model_state = None
    best_test_mse = float("inf")
    best_test_mae = float("inf")
    train_losses = []
    test_mse_history = []
    test_mae_history = []

    print(f"\n{'='*60}")
    print(f"训练 {model_name} | output_window={output_window} | epochs={epochs}")
    print(f"设备: {device} | 参数量: {n_params:,} | Multi-GPU: {torch.cuda.device_count()}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            is_sca, group_ids, grad_clip,
        )
        train_losses.append(train_loss)

        test_mse, test_mae = evaluate(model, test_loader, device, is_sca, group_ids)
        test_mse_history.append(test_mse)
        test_mae_history.append(test_mae)

        scheduler.step(test_mse)

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_test_mae = test_mae
            best_model_state = copy.deepcopy(_unwrap(model).state_dict())

        elapsed = time.time() - t0

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Test MSE: {test_mse:.6f} | "
                f"Test MAE: {test_mae:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s"
            )

        if early_stopping(test_mse):
            print(f"  早停于 epoch {epoch}")
            break

    result = {
        "best_test_mse": best_test_mse,
        "best_test_mae": best_test_mae,
        "total_epochs": epoch,
        "train_losses": train_losses,
        "test_mse_history": test_mse_history,
        "test_mae_history": test_mae_history,
    }
    print(f"  最佳结果 → MSE: {best_test_mse:.6f}, MAE: {best_test_mae:.6f}")

    # 保存训练日志
    if output_dir:
        log_path = os.path.join(output_dir, "training_log.json")
        log_data = {k: v for k, v in result.items() if k != "model_state"}
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

    result["model_state"] = best_model_state
    return result


def get_group_ids(feature_names: list = None) -> list:
    """根据实际特征名动态计算 SCA-Net 变量分组索引"""
    if feature_names is None:
        return [[0, 1, 4, 5, 6, 7], [2, 3], [8, 9, 10, 11, 12, 13]]

    power_keywords = ["active_power", "reactive_power", "sub_metering", "remainder"]
    elec_keywords = ["voltage", "intensity"]
    weather_keywords = ["sin_", "cos_", "RR", "NBJ", "BROU"]

    group0, group1, group2 = [], [], []
    for i, name in enumerate(feature_names):
        if any(k in name for k in power_keywords):
            group0.append(i)
        elif any(k in name for k in elec_keywords):
            group1.append(i)
        elif any(k in name for k in weather_keywords):
            group2.append(i)
        else:
            group2.append(i)

    return [group0, group1, group2]
