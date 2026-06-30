"""
可视化工具 — 预测曲线对比图、误差分布图等
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from utils.config import FIGURE_DIR


def plot_prediction_vs_ground_truth(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Prediction vs Ground Truth",
    save_name: str = "prediction_comparison.pdf",
    start_idx: int = 0,
    num_days: int = 90,
):
    """绘制预测值与真实值对比曲线

    Args:
        y_true: (N, T) 真实值
        y_pred: (N, T) 预测值
        title: 图标题
        save_name: 保存文件名
        start_idx: 起始样本索引
        num_days: 绘制天数
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, (ax, idx) in enumerate(zip(axes.flat, [start_idx, start_idx + 5, start_idx + 10, start_idx + 20])):
        if idx >= len(y_true):
            break
        days = np.arange(num_days)
        ax.plot(days, y_true[idx, :num_days], "b-", label="Ground Truth", linewidth=1.5)
        ax.plot(days, y_pred[idx, :num_days], "r--", label="Prediction", linewidth=1.5)
        ax.set_xlabel("Days")
        ax.set_ylabel("Active Power (normalized)")
        ax.set_title(f"Sample {idx}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存: {save_path}")


def plot_long_term_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_name: str = "long_term_pred.pdf",
):
    """长期预测（365天）的完整曲线对比"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))

    idx = 0  # 第一个测试样本
    days = np.arange(min(365, y_true.shape[1]))
    ax.plot(days, y_true[idx], "b-", label="Ground Truth", linewidth=1, alpha=0.8)
    ax.plot(days, y_pred[idx], "r--", label="Prediction", linewidth=1, alpha=0.8)
    ax.set_xlabel("Days")
    ax.set_ylabel("Active Power (normalized)")
    ax.set_title("Long-term Prediction (365 days): Prediction vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(FIGURE_DIR, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存: {save_path}")


def plot_model_comparison(
    results: dict,  # {"LSTM": (mse_mean, mse_std, mae_mean, mae_std), ...}
    task_name: str = "Short-term (90 days)",
    save_name: str = "model_comparison.pdf",
):
    """模型对比柱状图（MSE + MAE + 标准差）"""
    models = list(results.keys())
    mse_means = [results[m][0] for m in models]
    mse_stds = [results[m][1] for m in models]
    mae_means = [results[m][2] for m in models]
    mae_stds = [results[m][3] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(models))
    width = 0.35

    # MSE
    axes[0].bar(x, mse_means, width, yerr=mse_stds, capsize=5, color=["#4472C4", "#ED7D31", "#70AD47"])
    axes[0].set_ylabel("MSE")
    axes[0].set_title(f"MSE Comparison ({task_name})")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)

    # MAE
    axes[1].bar(x, mae_means, width, yerr=mae_stds, capsize=5, color=["#4472C4", "#ED7D31", "#70AD47"])
    axes[1].set_ylabel("MAE")
    axes[1].set_title(f"MAE Comparison ({task_name})")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)

    plt.suptitle(f"Model Comparison — {task_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(FIGURE_DIR, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  对比图已保存: {save_path}")


def plot_training_curves(
    train_losses: dict,  # {model_name: [losses]}
    test_mses: dict,  # {model_name: [mses]}
    save_name: str = "training_curves.pdf",
):
    """训练/测试损失曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, losses in train_losses.items():
        axes[0].plot(losses, label=name, alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train MSE Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, mses in test_mses.items():
        axes[1].plot(mses, label=name, alpha=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test MSE")
    axes[1].set_title("Test MSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  训练曲线已保存: {save_path}")
