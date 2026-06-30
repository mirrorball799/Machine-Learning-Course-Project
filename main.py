"""
主入口脚本 — 单模型单任务实验流水线

用法:
  python main.py --model lstm --task short       # LSTM 短期预测
  python main.py --model transformer --task long  # Transformer 长期预测
  python main.py --model sca --task both          # SCA-Net 短+长期

输出: outputs/<model_name>/<task>/
  - results.json      汇总结果 (MSE/MAE mean/std)
  - training_log.json 训练日志 (loss/history)
  - best_model.pt     最佳模型权重
  - figures/          预测曲线、对比图
"""

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processing.preprocess import build_dataset
from data_processing.dataset import create_dataloaders
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.improved_model import SCANet
from training.trainer import train_model, get_group_ids
from utils.config import (
    DATA_DIR,
    WEATHER_DIR,
    OUTPUT_DIR,
    INPUT_WINDOW,
    OUTPUT_WINDOW_SHORT,
    OUTPUT_WINDOW_LONG,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS_SHORT,
    EPOCHS_LONG,
    PATIENCE,
    NUM_ROUNDS,
    SEEDS,
    D_MODEL,
    N_HEADS,
    NUM_ENCODER_LAYERS,
    DIM_FF,
    DROPOUT,
    LSTM_HIDDEN,
    LSTM_NUM_LAYERS,
    DEVICE,
    USE_MULTI_GPU,
    GRAD_CLIP,
    set_seed,
    get_output_dir,
    log_system_info,
)
from utils.visualization import (
    plot_prediction_vs_ground_truth,
    plot_long_term_prediction,
    plot_model_comparison,
    plot_training_curves,
)


def build_model(model_name: str, input_dim: int, **kwargs) -> torch.nn.Module:
    if model_name == "lstm":
        return LSTMModel(
            input_dim=input_dim,
            hidden_dim=LSTM_HIDDEN,
            num_layers=LSTM_NUM_LAYERS,
            dropout=DROPOUT,
        )
    elif model_name == "transformer":
        return TransformerModel(
            input_dim=input_dim,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            dim_ff=DIM_FF,
            dropout=DROPOUT,
        )
    elif model_name == "sca":
        group_sizes = kwargs.pop("group_sizes", None) or [6, 2, 11]
        return SCANet(
            group_sizes=group_sizes,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            dim_ff=DIM_FF,
            dropout=DROPOUT,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_experiment(
    model_name: str,
    task: str,
    data_path: str,
    weather_path: Optional[str] = None,
    epochs_override: Optional[int] = None,
    quick: bool = False,
    dry_run: bool = False,
    skip_existing: bool = True,
):
    """运行一个模型在一个任务上的实验"""
    output_window = OUTPUT_WINDOW_SHORT if task == "short" else OUTPUT_WINDOW_LONG
    task_label = f"short ({INPUT_WINDOW}→{OUTPUT_WINDOW_SHORT})" if task == "short" \
        else f"long ({INPUT_WINDOW}→{OUTPUT_WINDOW_LONG})"
    output_dir = get_output_dir(model_name, task)
    results_path = os.path.join(output_dir, "results.json")

    # 断点续跑: 已有结果则跳过
    if skip_existing and not dry_run and os.path.exists(results_path):
        with open(results_path) as f:
            prev = json.load(f)
        print(f"\n⏭ 跳过 {model_name.upper()} {task_label} — 已有结果 "
              f"(MSE={prev['mse_mean']:.4f})")
        return prev

    num_rounds = 1 if quick else NUM_ROUNDS
    epochs = (EPOCHS_SHORT if task == "short" else EPOCHS_LONG)
    if quick:
        epochs = 5
    if epochs_override:
        epochs = epochs_override
    if quick:
        task_label += " [QUICK]"
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# 模型: {model_name.upper()} | 任务: {task_label} | 轮数: {num_rounds}")
    if dry_run:
        print(f"# DRY RUN: 仅验证数据+前向传播")
    print(f"{'#'*60}")
    log_system_info()

    # ——— 构建数据集 ———
    X_train, y_train, X_test, y_test, stats, feature_names = build_dataset(
        data_path=data_path,
        output_window=output_window,
        weather_path=weather_path,
    )

    # quick 模式: 只用前 300 天数据
    if quick:
        max_samples = 50
        X_train = X_train[:max_samples]
        y_train = y_train[:max_samples]
        X_test = X_test[:max(1, max_samples // 5)]
        y_test = y_test[:max(1, max_samples // 5)]

    print(f"  特征 ({len(feature_names)}): {feature_names}")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}, y_test: {y_test.shape}")

    # ——— SCA-Net 分组 ———
    sca_group_ids = get_group_ids(feature_names)
    sca_group_sizes = [len(g) for g in sca_group_ids]
    is_sca = (model_name == "sca")
    group_ids = sca_group_ids if is_sca else None
    group_sizes = sca_group_sizes if is_sca else None
    if is_sca:
        print(f"  SCA-Net 分组: {sca_group_sizes}")

    # ——— dry-run: 仅验证前向传播 ———
    if dry_run:
        input_dim = X_train.shape[-1]
        kwargs = {"group_sizes": group_sizes} if group_sizes else {}
        model = build_model(model_name, input_dim, **kwargs).to(DEVICE)

        x = torch.randn(2, INPUT_WINDOW, input_dim).to(DEVICE)
        t = torch.randn(2, output_window).to(DEVICE)
        model.eval()
        with torch.no_grad():
            if is_sca:
                out = model(x, group_ids=group_ids, target=t)
            else:
                out = model(x, target=t)
        print(f"  Dry-run 通过: input={x.shape} → output={out.shape}")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
        return {}

    # ——— 多轮实验 ———
    mse_list, mae_list = [], []
    best_model_state = None
    best_mse = float("inf")
    all_logs = []

    for round_idx in range(num_rounds):
        seed = SEEDS[round_idx]
        set_seed(seed)
        print(f"\n--- Round {round_idx + 1}/{num_rounds} (seed={seed}) ---")

        train_loader, test_loader = create_dataloaders(
            X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE
        )

        input_dim = X_train.shape[-1]
        kwargs = {"group_sizes": group_sizes} if group_sizes else {}
        model = build_model(model_name, input_dim, **kwargs)

        result = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            model_name=model_name,
            output_window=output_window,
            device=DEVICE,
            epochs=epochs,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            is_sca=is_sca,
            group_ids=group_ids,
            grad_clip=GRAD_CLIP,
            output_dir=output_dir,
        )

        mse_list.append(result["best_test_mse"])
        mae_list.append(result["best_test_mae"])
        all_logs.append(result)

        if result["best_test_mse"] < best_mse:
            best_mse = result["best_test_mse"]
            best_model_state = result["model_state"]

    # ——— 汇总统计 ———
    mse_array = np.array(mse_list)
    mae_array = np.array(mae_list)
    summary = {
        "model": model_name,
        "task": task,
        "input_window": INPUT_WINDOW,
        "output_window": output_window,
        "num_features": int(X_train.shape[-1]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "num_rounds": num_rounds,
        "seeds": SEEDS[:num_rounds],
        "mse_mean": float(mse_array.mean()),
        "mse_std": float(mse_array.std()) if num_rounds > 1 else 0.0,
        "mae_mean": float(mae_array.mean()),
        "mae_std": float(mae_array.std()) if num_rounds > 1 else 0.0,
        "mse_all": mse_array.tolist(),
        "mae_all": mae_array.tolist(),
    }

    print(f"\n{'='*60}")
    print(f"结果汇总 — {model_name.upper()} {task_label}")
    print(f"{'='*60}")
    print(f"  MSE: {summary['mse_mean']:.6f} ± {summary['mse_std']:.6f}")
    print(f"  MAE: {summary['mae_mean']:.6f} ± {summary['mae_std']:.6f}")
    for i in range(num_rounds):
        print(f"    Round {i+1} (seed={SEEDS[i]}): MSE={mse_array[i]:.6f}, MAE={mae_array[i]:.6f}")

    # ——— 保存结果 ———
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  结果已保存: {results_path}")

    # ——— 保存最佳模型 ———
    model_path = os.path.join(output_dir, "best_model.pt")
    torch.save(best_model_state, model_path)
    print(f"  模型已保存: {model_path}")

    # ——— 生成图表 ———
    try:
        _generate_plots(
            best_model_state, model_name, task, output_window,
            X_train, y_train, X_test, y_test,
            sca_group_ids, sca_group_sizes,
            figures_dir, all_logs,
        )
    except Exception as e:
        print(f"  图表生成失败 (不影响训练结果): {e}")


def _generate_plots(
    best_state, model_name, task, output_window,
    X_train, y_train, X_test, y_test,
    sca_group_ids, sca_group_sizes,
    figures_dir, all_logs,
):
    """生成预测对比图 + 训练曲线"""
    import matplotlib
    matplotlib.use('Agg')
    input_dim = X_train.shape[-1]
    is_sca = (model_name == "sca")
    group_sizes = sca_group_sizes if is_sca else None
    group_ids = sca_group_ids if is_sca else None

    with torch.no_grad():
        kwargs = {"group_sizes": group_sizes} if group_sizes else {}
        model = build_model(model_name, input_dim, **kwargs)
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
        model.eval()

        _, test_loader = create_dataloaders(
            X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE
        )
        all_preds, all_trues = [], []
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            if is_sca:
                pred = model(X_batch, group_ids=group_ids, target=y_batch)
            else:
                pred = model(X_batch, target=y_batch, teacher_forcing_ratio=0.0)
            all_preds.append(pred.detach().cpu().numpy())
            all_trues.append(y_batch.detach().cpu().numpy())

        y_pred_all = np.concatenate(all_preds, axis=0)
        y_true_all = np.concatenate(all_trues, axis=0)

        if output_window == OUTPUT_WINDOW_SHORT:
            plot_prediction_vs_ground_truth(
                y_true_all, y_pred_all,
                title=f"Short-term Prediction — {model_name.upper()}",
                save_name=os.path.join(figures_dir, "pred_vs_gt.pdf"),
                num_days=90,
            )
        else:
            plot_long_term_prediction(
                y_true_all, y_pred_all,
                save_name=os.path.join(figures_dir, "pred_vs_gt_long.pdf"),
            )

    # 训练曲线 (取第一轮)
    best_log = all_logs[0]
    plot_training_curves(
        {model_name.upper(): best_log["train_losses"]},
        {model_name.upper(): best_log["test_mse_history"]},
        save_name=os.path.join(figures_dir, "training_curve.pdf"),
    )
    print(f"  图表已保存: {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description="家庭电力消耗预测实验")
    parser.add_argument("--model", type=str, required=True,
                        choices=["lstm", "transformer", "sca"],
                        help="模型名称")
    parser.add_argument("--task", type=str, default="both",
                        choices=["short", "long", "both"],
                        help="预测任务")
    parser.add_argument("--data", type=str, default=DATA_DIR,
                        help="原始数据路径")
    parser.add_argument("--weather", type=str, default=WEATHER_DIR,
                        help="天气数据路径")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖训练轮数 (调试用)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出根目录")
    parser.add_argument("--quick", action="store_true",
                        help="快速验证模式: 1轮+5epochs+小数据量")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅验证数据加载和模型前向传播, 不训练")
    parser.add_argument("--force", action="store_true",
                        help="强制重新训练, 即使已有结果")
    args = parser.parse_args()

    # 允许通过环境变量或命令行覆盖路径
    global OUTPUT_DIR
    if args.output:
        OUTPUT_DIR = args.output

    if args.quick:
        print("⚡ 快速验证模式: 1轮 5epochs 小数据量")
    if args.dry_run:
        print("🔍 Dry-run: 仅验证数据+前向传播")

    print(f"设备: {DEVICE} | GPU 数: {torch.cuda.device_count()}")
    print(f"模型: {args.model} | 任务: {args.task}")
    print(f"数据: {args.data}")
    print(f"输出: {OUTPUT_DIR}")

    summaries = {}
    skip = not args.force
    if args.task in ("short", "both"):
        summaries["short"] = run_experiment(
            args.model, "short", args.data, args.weather, args.epochs,
            quick=args.quick, dry_run=args.dry_run, skip_existing=skip,
        )
    if args.task in ("long", "both"):
        summaries["long"] = run_experiment(
            args.model, "long", args.data, args.weather, args.epochs,
            quick=args.quick, dry_run=args.dry_run, skip_existing=skip,
        )

    # 汇总输出
    if summaries and not args.dry_run:
        print(f"\n{'='*60}")
        print("最终结果")
        print(f"{'='*60}")
        for task, s in summaries.items():
            if s:
                print(f"  [{task}] MSE: {s['mse_mean']:.6f} ± {s['mse_std']:.6f}  "
                      f"MAE: {s['mae_mean']:.6f} ± {s['mae_std']:.6f}")


if __name__ == "__main__":
    main()
