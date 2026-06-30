"""
SCA-Net 消融实验

针对短期预测任务 (90→90), 每种变体运行 5 轮, 报告 MSE/MAE 均值±标准差.
用法: python run_ablation.py [--quick]
"""

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from data_processing.preprocess import build_dataset
from data_processing.dataset import create_dataloaders
from models.improved_model import SCANet
from models.ablation_models import (
    SCANet_NoGroup, SCANet_NoMSC, SCANet_NoGate,
)
from training.trainer import train_model, get_group_ids
from utils.config import (
    DATA_DIR, WEATHER_DIR, OUTPUT_DIR, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS_SHORT, PATIENCE,
    SEEDS, D_MODEL, N_HEADS, NUM_ENCODER_LAYERS, DIM_FF, DROPOUT,
    DEVICE, INPUT_WINDOW, OUTPUT_WINDOW_SHORT, GRAD_CLIP,
    set_seed, log_system_info,
)

ABLATION_VARIANTS = {
    "full":     ("完整模型", "SCANet"),
    "no_group": ("去除变量分组", "SCANet_NoGroup"),
    "no_msc":   ("去除多尺度卷积", "SCANet_NoMSC"),
    "no_gate":  ("去除跨组门控", "SCANet_NoGate"),
}


def build_ablation_model(variant_key, input_dim, group_sizes):
    kwargs = dict(d_model=D_MODEL, n_heads=N_HEADS,
                  num_encoder_layers=NUM_ENCODER_LAYERS,
                  dim_ff=DIM_FF, dropout=DROPOUT)
    if variant_key == "full":
        return SCANet(group_sizes=group_sizes, **kwargs)
    elif variant_key == "no_group":
        return SCANet_NoGroup(input_dim=input_dim, **kwargs)
    elif variant_key == "no_msc":
        return SCANet_NoMSC(group_sizes=group_sizes, **kwargs)
    elif variant_key == "no_gate":
        return SCANet_NoGate(group_sizes=group_sizes, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant_key}")


def run_one_variant(variant_key, output_dir, X_train, y_train, X_test, y_test,
                    group_ids, group_sizes, quick):
    name_cn, name_en = ABLATION_VARIANTS[variant_key]
    num_rounds = 1 if quick else 5
    epochs = 5 if quick else EPOCHS_SHORT

    print(f"\n{'='*60}")
    print(f"消融: {name_cn} ({name_en}) | rounds={num_rounds} epochs={epochs}")
    print(f"{'='*60}")

    mse_list, mae_list = [], []
    best_state, best_mse = None, float("inf")

    for r in range(num_rounds):
        set_seed(SEEDS[r])
        print(f"  Round {r+1}/{num_rounds} (seed={SEEDS[r]})")

        train_loader, test_loader = create_dataloaders(
            X_train, y_train, X_test, y_test, BATCH_SIZE)

        model = build_ablation_model(variant_key, X_train.shape[-1], group_sizes)
        is_sca = (variant_key != "no_group")  # no_group doesn't use group_ids

        result = train_model(
            model=model, train_loader=train_loader, test_loader=test_loader,
            model_name=f"ablation_{variant_key}", output_window=OUTPUT_WINDOW_SHORT,
            device=DEVICE, epochs=epochs, lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY, patience=PATIENCE,
            is_sca=is_sca, group_ids=group_ids if is_sca else None,
            grad_clip=GRAD_CLIP, output_dir=output_dir,
        )
        mse_list.append(result["best_test_mse"])
        mae_list.append(result["best_test_mae"])
        if result["best_test_mse"] < best_mse:
            best_mse = result["best_test_mse"]
            best_state = result["model_state"]

    mse_a = np.array(mse_list)
    mae_a = np.array(mae_list)
    return {
        "variant": variant_key, "name": name_cn,
        "mse_mean": float(mse_a.mean()), "mse_std": float(mse_a.std()) if num_rounds > 1 else 0.0,
        "mae_mean": float(mae_a.mean()), "mae_std": float(mae_a.std()) if num_rounds > 1 else 0.0,
        "mse_all": mse_a.tolist(), "mae_all": mae_a.tolist(),
        "best_state": best_state,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--weather", default=WEATHER_DIR)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--variants", nargs="+",
                        default=list(ABLATION_VARIANTS.keys()),
                        choices=list(ABLATION_VARIANTS.keys()))
    args = parser.parse_args()

    output_root = args.output or os.path.join(OUTPUT_DIR, "ablation")
    os.makedirs(output_root, exist_ok=True)

    print(f"消融实验 | 变体: {args.variants} | quick={args.quick}")
    log_system_info()

    # 加载数据 (短期)
    X_train, y_train, X_test, y_test, _, features = build_dataset(
        data_path=args.data, output_window=OUTPUT_WINDOW_SHORT, weather_path=args.weather)
    group_ids = get_group_ids(features)
    group_sizes = [len(g) for g in group_ids]
    print(f"  features={len(features)} groups={group_sizes}")

    results = []
    for vk in args.variants:
        out_dir = os.path.join(output_root, vk)
        os.makedirs(out_dir, exist_ok=True)
        r = run_one_variant(vk, out_dir, X_train, y_train, X_test, y_test,
                            group_ids, group_sizes, args.quick)
        # 保存最佳模型
        if r["best_state"]:
            torch.save(r["best_state"], os.path.join(out_dir, "best_model.pt"))
        # 保存结果
        results.append({k: v for k, v in r.items() if k != "best_state"})
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results[-1], f, indent=2, ensure_ascii=False)
        print(f"  {r['name']}: MSE={r['mse_mean']:.4f}±{r['mse_std']:.4f}  "
              f"MAE={r['mae_mean']:.4f}±{r['mae_std']:.4f}")

    # 汇总
    print(f"\n{'='*60}")
    print("消融实验结果汇总")
    print(f"{'='*60}")
    print(f"{'变体':<20s}  {'MSE':>14s}  {'MAE':>14s}")
    print("-" * 52)
    for r in results:
        print(f"{r['name']:<20s}  {r['mse_mean']:>6.4f}±{r['mse_std']:<8.4f}  "
              f"{r['mae_mean']:>6.4f}±{r['mae_std']:<8.4f}")

    summary_path = os.path.join(output_root, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
