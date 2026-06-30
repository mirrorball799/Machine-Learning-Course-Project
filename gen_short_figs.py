"""为三个模型生成短期预测对比图"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data_processing.preprocess import build_dataset
from data_processing.dataset import create_dataloaders
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.improved_model import SCANet
from training.trainer import get_group_ids
from utils.config import DEVICE, set_seed

REPORT_FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report", "figures")
OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "household_power_consumption.txt")
WEATHER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "weather_monthly.csv")

os.makedirs(REPORT_FIG, exist_ok=True)
set_seed(42)

X_train, y_train, X_test, y_test, _, features = build_dataset(
    data_path=DATA, output_window=90, weather_path=WEATHER)
input_dim = X_train.shape[-1]
sca_group_ids = get_group_ids(features)
sca_group_sizes = [len(g) for g in sca_group_ids]

models = {
    "lstm": (LSTMModel(input_dim=input_dim), False, None),
    "transformer": (TransformerModel(input_dim=input_dim), False, None),
    "sca": (SCANet(group_sizes=sca_group_sizes), True, sca_group_ids),
}

name_map = {"lstm": "LSTM", "transformer": "Transformer", "sca": "SCA-Net"}

for mname, (model, is_sca, gids) in models.items():
    ckpt = os.path.join(OUTPUTS, mname, "short", "best_model.pt")
    if not os.path.exists(ckpt):
        print(f"SKIP {mname}: {ckpt} not found")
        continue

    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(DEVICE)
    model.eval()

    _, loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32)
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            out = model(Xb, group_ids=gids, target=yb) if is_sca else model(Xb, target=yb)
            preds.append(out.cpu().numpy())
            trues.append(yb.cpu().numpy())
    yp = np.concatenate(preds, axis=0)
    yt = np.concatenate(trues, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{name_map[mname]} — Short-term Prediction (90→90 days)",
                 fontsize=13, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        idx = i * 5
        if idx >= len(yt): break
        n = min(90, yt.shape[1])
        ax.plot(range(n), yt[idx, :n], "b-", label="Ground Truth", lw=1)
        ax.plot(range(n), yp[idx, :n], "r--", label="Prediction", lw=1)
        ax.set_xlabel("Days"); ax.set_ylabel("Active Power (normalized)")
        ax.set_title(f"Test Sample {idx}"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(REPORT_FIG, f"pred_vs_gt_short_{mname}.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"OK  {mname} → {out}")
