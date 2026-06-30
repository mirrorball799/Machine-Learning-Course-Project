"""统一生成三模型长期预测图（相同尺寸: 14x4 inches 单样本全365天）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

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
    data_path=DATA, output_window=365, weather_path=WEATHER)
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
    ckpt = os.path.join(OUTPUTS, mname, "long", "best_model.pt")
    if not os.path.exists(ckpt):
        print(f"SKIP {mname}: {ckpt}")
        continue
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(DEVICE)
    model.eval()

    _, loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=8)
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            out = model(Xb, group_ids=gids, target=yb) if is_sca else model(Xb, target=yb)
            preds.append(out.cpu().numpy())
            trues.append(yb.cpu().numpy())
    yp = np.concatenate(preds, axis=0)
    yt = np.concatenate(trues, axis=0)

    # 统一尺寸: 14x4 inches, 单样本, 全365天
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    n = min(365, yt.shape[1])
    ax.plot(range(n), yt[0, :n], "b-", label="Ground Truth", lw=1, alpha=0.8)
    ax.plot(range(n), yp[0, :n], "r--", label=f"{name_map[mname]} Prediction", lw=1, alpha=0.8)
    ax.set_xlabel("Days")
    ax.set_ylabel("Active Power (normalized)")
    ax.set_title(f"{name_map[mname]} — Long-term Prediction (90→365 days)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(REPORT_FIG, f"pred_vs_gt_long_{mname}.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"OK {mname}")
