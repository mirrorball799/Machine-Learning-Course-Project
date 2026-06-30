"""为 Transformer 生成长期预测图"""
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
from models.transformer_model import TransformerModel
from utils.config import DEVICE, set_seed

REPORT_FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report", "figures")
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "transformer", "long", "best_model.pt")
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "household_power_consumption.txt")
WEATHER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "weather_monthly.csv")

set_seed(42)

X_train, y_train, X_test, y_test, _, features = build_dataset(
    data_path=DATA, output_window=365, weather_path=WEATHER)

model = TransformerModel(input_dim=X_train.shape[-1])
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model = model.to(DEVICE)
model.eval()

_, loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=8)
preds, trues = [], []
with torch.no_grad():
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        preds.append(model(Xb, target=yb).cpu().numpy())
        trues.append(yb.cpu().numpy())
yp = np.concatenate(preds, axis=0)
yt = np.concatenate(trues, axis=0)

fig, ax = plt.subplots(1, 1, figsize=(14, 4))
n = min(365, yt.shape[1])
ax.plot(range(n), yt[0, :n], "b-", label="Ground Truth", lw=1, alpha=0.8)
ax.plot(range(n), yp[0, :n], "r--", label="Transformer Prediction", lw=1, alpha=0.8)
ax.set_xlabel("Days"); ax.set_ylabel("Active Power (normalized)")
ax.set_title("Transformer — Long-term Prediction (90→365 days)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(REPORT_FIG, "pred_vs_gt_long_transformer.pdf")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"OK → {out}")
