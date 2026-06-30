"""
全局配置文件 — 支持本地 Mac 和远程 RTX3090 服务器
"""

import os
import random

import numpy as np
import torch

# ============ 路径 (通过环境变量覆盖) ============
ROOT_DIR = os.environ.get(
    "PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA_DIR = os.environ.get(
    "DATA_PATH",
    os.path.join(os.path.dirname(ROOT_DIR), "household_power_consumption.txt"),
)
WEATHER_DIR = os.environ.get(
    "WEATHER_PATH",
    os.path.join(ROOT_DIR, "data", "weather_monthly.csv"),
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(ROOT_DIR, "outputs"),
)

# ============ 设备 ============
if torch.cuda.is_available():
    DEVICE = "cuda"
    N_GPU = torch.cuda.device_count()
    USE_MULTI_GPU = N_GPU > 1
else:
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    N_GPU = 0
    USE_MULTI_GPU = False

# ============ 数据参数 ============
INPUT_WINDOW = 90
OUTPUT_WINDOW_SHORT = 90
OUTPUT_WINDOW_LONG = 365
TRAIN_RATIO = 0.8

AGGREGATION_RULES = {
    "global_active_power": "sum",
    "global_reactive_power": "sum",
    "voltage": "mean",
    "global_intensity": "mean",
    "sub_metering_1": "sum",
    "sub_metering_2": "sum",
    "sub_metering_3": "sum",
}

# ============ 模型参数 ============
D_MODEL = 256
N_HEADS = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 2
DROPOUT = 0.1
DIM_FF = 512

LSTM_HIDDEN = 256
LSTM_NUM_LAYERS = 3

FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# ============ 训练参数 ============
# RTX3090 24GB → batch_size 可设大一些
BATCH_SIZE = 128 if USE_MULTI_GPU else 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS_SHORT = 200
EPOCHS_LONG = 300
PATIENCE = 30
NUM_ROUNDS = 5
GRAD_CLIP = 1.0

# ============ 随机种子 ============
SEEDS = [42, 123, 777, 2024, 9999]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_output_dir(model_name: str, task: str) -> str:
    """每个模型+任务的输出目录"""
    d = os.path.join(OUTPUT_DIR, model_name, task)
    os.makedirs(d, exist_ok=True)
    return d


def log_system_info():
    """打印系统信息用于调试"""
    print(f"  Python: {torch.__version__} | Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU 数量: {N_GPU}")
        for i in range(N_GPU):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
            print(f"    GPU {i}: {name} ({mem:.1f} GB)")
    print(f"  Batch Size: {BATCH_SIZE} | Multi-GPU: {USE_MULTI_GPU}")
