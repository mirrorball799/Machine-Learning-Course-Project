#!/bin/bash
# ============================================================
# SCA-Net 消融实验
# 用法: bash scripts/run_ablation.sh [--quick]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export PROJECT_ROOT="$PROJECT_DIR"
export DATA_PATH="${DATA_PATH:-$PROJECT_DIR/../household_power_consumption.txt}"
export WEATHER_PATH="${WEATHER_PATH:-$PROJECT_DIR/data/weather_monthly.csv}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

QUICK="${1:-}"
QUICK_FLAG=""
if [ "$QUICK" = "--quick" ]; then
    QUICK_FLAG="--quick"
    echo "消融实验 [快速验证模式]"
else
    echo "消融实验 [完整模式]"
fi

echo "=============================================="
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  输出: $OUTPUT_DIR/ablation/"
echo "=============================================="

cd "$PROJECT_DIR"
python run_ablation.py $QUICK_FLAG

echo "消融实验完成。结果: $OUTPUT_DIR/ablation/ablation_summary.json"
