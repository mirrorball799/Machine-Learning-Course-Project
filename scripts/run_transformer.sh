#!/bin/bash
# ============================================================
# Transformer 模型实验脚本
# 用法: bash scripts/run_transformer.sh [short|long|both]
# ============================================================
set -euo pipefail

TASK="${1:-both}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export PROJECT_ROOT="$PROJECT_DIR"
export DATA_PATH="${DATA_PATH:-$PROJECT_DIR/../household_power_consumption.txt}"
export WEATHER_PATH="${WEATHER_PATH:-$PROJECT_DIR/data/weather_monthly.csv}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

echo "=============================================="
echo " Transformer 实验"
echo "=============================================="
echo "  任务: $TASK"
echo "  GPU:  $CUDA_VISIBLE_DEVICES"
echo "  输出: $OUTPUT_DIR/transformer/"
echo "=============================================="

cd "$PROJECT_DIR"
python main.py \
    --model transformer \
    --task "$TASK" \
    --data "$DATA_PATH" \
    --weather "$WEATHER_PATH" \
    --output "$OUTPUT_DIR"

echo "Transformer 实验完成。结果: $OUTPUT_DIR/transformer/"
