#!/bin/bash
# ============================================================
# 快速验证脚本: 依次验证 LSTM / Transformer / SCA-Net 能跑通
# 用法: bash scripts/run_quick_test.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export PROJECT_ROOT="$PROJECT_DIR"
export DATA_PATH="${DATA_PATH:-$PROJECT_DIR/../household_power_consumption.txt}"
export WEATHER_PATH="${WEATHER_PATH:-$PROJECT_DIR/data/weather_monthly.csv}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$PROJECT_DIR"

for MODEL in lstm transformer sca; do
    echo ""
    echo "============================================"
    echo "  验证: $MODEL"
    echo "============================================"

    # Step 1: dry-run
    echo "--- dry-run: 前向传播测试 ---"
    python main.py --model "$MODEL" --task short --dry-run

    # Step 2: quick train
    echo "--- quick: 小数据量训练 ---"
    python main.py --model "$MODEL" --task short --quick
done

echo ""
echo "============================================"
echo "全部模型验证通过"
echo "============================================"

# 打印结果
for MODEL in lstm transformer sca; do
    R="$OUTPUT_DIR/$MODEL/short/results.json"
    if [ -f "$R" ]; then
        python3 -c "
import json
with open('$R') as f:
    d = json.load(f)
print(f'$MODEL: MSE={d[\"mse_mean\"]:.4f} MAE={d[\"mae_mean\"]:.4f} ({d[\"train_samples\"]}训练/{d[\"test_samples\"]}测试样本)')
"
    fi
done
