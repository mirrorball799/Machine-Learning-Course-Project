#!/bin/bash
# ============================================================
# 全部模型实验 (LSTM → Transformer → SCA-Net)
# 用法: bash scripts/run_all.sh [short|long|both]
# ============================================================
set -euo pipefail

TASK="${1:-both}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "##############################################"
echo "# 全部模型实验 — 任务: $TASK"
echo "# 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "##############################################"

for MODEL in lstm transformer sca; do
    echo ""
    echo "============================================"
    echo "  运行: $MODEL"
    echo "============================================"
    bash "$SCRIPT_DIR/run_${MODEL}.sh" "$TASK"
    echo "  $MODEL 完成: $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "##############################################"
echo "# 全部实验完成"
echo "# 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "##############################################"

# 收集所有结果
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/../outputs}"
echo ""
echo "=== 结果汇总 ==="
for MODEL in lstm transformer sca; do
    echo "--- $MODEL ---"
    for T in short long; do
        RESULT_FILE="$OUTPUT_DIR/$MODEL/$T/results.json"
        if [ -f "$RESULT_FILE" ]; then
            python3 -c "
import json
with open('$RESULT_FILE') as f:
    d = json.load(f)
print(f\"  [$T] MSE: {d['mse_mean']:.6f} ± {d['mse_std']:.6f}  MAE: {d['mae_mean']:.6f} ± {d['mae_std']:.6f}\")
"
        fi
    done
done
