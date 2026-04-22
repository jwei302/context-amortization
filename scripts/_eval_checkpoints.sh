#!/bin/bash
# Eval every 10k checkpoint at fixed horizon=32 to build FVD-vs-step curve.
set -u
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"

H=32
STEPS=(10000 20000 30000 40000 50000 60000 70000 80000 90000 100000)

declare -A RUN_DIR=(
    [df]="outputs/2026-04-20/09-32-55"
    [ca_k8]="outputs/2026-04-20/20-48-57"
    [ca_k4]="outputs/2026-04-21/16-06-17"
)

for NAME in df ca_k8 ca_k4; do
    DIR="${RUN_DIR[$NAME]}"
    for S in "${STEPS[@]}"; do
        CKPT="${DIR}/checkpoints/epoch=0-step=${S}.ckpt"
        OUT="eval_out/${NAME}_steps/step${S}_h${H}"
        if [ ! -f "$CKPT" ]; then
            echo "=== $(date) :: ${NAME} step=${S} ckpt missing — skipping ==="
            continue
        fi
        if [ -f "$OUT/metrics.json" ]; then
            echo "=== $(date) :: ${NAME} step=${S} ALREADY DONE — skipping ==="
            continue
        fi
        echo "=== $(date) :: ${NAME} step=${S} h=${H} starting ==="
        rm -rf "$OUT"
        $PY eval.py --ckpt "$CKPT" --horizon $H --n-contexts 16 --batch-size 2 \
            --no-save-mp4 --out-dir "$OUT" 2>&1 | tail -15
        echo "=== $(date) :: ${NAME} step=${S} done ==="
    done
done
echo "=== CHECKPOINT-SWEEP EVAL COMPLETE $(date) ==="
