#!/bin/bash
# Sequential eval: DF + CA across horizons 16/300/1000. Runs alongside the
# k=4 training job — uses batch_size=2 to leave GPU memory for both.
set -u
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"

DF_CKPT="outputs/2026-04-20/09-32-55/checkpoints/epoch=0-step=100000.ckpt"
CA_CKPT="outputs/2026-04-20/20-48-57/checkpoints/epoch=0-step=100000.ckpt"

for CONFIG in "df:$DF_CKPT" "ca:$CA_CKPT"; do
    NAME="${CONFIG%%:*}"
    CKPT="${CONFIG#*:}"
    for H in 32 128 512; do
        OUT="eval_out/${NAME}/h${H}"
        echo "=== $(date) :: ${NAME} horizon=${H} ==="
        $PY eval.py --ckpt "$CKPT" --horizon $H --n-contexts 16 --batch-size 2 \
            --out-dir "$OUT" 2>&1 | tail -40
        echo "=== $(date) :: ${NAME} horizon=${H} DONE ==="
    done
done
echo "=== ALL EVALS COMPLETE $(date) ==="
