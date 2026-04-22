#!/bin/bash
# Runs all evals not yet completed. Solo-GPU only — assumes training is done.
set -u
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"

DF_CKPT="outputs/2026-04-20/09-32-55/checkpoints/epoch=0-step=100000.ckpt"
CA_K8_CKPT="outputs/2026-04-20/20-48-57/checkpoints/epoch=0-step=100000.ckpt"
CA_K4_CKPT="outputs/2026-04-21/16-06-17/checkpoints/epoch=0-step=100000.ckpt"

run_eval() {
    local NAME=$1 CKPT=$2 H=$3
    local OUT="eval_out/${NAME}/h${H}"
    if [ -f "$OUT/metrics.json" ]; then
        echo "=== $(date) :: ${NAME} h=${H} ALREADY DONE — skipping ==="
        return
    fi
    echo "=== $(date) :: ${NAME} h=${H} starting ==="
    rm -rf "$OUT"
    $PY eval.py --ckpt "$CKPT" --horizon $H --n-contexts 16 --batch-size 2 \
        --out-dir "$OUT" 2>&1 | tail -25
    echo "=== $(date) :: ${NAME} h=${H} done ==="
}

# DF: h=64 new (h=32, h=128 already done; h=512 partial kept as bonus)
run_eval df "$DF_CKPT" 64

# CA k=8: h=64 new (h=32, h=128 already done)
run_eval ca_k8 "$CA_K8_CKPT" 64

# CA k=4: all three horizons
for H in 32 64 128; do run_eval ca_k4 "$CA_K4_CKPT" $H; done

echo "=== ALL EVALS COMPLETE $(date) ==="
