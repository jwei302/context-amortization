#!/bin/bash
# Polls until k=4 100k checkpoint exists, then runs all remaining evals + plots.
set -u
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"

CKPT="outputs/2026-04-21/16-06-17/checkpoints/epoch=0-step=100000.ckpt"
echo "=== $(date) :: waiting for $CKPT ==="
while [ ! -f "$CKPT" ]; do
    sleep 60
done
# Give Lightning a moment to flush the file
sleep 30
echo "=== $(date) :: k=4 ckpt found, starting eval ==="

bash scripts/_eval_remaining.sh
echo "=== $(date) :: horizon-sweep done, starting checkpoint sweep ==="

bash scripts/_eval_checkpoints.sh
echo "=== $(date) :: checkpoint-sweep done, generating plots ==="

$PY scripts/make_plots.py
echo "=== $(date) :: ALL DONE ==="
