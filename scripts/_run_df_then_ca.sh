#!/bin/bash
# Sequential runner for DF then CA. Launch inside a tmux session so the run
# survives SSH disconnects. Outputs go to outputs/train_{df,ca}.log.

set -u
cd "$(dirname "$0")/.."
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "=== DF starting at $(date) ===" | tee -a outputs/train_orchestrator.log
bash scripts/train_dmlab_df.sh > outputs/train_df.log 2>&1
rc_df=$?
echo "=== DF finished at $(date) with rc=$rc_df ===" | tee -a outputs/train_orchestrator.log

if [ "$rc_df" -ne 0 ]; then
    echo "=== DF failed, aborting CA ===" | tee -a outputs/train_orchestrator.log
    exit 1
fi

echo "=== CA starting at $(date) ===" | tee -a outputs/train_orchestrator.log
bash scripts/train_dmlab_ca.sh > outputs/train_ca.log 2>&1
rc_ca=$?
echo "=== CA finished at $(date) with rc=$rc_ca ===" | tee -a outputs/train_orchestrator.log

exit $rc_ca
