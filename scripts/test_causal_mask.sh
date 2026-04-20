#!/bin/bash
#SBATCH --job-name=df-causal-mask-test
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:15:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

cd /nfs/roberts/project/pi_jks79/jw2933/rhoda
PY=/nfs/roberts/project/pi_jks79/jw2933/rhoda/.venv/bin/python
export PYTHONPATH=/nfs/roberts/project/pi_jks79/jw2933/rhoda:${PYTHONPATH:-}

$PY scripts/test_causal_mask.py
