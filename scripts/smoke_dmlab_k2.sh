#!/bin/bash
#SBATCH --job-name=df-smoke-k2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
PY="$REPO_ROOT/.venv/bin/python"
export PYTHONPATH="$REPO_ROOT"

nvidia-smi | head -15 || true

STEPS=300

# DMLab CA k=2 smoke — blog-faithful CA matching context_length=2
$PY -m main "+name=smoke_ca_k2" \
    algorithm=df_video \
    dataset=video_dmlab \
    algorithm.weight_decay=1e-3 \
    algorithm.diffusion.architecture.network_size=48 \
    algorithm.diffusion.architecture.attn_dim_head=32 \
    algorithm.diffusion.architecture.attn_resolutions=[8,16,32,64] \
    algorithm.diffusion.beta_schedule=cosine \
    experiment.training.max_steps=$STEPS \
    experiment.validation.val_every_n_step=100000 \
    algorithm.noise_level=context_amortization \
    algorithm.anchor_prefix_size=2
