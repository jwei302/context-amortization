#!/bin/bash
#SBATCH --job-name=df-train-dmlab-baseline
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

cd /nfs/roberts/project/pi_jks79/jw2933/rhoda
PY=/nfs/roberts/project/pi_jks79/jw2933/rhoda/.venv/bin/python

nvidia-smi | head -15 || true

# DF baseline — random_all noise levels, no loss mask
$PY -m main "+name=train_df_baseline_100k" \
    algorithm=df_video \
    dataset=video_dmlab \
    algorithm.weight_decay=1e-3 \
    algorithm.diffusion.architecture.network_size=48 \
    algorithm.diffusion.architecture.attn_dim_head=32 \
    algorithm.diffusion.architecture.attn_resolutions=[8,16,32,64] \
    algorithm.diffusion.beta_schedule=cosine \
    experiment.training.max_steps=100005 \
    experiment.training.checkpointing.every_n_train_steps=10000 \
    experiment.validation.val_every_n_step=5000 \
    algorithm.noise_level=random_all
