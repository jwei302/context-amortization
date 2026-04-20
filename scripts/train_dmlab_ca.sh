#!/bin/bash
#SBATCH --job-name=df-train-dmlab-ca-k2
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

# CA k=2 — blog-faithful: 2 clean anchor frames (matches DMLab context_length=2),
# random noise on suffix, loss masked on anchors.
$PY -m main "+name=train_ca_k2_100k" \
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
    algorithm.noise_level=context_amortization \
    algorithm.anchor_prefix_size=2
