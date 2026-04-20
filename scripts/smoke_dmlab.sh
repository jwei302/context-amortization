#!/bin/bash
#SBATCH --job-name=df-smoke-dmlab
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

cd /nfs/roberts/project/pi_jks79/jw2933/rhoda
source .venv/bin/activate

nvidia-smi | head -15 || true

STEPS=300

# DMLab command flags per README (network_size=48 etc)
COMMON_FLAGS=(
    algorithm=df_video
    dataset=video_dmlab
    algorithm.weight_decay=1e-3
    algorithm.diffusion.architecture.network_size=48
    algorithm.diffusion.architecture.attn_dim_head=32
    algorithm.diffusion.architecture.attn_resolutions=[8,16,32,64]
    algorithm.diffusion.beta_schedule=cosine
    experiment.training.max_steps=$STEPS
    experiment.validation.val_every_n_step=100000   # skip val inside the smoke window
)

# 1. DF baseline
python -m main "+name=smoke_df_baseline" \
    "${COMMON_FLAGS[@]}" \
    algorithm.noise_level=random_all

# 2. CA(k=0) — should track DF baseline closely (same sampling, no loss mask)
python -m main "+name=smoke_ca_k0" \
    "${COMMON_FLAGS[@]}" \
    algorithm.noise_level=context_amortization \
    algorithm.anchor_prefix_size=0

# 3. CA(k=1) — real CA; should diverge from the above two
python -m main "+name=smoke_ca_k1" \
    "${COMMON_FLAGS[@]}" \
    algorithm.noise_level=context_amortization \
    algorithm.anchor_prefix_size=1
