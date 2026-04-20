#!/bin/bash
#SBATCH --job-name=df-repro-minecraft
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

cd /nfs/roberts/project/pi_jks79/jw2933/rhoda
source .venv/bin/activate

nvidia-smi | head -20

python -m main \
    +name=repro_minecraft_pretrained \
    load=outputs/minecraft.ckpt \
    experiment.tasks=[validation]
