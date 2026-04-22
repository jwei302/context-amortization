# Context Amortization vs. Diffusion Forcing on DMLab

A controlled comparison of **Context Amortization (CA)** against the **Diffusion
Forcing (DF)** baseline for autoregressive video prediction on the DMLab
dataset. Forked from [boyuanchen/diffusion-forcing](https://github.com/buoyancy99/diffusion-forcing).

## Task

Given an 8-frame clean prefix from a DMLab clip, generate the next N frames
autoregressively and evaluate quality with FVD, LPIPS, PSNR, SSIM.

- **DF** (baseline): every training frame gets an independent random noise
  level; loss is computed on every position. At inference, clean context
  frames are held fixed while the remainder is denoised.
- **CA** (this work): the first `k` frames are kept at ~zero noise during
  training (the "anchor prefix"); the remainder gets random noise as in DF.
  **Loss is masked on the anchor positions** — the model only learns to
  denoise the tail conditioned on clean anchors. This matches the inference
  distribution exactly, which we hypothesize improves long-horizon rollout
  quality.

Three configurations trained for 100k steps on a single GH200:
- **DF** (effectively k=1, no real anchor)
- **CA k=4** (4 clean anchors + 28 noised, of 32)
- **CA k=8** (8 clean anchors + 24 noised, of 32)

## Results

FVD (lower is better), `n_contexts=16` held-out DMLab validation clips:

| Config  | h=32   | h=64  | h=128 |
|---------|-------:|------:|------:|
| DF      | 1500   | 807   | 852   |
| CA k=4  | **799**| 825   | **580** |
| CA k=8  | 1062   | **779** | 635 |

CA beats DF at every horizon; CA k=4 is the strongest config on this sweep.
Full FVD/LPIPS/PSNR/SSIM per (config, horizon) in
`eval_out/{df,ca_k4,ca_k8}/h{32,64,128}/metrics.json`.

Figures: `figs/fvd_vs_horizon.png` (hero chart), `figs/loss_curves.png`
(training), `figs/frame_strip_h{32,64,128}.png` (qualitative).

## Pretrained checkpoints

The three 100k-step checkpoints (214 MB each) are hosted on HuggingFace:

> **https://huggingface.co/jwei302/context-amortization**

Files: `df.ckpt`, `ca_k4.ckpt`, `ca_k8.ckpt`. Each checkpoint's matching
Hydra config snapshot lives in this repo at `artifacts/{df,ca_k4,ca_k8}/config.yaml`
and `eval.py` auto-loads it.

Download:
```bash
pip install -U "huggingface_hub[cli]"
hf download jwei302/context-amortization df.ckpt ca_k4.ckpt ca_k8.ckpt --local-dir checkpoints/
```

## Reproducing from scratch

### 1. Environment

Tested on Linux aarch64 (NVIDIA GH200). `uv` recommended.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install "moviepy<2" "imageio<3" imageio-ffmpeg
```

Gotchas:
- On aarch64, `torch==2.4.1+cu121` has no wheels. Install torch 2.5.1+cu124
  from the PyTorch index: `--index-url https://download.pytorch.org/whl/cu124`.
- Pin `numpy<2` (torchmetrics/moviepy break on numpy 2.x).
- Pin `setuptools<81` (torchmetrics 0.11.4 imports `pkg_resources`).
- Flash-attention on GH200 requires bf16 inputs. `eval.py` wraps rollout in
  `@torch.autocast("cuda", dtype=torch.bfloat16)`; don't drop that.

### 2. Data

DMLab autodownloads from the TECO source on first training run (~57 GB into
`data/dmlab/`). The upstream dataloader has a bug where
`datasets/video/dmlab_video_dataset.py` calls `rmdir()` on a regular file
during extraction. Workaround — download and extract manually before the
first run:

```bash
# Fetch the three parts directly, then stream-extract
wget <dmlab_dataset_aa_url> <ab_url> <ac_url>
cat dmlab.tar.partaa dmlab.tar.partab dmlab.tar.partac | tar -xf - -C data/
# Hand-write metadata: {"training": [300]*39375, "validation": [300]*625}
```

### 3. Training

`scripts/_run_df_then_ca.sh` runs DF and CA k=8 back-to-back (use `tmux`
to survive SSH disconnects). `scripts/train_dmlab_ca_k4.sh` runs the k=4
ablation separately. Each run is ~11 h on a GH200 at bf16-mixed.

```bash
tmux new-session -d -s train "WANDB_MODE=offline bash scripts/_run_df_then_ca.sh"
tmux new-session -d -s train_k4 "WANDB_MODE=offline bash scripts/train_dmlab_ca_k4.sh"
```

Outputs land in `outputs/<date>/<time>/` with `loss.csv`, `.hydra/config.yaml`,
and `checkpoints/`.

**Known issue — intermediate checkpoints are deleted**: Lightning defaults to
`save_top_k=1` on the `ModelCheckpoint` callback, so only the latest step is
retained even though `every_n_train_steps=10000` is set. If you want an
FVD-vs-training-step chart, patch `experiments/exp_base.py` to pass
`save_top_k=-1` to ModelCheckpoint before running.

### 4. Evaluation

```bash
python eval.py \
  --ckpt <path/to/checkpoint.ckpt> \
  --horizon 128 --n-contexts 16 --batch-size 2 \
  --out-dir eval_out/<name>/h128
```

`eval.py` auto-loads the Hydra config from `<ckpt_dir>/../.hydra/config.yaml`
(or the HuggingFace-downloaded checkpoint works if you copy the matching
`artifacts/<name>/config.yaml` next to it).

For the full sweep:
```bash
bash scripts/_eval_remaining.sh       # DF + CA k=8 + CA k=4 × {32, 64, 128}
```

### 5. Plots

```bash
python scripts/make_plots.py          # reads eval_out/ and artifacts/
```

Writes all figures into `figs/`. The training-loss plot reads from
`artifacts/{df,ca_k8,ca_k4}/loss.csv` so it works off a fresh clone.

## Repo layout

```
algorithms/diffusion_forcing/    # DF + CA (noise_level=context_amortization)
artifacts/                       # loss CSVs + Hydra configs, checkpoint-paired
configurations/                  # Hydra configs (datasets, algorithms, experiments)
datasets/                        # DMLab dataset loader (with upstream bug noted above)
eval.py                          # long-horizon rollout + metrics
eval_out/                        # per-(config, horizon) metrics.json + mp4s
experiments/                     # Lightning experiment wrappers
figs/                            # generated plots
scripts/
  train_dmlab_{df,ca,ca_k4}.sh   # training entrypoints
  _run_df_then_ca.sh             # DF → CA orchestrator
  _eval_remaining.sh             # batch eval across configs/horizons
  make_plots.py                  # all figures
```

## Credit

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research
template. By its MIT license, this credit line must stay. See `LICENSE`.

```
@article{chen2025diffusion,
  title={Diffusion forcing: Next-token prediction meets full-sequence diffusion},
  author={Chen, Boyuan and Mart{\'\i} Mons{\'o}, Diego and Du, Yilun and Simchowitz, Max and Tedrake, Russ and Sitzmann, Vincent},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={24081--24125},
  year={2025}
}
```
