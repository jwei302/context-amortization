# Context Amortization vs. Diffusion Forcing on DMLab

Forked from [boyuanchen/diffusion-forcing](https://github.com/buoyancy99/diffusion-forcing).

## Motivation

This project is inspired by [Rhoda AI's research blog](https://rhoda.ai) on
context amortization and by how autoregressive language models train. Both
point to the same idea: **condition on clean context, supervise only the
tokens you actually want to generate**. LMs do this with teacher forcing and
a prompt mask — the prompt incurs no loss, uncertainty lives only in the
targets. Context Amortization is the video-diffusion analog.

### The train–test gap

Standard **Diffusion Forcing (DF)** samples an independent random noise
level for every frame during training, including the frames that will serve
as context. At inference, however, the context is clean (noise level 0) and
only the future is denoised. The "clean context + noisy future" joint
distribution is never seen during training — the model is rolled out from
outside its training support, and autoregressive drift follows.

**Context Amortization (CA)** closes the gap: the first `k` anchor frames
are held at near-zero noise during training, the remaining frames get
random noise as in DF, and **the loss is masked on the anchor positions**.
Every gradient step goes toward the exact conditional the model computes
at inference, `p(x_{k+1:T} | x_{0:k})` with clean `x_{0:k}` — nothing is
wasted on a regime the model never encounters.

## Task & results

On DMLab, predict N frames autoregressively from an 8-frame clean prefix.
Compare three runs, all 100k steps, identical architecture, optimizer,
batch size, and wall-clock on a single GH200:

- **DF** (no real anchor, k=1)
- **CA k=4** (4 clean anchors, 28 noised)
- **CA k=8** (8 clean anchors, 24 noised)

FVD (lower is better), 16 held-out validation clips:

| Config  | h=32     | h=64     | h=128    |
|---------|---------:|---------:|---------:|
| DF      | 1500     | 807      | 852      |
| CA k=4  | **799**  | 825      | **580**  |
| CA k=8  | 1062     | **779**  | 635      |

CA beats DF at every horizon (25–47% lower FVD at matched compute). Full
metrics in `eval_out/{df,ca_k4,ca_k8}/h{32,64,128}/metrics.json`; plots in
`figs/`.

## Pretrained checkpoints

Hosted on HuggingFace: **https://huggingface.co/jwei302/context-amortization**
(`df.ckpt`, `ca_k4.ckpt`, `ca_k8.ckpt`). Matching Hydra configs are in
`artifacts/{df,ca_k4,ca_k8}/config.yaml`.

```bash
pip install -U "huggingface_hub[cli]"
hf download jwei302/context-amortization df.ckpt ca_k4.ckpt ca_k8.ckpt --local-dir checkpoints/
```

## Reproducing from scratch

### Environment (Linux aarch64, tested on GH200)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install "moviepy<2" "imageio<3" imageio-ffmpeg
```

On aarch64, install torch 2.5.1+cu124 (`--index-url
https://download.pytorch.org/whl/cu124`); pin `numpy<2` and `setuptools<81`.
`eval.py` wraps rollout in `@torch.autocast("cuda", dtype=torch.bfloat16)` —
required for Flash Attention on GH200.

### Data

DMLab autodownloads on first training run (~57 GB → `data/dmlab/`). The
upstream loader has a bug (`part_file.rmdir()` on a regular file); if it
breaks, download the three tar parts directly, `cat` them into `tar -xf -`,
and write `data/dmlab/metadata.json` = `{"training":[300]*39375,"validation":[300]*625}`.

### Training

```bash
tmux new-session -d -s train    "WANDB_MODE=offline bash scripts/_run_df_then_ca.sh"
tmux new-session -d -s train_k4 "WANDB_MODE=offline bash scripts/train_dmlab_ca_k4.sh"
```

~11 h per run on a GH200 at bf16-mixed.

### Evaluation & plots

```bash
python eval.py --ckpt <ckpt> --horizon 128 --n-contexts 16 --batch-size 2 --out-dir eval_out/<name>/h128
bash scripts/_eval_remaining.sh       # full sweep across configs × {32,64,128}
python scripts/make_plots.py          # writes figs/
```

## Credit

Forked from [Boyuan Chen](https://boyuan.space/)'s research template; MIT
license requires keeping this line.

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
