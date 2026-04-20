# Context Amortization vs Diffusion Forcing on DMLab

## Thesis

Language-model training is efficient because a single forward pass over $T$
tokens with a causal mask produces $T$ supervised predictions in parallel —
every position is both a target and context for later positions. Diffusion
Forcing (DF), the natural causal-video-diffusion training recipe, does not do
this: it noises every frame independently, so training never sees the
"clean past, noised present" distribution it encounters at inference.

Context Amortization (CA) ports the LM trick over:
- Keep the first $k$ frames clean (match DMLab `context_length=2` → $k=2$).
- Noise the remaining frames with independent random levels.
- Compute loss at every non-anchor position; the causal mask makes each of
  those losses correspond to a different inference-time prefix length.

The blog post should show that CA beats DF on (Q1) long-horizon rollout
quality and (Q2) training efficiency to a given FVD target.

## Status (as of 2026-04-19, end of day 1)

Completed:
- Repo cloned at `/nfs/roberts/project/pi_jks79/jw2933/rhoda/` (upstream
  `buoyancy99/diffusion-forcing` main; v1.5-transformer branch does not exist).
- Venv at `.venv/` (uv, Python 3.10, torch 2.4.1+cu121).
- CA wired in — `algorithms/diffusion_forcing/df_base.py`:
  - `_generate_noise_levels()` zeroes the first `k = anchor_prefix_size` rows
    to `stabilization_level - 1` when `noise_level == "context_amortization"`.
  - `training_step` zeroes the loss mask on anchor positions before
    `reweight_loss(...)`.
  - Config flag: `algorithm.anchor_prefix_size` in
    `configurations/algorithm/df_base.yaml`.
- Causal attention verified: perturbation test on `TemporalAttentionBlock`
  (`scripts/test_causal_mask.py`) — positions `< k` bitwise-unchanged,
  positions `>= k` differ. (Note: the full UNet3D has a soft non-causality
  from `nn.GroupNorm` in ResnetBlock aggregating statistics across the time
  dim; architectural and present in the upstream unmodified model.)
- Smoke tests pass: `scripts/smoke_dmlab_k2.sh` runs 300 steps, loss
  ~0.32 at step 299 (sits between k=0 and k=1, as expected).
- Training scripts ready: `scripts/train_dmlab_df.sh` and `train_dmlab_ca.sh`
  at 100k steps, batch 8, LR 8e-5, warmup 10k, `weight_decay=1e-3`,
  `network_size=48`, `attn_resolutions=[8,16,32,64]`, checkpoint every 10k.

Blocker: Yale gpu_h200 queue estimated start is **2026-04-24** (~5 days out),
even after trimming `--time` to 14h. Moving compute to Lambda Labs.

## Moving to Lambda Labs

Repo is pushed to <https://github.com/jwei302/context-amortization>.

On Lambda:
```bash
# Launch a 1× or 2× H100 instance in the Lambda console.
ssh ubuntu@<LAMBDA_IP>

git clone https://github.com/jwei302/context-amortization.git rhoda
cd rhoda

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install "moviepy<2" "imageio<3" imageio-ffmpeg

wandb login                        # API key from wandb.ai/authorize
# Memory says: wandb team is `jeffrey-wei-yale`; personal entity is disabled.

# DMLab auto-downloads on first run (~30M).

# Launch (use tmux so it survives SSH disconnects):
tmux new -s df
bash scripts/train_dmlab_df.sh     # or call python -m main ... directly
# Ctrl+b d to detach.

tmux new -s ca
bash scripts/train_dmlab_ca.sh
```

Both `scripts/train_dmlab_*.sh` are SLURM scripts; on Lambda, either
`bash` them directly (SBATCH lines are ignored as comments), or extract
the `python -m main` command. A 1×H100 can run only one job at a time;
for parallelism rent a 2×H100 instance and prefix with
`CUDA_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=1`.

## Experiment sweep (Day 2)

Two configs, 100k steps each, identical hyperparameters except the
noise-level / anchor knobs. Training window is `n_frames=32` for both
(longer than upstream default of 16), so the comparison is at matched
context size:

| Name        | `noise_level`          | `anchor_prefix_size` | `n_frames` | `context_length` (inference) |
|-------------|------------------------|----------------------|------------|------------------------------|
| DF baseline | `random_all`           | —                    | 32         | 2                            |
| CA          | `context_amortization` | 8                    | 32         | 8                            |

Rationale: DF noises every frame independently. CA clamps the first 8
positions to clean anchors (at `stabilization_level - 1`) and masks the
loss on them. Longer `n_frames` gives CA more non-anchor positions to
amortize loss across (the core efficiency lever), and the larger anchor
prefix matches the blog's "long history of noise-free captured context."

At inference, match `dataset.context_length` to `anchor_prefix_size` so
the model sees the distribution it was trained on.

## Evaluation (Day 3, `eval.py` to be written)

Autoregressive rollout at horizons 16 / 300 / 1000 / 2000 on a held-out
set of starting contexts. Reuse `algorithms/common/metrics/fvd.py`
(`FrechetVideoDistance`, input shape `(T, B, C, H, W)` in `[-1, 1]`)
and `algorithms/common/metrics/lpips.py`. Save mp4s per config per
horizon.

No KV cache upstream → 2000-frame rollouts are quadratic. Keep the
held-out set to ~16 contexts to fit compute budget.

## Key files

- `configurations/algorithm/df_base.yaml` — `anchor_prefix_size` knob.
- `configurations/algorithm/df_video.yaml` — DMLab architecture.
- `configurations/dataset/video_dmlab.yaml` — `n_frames`, `context_length`,
  `frame_skip`.
- `algorithms/diffusion_forcing/df_base.py` — `training_step` loss mask,
  `_generate_noise_levels` noise-sampling branch.
- `algorithms/diffusion_forcing/models/attention.py` —
  `TemporalAttentionBlock` (the piece we verified is causal).
- `experiments/exp_base.py` — `_LossCSVCallback` writes `loss.csv` per run
  (survives offline wandb).
- `scripts/train_dmlab_df.sh`, `scripts/train_dmlab_ca.sh` — main training.
- `scripts/test_causal_mask.py` — causal-attention sanity check.

## Risks

- **CA doesn't beat DF meaningfully.** Honest null result still ships.
- **GroupNorm leakage is a confound.** If CA wins, some of the "clean
  context" statistics flow through GroupNorm to anchor positions during
  training too, so the attribution isn't clean. Flag in the writeup.
- **Runs take longer than expected.** Cut step count before cutting
  methods (50k × 3 configs > 100k × 2).
