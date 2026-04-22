"""
Long-horizon autoregressive rollout + FVD/LPIPS/PSNR/SSIM eval for DF/CA runs.

Usage:
    python eval.py \
        --ckpt outputs/<run>/checkpoints/last.ckpt \
        --horizon 16 \
        --n-contexts 16 \
        --out-dir eval_out/<run>/h16

Loads the Hydra config from the checkpoint's sibling .hydra/ dir, instantiates
the DiffusionForcingVideo model, loads state_dict, pulls n_contexts clips from
the DMLab validation split, and runs sliding-window rollout.

- For H <= n_tokens (usually 32), this is a single sampling pass in the model's
  native window — equivalent to what validation_step does during training.
- For H > n_tokens, the window slides: each step emits (n_tokens - anchor) new
  frames, using the previous window's last `anchor` frames as the clean prefix.
  Cost is linear in H (not quadratic), since attention is always over n_tokens.

FVD is computed on positions >= context_length at all horizons (distributional,
no per-frame alignment needed). LPIPS/PSNR/SSIM require GT at every position
and are only computed when horizon <= (clip_len - context_length).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to Lightning .ckpt file")
    p.add_argument("--horizon", type=int, required=True,
                   help="Number of frames to generate past the context")
    p.add_argument("--n-contexts", type=int, default=16,
                   help="Number of held-out contexts to evaluate")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch dim for rollout")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-mp4", action="store_true", default=True)
    p.add_argument("--no-save-mp4", dest="save_mp4", action="store_false")
    return p.parse_args()


def load_cfg(ckpt_path: Path):
    """Hydra snapshot lives at <output_dir>/.hydra/config.yaml, where
    output_dir is the parent of the checkpoints/ directory."""
    output_dir = ckpt_path.parent
    if output_dir.name == "checkpoints":
        output_dir = output_dir.parent
    hydra_cfg = output_dir / ".hydra" / "config.yaml"
    if not hydra_cfg.exists():
        raise FileNotFoundError(
            f"Could not find hydra config snapshot at {hydra_cfg}. "
            "Expected ckpt at <output_dir>/checkpoints/X.ckpt."
        )
    return OmegaConf.load(hydra_cfg), output_dir


def build_algo(cfg):
    """Instantiate the DiffusionForcingVideo LightningModule from config.
    Mirrors what experiments/exp_video.py does via compatible_algorithms."""
    from algorithms.diffusion_forcing.df_video import DiffusionForcingVideo
    algo = DiffusionForcingVideo(cfg.algorithm)
    return algo


def load_state(algo, ckpt_path: Path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = algo.load_state_dict(state["state_dict"], strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys, first 3: {missing[:3]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys, first 3: {unexpected[:3]}")
    return algo


def build_eval_loader(cfg, horizon: int, context_length: int, batch_size: int,
                      n_contexts: int):
    """Pull held-out DMLab clips long enough to cover context + horizon where
    possible. DMLab source videos are 300 frames; capped by frame_skip."""
    from datasets.video.dmlab_video_dataset import DmlabVideoDataset

    frame_skip = cfg.dataset.frame_skip
    # We need context + min(horizon, clip_max) GT frames for per-frame metrics.
    # Cap at 300 / frame_skip = 150 subsampled frames (the DMLab source limit).
    max_clip = 300 // frame_skip
    needed = context_length + horizon
    eval_n_frames = min(needed, max_clip)

    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset))
    ds_cfg.n_frames = eval_n_frames
    ds_cfg.validation_multiplier = 1

    dataset = DmlabVideoDataset(ds_cfg, split="validation")
    # Subset to first n_contexts clips (idx_remap in the dataset is seeded).
    indices = list(range(min(n_contexts, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=2,
    )
    return loader, eval_n_frames


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def rollout(algo, xs_context, conditions_full, horizon: int, anchor: int):
    """Autoregressive sliding-window rollout.

    Args:
        algo: DiffusionForcingVideo (on device, eval mode).
        xs_context: (T_ctx, B, C_stacked, H, W) — normalized clean context.
            T_ctx = anchor frames.
        conditions_full: list of length (T_ctx + horizon), per-position
            conditioning (None for DMLab since external_cond_dim=0).
        horizon: number of NEW frames to generate past the context.
        anchor: number of clean anchor frames kept at the front of each window
            (== context_length == algo.context_frames // frame_stack).

    Returns:
        xs_pred: (T_ctx + horizon, B, C_stacked, H, W), normalized.
    """
    device = xs_context.device
    batch_size = xs_context.shape[1]
    n_tokens = algo.n_tokens            # 32 for DMLab
    # How many new frames per sampling pass. With chunk_size=1 we regenerate a
    # single frame per pass (the upstream default). For eval we want (n_tokens
    # - anchor) frames per pass to make long rollouts tractable.
    chunk_size = min(algo.chunk_size if algo.chunk_size > 0 else n_tokens,
                     n_tokens - anchor)
    if chunk_size <= 0:
        chunk_size = max(1, n_tokens - anchor)

    xs_pred = xs_context.clone()
    curr_frame = xs_pred.shape[0]           # == anchor
    target = curr_frame + horizon

    pbar = tqdm(total=target, initial=curr_frame, desc="rollout")
    while curr_frame < target:
        h = min(target - curr_frame, chunk_size)
        assert h <= n_tokens

        # Sliding window: only attend to the last n_tokens frames.
        start = max(0, curr_frame + h - n_tokens)

        # Initialize h new positions at full noise, clipped.
        chunk = torch.randn(
            (h, batch_size, *algo.x_stacked_shape), device=device,
        ).clamp_(-algo.clip_noise, algo.clip_noise)
        xs_pred = torch.cat([xs_pred, chunk], dim=0)

        sched = algo._generate_scheduling_matrix(h)
        for m in range(sched.shape[0] - 1):
            # Positions before curr_frame are "clean context" at noise level 0;
            # positions in the generation window track the scheduling matrix.
            from_levels = np.concatenate(
                (np.zeros((curr_frame,), dtype=np.int64), sched[m]),
            )[:, None].repeat(batch_size, axis=1)
            to_levels = np.concatenate(
                (np.zeros((curr_frame,), dtype=np.int64), sched[m + 1]),
            )[:, None].repeat(batch_size, axis=1)
            from_levels = torch.from_numpy(from_levels).to(device)
            to_levels = torch.from_numpy(to_levels).to(device)

            cond_slice = conditions_full[start : curr_frame + h]
            xs_pred[start:] = algo.diffusion_model.sample_step(
                xs_pred[start:],
                cond_slice,
                from_levels[start:],
                to_levels[start:],
            )

        curr_frame += h
        pbar.update(h)
    pbar.close()
    return xs_pred


def save_mp4(frames_btchw: torch.Tensor, out_path: Path, fps: int = 8):
    """frames_btchw in [0, 1], shape (T, B, C, H, W). Saves one mp4 per B."""
    import imageio
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = (frames_btchw.clamp(0, 1) * 255).byte().cpu().numpy()
    t, b, c, h, w = frames.shape
    frames = np.transpose(frames, (1, 0, 3, 4, 2))  # (B, T, H, W, C)
    for i in range(b):
        imageio.mimwrite(
            out_path.with_name(f"{out_path.stem}_sample{i}.mp4"),
            frames[i], fps=fps, codec="libx264",
        )


def compute_metrics(xs_pred, xs_gt, context_length, horizon):
    """xs_pred, xs_gt: (T, B, C, H, W) in [-1, 1] (post-unnormalize, we shift).

    FVD is always computed on positions >= context_length.
    LPIPS/PSNR/SSIM only computed when GT covers the full horizon.
    """
    from utils.logging_utils import get_validation_metrics_for_videos
    from algorithms.common.metrics.fvd import FrechetVideoDistance
    from algorithms.common.metrics.lpips import LearnedPerceptualImagePatchSimilarity

    device = xs_pred.device
    gen = xs_pred[context_length : context_length + horizon]
    # Rescale [0, 1] -> [-1, 1] for LPIPS/FVD
    gen = gen * 2 - 1
    gt_available = xs_gt.shape[0] >= context_length + horizon
    if gt_available:
        gt = xs_gt[context_length : context_length + horizon] * 2 - 1
    else:
        # Use whatever GT exists up to clip boundary; FVD still works by sampling
        # additional real clips, but for the scaffold we just use what's there.
        gt = xs_gt[context_length:] * 2 - 1
        gen_for_fvd = gen[: gt.shape[0]]
    fvd_model = FrechetVideoDistance().to(device) if gen.shape[0] >= 9 else None

    metrics = {}
    if gt_available:
        lpips_model = LearnedPerceptualImagePatchSimilarity().to(device)
        out = get_validation_metrics_for_videos(
            gen, gt,
            lpips_model=lpips_model,
            fvd_model=fvd_model,
        )
        metrics.update({k: float(v) for k, v in out.items()})
    elif fvd_model is not None:
        # FVD-only path for horizons past GT coverage. Align lengths for now.
        # TODO: replace with distributional FVD over sampled real clips once
        # we have the loader wired to emit full 300-frame source clips.
        metrics["fvd_partial"] = float(fvd_model.compute(
            torch.clamp(gen_for_fvd, -1, 1),
            torch.clamp(gt, -1, 1),
        ))
        metrics["fvd_gt_frames"] = int(gt.shape[0])
    return metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg, output_dir = load_cfg(args.ckpt)
    print(f"Loaded config from {output_dir}")
    print(f"  noise_level={cfg.algorithm.noise_level}, "
          f"anchor_prefix_size={cfg.algorithm.anchor_prefix_size}, "
          f"n_frames={cfg.dataset.n_frames}, "
          f"context_length={cfg.dataset.context_length}")

    context_length = cfg.dataset.context_length
    anchor = cfg.algorithm.anchor_prefix_size

    algo = build_algo(cfg).to(device).eval()
    load_state(algo, args.ckpt)

    loader, eval_n_frames = build_eval_loader(
        cfg, args.horizon, context_length, args.batch_size, args.n_contexts,
    )
    print(f"Eval loader: n_frames={eval_n_frames} "
          f"({'covers horizon' if eval_n_frames >= context_length + args.horizon else 'capped by DMLab 300f source'})")

    all_pred, all_gt = [], []
    for bi, batch in enumerate(loader):
        # batch: (xs, actions, nonterminal) for DMLab — we ignore cond/nonterm.
        xs = batch[0].to(device)  # (B, T_raw, C, H, W) in [0,1]

        # Mirror _preprocess_batch: normalize and pack into frame_stack tokens.
        xs_norm = algo._normalize_x(xs)
        xs_stacked = rearrange(
            xs_norm, "b (t fs) c ... -> t b (fs c) ...",
            fs=algo.frame_stack,
        ).contiguous()
        # Take only the anchor prefix as clean context.
        xs_ctx = xs_stacked[: context_length // algo.frame_stack]

        n_tokens_full = xs_stacked.shape[0]
        conditions = [None] * n_tokens_full  # DMLab has no external cond

        target_tokens = (context_length + args.horizon) // algo.frame_stack
        # Pad conditions if rollout extends past available GT tokens
        if target_tokens > n_tokens_full:
            conditions = conditions + [None] * (target_tokens - n_tokens_full)

        xs_pred_stacked = rollout(
            algo, xs_ctx, conditions, args.horizon // algo.frame_stack, anchor,
        )

        # Unpack frame_stack and unnormalize to [0, 1]
        xs_pred_unpacked = algo._unstack_and_unnormalize(xs_pred_stacked)
        xs_gt_unpacked = algo._unstack_and_unnormalize(xs_stacked)

        all_pred.append(xs_pred_unpacked.cpu())
        all_gt.append(xs_gt_unpacked.cpu())

        if args.save_mp4:
            save_mp4(
                xs_pred_unpacked,
                args.out_dir / f"pred_batch{bi}",
                fps=8,
            )
            save_mp4(
                xs_gt_unpacked,
                args.out_dir / f"gt_batch{bi}",
                fps=8,
            )

    xs_pred = torch.cat(all_pred, dim=1)  # (T, N_total, C, H, W)
    xs_gt = torch.cat(all_gt, dim=1)

    metrics = compute_metrics(
        xs_pred.to(device), xs_gt.to(device),
        context_length, args.horizon,
    )
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Write metrics.json for aggregation across runs/horizons.
    import json
    (args.out_dir / "metrics.json").write_text(json.dumps({
        "ckpt": str(args.ckpt),
        "horizon": args.horizon,
        "n_contexts": xs_pred.shape[1],
        "context_length": context_length,
        "anchor_prefix_size": anchor,
        "noise_level": cfg.algorithm.noise_level,
        **metrics,
    }, indent=2))
    print(f"\nWrote {args.out_dir}/metrics.json")


if __name__ == "__main__":
    main()
