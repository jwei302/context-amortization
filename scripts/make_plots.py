"""Generate slideshow figures from eval_out/ + training loss CSVs."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio.v3 as iio

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval_out"
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

CONFIGS = [
    ("df",    "DF (k=1)",   "tab:red"),
    ("ca_k4", "CA (k=4)",   "tab:orange"),
    ("ca_k8", "CA (k=8)",   "tab:blue"),
]
HORIZONS = [32, 64, 128]

LOSS_CSVS = {
    "DF":      ROOT / "artifacts/df/loss.csv",
    "CA k=8":  ROOT / "artifacts/ca_k8/loss.csv",
    "CA k=4":  ROOT / "artifacts/ca_k4/loss.csv",
}
LOSS_COLORS = {"DF": "tab:red", "CA k=8": "tab:blue", "CA k=4": "tab:orange"}


def load_metric(name: str, h: int, key: str) -> float | None:
    f = EVAL / name / f"h{h}" / "metrics.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get(key)


def plot_metric_vs_horizon(metric: str, ylabel: str, out: Path, log_y: bool):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, label, color in CONFIGS:
        ys = [load_metric(name, h, metric) for h in HORIZONS]
        xs = [h for h, y in zip(HORIZONS, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        if not ys:
            continue
        ax.plot(xs, ys, "o-", label=label, color=color, lw=2.2, ms=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(HORIZONS); ax.set_xticklabels([str(h) for h in HORIZONS])
    if log_y: ax.set_yscale("log")
    ax.set_xlabel("rollout horizon (frames)")
    ax.set_ylabel(ylabel)
    ax.set_title("FVD vs Rollout Horizon")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_loss_curves(out: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, csv in LOSS_CSVS.items():
        if not csv.exists(): continue
        df = pd.read_csv(csv)
        # smooth with rolling mean
        smooth = df["loss"].rolling(window=500, min_periods=1).mean()
        ax.plot(df["step"], smooth, label=label, color=LOSS_COLORS[label], lw=1.6)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def extract_frames(mp4: Path, idxs: list[int]) -> list[np.ndarray]:
    vid = iio.imread(mp4)  # (T, H, W, C)
    return [vid[min(i, len(vid)-1)] for i in idxs]


def plot_frame_strip(out: Path, h: int, sample: int = 0, batch: int = 0,
                     positions: list[int] | None = None):
    """Side-by-side strip: rows = configs, cols = sampled positions."""
    if positions is None:
        positions = sorted({0, 8, 16, h // 4, h // 2, h - 1})
    rows = []
    for name, label, _ in CONFIGS:
        mp4 = EVAL / name / f"h{h}" / f"pred_batch{batch}_sample{sample}.mp4"
        if not mp4.exists():
            print(f"  skip {label}: {mp4} missing")
            continue
        rows.append((label, extract_frames(mp4, positions)))
    # GT row
    gt_mp4 = EVAL / "df" / f"h{h}" / f"gt_batch{batch}_sample{sample}.mp4"
    if gt_mp4.exists():
        rows.append(("GT", extract_frames(gt_mp4, positions)))

    if not rows:
        print(f"no rows for h={h}")
        return
    n_cols = len(positions)
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows))
    if n_rows == 1: axes = axes[None, :]
    for r, (label, frames) in enumerate(rows):
        for c, frame in enumerate(frames):
            axes[r, c].imshow(frame)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if c == 0:
                axes[r, c].set_ylabel(label, fontsize=11)
            if r == 0:
                axes[r, c].set_title(f"t={positions[c]}")
    fig.suptitle(f"Rollout frames at horizon={h}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_fvd_vs_step(out: Path, h: int = 32):
    """Eval-FVD-vs-training-step curve, one line per config. Reads
    eval_out/{name}_steps/step{N}_h{h}/metrics.json files."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, label, color in CONFIGS:
        run_dir = EVAL / f"{name}_steps"
        if not run_dir.exists():
            continue
        pts = []
        for step_dir in sorted(run_dir.glob(f"step*_h{h}")):
            mj = step_dir / "metrics.json"
            if not mj.exists(): continue
            d = json.loads(mj.read_text())
            step = int(step_dir.name.split("_")[0].removeprefix("step"))
            pts.append((step, d.get("fvd")))
        pts = [(s, v) for s, v in pts if v is not None]
        if not pts: continue
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", label=label, color=color, lw=2.0, ms=6)
    ax.set_xlabel("training step")
    ax.set_ylabel(f"FVD (h={h}, lower better)")
    ax.set_title(f"Training efficiency: FVD@h={h} vs. step")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_metric_vs_horizon("fvd", "FVD (lower better)", FIGS / "fvd_vs_horizon.png", log_y=False)
    plot_metric_vs_horizon("lpips", "LPIPS (lower better)", FIGS / "lpips_vs_horizon.png", log_y=False)
    plot_metric_vs_horizon("psnr", "PSNR (higher better)", FIGS / "psnr_vs_horizon.png", log_y=False)
    plot_loss_curves(FIGS / "loss_curves.png")
    plot_fvd_vs_step(FIGS / "fvd_vs_step.png", h=32)
    for h in HORIZONS:
        plot_frame_strip(FIGS / f"frame_strip_h{h}.png", h=h)
