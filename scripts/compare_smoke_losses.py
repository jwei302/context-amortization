"""Compare training loss across the three smoke runs.

Expected:
 - DF baseline == CA(k=0) *exactly* (same seed, same noise sampling, same loss mask).
 - CA(k=1) diverges (first row of noise levels fixed to stab-1, first frame masked out of loss).
"""
import csv
from pathlib import Path
import numpy as np

RUNS = {
    "df_baseline": "outputs/2026-04-19/15-54-44/loss.csv",
    "ca_k0":       "outputs/2026-04-19/15-56-03/loss.csv",
    "ca_k1":       "outputs/2026-04-19/15-57-24/loss.csv",
}


def load(path: str):
    steps, losses = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return np.array(steps), np.array(losses)


def main():
    s = {name: load(p) for name, p in RUNS.items()}
    for name, (steps, v) in s.items():
        print(f"{name:12s}  n={len(v):3d}  first={v[0]:.4f}  step50={v[50]:.4f}  step150={v[150]:.4f}  last={v[-1]:.4f}")

    df = s["df_baseline"][1]
    ca0 = s["ca_k0"][1]
    ca1 = s["ca_k1"][1]
    n = min(len(df), len(ca0), len(ca1))
    df, ca0, ca1 = df[:n], ca0[:n], ca1[:n]
    print()
    print(f"max|DF - CA(k=0)| = {np.abs(df - ca0).max():.6g}")
    print(f"mean|DF - CA(k=0)| = {np.abs(df - ca0).mean():.6g}")
    print(f"max|DF - CA(k=1)| = {np.abs(df - ca1).max():.6g}")
    print(f"mean|DF - CA(k=1)| = {np.abs(df - ca1).mean():.6g}")

    eq_ok = np.allclose(df, ca0, atol=1e-5)
    diff_ok = np.abs(df - ca1).mean() > 1e-3
    print()
    print(f"Equivalence DF == CA(k=0): {'PASS' if eq_ok else 'FAIL'}")
    print(f"Divergence DF != CA(k=1): {'PASS' if diff_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
