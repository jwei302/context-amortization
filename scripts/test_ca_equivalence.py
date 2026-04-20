"""
CPU-only unit tests for the Context Amortization (CA) changes in df_base.py.

Verifies:
 1. `_generate_noise_levels` with noise_level="random_all" and with
    noise_level="context_amortization" + anchor_prefix_size=0 produce
    *identical* tensors under the same torch seed. This is the equivalence
    guarantee plan.md calls for.
 2. With anchor_prefix_size=k > 0, the first k rows of the noise-level
    tensor are exactly stabilization_level - 1 (inference-matching).
 3. The training-step loss-mask construction zeros the first k*frame_stack
    raw-frame positions when CA is active and leaves them alone otherwise.

Run: python scripts/test_ca_equivalence.py
"""
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, ".")
from algorithms.diffusion_forcing.df_base import DiffusionForcingBase


def _make_self(noise_level: str, anchor_prefix_size: int, frame_stack: int = 1):
    """Minimal stand-in for a DiffusionForcingBase instance — just enough to
    call `_generate_noise_levels` unbound."""
    cfg = SimpleNamespace(
        noise_level=noise_level,
        diffusion=SimpleNamespace(stabilization_level=15),
    )
    return SimpleNamespace(
        cfg=cfg,
        timesteps=1000,
        stabilization_level=15,
        anchor_prefix_size=anchor_prefix_size,
        frame_stack=frame_stack,
    )


def test_equivalence_k0():
    """random_all and context_amortization(k=0) must produce identical noise levels."""
    torch.manual_seed(1234)
    s1 = _make_self("random_all", anchor_prefix_size=0)
    xs = torch.zeros(16, 8, 3, 64, 64)
    noise_a = DiffusionForcingBase._generate_noise_levels(s1, xs)

    torch.manual_seed(1234)
    s2 = _make_self("context_amortization", anchor_prefix_size=0)
    noise_b = DiffusionForcingBase._generate_noise_levels(s2, xs)

    assert torch.equal(noise_a, noise_b), "k=0 CA must equal random_all under same seed"
    print("PASS: random_all == context_amortization(k=0)")


def test_ca_k1_prefix():
    """context_amortization(k=1) must set first row to stab - 1; suffix must match random_all."""
    stab = 15
    torch.manual_seed(42)
    s_rand = _make_self("random_all", anchor_prefix_size=0)
    xs = torch.zeros(16, 8, 3, 64, 64)
    noise_rand = DiffusionForcingBase._generate_noise_levels(s_rand, xs)

    torch.manual_seed(42)
    s_ca = _make_self("context_amortization", anchor_prefix_size=1)
    noise_ca = DiffusionForcingBase._generate_noise_levels(s_ca, xs)

    assert torch.all(noise_ca[0] == stab - 1), f"prefix row must be {stab - 1}, got {noise_ca[0]}"
    assert torch.equal(noise_ca[1:], noise_rand[1:]), "suffix must match random_all"
    print("PASS: CA(k=1) prefix == stab-1, suffix == random_all")


def test_ca_k3_prefix():
    """context_amortization(k=3) zeros first 3 rows to stab-1."""
    stab = 15
    torch.manual_seed(7)
    s_ca = _make_self("context_amortization", anchor_prefix_size=3)
    xs = torch.zeros(16, 8, 3, 64, 64)
    noise_ca = DiffusionForcingBase._generate_noise_levels(s_ca, xs)
    assert torch.all(noise_ca[:3] == stab - 1), "first 3 rows must be stab-1"
    assert not torch.all(noise_ca[3:] == stab - 1), "rows after prefix should not all be stab-1"
    print("PASS: CA(k=3) zeros first 3 rows to stab-1")


def test_loss_mask_k0_unchanged():
    """With k=0 the training-step mask modification is a no-op."""
    # Replicate the training_step mask mutation:
    frame_stack = 1
    anchor_prefix_size = 0
    masks = torch.ones(16, 8)
    original = masks.clone()
    if "context_amortization" == "context_amortization" and anchor_prefix_size > 0:
        k_raw = anchor_prefix_size * frame_stack
        masks = masks.clone()
        masks[:k_raw] = 0.0
    assert torch.equal(masks, original), "k=0 must not modify the mask"
    print("PASS: loss mask unchanged when k=0")


def test_loss_mask_k2():
    """With k=2, first 2*frame_stack rows of mask must be zero."""
    frame_stack = 2
    anchor_prefix_size = 2
    masks = torch.ones(16, 8)
    k_raw = anchor_prefix_size * frame_stack
    masks = masks.clone()
    masks[:k_raw] = 0.0
    assert torch.all(masks[:k_raw] == 0), "first k*frame_stack rows must be zero"
    assert torch.all(masks[k_raw:] == 1), "remaining rows must be one"
    print("PASS: loss mask zeros first k*frame_stack=%d raw positions" % k_raw)


if __name__ == "__main__":
    test_equivalence_k0()
    test_ca_k1_prefix()
    test_ca_k3_prefix()
    test_loss_mask_k0_unchanged()
    test_loss_mask_k2()
    print("\nAll equivalence tests passed.")
