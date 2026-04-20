"""Causal-mask sanity check for the TemporalAttentionBlock used in the UNet3D.

The full UNet3D is NOT strictly causal per-frame: nn.GroupNorm in ResnetBlock
normalizes across the time dimension, so a perturbation at frame t leaks into
frames < t through the normalization statistics. That leak is architectural
and present in the unmodified upstream model — it's not a bug in our CA wiring.

The CA wiring depends specifically on the ATTENTION being causal (so a clean
anchor at position < k can be attended to by a noised frame at position >= k
without the anchor seeing the noised frame). We test that in isolation.

Perturb input at position t; verify:
  - outputs of TemporalAttentionBlock at positions <  t are bitwise-identical
  - outputs at positions >= t differ
"""
import torch

from algorithms.diffusion_forcing.models.attention import TemporalAttentionBlock


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dim = 48
    attn = TemporalAttentionBlock(
        dim=dim,
        heads=4,
        dim_head=32,
        is_causal=True,
        rotary_emb=None,
    ).to(device).eval()

    B, C, F, H, W = 2, dim, 8, 16, 16
    x = torch.randn(B, C, F, H, W, device=device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        y_a = attn(x)
        y_a2 = attn(x)  # noise-floor reference

        perturb_idx = 4
        x_b = x.clone()
        x_b[:, :, perturb_idx] = torch.randn_like(x_b[:, :, perturb_idx])

        y_b = attn(x_b)

    noise_floor = (y_a - y_a2).abs().max().item()
    before = (y_a[:, :, :perturb_idx] - y_b[:, :, :perturb_idx]).abs().max().item()
    after = (y_a[:, :, perturb_idx:] - y_b[:, :, perturb_idx:]).abs().max().item()

    print(f"output shape: {tuple(y_a.shape)}")
    print(f"perturb at temporal index {perturb_idx}")
    print(f"noise floor (same input, two passes):        {noise_floor:.3e}")
    print(f"max|diff| positions <  {perturb_idx} (should ~= noise floor): {before:.3e}")
    print(f"max|diff| positions >= {perturb_idx} (should be >> noise floor): {after:.3e}")

    pass_before = before <= max(3 * noise_floor, 1e-5)
    pass_after = after > 10 * max(noise_floor, 1e-4)
    print()
    print(f"Causal attention: before-perturb unchanged: {'PASS' if pass_before else 'FAIL'}")
    print(f"Causal attention: at/after-perturb changed:  {'PASS' if pass_after else 'FAIL'}")

    if not (pass_before and pass_after):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
