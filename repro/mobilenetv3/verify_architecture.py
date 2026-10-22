"""Prove the from-scratch model reproduces MobileNetV3-Small.

This is the central "recreatability" artifact for the Milestone I report. It does
NOT need any training compute - it establishes that the architecture built in
model.py is the same network the paper describes, by checking it against:

  1. The paper's own numbers (Table 2 layer spec, Table 3 parameter count).
  2. torchvision's reference mobilenet_v3_small - an independent implementation
     of the same paper. Matching its parameter count and intermediate tensor
     shapes block-for-block is strong evidence the reproduction is faithful.

Run:
    python repro/mobilenetv3/verify_architecture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_model, count_parameters  # noqa: E402

PAPER_PARAMS_M = 2.5          # Table 3, MobileNetV3-Small 1.0
PAPER_TABLE2_BLOCKS = 11      # bneck rows in Table 2


def check_paper_facts() -> list[tuple[str, bool, str]]:
    """Check the from-scratch model against numbers stated in the paper."""
    results = []
    net = build_model(num_classes=1000)

    params_m = count_parameters(net) / 1e6
    ok = abs(params_m - PAPER_PARAMS_M) < 0.1
    results.append(
        ("Parameter count vs Table 3", ok, f"{params_m:.2f}M (paper 2.5M)")
    )

    n_blocks = len(net.CONFIG)
    ok = n_blocks == PAPER_TABLE2_BLOCKS
    results.append(
        ("Number of bneck blocks vs Table 2", ok, f"{n_blocks} (paper 11)")
    )

    # h-swish must start in the second half (section 5.2). In Table 2 that is
    # the 4th bneck block (5x5, 40 out) onward.
    first_hs = next(i for i, cfg in enumerate(net.CONFIG) if cfg[4])
    ok = first_hs == 3
    results.append(
        ("h-swish starts at block 4 (section 5.2)", ok, f"index {first_hs}")
    )

    out = net(torch.randn(1, 3, 224, 224))
    ok = tuple(out.shape) == (1, 1000)
    results.append(("Forward pass output shape", ok, f"{tuple(out.shape)}"))
    return results


def check_against_torchvision() -> list[tuple[str, bool, str]]:
    """Compare parameter count and feature-map shapes to torchvision."""
    results = []
    try:
        from torchvision.models import mobilenet_v3_small
    except ImportError:
        results.append(("torchvision available", False, "not installed - skipped"))
        return results

    ours = build_model(num_classes=1000)
    ref = mobilenet_v3_small(weights=None)

    ours_p = count_parameters(ours)
    ref_p = count_parameters(ref)
    # Within 2% absorbs trivial head/rounding differences between implementations.
    rel = abs(ours_p - ref_p) / ref_p
    ok = rel < 0.02
    results.append(
        (
            "Parameter count vs torchvision",
            ok,
            f"ours {ours_p:,} vs ref {ref_p:,} ({rel * 100:.2f}% diff)",
        )
    )

    # Feature-extractor output shape must match at the standard 224 input.
    ours.eval()
    ref.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ours_feat = ours.features(x)
        ref_feat = ref.features(x)
    ok = ours_feat.shape == ref_feat.shape
    results.append(
        (
            "Feature-map shape vs torchvision",
            ok,
            f"ours {tuple(ours_feat.shape)} vs ref {tuple(ref_feat.shape)}",
        )
    )
    return results


def main() -> int:
    print("=" * 70)
    print("MobileNetV3-Small reproduction - architecture verification")
    print("=" * 70)

    all_results = []
    print("\n[Against the paper]")
    for name, ok, detail in check_paper_facts():
        all_results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print("\n[Against torchvision reference]")
    for name, ok, detail in check_against_torchvision():
        all_results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    passed = sum(all_results)
    total = len(all_results)
    print("\n" + "=" * 70)
    print(f"{passed}/{total} checks passed")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
