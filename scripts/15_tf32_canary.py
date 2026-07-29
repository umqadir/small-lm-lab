"""Measure whether float32 matmuls are really float32, or silently TF32.

Suppressing bf16 autocast does not by itself produce the registered float32 on
an NVIDIA accelerator. Ampere and later can execute a float32 matmul in TF32: same range,
10 explicit mantissa bits instead of 23, relative error near 1e-3 rather than
1e-7. The tensors still say torch.float32, so nothing in the model or the
optimizer reveals it. The only way to know is to measure against a reference
computed in higher precision.

This is the torch counterpart of the MLX leg's MLX_ENABLE_TF32=0 canary, which
measured 1.145e-06 against a float64 reference on a 1e-5 threshold.

Method: build one matrix pair, multiply in float64 on the CPU for the reference,
multiply the float32 casts of the same matrices on the target device, and compare.
Relative Frobenius error is the headline because a max elementwise ratio is
dominated by cancellation near zero and is not a stable discriminator.

  true float32   ~1e-7   (well under the threshold)
  TF32           ~1e-3   (roughly four orders of magnitude worse)

Exits nonzero if the measured error fails the threshold, so it can gate a run.

Usage:
  python scripts/15_tf32_canary.py --device cuda
  python scripts/15_tf32_canary.py --device cuda --no-enforce   # measure the raw default
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from small_lm_lab.train import disable_tf32

# TF32 lands near 1e-3 and true float32 near 1e-7, so anything in between is
# unambiguous. Same 1e-5 bar the MLX leg used.
THRESHOLD = 1e-5
N = 4096


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--size", type=int, default=N)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument(
        "--no-enforce",
        action="store_true",
        help="measure the framework default instead of applying disable_tf32() first",
    )
    args = p.parse_args()

    report: dict = {"device": args.device, "size": args.size, "threshold": args.threshold}

    if args.no_enforce:
        report["policy"] = "framework default (not enforced)"
    else:
        report["policy"] = disable_tf32()

    # Report the flags as the framework sees them, alongside the measurement.
    observed = {}
    try:
        observed["cuda.matmul.allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception as exc:
        observed["cuda.matmul.allow_tf32"] = f"unavailable: {exc}"
    try:
        observed["cudnn.allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception as exc:
        observed["cudnn.allow_tf32"] = f"unavailable: {exc}"
    try:
        observed["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except Exception as exc:
        observed["float32_matmul_precision"] = f"unavailable: {exc}"
    report["observed_flags"] = observed

    torch.manual_seed(0)
    a64 = torch.randn(args.size, args.size, dtype=torch.float64)
    b64 = torch.randn(args.size, args.size, dtype=torch.float64)
    ref = a64 @ b64  # float64 on CPU: the reference

    dev = torch.device(args.device)
    # Move to host before widening to float64: MPS has no float64, and the widening
    # is exact in either order, so the value is unchanged and the repo's static
    # invariant (widening must follow a host transfer) is honoured.
    got = (a64.float().to(dev) @ b64.float().to(dev)).cpu().double()

    err = got - ref
    rel_fro = (err.norm() / ref.norm()).item()
    # Scale-relative max, guarded so near-zero entries do not dominate.
    denom = ref.abs().max().clamp_min(1e-300)
    rel_max = (err.abs().max() / denom).item()

    report["rel_frobenius_error"] = rel_fro
    report["rel_max_error"] = rel_max
    report["verdict"] = "PASS true-float32" if rel_fro < args.threshold else "FAIL likely-TF32"

    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if rel_fro < args.threshold else 1)


if __name__ == "__main__":
    main()
