"""Framework throughput pilot: PyTorch-MPS vs MLX training steps.

Benchmarks real training throughput (forward, next-token cross-entropy, backward,
AdamW step) on synthetic random token batches at vocab 16384, sequence length
512. Runs every (framework, config, batch_size) cell, catching errors per cell,
and writes a JSON record plus a readable table sorted by tokens per second.

Precision policy:
  torch: try bf16 autocast on MPS, fall back to fp16 autocast, then fp32.
  mlx:   try bfloat16 params and compute, fall back to fp32.

Timing: 10 warmup steps, then 30 timed steady-state steps. If a single step is
slower than about 5 seconds the timed count drops to 8. size120m batch 64 is
skipped for a framework if that framework's size120m batch 32 exceeded about
3 seconds per step, to keep total runtime bounded.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim

from small_lm_lab.config import get_config, ModelConfig
from small_lm_lab import model_torch, model_mlx

# Benchmark grid.
CONFIGS = ["size30m", "size120m"]
BATCH_SIZES = [16, 32, 64]
SEQ_LEN = 512
VOCAB = 16384
LR = 1e-4

WARMUP_STEPS = 10
TIMED_STEPS_FULL = 30
TIMED_STEPS_SLOW = 8
SLOW_STEP_SECONDS = 5.0
SKIP_BATCH64_THRESHOLD = 3.0  # sec/step on size120m bs32 above which bs64 is skipped

# Repeats of the whole grid. The two frameworks are measured back to back inside
# each cell, so a round gives one paired comparison per cell and the median over
# rounds is what gets reported.
ROUNDS = 3

OUT_PATH = Path(__file__).resolve().parents[1] / "analysis" / "pilot" / "throughput.json"

_PROC = psutil.Process()


def rss_gb() -> float:
    return _PROC.memory_info().rss / 1e9


# ----------------------------------------------------------------------------
# torch-mps
# ----------------------------------------------------------------------------

def _torch_pick_precision(cfg: ModelConfig, batch_size: int):
    """Probe autocast dtypes on a real step; return (dtype_str, autocast_dtype)."""
    device = "mps"
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", None)):
        model = opt = tokens = None
        try:
            model = model_torch.TransformerLM(cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=LR)
            tokens = torch.randint(0, VOCAB, (batch_size, SEQ_LEN), device=device)
            _torch_one_step(model, opt, tokens, dtype)
            torch.mps.synchronize()
            return name, dtype
        except Exception:
            continue
        finally:
            model = opt = tokens = None
            gc.collect()
            torch.mps.empty_cache()
    return "fp32", None


def _torch_one_step(model, opt, tokens, autocast_dtype: Optional[torch.dtype]) -> float:
    opt.zero_grad(set_to_none=True)
    if autocast_dtype is not None:
        with torch.autocast(device_type="mps", dtype=autocast_dtype):
            logits = model(tokens)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                tokens[:, 1:].reshape(-1),
            )
    else:
        logits = model(tokens)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            tokens[:, 1:].reshape(-1),
        )
    loss.backward()
    opt.step()
    return float(loss.detach().to("cpu"))


def run_torch_cell(cfg_name: str, batch_size: int) -> dict:
    cfg = get_config(cfg_name)
    device = "mps"
    cell = {
        "framework": "torch-mps",
        "config": cfg_name,
        "batch_size": batch_size,
        "precision": None,
        "tokens_per_sec": None,
        "sec_per_step": None,
        "peak_mem_gb": None,
        "rss_gb": None,
        "loss_start": None,
        "loss_end": None,
        "timed_steps": None,
        "error": None,
    }
    try:
        gc.collect()
        torch.mps.empty_cache()
        prec_name, autocast_dtype = _torch_pick_precision(cfg, batch_size)
        cell["precision"] = prec_name

        model = model_torch.TransformerLM(cfg).to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=LR)

        def fresh_tokens():
            return torch.randint(0, VOCAB, (batch_size, SEQ_LEN), device=device)

        # Warmup.
        loss_start = None
        for i in range(WARMUP_STEPS):
            loss_val = _torch_one_step(model, opt, fresh_tokens(), autocast_dtype)
            if i == 0:
                loss_start = loss_val
        torch.mps.synchronize()

        # Decide timed step count from one measured step.
        t0 = time.perf_counter()
        _torch_one_step(model, opt, fresh_tokens(), autocast_dtype)
        torch.mps.synchronize()
        probe = time.perf_counter() - t0
        timed_steps = TIMED_STEPS_SLOW if probe > SLOW_STEP_SECONDS else TIMED_STEPS_FULL

        # Timed run.
        torch.mps.synchronize()
        t0 = time.perf_counter()
        loss_end = None
        for i in range(timed_steps):
            loss_val = _torch_one_step(model, opt, fresh_tokens(), autocast_dtype)
            if i == timed_steps - 1:
                loss_end = loss_val
        torch.mps.synchronize()
        elapsed = time.perf_counter() - t0

        cell["timed_steps"] = timed_steps
        cell["sec_per_step"] = elapsed / timed_steps
        cell["tokens_per_sec"] = batch_size * SEQ_LEN * timed_steps / elapsed
        cell["peak_mem_gb"] = torch.mps.driver_allocated_memory() / 1e9
        cell["rss_gb"] = rss_gb()
        cell["loss_start"] = loss_start
        cell["loss_end"] = loss_end

        del model, opt
        gc.collect()
        torch.mps.empty_cache()
    except Exception as exc:  # noqa: BLE001 - record and continue
        cell["error"] = f"{type(exc).__name__}: {exc}"
        gc.collect()
        torch.mps.empty_cache()
    return cell


# ----------------------------------------------------------------------------
# mlx
# ----------------------------------------------------------------------------

def _mlx_loss_fn(model, tokens):
    logits = model(tokens)
    v = logits.shape[-1]
    logits = logits[:, :-1, :].reshape(-1, v).astype(mx.float32)
    targets = tokens[:, 1:].reshape(-1)
    return mx.mean(mlx_nn.losses.cross_entropy(logits, targets, reduction="none"))


def run_mlx_cell(cfg_name: str, batch_size: int) -> dict:
    cfg = get_config(cfg_name)
    cell = {
        "framework": "mlx",
        "config": cfg_name,
        "batch_size": batch_size,
        "precision": None,
        "tokens_per_sec": None,
        "sec_per_step": None,
        "peak_mem_gb": None,
        "rss_gb": None,
        "loss_start": None,
        "loss_end": None,
        "timed_steps": None,
        "error": None,
    }

    def build(dtype):
        model = model_mlx.TransformerLM(cfg)
        if dtype is not None:
            model.set_dtype(dtype)
        mx.eval(model.parameters())
        return model

    try:
        mx.clear_cache()
        # Pick precision by probing one full step.
        precision = None
        model = None
        for name, dtype in (("bf16", mx.bfloat16), ("fp32", None)):
            try:
                model = build(dtype)
                opt = mlx_optim.AdamW(learning_rate=LR)
                lg = mlx_nn.value_and_grad(model, _mlx_loss_fn)
                tokens = mx.random.randint(0, VOCAB, (batch_size, SEQ_LEN))
                loss, grads = lg(model, tokens)
                opt.update(model, grads)
                mx.eval(model.parameters(), opt.state, loss)
                precision = name
                break
            except Exception:
                model = None
                mx.clear_cache()
                continue
        if precision is None:
            raise RuntimeError("mlx failed at both bf16 and fp32")
        cell["precision"] = precision

        # Fresh model and optimizer for the actual measurement.
        dtype = mx.bfloat16 if precision == "bf16" else None
        model = build(dtype)
        opt = mlx_optim.AdamW(learning_rate=LR)
        loss_and_grad = mlx_nn.value_and_grad(model, _mlx_loss_fn)

        def one_step():
            tokens = mx.random.randint(0, VOCAB, (batch_size, SEQ_LEN))
            loss, grads = loss_and_grad(model, tokens)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            return float(loss)

        mx.reset_peak_memory()

        # Warmup.
        loss_start = None
        for i in range(WARMUP_STEPS):
            v = one_step()
            if i == 0:
                loss_start = v

        # Probe one step to decide timed count.
        t0 = time.perf_counter()
        one_step()
        probe = time.perf_counter() - t0
        timed_steps = TIMED_STEPS_SLOW if probe > SLOW_STEP_SECONDS else TIMED_STEPS_FULL

        # Timed run.
        t0 = time.perf_counter()
        loss_end = None
        for i in range(timed_steps):
            v = one_step()
            if i == timed_steps - 1:
                loss_end = v
        elapsed = time.perf_counter() - t0

        cell["timed_steps"] = timed_steps
        cell["sec_per_step"] = elapsed / timed_steps
        cell["tokens_per_sec"] = batch_size * SEQ_LEN * timed_steps / elapsed
        cell["peak_mem_gb"] = mx.get_peak_memory() / 1e9
        cell["rss_gb"] = rss_gb()
        cell["loss_start"] = loss_start
        cell["loss_end"] = loss_end

        del model, opt, loss_and_grad
        gc.collect()
        mx.clear_cache()
    except Exception as exc:  # noqa: BLE001 - record and continue
        cell["error"] = f"{type(exc).__name__}: {exc}"
        gc.collect()
        mx.clear_cache()
    return cell


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def format_table(cells: list[dict]) -> str:
    ranked = sorted(
        cells,
        key=lambda c: (c["tokens_per_sec"] is not None, c["tokens_per_sec"] or 0.0),
        reverse=True,
    )
    header = (
        f"{'framework':<10} {'config':<9} {'bs':>3} {'prec':>5} "
        f"{'tok/s':>10} {'s/step':>8} {'peakGB':>7} {'rssGB':>6} "
        f"{'loss0':>7} {'loss1':>7} {'steps':>5}  error"
    )
    lines = [header, "-" * len(header)]
    for c in ranked:
        tps = f"{c['tokens_per_sec']:,.0f}" if c["tokens_per_sec"] is not None else "-"
        sps = f"{c['sec_per_step']:.3f}" if c["sec_per_step"] is not None else "-"
        pk = f"{c['peak_mem_gb']:.2f}" if c["peak_mem_gb"] is not None else "-"
        rss = f"{c['rss_gb']:.2f}" if c["rss_gb"] is not None else "-"
        l0 = f"{c['loss_start']:.3f}" if c["loss_start"] is not None else "-"
        l1 = f"{c['loss_end']:.3f}" if c["loss_end"] is not None else "-"
        steps = c["timed_steps"] if c["timed_steps"] is not None else "-"
        err = c["error"] or ""
        lines.append(
            f"{c['framework']:<10} {c['config']:<9} {c['batch_size']:>3} "
            f"{str(c['precision']):>5} {tps:>10} {sps:>8} {pk:>7} {rss:>6} "
            f"{l0:>7} {l1:>7} {str(steps):>5}  {err}"
        )
    return "\n".join(lines)


def summarize(cells: list[dict]) -> tuple[str, dict]:
    """Median throughput per cell and the paired framework ratio.

    The median over rounds is reported rather than the mean because a single
    round that collided with a burst of background load would drag a mean down.
    The ratio is computed within a round, where both frameworks saw nearly the
    same load, and then the median is taken over rounds. That is the number the
    framework decision rests on.
    """
    by_key: dict[tuple[str, int, str], list[float]] = {}
    for c in cells:
        if c["tokens_per_sec"] is None:
            continue
        by_key.setdefault((c["config"], c["batch_size"], c["framework"]), []).append(
            c["tokens_per_sec"]
        )

    lines = [
        "",
        "Median tokens per second over rounds, and the within-round MLX to torch ratio.",
        "A ratio above 1 means MLX was faster on the same machine at the same moment.",
        "",
        f"{'config':<9} {'bs':>3} {'torch tok/s':>12} {'mlx tok/s':>12} {'mlx/torch':>10}",
        "-" * 52,
    ]
    summary: dict = {"per_cell": {}, "paired_ratio_median": None}
    ratios: list[float] = []
    for cfg_name in CONFIGS:
        for bs in BATCH_SIZES:
            t = by_key.get((cfg_name, bs, "torch-mps"), [])
            m = by_key.get((cfg_name, bs, "mlx"), [])
            if not t and not m:
                continue
            t_med = statistics.median(t) if t else None
            m_med = statistics.median(m) if m else None
            # Pair within a round, then take the median of the ratios.
            cell_ratios = [
                mm / tt
                for rr in range(ROUNDS)
                for tt in [
                    c["tokens_per_sec"]
                    for c in cells
                    if c.get("round") == rr
                    and c["config"] == cfg_name
                    and c["batch_size"] == bs
                    and c["framework"] == "torch-mps"
                    and c["tokens_per_sec"]
                ]
                for mm in [
                    c["tokens_per_sec"]
                    for c in cells
                    if c.get("round") == rr
                    and c["config"] == cfg_name
                    and c["batch_size"] == bs
                    and c["framework"] == "mlx"
                    and c["tokens_per_sec"]
                ]
            ]
            r_med = statistics.median(cell_ratios) if cell_ratios else None
            ratios.extend(cell_ratios)
            summary["per_cell"][f"{cfg_name}_bs{bs}"] = {
                "torch_tokens_per_sec_median": t_med,
                "mlx_tokens_per_sec_median": m_med,
                "mlx_over_torch_ratio_median": r_med,
                "n_rounds_torch": len(t),
                "n_rounds_mlx": len(m),
            }
            lines.append(
                f"{cfg_name:<9} {bs:>3} "
                f"{(f'{t_med:,.0f}' if t_med else '-'):>12} "
                f"{(f'{m_med:,.0f}' if m_med else '-'):>12} "
                f"{(f'{r_med:.2f}' if r_med else '-'):>10}"
            )
    if ratios:
        summary["paired_ratio_median"] = statistics.median(ratios)
        lines.append("")
        lines.append(
            f"Median MLX to torch ratio across all cells: {statistics.median(ratios):.3f}"
        )
    loads = [c["load_avg_1min"] for c in cells if c.get("load_avg_1min") is not None]
    if loads:
        summary["load_avg_1min_median"] = statistics.median(loads)
        lines.append(
            f"Median 1 minute load average during the pilot: {statistics.median(loads):.2f} "
            f"(this machine is shared, so absolute throughput is a lower bound)"
        )
    return "\n".join(lines), summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Framework throughput pilot.")
    p.add_argument(
        "--frameworks",
        default="torch-mps,mlx",
        help="comma separated subset of torch-mps,mlx",
    )
    p.add_argument("--configs", default=",".join(CONFIGS), help="comma separated config names")
    p.add_argument(
        "--batch-sizes",
        default=",".join(str(b) for b in BATCH_SIZES),
        help="comma separated batch sizes",
    )
    p.add_argument("--rounds", type=int, default=ROUNDS, help="repeats of the whole grid")
    p.add_argument("--out", default=str(OUT_PATH), help="where to write the results JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frameworks = [f.strip() for f in args.frameworks.split(",") if f.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    rounds = args.rounds
    out_path = Path(args.out)

    print(f"Platform: {platform.platform()}  machine={platform.machine()}")
    print(f"torch {torch.__version__}  mlx {version('mlx')}  mps={torch.backends.mps.is_available()}")
    print(f"grid: frameworks={frameworks} configs={configs} batch_sizes={batch_sizes}")
    print(f"rounds: {rounds}, frameworks paired within a cell\n")

    cells: list[dict] = []
    # Track per-framework size120m bs32 sec/step to gate bs64.
    slow_120m_bs32: dict[str, Optional[float]] = {"torch-mps": None, "mlx": None}

    all_runners = {"torch-mps": run_torch_cell, "mlx": run_mlx_cell}
    runners = {k: v for k, v in all_runners.items() if k in frameworks}

    # This machine is shared with other projects, so background load varies over
    # time and cannot be assumed away. Running every torch cell and then every
    # MLX cell would confound the framework comparison with whatever else was
    # running at the time, which is the one thing this pilot must not do, since
    # the framework choice is the only thing it decides.
    #
    # So the two frameworks are measured back to back inside a cell, and the
    # whole grid repeats for several rounds. Adjacent measurements see nearly the
    # same background load, which makes the within-round ratio a fair comparison
    # even when the absolute numbers are depressed by contention. Absolute
    # tokens per second is then a lower bound on what a quiet machine would give,
    # and the load average recorded with each cell says how contended it was.
    def write_results() -> None:
        """Write the results file from whatever has been measured so far.

        Called after every cell rather than once at the end. A slow cell on a
        contended machine can take half an hour, so a run that is interrupted
        part way through has still bought real measurements, and they should not
        die with the process.
        """
        metadata = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_version": torch.__version__,
            "mlx_version": version("mlx"),
            "mps_available": bool(torch.backends.mps.is_available()),
            "seq_len": SEQ_LEN,
            "vocab": VOCAB,
            "lr": LR,
            "warmup_steps": WARMUP_STEPS,
            "timed_steps_full": TIMED_STEPS_FULL,
            "timed_steps_slow": TIMED_STEPS_SLOW,
            "rounds_requested": rounds,
            "grid": {
                "frameworks": frameworks,
                "configs": configs,
                "batch_sizes": batch_sizes,
            },
            "paired_design": (
                "The two frameworks are measured back to back within each cell and "
                "the grid repeats for several rounds, so the framework comparison is "
                "not confounded by background load on this shared machine."
            ),
            "complete": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _, summary = summarize(cells)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(
                {"metadata": metadata, "cells": cells, "summary": summary}, f, indent=2
            )

    for round_idx in range(rounds):
        print(f"--- round {round_idx + 1} of {rounds} ---", flush=True)
        for cfg_name in configs:
            for bs in batch_sizes:
                for framework, runner in runners.items():
                    if (
                        cfg_name == "size120m"
                        and bs == 64
                        and slow_120m_bs32[framework] is not None
                        and slow_120m_bs32[framework] > SKIP_BATCH64_THRESHOLD
                    ):
                        cell = {
                            "framework": framework,
                            "config": cfg_name,
                            "batch_size": bs,
                            "round": round_idx,
                            "precision": None,
                            "tokens_per_sec": None,
                            "sec_per_step": None,
                            "peak_mem_gb": None,
                            "rss_gb": None,
                            "loss_start": None,
                            "loss_end": None,
                            "timed_steps": None,
                            "load_avg_1min": os.getloadavg()[0],
                            "error": (
                                f"skipped: size120m bs32 was "
                                f"{slow_120m_bs32[framework]:.2f}s/step > {SKIP_BATCH64_THRESHOLD}s"
                            ),
                        }
                        cells.append(cell)
                        write_results()
                        print(f"[skip] {framework} {cfg_name} bs{bs}: {cell['error']}")
                        continue

                    print(f"[run ] r{round_idx} {framework} {cfg_name} bs{bs} ...", flush=True)
                    load_before = os.getloadavg()[0]
                    cell = runner(cfg_name, bs)
                    cell["round"] = round_idx
                    cell["load_avg_1min"] = (load_before + os.getloadavg()[0]) / 2.0
                    cells.append(cell)
                    write_results()
                    if cfg_name == "size120m" and bs == 32 and cell["sec_per_step"] is not None:
                        slow_120m_bs32[framework] = cell["sec_per_step"]
                    status = cell["error"] or (
                        f"{cell['tokens_per_sec']:,.0f} tok/s, "
                        f"{cell['sec_per_step']:.3f} s/step, {cell['precision']}, "
                        f"load {cell['load_avg_1min']:.1f}"
                    )
                    print(f"       -> {status}", flush=True)

    metadata = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "mlx_version": version("mlx"),
        "mps_available": bool(torch.backends.mps.is_available()),
        "seq_len": SEQ_LEN,
        "vocab": VOCAB,
        "lr": LR,
        "warmup_steps": WARMUP_STEPS,
        "timed_steps_full": TIMED_STEPS_FULL,
        "timed_steps_slow": TIMED_STEPS_SLOW,
        "rounds": ROUNDS,
        "paired_design": (
            "The two frameworks are measured back to back within each cell and the "
            "grid repeats for several rounds, so the framework comparison is not "
            "confounded by background load on this shared machine."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    table, summary = summarize(cells)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(
            {"metadata": metadata, "cells": cells, "summary": summary}, f, indent=2
        )

    print("\n" + format_table(cells))
    print(table)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
