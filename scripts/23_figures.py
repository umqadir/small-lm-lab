"""The two curves of the Stage A result, drawn from the measured artifacts.

Reads the trajectory JSON from scripts/19_emergence.py and the validation-loss
curve from scripts/21_val_curve.py and writes two PNGs. It computes nothing: the
registered bounds, the chance level and the crossing interval are read from the
files, so a figure cannot disagree with the numbers it illustrates.

  emergence.png   prefix-matching score against tokens. All 64 heads as thin
                  lines, the maximum over heads in bold, the registered 0.1,
                  0.2 and 0.3 levels, the analytic chance level, and the
                  crossing interval shaded.
  loss.png        validation loss per domain against tokens, on the same token
                  axis and with the same interval shaded, so the induction
                  transition can be located against ordinary learning by eye.
                  That comparison is the comparator F of the abruptness
                  criterion, and the figure is where a reader sees what the
                  number is a summary of.

Usage:
  python scripts/23_figures.py --trajectory TRAJ.json --curve CURVE.json \
      --out-dir analysis/emergence/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIGSIZE = (7.0, 4.2)
DPI = 160


def load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def placed(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Points carrying a token count, oldest first. A random-init control sits
    on no point of the token axis and is not the first point of a curve."""
    return sorted(
        (p for p in points if p.get("tokens") is not None), key=lambda p: p["tokens"]
    )


def shade_interval(ax, interval: dict[str, Any] | None) -> None:
    if not interval or not interval.get("crossed"):
        return
    ax.axvspan(
        interval["start_tokens"],
        interval["end_tokens"],
        color="0.85",
        zorder=0,
        label="crossing interval",
    )


def emergence_figure(trajectory: dict[str, Any], out: Path) -> None:
    points = placed(trajectory["points"])
    tokens = np.array([p["tokens"] for p in points], dtype=np.float64)
    per_head = np.array(
        [np.asarray(p["heads"]["prefix_matching"], dtype=np.float64) for p in points]
    )
    n_points, n_layers, n_heads = per_head.shape
    maxima = np.array([p["max_prefix_matching"] for p in points])
    meta = trajectory["metadata"]
    low, high = meta["phase_bounds"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    shade_interval(ax, trajectory["trajectory_summary"]["phase_change"]["interval"])
    flat = per_head.reshape(n_points, n_layers * n_heads)
    for column in range(flat.shape[1]):
        ax.plot(tokens, flat[:, column], color="0.72", linewidth=0.6, zorder=1)
    ax.plot(
        tokens,
        maxima,
        color="black",
        linewidth=1.8,
        marker="o",
        markersize=3.0,
        zorder=3,
        label=f"max over {n_layers * n_heads} heads",
    )
    for level, style in ((low, ":"), (meta["induction_threshold"], "-."), (high, "--")):
        ax.axhline(level, color="0.35", linestyle=style, linewidth=0.9, zorder=2)
        ax.annotate(
            f"{level:g}",
            xy=(tokens[0], level),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=7,
            color="0.35",
        )
    chance = meta["prefix_matching_chance"]
    ax.axhline(chance, color="0.55", linestyle="-", linewidth=0.7, zorder=2)
    ax.annotate(
        f"chance {chance:.4f}",
        xy=(tokens[0], chance),
        xytext=(2, 3),
        textcoords="offset points",
        fontsize=7,
        color="0.55",
    )

    ax.set_xscale("log")
    ax.set_xlabel("training tokens")
    ax.set_ylabel("prefix-matching score")
    ax.set_ylim(0.0, max(1.0, float(maxima.max()) * 1.05))
    ax.set_title(
        f"Induction-head emergence, {meta['config']}, "
        f"{meta['n_sequences_used']} sequences"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=1.0)
    ax.grid(True, which="major", axis="both", color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def loss_figure(
    curve: dict[str, Any], trajectory: dict[str, Any], out: Path
) -> None:
    points = placed(curve["points"])
    tokens = np.array([p["tokens"] for p in points], dtype=np.float64)
    domains = curve["metadata"]["domains"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    shade_interval(ax, trajectory["trajectory_summary"]["phase_change"]["interval"])
    styles = {"tinystories": ("black", "-"), "fineweb_edu": ("0.4", "--")}
    for domain in domains:
        losses = np.array([p["domains"][domain]["mean_nll"] for p in points])
        color, style = styles.get(domain, ("0.2", "-"))
        ax.plot(
            tokens,
            losses,
            color=color,
            linestyle=style,
            linewidth=1.6,
            marker="o",
            markersize=3.0,
            zorder=3,
            label=domain,
        )

    ax.set_xscale("log")
    ax.set_xlabel("training tokens")
    ax.set_ylabel("validation loss, nats per token")
    n_windows = points[0]["domains"][domains[0]]["n_windows"]
    ax.set_title(
        f"Validation loss, {curve['metadata']['config']}, "
        f"{n_windows} windows per domain"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=1.0)
    ax.grid(True, which="major", axis="both", color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--curve", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    trajectory = load(args.trajectory)
    curve = load(args.curve)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    emergence_figure(trajectory, args.out_dir / "emergence.png")
    loss_figure(curve, trajectory, args.out_dir / "loss.png")


if __name__ == "__main__":
    main()
