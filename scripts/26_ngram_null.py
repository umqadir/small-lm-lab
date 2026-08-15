"""Score the registered interpolated 5-gram Kneser-Ney ICL null.

One KenLM binary is supplied per domain, trained on that domain's complete
training split.  The held-out windows and late-minus-early positions are
identical to the transformer ICL evaluation.  The model state resets after the
tokenizer's end-of-text token, matching the document lines used by lmplz.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from small_lm_lab.bootstrap import DEFAULT_ALPHA, DEFAULT_N_RESAMPLES, bootstrap_ci
from small_lm_lab.data import DEFAULT_TOKENIZED_ROOT, iter_eval_split
from small_lm_lab.evaluate import DOMAINS, EOT_ID, ICL_EARLY, ICL_LATE, _icl_statistic
from small_lm_lab.paths import portable_path

ORDER = 5
CONTEXT_LEN = 512
PRUNING = [0, 0, 1, 1, 1]


def parse_model(value: str) -> tuple[str, Path]:
    domain, separator, path = value.partition("=")
    if not separator or domain not in DOMAINS or not path:
        raise argparse.ArgumentTypeError(
            f"expected DOMAIN=PATH with DOMAIN in {DOMAINS}, got {value!r}"
        )
    return domain, Path(path)


def score_window(model, tokens: np.ndarray, state_factory=None) -> np.ndarray:
    """Natural-log NLL for targets tokens[1:] under a KenLM model."""
    if state_factory is None:
        import kenlm

        state_factory = kenlm.State
    state = state_factory()
    model.BeginSentenceWrite(state)
    first = int(tokens[0])
    if first != EOT_ID:
        next_state = state_factory()
        model.BaseScore(state, str(first), next_state)
        state = next_state
    nll = np.empty(len(tokens) - 1, dtype=np.float64)
    for index, target in enumerate(tokens[1:]):
        next_state = state_factory()
        log10_probability = model.BaseScore(state, str(int(target)), next_state)
        nll[index] = -float(log10_probability) * math.log(10.0)
        if int(target) == EOT_ID:
            model.BeginSentenceWrite(state)
        else:
            state = next_state
    return nll


def evaluate_domain(
    model,
    domain: str,
    split: str,
    root: Path,
    max_windows: int | None,
    n_resamples: int,
    seed: int,
) -> dict:
    rows = []
    n_seen = 0
    for x_batch, y_batch in iter_eval_split(
        domain, split, CONTEXT_LEN, batch_size=32, root=root
    ):
        for x, y in zip(x_batch, y_batch):
            sequence = np.concatenate([x[:1], y])
            nll = score_window(model, sequence)
            rows.append(
                [
                    float(nll[ICL_LATE[0] : ICL_LATE[1]].mean()),
                    float(nll[ICL_EARLY[0] : ICL_EARLY[1]].mean()),
                ]
            )
            n_seen += 1
            if max_windows is not None and n_seen >= max_windows:
                break
        if max_windows is not None and n_seen >= max_windows:
            break
    values = np.asarray(rows, dtype=np.float64)
    point, low, high = bootstrap_ci(
        values,
        _icl_statistic,
        n_resamples=n_resamples,
        seed=seed,
        alpha=DEFAULT_ALPHA,
    )
    return {
        "domain": domain,
        "split": split,
        "icl_score": point,
        "ci_low": low,
        "ci_high": high,
        "n_windows": int(len(values)),
        "mean_early_loss": float(values[:, 1].mean()),
        "mean_late_loss": float(values[:, 0].mean()),
        "max_windows": max_windows,
        "windows_capped": max_windows is not None and len(values) >= max_windows,
    }


def main() -> None:
    import kenlm

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--tokenized-root", type=Path, default=DEFAULT_TOKENIZED_ROOT)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model_paths = dict(args.model)
    missing = sorted(set(DOMAINS) - set(model_paths))
    if missing:
        raise SystemExit(f"missing --model entries for {missing}")
    result = {
        "metadata": {
            "order": ORDER,
            "smoothing": "interpolated modified Kneser-Ney (KenLM)",
            "pruning": PRUNING,
            "training": "one model per domain, complete train split",
            "model_paths": {
                key: portable_path(value) for key, value in model_paths.items()
            },
            "split": args.split,
            "context_len": CONTEXT_LEN,
            "icl_windows": {"early": list(ICL_EARLY), "late": list(ICL_LATE)},
            "eot_state_reset": True,
            "n_resamples": args.n_resamples,
            "seed": args.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "domains": {},
    }
    for domain in DOMAINS:
        model = kenlm.Model(str(model_paths[domain]))
        if model.order != ORDER:
            raise SystemExit(
                f"{model_paths[domain]} has order {model.order}, expected {ORDER}"
            )
        result["domains"][domain] = evaluate_domain(
            model,
            domain,
            args.split,
            args.tokenized_root,
            args.max_windows,
            args.n_resamples,
            args.seed,
        )
        print(
            f"{domain}: ICL {result['domains'][domain]['icl_score']:.6f}",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
