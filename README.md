# small-lm-lab

Locating induction-head emergence in from-scratch language model pretraining, on a pre-registered checkpoint grid.

## Model

| | |
|---|---|
| Architecture | decoder-only transformer, pre-norm RMSNorm, RoPE, SwiGLU, no biases, tied embeddings |
| Context | 512 tokens |
| Headline size | 34,087,424 parameters (`d_model` 512, 8 layers, 8 heads) |
| Optional sizes | 60M, 120M |
| Vocabulary | byte-level BPE, 16,384 |
| Optimizer batch | 16,384 tokens/step, held exact under gradient accumulation |

## Data

| Corpus | License | Train tokens |
|---|---|---|
| TinyStories V2 (GPT-4) | CDLA-Sharing-1.0 | 531,101,051 |
| FineWeb-Edu (sample-10BT) | ODC-By 1.0 | 167,948,480 |
| Web extension | ODC-By 1.0 | 2,610,318,560 |

Mix 70/30 stories to web. Web train stream with extension: 2,778,267,040 tokens. Document-level splits, assigned before tokenization.

200-gram contamination check, with positive control. Pre-decontamination: 9/128 validation, 7/128 test. Post: 0/128, 0/128. See `docs/DATA.md`, `docs/DECONTAMINATION.md`.

## Checkpoint grid

34 checkpoints, 1,000,000 to 2,000,000,000 tokens, front-loaded. Fixed in `configs/stage_a_checkpoints.json`.

## Pre-registration

- Registered decisions, criteria, thresholds, and emergence-location predictions, condensed: `docs/PROTOCOL.md`.
- Registration written before training and before the measuring harness existed, amended only by dated entries. Full registered text retained offline.
- Every number traces to a pipeline output.
- Abandonment criteria registered alongside success criteria.
- Amendment history: `docs/METHODOLOGY.md`.

## Results

Peak learning rate, selected by held-out TinyStories perplexity at 40,009,728 tokens, 1000-resample bootstrap:

| Peak LR | TinyStories val perplexity | 95% interval |
|---|---|---|
| 6e-4 | 5.2189 | 5.1917 to 5.2448 |
| 1.2e-3 | 5.0160 | 4.9908 to 5.0400 |
| 2.4e-3 | 6.1681 | 6.1341 to 6.1996 |

Record: `analysis/ablations/lr_winner.json`. Per-run output: `analysis/ablations/valppl_abl_lr*.json`.

### Cross-backend validation gate

Registered tolerance failed: end difference 0.061059 against 0.02 bound, mean difference steps 400-500 of 0.043621 against 0.015. Five-seed control: all ten same-configuration pairs also fail, smallest end difference 0.095183 (4.76x bound). TF32 explanation refuted at 5.75e-07 relative Frobenius error against float64, threshold 1e-05. Derived tolerance 1.246862 exceeds registered abandonment bound 0.25, so no replacement gate adopted. Backend fidelity established instead by identical-weight comparison: 3.5e-06 maximum absolute logit difference. `docs/GATE_FINDING.md`.

### Positive controls

- Contamination check paired with planted-contamination control.
- Induction-presence statistical test: on random-init models, gap-to-standard-error rose 0.61 (8 sequences) to 5.31 (1024), gate opened on 47.5% of 40 random-init models at the registered 256. Demoted to necessary-not-sufficient. `docs/PROTOCOL.md`, amendment 2026-07-17.

## Throughput and cost

Stage A: 2e9 tokens, 122,071 steps. RTX 3070, float32, from `analysis/desktop_bench/train_log_mb{2,4,8}.jsonl`:

| Micro-batch | Accum | Tok/s | Train peak VRAM | Val peak VRAM |
|---|---|---|---|---|
| 2 | 16 | 28,165 | 1.14 GB | 2.72 GB |
| 4 | 8 | 30,506 | 1.72 GB | 2.73 GB |
| 8 | 4 | 31,094 | 2.81 GB | 2.81 GB |

Effective 29,840 tok/s at micro-batch 4: Stage A about 18.6 hours, full 5e9-token budget about 46.5 hours. Apple M4 via MLX: 3,785 tok/s, Stage A about six days. Micro-batch 2/4/8 agree on loss to about 1e-7.

Disk, under `SMALL_LM_LAB_BULK_ROOT`:

| Artifact | Size |
|---|---|
| Tokenized corpus, base splits | 1.44 GB |
| Tokenized corpus, web extension | 5.22 GB |
| Stage A checkpoints (34 bf16) | 2.3 GB |
| Resume state | 442 MB |

Web extension stages ~11.2 GB raw before tokenization, deleted after. Peak requirement ~20 GB.

## Layout

```
src/small_lm_lab/    model (torch and mlx), training, evaluation, interpretability
scripts/             numbered entry points, run in order
configs/             checkpoint grid
docs/                pre-registration, data provenance, findings
tests/               correctness and regression suite
analysis/            measured outputs
```

Order: throughput pilot (01), download and tokenize (02-05), evaluate (06), train (07), causal battery (08), schedule runner (09), copying probe (10), curve comparison (11), contamination (12), decontamination (13), backend diagnostics (15-17), emergence sweep (19), vocabulary coverage (20). `scripts/smoke_gate.py` runs downstream stages against a tiny model first.

## Running

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

```bash
uv run python scripts/07_train.py \
  --size size30m --framework torch --device cuda --torch-precision fp32 \
  --lr 1.2e-3 --lr-schedule constant --warmup-tokens 4000000 \
  --tokens 2e9 --seed 1 --micro-batch 4 \
  --checkpoint-schedule configs/stage_a_checkpoints.json \
  --run-name size30m_staged_seed1
```

## Status

Stage A in progress. Outstanding, content fixed by `docs/PROTOCOL.md`:

- Loss against tokens, per size and per checkpoint.
- Test perplexity per domain with bootstrap intervals; ten-paradigm BLiMP subset against 0.50 chance.
- Per-checkpoint prefix-matching, synthetic copying, and in-context learning scores; crossing interval and width against the registered abruptness criterion.
- Activation patching and mean-ablation at registered anchor checkpoints, with effect sizes and matched control heads.

## License

MIT (`LICENSE`). Corpora are not redistributed. `data/tokenizer/tokenizer.json` carries the source datasets' terms in addition to MIT; see `NOTICE`.
