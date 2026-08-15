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

### Stage A headline

Stage A finished at 2,000,011,264 tokens with all 34 registered checkpoints.
At the registered 256-sequence reading, maximum prefix matching moves from
0.0100 at 24,002,560 tokens to 0.4918 at 48,005,120. The 1,000-resample crossing
time is 48,005,120 tokens with both percentile bounds at the same grid point.
Four heads remain above 0.2 at the final checkpoint, in layers 4, 5 and 6.

The registered from-scratch second seed crosses at 64,012,288 tokens, with its
1,000-resample interval at that same grid point and two layer-5 induction heads.
The crossing-location ratio between seeds is 1.33, inside the registered 2x
replication bound.

Neither registered depth control crosses by 64,012,288 tokens. The one-layer
model peaks at 0.0064 and the two-layer attention-only model at 0.0132; every
one of their 1,000 bootstrap trajectories remains uncrossed. This rules out a
generic training artifact at the measured horizon, but does not separate an
MLP requirement from a later transition in the minimal attention-only system.

The full 2,048-window causal battery supports the induction interpretation. At
48,005,120 tokens, joint per-position mean ablation of the five heads above 0.2
removes 0.00684 copying exact-match accuracy from a 0.00701 baseline and erases
0.0914 nats of FineWeb-Edu ICL, versus +0.00011 and 0.0030 for matched controls.
At the final checkpoint, the corresponding copying drops are 0.11665 versus
0.01779 and the ICL damage is 0.0853 versus 0.0079. Global-mean ablation gives
the same reading. Single-head activation patches recover 5.7 to 35.3 percent of
the clean-corrupted gap at 48M and 10.6 to 22.6 percent at 2B; matched controls
are near zero except for one final layer-6 control at 5.5 percent.

The primary prediction is an exact grid-point hit: the crossing closes at the
48M checkpoint, below the registered 200M deadline. Six of eight predictions
hold. P5 fails because the capped FineWeb-Edu ICL curve falls by 0.0408 nats
across the crossing, below 0.05, and its steepest step occurs later. P7 fails
because neither head above 0.3 at the crossing has an OV copying score above
0.5 there or one grid step earlier. The untrained control stays at chance and
passes all three P8 clauses.

The registered abruptness verdict is **INDETERMINATE AT THIS RESOLUTION**. The
crossing spans two grid intervals, which meets the abrupt width rule, but it
contains 13.57 percent of the run's total FineWeb-Edu validation-loss
improvement, above the 10 percent abrupt cutoff and below the 30 percent gradual
cutoff. Neither domain shows a registered loss bump.

| split | TinyStories perplexity | FineWeb-Edu perplexity | TinyStories ICL | FineWeb-Edu ICL |
|---|---:|---:|---:|---:|
| validation | 3.791 | 52.289 | -0.0691 | -0.1705 |
| test | 3.777 | 53.225 | -0.0780 | -0.1673 |

Mean accuracy across the ten fixed BLiMP paradigms is 0.8021; nine are above
0.75 and `npi_present_1` is 0.475. Final copying exact-match accuracy is 0.1173
[0.1124, 0.1224], against chance 0.000061. The pre/mid/post anchor rule cannot
select anchors because the post-rise curve never has a registered
three-checkpoint plateau; this failure is recorded rather than repaired after
seeing the curve.

The registered full-train interpolated 5-gram Kneser-Ney null does not reproduce
the neural ICL effect. On the same exhaustive validation windows it scores
+0.01345 [-0.00587, 0.03026] on TinyStories (9,827 windows) and -0.00122
[-0.01864, 0.01697] on FineWeb-Edu (9,801 windows); both intervals include zero,
whereas the neural scores are -0.0691 and -0.1705. For storage feasibility the
KenLM build prunes singleton 3-grams and count-at-most-two 4/5-grams
(`0 1 1 2 2`). Pruning was not fixed by the protocol, so this implementation
choice is disclosed; the registered order, smoothing, complete domain-specific
train splits, held-out windows and late-minus-early statistic are unchanged.

Primary artifacts: `analysis/emergence/final/` and `analysis/evaluation/`.

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
| Stage A checkpoints (34 bf16) | 2.89 GB |
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

Order: throughput pilot (01), download and tokenize (02-05), evaluate (06), train
(07), causal battery (08), schedule runner (09), copying probe (10), curve
comparison (11), contamination (12), decontamination (13), backend diagnostics
(15-17), emergence sweep (19), vocabulary coverage (20), validation-loss curve
(21), abruptness verdict (22), figures (23), prediction scorecard (24), anchor
selection (25), 5-gram null (26), and replication/control summary (27).
`scripts/smoke_gate.py` runs downstream stages against a tiny model first.

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

Stage A and its registered replication, depth-control, causal, evaluation and
null-model follow-ups are complete. The primary run crossed the induction
threshold by 48,005,120 tokens, so the registered no-crossing condition for
Stage B was not met and Stage B was not activated.

## License

MIT (`LICENSE`). Corpora are not redistributed. `data/tokenizer/tokenizer.json` carries the source datasets' terms in addition to MIT; see `NOTICE`.
