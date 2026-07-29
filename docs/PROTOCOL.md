# Registered protocol
Condensed from the registered documents. Decisions and criteria only.

## Hypotheses
- Phase A: loss falls with params at fixed tokens, falls with tokens at fixed params, size curves cross as planned. Not a claim of model quality.
- Phase B: induction heads form in an abrupt phase change; heads causally responsible for in-context copying. Correlation alone = negative result.

## Architecture
| name | d_model | layers | heads | params |
|---|---:|---:|---:|---:|
| size30m | 512 | 8 | 8 | 34,087,424 |
| size60m | 576 | 12 | 9 | 57,227,328 |
| size120m | 768 | 16 | 12 | 125,854,464 |

Decoder-only, pre-norm RMSNorm, RoPE half-rotation, SwiGLU, no biases, tied embeddings, head_dim 64, context 512, vocab 16,384.

## Training, fixed across sizes
- AdamW b1 0.9, b2 0.95, eps 1e-8, wd 0.1, applied to matrices only, not RMSNorm/embedding.
- Grad clip global norm 1.0. Cosine decay to 10% peak, warmup 2% of steps. Seed 1.
- Tokens/step = 32 x 512 = 16,384. Mix 0.70 TinyStories / 0.30 FineWeb-Edu, fixed token order.

## Token budget rule (pre-pilot)
1. Framework = higher tok/s of PyTorch-MPS vs MLX, uncontended, same config/batch.
2. B equal for all sizes = largest round number with size120m projected <=48h at measured throughput, cap 500M.
3. Cap keeps B under 699.0M available train tokens (no within-epoch repeat).
4. If 200M infeasible for size120m in 48h: report as blocker, do not shrink model.

## Budget resolved, 2026-07-17: B = 200M
MLX wins every cell: 1.77x (size30m b16), 1.56x (size30m b32), 1.60x (size120m b16). Eval/interp in PyTorch, converter agrees 3.5e-06 on logits. MLX tok/s @16,384/step: 4,568 / 2,762 / 1,365 (30/60/120m). 1,365 tok/s -> 236M tokens/48h -> B=200M. size120m b32 first measured 760 tok/s (0.096e12, 18.51GB/32GB, memory-bound); micro-batch 16 + 2 grad-accum -> 1,365 tok/s (0.172e12), compute-bound, same 16,384 tok/step. Compute throughput flat: 0.156e12 (30m), 0.158e12 (60m). Tokens/param at B: 5.9/3.5/1.6. Total schedule ~83 GPU-hours.

## Ablations (size30m only)
Peak LR in {6e-4, 1.2e-3, 2.4e-3}, warmup 2%; then warmup in {0, 2%} at winning LR. Each: 20% of B, scored on TinyStories val loss, validation only never test. Winning LR used for headline runs. Stability check @size120m: no divergence in first 2% steps; divergence -> one halving, reported.

## Evaluation battery
1. Held-out perplexity (primary): TinyStories/FineWeb-Edu separate, never pooled. Exhaustive non-overlapping 512-token windows. Per size/checkpoint. Bootstrap 1000 resamples, percentile interval. Non-monotonic size ordering = negative result.
2. BLiMP (secondary), 10 fixed paradigms: anaphor_number_agreement, anaphor_gender_agreement, determiner_noun_agreement_1, determiner_noun_agreement_2, regular_plural_subject_verb_agreement_1, irregular_plural_subject_verb_agreement_1, irregular_past_participle_verbs, animate_subject_trans, npi_present_1, wh_questions_object_gap. Score: sum token log-probs/sentence, correct if grammatical>ungrammatical. Bootstrap interval per paradigm. Chance 0.50.
3. In-context learning score: mean loss tok 450-500 minus mean loss tok 50-100, per domain, per checkpoint.
4. Synthetic copying: random tokens repeated twice, 128/repeat, 512 sequences, bootstrap over sequences. Metrics: mean loss(copy2-copy1); exact-match accuracy of copy2. Per checkpoint.

## Checkpoint schedule
Original grid (tokens): 1M,2M,4M,8M,16M,24M,32M,48M,64M,96M,128M,160M,200M,250M,300M,350M,400M,450M,B. Weights bf16; optimizer state kept only for most recent checkpoint. Phase change falling in a gap -> re-run with extra bracketing checkpoints, reported.

## Phase B instruments
- Induction head score: prefix-matching, 256 sequences, per head/layer/checkpoint = mean attention at 2nd occurrence to token following earlier occurrence. Threshold: score > 0.2.
- Phase change: token interval where max prefix-matching goes <0.1 to >0.3. Gradual result = negative against phase-change framing.
- Head ablation: mean-ablate (per-evaluation-distribution mean, not zero); measure change in copying accuracy, ICL score, test perplexity. Conditions: each induction head singly; all jointly; equal-count matched non-induction controls, same layers. Bootstrap over eval sequences.
- Activation patching: clean->corrupted (earlier occurrence replaced). Metric = logit(correct continuation) - logit(token corrupted prefix would induct to). Per head.
- Phase B claim: joint ablation degrades copying accuracy substantially more than matched controls, non-overlapping intervals. Equal damage = claim fails, reported failed.

## Amendment 2026-07-17: ambiguities resolved pre-data
1. Mean ablation = per-position mean, not global vector; global-mean reported only as sensitivity check.
2. Patching metric = logit(correct continuation) - logit(token following corrupted first-copy occurrence); both fixed by construction.
3. Repeated ids in random block resolve to construction-defined occurrence; bias small, downward.
4. Causal analyses run at final checkpoint + checkpoints bracketing phase-change interval; patching uses 256 sequences.
5. Recovered fraction withheld (reported absent) whenever bootstrap interval on clean-minus-corrupted gap includes zero. One verdict per run.

## Amendment 2026-07-17 (second): negative result vs dead instrument
- Copying accuracy at/near chance + withheld fraction + null ablation = NEGATIVE RESULT, induction claim fails.
- Copying accuracy clearly above chance + gap straddles zero = UNDERPOWERED, not absence of mechanism.
- Escalation, underpowered case only: re-run patching ONCE at 1024 sequences. Clears zero -> report fraction, disclose escalation. Still straddles with demonstrated copying -> report as instrument limit, not evidence against induction. No further escalation.

## Amendment 2026-07-17 (third): gap-interval correction
| sequences | gap | SE | ratio | gate |
|---:|---:|---:|---:|---|
| 8 | -2.80e-04 | 4.60e-04 | 0.61 | closed |
| 64 | -2.82e-04 | 1.69e-04 | 1.67 | closed |
| 256 | -2.51e-04 | 9.29e-05 | 2.71 | OPENS |
| 1024 | -2.47e-04 | 4.65e-05 | 5.31 | OPENS |

40 random-init models: gate opened 47.5% at 256 sequences; alpha 0.01 -> 35%. Correction: (1) fraction withheld unless copying exact-match interval clearly above chance, positive control gates the instrument. (2) gap-interval test retained as necessary-not-sufficient only, catches degenerate denominator. (3) 1024-sequence escalation reachable only once copying gate open.

## Limitations, acknowledged in advance
Single seed for headline runs, variance uncharacterized. Fixed hyperparameters disadvantage largest model. Fixed budget: larger models undertrained vs compute-optimal, no scaling coefficient fitted. Two narrow domains, one synthetic. Context 512 short, bounds ICL claims. BLiMP evaluated below intended scale, near-chance expected on several paradigms.

## Excluded methods
No t-SNE, no saliency maps, no single-neuron anecdotes, no cherry-picked samples as evidence. Generated text only as labelled illustration, never as result.

## Amendment 2026-07-21: headline rescoped, staged size30m run
3-size comparison at B=200M demoted; headline = size30m into emergence window (~2.5-5B tokens per literature, ~12.5x above B). size60m/120m optional.
- Schedule: Stage A 0->2B, checkpoints at original grid to 480M then every 100M. Stage B: 500M segments to 5B max.
- Trigger: copying-probe rises 3 consecutive checkpoints above pre-transition band -> checkpoint interval drops to 25M for next 500M.
- Stop: probe plateau 3+ consecutive checkpoints + 20% token margin, OR 5B cap, OR resumable budget pause.
- Anchors from probe curve: pre = last checkpoint below 10% rise from band; mid = nearest 50% of plateau delta; post = first >=90% of plateau; plus final checkpoint. Causal battery runs all anchors.
- scripts/08_interp.py corrected to run across checkpoints (prior final-only behavior = bug).
- LR: 1.2e-3, linear warmup 4M tokens, then CONSTANT (differs from ablation's cosine/40M; disclosed as hyperparameter choice not compared result).
- Corpus: FineWeb-Edu train stream expanded from same ODC-By sample-10BT (no in-run repeat); val/test shards untouched; 200-gram leakage check re-run on new train shards.
- TinyStories not expanded (531M train tokens); 70/30 mix kept; repetition (~6.6 passes at 5B) REGISTERED as experimental factor; per-domain val perplexity tracked all checkpoints.
- Hardware: rented RTX 4090, MLX CUDA backend, float32, TF32 disabled + canary-checked every process start.
- Cross-backend gate (pass required before run start): full test suite on target machine; 500-step same-seed rerun of abl_lr1.2e-3, |loss delta|<=0.02 at step 500, mean<=0.015 over steps 400-500; checkpoint round-trip to PyTorch eval path.
- Probe = monitoring instrument only; reported numbers come from registered battery.

## Amendment 2026-07-24: replacement cross-backend gate
2026-07-21 gate passed on MLX-Metal-vs-CUDA pod leg: end delta 0.001119. Same gate vs TORCH leg FAILED: end delta 0.061059, mean 0.043621; recorded as failure, not amended away. On-device canary: relative Frobenius error 5.753e-07 vs float64, threshold 1e-05: true float32, not precision artifact. torch/MLX use different PRNGs (nn.init.normal_ vs mx.random.normal): different init points.
Replacement rule, fixed pre-data:
1. Within-framework spread: torch-vs-torch, 5 seeds, all 10 pairs, steps 400-500, end/mean delta each pair.
2. tol_end_new = 1.5 x max(within-framework end deltas); tol_mean_new = 1.5 x max(within-framework mean deltas).
3. Gate not adopted if derived tolerance exceeds 0.25 end-delta.
4. Replacement test: banked MLX checkpoint -> torch via converter (3.5e-06 max abs logit agreement), continue from identical weights, same steps 400-500 window.
5. If within-framework spread small vs observed 0.061: no gate adopted, torch-on-CUDA reported unsuitable pending cause.
Failed gate appears in writeup either way.

## Amendment 2026-07-26 (folded into the registration)
Written after Stage A passed ~8,192,000 of 2,000,000,000 tokens, before any interp quantity read.

1. Abruptness: G = grid intervals spanned by crossing (34-point grid). W = log10(end/start), decades (not admissible alone, grid non-uniform: ~1.5x near 16M, ~1.05x near 1.9B). F = fraction of total FineWeb-Edu val-loss improvement occurring within crossing interval (total = first to final grid checkpoint). Verdict: ABRUPT if G<=3 and F<=0.10; GRADUAL if G>=7 or F>=0.30; else INDETERMINATE AT THIS RESOLUTION. If G==1: resolution_limited=true, W reported as bound only, ABRUPT still assignable if F allows.

2. Anchor bands measured: probe at 10 seeds x 4 grid checkpoints <=8,000,000 tokens = 40 readings; mu, sigma = mean, SD. Pre-transition band [0, mu+3*sigma]. Rise = statistic > mu+3*sigma (3-checkpoint confirmation unchanged). Plateau = 3 consecutive checkpoints within 3*sigma of running max. 10%-rise anchor = mu + 0.10*(plateau_value-mu). Trigger statistic changed from copying exact-match accuracy (single point 0.000984251968503937 at 40,009,728 tokens = 8/8,128, Poisson regime) to max prefix-matching score (32,512 attention weights/head, continuous); copying retained alongside; no reported number changes.

3. Anchor sensitivity: causal battery also run one grid step below/above each anchor; claim read from anchor; reversal at both neighbours = reported anchor-sensitive.

4. Loss-bump: power law fit (least squares, log loss vs log tokens) per domain, grid points 1,000,000 to final. Bump statistic = signed residual, nats. Bump = >=2 consecutive checkpoints, residual positive and >3x residual SD over fitted window. TinyStories-epoch-boundary check (~0.76B/1.5B/2.3B/3.0B) NOT registered: MixtureBatcher.next_batch (src/small_lm_lab/data.py) draws domain i.i.d. per row, uniform window start, sampling with replacement, no epoch structure -> cannot produce sharp bump.

5. TinyStories repetition (531,101,051 train tokens): at 2,000,000,000 mixture tokens, expected multiplicity/token 2.64, coverage 0.9284; at 5,000,000,000, multiplicity 6.59, coverage 0.9986. Multiplicity Poisson(mean), coverage = 1-exp(-mean).

6. ICL headline domain = FineWeb-Edu (mean tokens/doc 1130.1, 0.45 docs/512-tok window) not TinyStories (197.1 tokens/doc, 2.60 docs/window), for every ICL claim including P5. scripts/10_probe_copying.py default --domain changed tinystories -> fineweb_edu (monitoring only); existing probe point (ICL score -0.04071552073583007 at 40,009,728 tokens, tinystories) kept with domain recorded.

7. Planned additions, no registered claim read from them: previous-token-head score/head (attention j to j-1, same forward passes); off-by-one controls at key i and key i+2 vs registered key i+1; weights-only OV copying score = sum(real parts of OV eigenvalues)/sum(magnitudes), 16,384x16,384 circuit reduced to 64x64 non-degenerate part; head churn = count of heads entering/leaving 0.2-threshold set between consecutive checkpoints.

8. Crossing-time uncertainty: per-sequence prefix-matching array retained every checkpoint (131KB/checkpoint, <5MB total). Bootstrap 1000 resamples: sequence indices drawn once/resample, per-head means and max recomputed per checkpoint, first checkpoint with resampled max >0.3 recorded. Reported crossing time = median + percentile interval, alongside point estimate; upward bias of max disclosed, not corrected.

9. Late divergence under constant LR: training loss diverges OR gradient norm exceeds 1.0 clip on majority of steps over any 1000-step post-warmup window -> stop run, keep last clean checkpoint, report event + token count. LR not lowered, run not restarted.

10. Replication/controls: second seed, from-scratch size30m, through located onset + 20% token margin, after Stage A GPU release; crossing interval moving >2x between seeds -> reported as headline emergence-location uncertainty. Depth controls: 1-layer model (cannot compose induction circuit) and 2-layer attention-only model (minimal induction-forming system), same d_model/tokenizer/data/mixture/LR. ICL null: interpolated 5-gram Kneser-Ney, train split, identical late-minus-early statistic and windows.

11. Cross-backend gate numbers, failure, five-seed noise floor, adjudication are in docs/GATE_FINDING.md, not STATE.md (not part of final tree).

## Prediction document: design numbers
| quantity | value |
|---|---|
| parameters | 34,087,424 |
| d_model, layers, heads, head_dim | 512, 8, 8, 64 |
| context | 512 |
| vocabulary | 16,384 |
| tokens per optimizer step | 16,384 |
| optimizer steps at 2B tokens | 122,070 |
| tokens per parameter at 2B | 58.7 |
| peak learning rate | 1.2e-3, constant after 4M-token warmup |
| mix | 0.70 TinyStories, 0.30 FineWeb-Edu |
| checkpoint grid | 34 points, 13 at or below 200M tokens |
| prefix-matching chance (repeat length 128) | 0.005412 |
| copying exact-match chance | 6.104e-05 |

Corpus (367,948,480-token prefix): 16,331/16,384 ids occur; 16,162 occur >=100x; 14,855 occur >=1,000x; uniform draw lands on id seen <100x 1.355% of time, on unseen id 0.324%. Concentration: top 100 ids = 0.5027 of mass, top 1,000 = 0.7633. Restated same day against analysis/vocab_coverage.json (full 167,948,480-token web stream vs earlier 167,000,000-token stop); moved 3rd significant figure of three values; no prediction depends on that precision.

## Predictions P1-P8
- P1 (primary): phase-change interval closes at or below 200,000,000 tokens; point prediction 48,000,000 tokens; read from phase_change.interval.end_tokens. Fails if end_tokens>200M or crossed=false at 2B.
- P2: >=1 head scores >0.2 at final Stage A checkpoint; point prediction 1-4 heads, layers 2-6. Fails if induction_heads empty at 2B.
- P3: phase-change interval spans at most 3 grid steps. Fails if >3 intervening grid points.
- P4: at last checkpoint with max prefix-matching <0.1, some head already >0.3 on previous-token score. Fails if none exceed 0.3 at/before that checkpoint.
- P5: fineweb_edu ICL score more negative by >=0.05 nats between start_tokens and first grid point >=end_tokens; steepest per-step drop within one grid step of end_tokens. Fails if drop <0.05 nats or steepest step >1 grid point away.
- P6: copying exact-match accuracy, lower bootstrap bound above chance (6.104e-05), at checkpoint <=200M tokens. Fails if not cleared by 200M; failure also closes the recovered-fraction instrument (amendment 3).
- P7: for head first crossing prefix-matching 0.3, its OV copying score already >0.5 at checkpoint at/before its own crossing. Fails if OV score crosses 0.5 strictly after.
- P8 (negative control): untrained size30m, same seed: max prefix-matching <0.05, no head >0.2, copying exact-match interval lower bound below chance. Fails if any of the three fire on random weights; metric then deemed wrong, no trained-model number admissible.

## P1 failure disposition, decided in advance
No crossing of 0.3 by 2e9 tokens: P1 reported failed, reasoning reported wrong; not evidence instrument broken if P8 passed; not grounds for lowering 0.3; Stage B (5B cap) continues. Reverse failure: max prefix-matching already >0.1 at 1,000,000-token checkpoint -> crossed=false for that reason, re-run with extra checkpoints below 1M tokens (61 optimizer steps).

## Supplementary diagnostics (predictions doc, no registered claim read from them)
Off-by-one attention controls (key i, key i+2 vs registered key i+1); attention entropy, first-token attention, self-attention per head; OV copying score per head (same construction as section-7 item above).
