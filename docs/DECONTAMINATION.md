# Web corpus extension and its decontamination

Registered holdout closed at the document level before tokenization: base corpus (`docs/DATA.md`) shares no text between train and evaluation. The web extension does not inherit that property.

## Purpose of the extension

- Base FineWeb-Edu train stream: 167,948,480 tokens.
- Staged run: 30 percent of a multi-billion-token budget from the web domain.
- Extension continues `sample-10BT` past the same 157,616-document prefix used by the base slice, tokenized into train-only shards.

## The check

`scripts/12_leakage_check.py`: samples 128 n-grams of length 200 at seeded random positions per holdout split, searches train shards for an exact token match (verbatim match, not hash collision). Positive control every invocation: same n-grams searched in their own split, must be found. Zero-leak result reported as failure unless the control confirms a present n-gram is found.

## Result before decontamination

4.7 billion token extension, built by continuing the stream with no holdout exclusion:

| Split | Sampled n-grams found in train |
| --- | ---: |
| val | 9 / 128 |
| test | 7 / 128 |

Positive controls: pass, both splits. Run log not retained; counts recorded at the time.

Cause: `sample-10BT` contains near-duplicate documents of the held-out documents; base slice was clean only because holdout documents were excluded from that prefix at the document level.

Registered response, decided before any training on the extension: rebuild with a document-level decontamination pass, require the gate to pass before use.

## The decontamination pass

`scripts/13_filter_ext.py` replaces `scripts/04_tokenize.py --extra-web` for the extension: tokenizes each extension document, drops it if its token stream contains any 200-token n-gram present in the closed `fineweb_edu` val or test stream, writes the same shard files. Contaminated tokenized artifact never written to disk.

Holdout n-grams compared by 64-bit polynomial hash, not stored verbatim (10,188,221 windows of 200 token ids would be about 4 GB). A hash collision can only drop a clean document, never admit a leak.

## Run record

Rebuild request: 2.5e9 extension tokens, down from 4.5e9 for the first build. Registered cap: 5 billion tokens at the 70/30 mix.

| Quantity | Value |
| --- | ---: |
| Extension documents downloaded | 2,329,044 |
| Extension characters downloaded | 11,051,250,848 |
| Unique holdout 200-grams in the hash table | 10,188,221 |
| Documents dropped, first 2,000,000 seen | 1,699 |
| Shards written | 6 |
| Tokens kept | 2,610,318,560 |

Progress logged every 500,000 documents; 1,699 is the last-checkpoint count, not the final total. Drop rate over that range: 8.5e-4.

Logs: `analysis/leakage/fineweb_edu_extension_filter.log` (run), `analysis/corpus_extension.json` (per-shard token counts), `analysis/tokenized_corpus_sha256.txt` (shard SHA-256 digests).

## Self-verification of the written shards

Per-document filter cannot see a window straddling the boundary between two kept documents. Script hash-scans written shards end to end for residual holdout n-grams: about 2.61e9 windows scanned, 2 matches, against an assertion of zero. Script exited before writing extension `meta.json` and its decontamination report; neither file exists for this build.

Two mechanisms for such a match: a real boundary-straddling window, or a 64-bit hash collision (expected count: 2.61e9 x 1.0188e7 / 2**64, or 1.4e-3). Arithmetic favors the first.

Residual: 2 of 10,188,221 holdout 200-grams, 2.0e-7 of the holdout n-gram population.

## Result after decontamination

Registered gate, six written shards:

| Split | Sampled n-grams found in train | Positive control | Verdict |
| --- | ---: | --- | --- |
| val | 0 / 128 | pass | pass |
| test | 0 / 128 | pass | pass |

Log: `analysis/leakage/fineweb_edu_extension_200gram_gate.log`.

At a residual rate of 2.0e-7, a 128-sample draw has a probability of about 2.5e-5 of touching a residual n-gram.

## Standing conditions

Extension shards pass the registered gate and are the shards the staged run reads. Extension `meta.json` and decontamination report not written: zero-residual assertion is a hard condition, not met. Reproducing the extension from scratch reproduces that outcome unless the filter is extended to remove boundary-straddling windows (post-write pass over shard boundaries, or re-run seeded with residual positions).
