"""Train the byte-level BPE tokenizer on a proportional sample of the training mix.

The tokenizer is trained on this corpus, not borrowed from another model, so the
vocabulary reflects the actual mixture the models will be trained on. The
training sample is drawn proportionally to the intended TOKEN mix (about 480M
TinyStories tokens to 150M FineWeb-Edu tokens), approximated by characters,
because chars per token is close enough across the two domains for the sample
proportion to hold. The measured ratio per domain is reported at the end.

Sampling strides through each domain rather than taking a prefix, so the sample
spans the whole corpus instead of only its first shards.

Byte-level BPE with the full 256-byte initial alphabet means any input round
trips exactly, including text the trainer never saw. <|endoftext|> is the only
special token and takes id 0; it is the document separator written by
04_tokenize.py.

The trained tokenizer is small and belongs in version control, so it is written
inside the repo, not to the bulk artifact root.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors, trainers
from tokenizers.models import BPE

from small_lm_lab.paths import RAW_ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizer" / "tokenizer.json"

VOCAB_SIZE = 16384
EOT_TOKEN = "<|endoftext|>"
MIN_FREQUENCY = 2

# Intended token mix, matching the lab plan: ~480M TinyStories + ~150M
# FineWeb-Edu. The tokenizer sample follows these proportions.
MIX_WEIGHTS = {"tinystories": 480.0 / 630.0, "fineweb_edu": 150.0 / 630.0}

# Total characters of training text for the tokenizer. A few hundred MB is far
# more than a 16k vocab needs and still trains in a few minutes.
SAMPLE_TOTAL_CHARS = 300_000_000

# Held-out-ish measurement sample for the chars/token report and round trip.
MEASURE_DOCS_PER_DOMAIN = 2000

DOMAINS = ["tinystories", "fineweb_edu"]


def load_manifest(domain: str) -> dict:
    path = RAW_ROOT / domain / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run scripts/02_download.py first")
    with path.open() as f:
        return json.load(f)


def iter_domain_docs(domain: str) -> Iterator[str]:
    """Yield every document of a domain in shard order."""
    manifest = load_manifest(domain)
    for shard_name in manifest["shards"]:
        with (RAW_ROOT / domain / shard_name).open("r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]


def strided_sample(domain: str, budget_chars: int, offset: int = 0) -> Iterator[str]:
    """Yield documents spread across the domain until budget_chars is reached.

    The stride is chosen from the manifest so that taking every stride-th
    document covers the whole corpus while landing near the character budget.
    """
    manifest = load_manifest(domain)
    total_docs = manifest["documents"]
    avg_doc_chars = manifest["chars"] / max(total_docs, 1)
    docs_needed = max(int(budget_chars / max(avg_doc_chars, 1.0)), 1)
    stride = max(total_docs // docs_needed, 1)

    taken_chars = 0
    for i, text in enumerate(iter_domain_docs(domain)):
        if i % stride != offset % stride:
            continue
        yield text
        taken_chars += len(text)
        if taken_chars >= budget_chars:
            return


def build_training_iterator() -> Iterator[str]:
    """Yield the proportional tokenizer training sample across all domains."""
    for domain in DOMAINS:
        budget = int(SAMPLE_TOTAL_CHARS * MIX_WEIGHTS[domain])
        print(f"       sampling {domain}: target {budget/1e6:.0f} M chars", flush=True)
        taken = 0
        docs = 0
        for text in strided_sample(domain, budget):
            taken += len(text)
            docs += 1
            yield text
        print(f"       sampled  {domain}: {docs:,} docs, {taken/1e6:.0f} M chars", flush=True)


def train_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=None))
    # add_prefix_space=False keeps a leading space out of the first token; the
    # ByteLevel alphabet guarantees every byte is representable.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        # Listed first so it takes id 0.
        special_tokens=[EOT_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(build_training_iterator(), trainer=trainer)
    return tokenizer


def measure_domain(tokenizer: Tokenizer, domain: str) -> dict:
    """Round trip and measure chars per token on a sample of one domain.

    The measurement sample uses a different stride offset from the training
    sample so it is not the exact same documents the vocabulary was fit on.
    """
    texts: list[str] = []
    for i, text in enumerate(strided_sample(domain, 20_000_000, offset=1)):
        texts.append(text)
        if len(texts) >= MEASURE_DOCS_PER_DOMAIN:
            break

    total_chars = 0
    total_tokens = 0
    round_trip_failures = 0
    for text in texts:
        encoding = tokenizer.encode(text)
        decoded = tokenizer.decode(encoding.ids)
        if decoded != text:
            round_trip_failures += 1
        total_chars += len(text)
        total_tokens += len(encoding.ids)

    return {
        "domain": domain,
        "measure_docs": len(texts),
        "chars": total_chars,
        "tokens": total_tokens,
        "chars_per_token": total_chars / max(total_tokens, 1),
        "round_trip_failures": round_trip_failures,
    }


def main() -> None:
    print(f"tokenizer out: {TOKENIZER_PATH}")
    print(f"vocab_size={VOCAB_SIZE} special={EOT_TOKEN!r} sample={SAMPLE_TOTAL_CHARS/1e6:.0f} M chars")
    for domain in DOMAINS:
        m = load_manifest(domain)
        print(
            f"  {domain}: {m['documents']:,} docs, {m['chars']/1e6:.0f} M chars available, "
            f"mix weight {MIX_WEIGHTS[domain]:.3f}"
        )
    print()

    print("[run ] training BPE", flush=True)
    tokenizer = train_tokenizer()

    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_PATH))

    vocab_size = tokenizer.get_vocab_size()
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    digest = hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest()

    print(f"\nvocab size: {vocab_size}")
    print(f"{EOT_TOKEN} id: {eot_id}")
    print(f"tokenizer sha256: {digest}")
    print(f"file size: {TOKENIZER_PATH.stat().st_size/1e6:.2f} MB")

    if vocab_size != VOCAB_SIZE:
        print(f"WARNING: vocab size {vocab_size} != requested {VOCAB_SIZE}")
    if eot_id != 0:
        print(f"WARNING: {EOT_TOKEN} has id {eot_id}, expected 0")
    if vocab_size >= 65536:
        raise SystemExit("vocab too large for uint16 token storage")

    print("\n[run ] sanity check: round trip and chars/token per domain", flush=True)
    stats = [measure_domain(tokenizer, d) for d in DOMAINS]

    print(f"\n{'domain':<14} {'docs':>7} {'chars':>12} {'tokens':>12} {'chars/token':>12} {'rt fails':>9}")
    print("-" * 72)
    for s in stats:
        print(
            f"{s['domain']:<14} {s['measure_docs']:>7,} {s['chars']:>12,} "
            f"{s['tokens']:>12,} {s['chars_per_token']:>12.3f} {s['round_trip_failures']:>9}"
        )

    failures = sum(s["round_trip_failures"] for s in stats)
    if failures:
        raise SystemExit(f"round trip failed on {failures} documents")
    print("\nround trip exact on every sampled document in both domains")

    # A short explicit round trip, printed so the check is visible in the log.
    for domain, probe in (
        ("tinystories", "Once upon a time, a small dog found a red ball."),
        ("fineweb_edu", "Photosynthesis converts light energy into chemical energy (C6H12O6)."),
    ):
        ids = tokenizer.encode(probe).ids
        back = tokenizer.decode(ids)
        print(f"  {domain}: {len(probe)} chars -> {len(ids)} tokens -> exact={back == probe}")


if __name__ == "__main__":
    main()
