#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "tokenizers"]
# ///
"""
Print stats and decoded examples from each v3_small_ppl_validation source.

Usage:
    uv run src/scripts/inspect_ppl_val_sources.py
    uv run src/scripts/inspect_ppl_val_sources.py --sources c4_en dolma_wiki
    uv run src/scripts/inspect_ppl_val_sources.py --num-tokens 200  # more context per example
"""

import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

DATA_ROOT = Path("/weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer")
TOKENIZER_ID = "allenai/dolma2-tokenizer"

ALL_SOURCES = [
    "c4_en",
    "dolma_books",
    "dolma_common-crawl",
    "dolma_pes2o",
    "dolma_reddit",
    "dolma_stack",
    "dolma_wiki",
    "ice",
    "m2d2_s2orc",
    "pile",
    "wikitext_103",
]


def load_tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(TOKENIZER_ID)


def inspect_source(
    source: str,
    tokenizer: Tokenizer,
    split: str = "val",
    num_tokens: int = 100,
    num_examples: int = 3,
):
    npy_path = DATA_ROOT / source / split / "part-0-00000.npy"
    if not npy_path.exists():
        print(f"  [not found: {npy_path}]")
        return

    tokens = np.memmap(npy_path, dtype=np.uint32, mode="r")
    total_tokens = len(tokens)
    unique_tokens = len(np.unique(tokens[:100_000]))  # sample for speed

    print(f"  File:          {npy_path}")
    print(f"  Total tokens:  {total_tokens:,}")
    print(f"  File size:     {npy_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Unique tokens (first 100k): {unique_tokens:,}")
    print()

    # Decode a few spans from different positions
    positions = [0, total_tokens // 3, 2 * total_tokens // 3]
    for i, pos in enumerate(positions):
        if i >= num_examples:
            break
        end = min(pos + num_tokens, total_tokens)
        token_ids = tokens[pos:end].tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        # Truncate display to avoid flooding terminal
        display = text[:500]
        if len(text) > 500:
            display += " [...]"
        print(f"  --- Example {i + 1} (tokens {pos:,}–{end:,}) ---")
        print(f"  {display!r}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Inspect v3_small_ppl_validation sources")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=ALL_SOURCES,
        help="Which sources to inspect (default: all)",
    )
    parser.add_argument(
        "--split", default="val", choices=["val", "test"], help="Which split to inspect"
    )
    parser.add_argument(
        "--num-tokens", type=int, default=100, help="Tokens to decode per example"
    )
    parser.add_argument(
        "--num-examples", type=int, default=3, help="Number of example spans per source"
    )
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    print()

    for source in args.sources:
        print(f"{'=' * 70}")
        print(f"  SOURCE: {source}")
        print(f"{'=' * 70}")
        inspect_source(
            source,
            tokenizer,
            split=args.split,
            num_tokens=args.num_tokens,
            num_examples=args.num_examples,
        )
        print()


if __name__ == "__main__":
    main()
