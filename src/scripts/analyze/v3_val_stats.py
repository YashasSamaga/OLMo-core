"""
Print statistics for the v3_small_ppl_validation dataset.

Reports per-source and aggregate numbers: document count, total tokens,
padding tokens, padding fraction, and sequence-length distribution.

Usage:
    python src/scripts/analyze/v3_val_stats.py [--sequence-length 4096] [--data-root /path/to/ai2-llm]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from olmo_core.data import (
    DataMix,
    NumpyPaddedFSLDataset,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)

DEFAULT_DATA_ROOT = "/weka/oe-training-default/ai2-llm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=4096,
        help="Sequence length to pad to (default: 4096).",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Root directory for eval data (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp/v3_val_stats",
        help="Local working directory for dataset index cache.",
    )
    return parser.parse_args()


def compute_stats(
    dataset: NumpyPaddedFSLDataset, sequence_length: int
) -> Tuple[Dict[str, Dict], Dict]:
    """
    Iterate over the dataset and compute per-source and aggregate statistics.

    Returns (per_source_stats, aggregate_stats).
    """
    # Map source paths back to short names.
    # NumpyPaddedFSLDataset stores paths; the source name is the
    # second-to-last component (e.g. .../c4_en/val/part-0-00000.npy -> c4_en).
    source_for_path: Dict[int, str] = {}
    for path_idx, path_str in enumerate(dataset.paths):
        parts = Path(path_str).parts
        # Look for the source directory name (two levels up from file).
        source_name = parts[-3] if len(parts) >= 3 else Path(path_str).stem
        source_for_path[path_idx] = source_name

    # Per-source instance counts from dataset offsets (before scanning).
    instances_per_source: Dict[str, int] = {}
    for path_idx, (start, end) in enumerate(dataset.offsets):
        source = source_for_path.get(path_idx, f"path_{path_idx}")
        instances_per_source[source] = instances_per_source.get(source, 0) + (end - start)

    # Per-source accumulators
    source_stats: Dict[str, Dict] = defaultdict(
        lambda: {
            "num_instances": 0,
            "num_docs": 0,
            "total_tokens": 0,
            "real_tokens": 0,
            "doc_lengths": [],
        }
    )

    # Pre-fill instance counts from offsets.
    for source, count in instances_per_source.items():
        source_stats[source]["num_instances"] = count

    num_docs = len(dataset)
    print(f"Scanning {num_docs:,} documents ...", flush=True)

    for idx in range(num_docs):
        item = dataset[idx]
        input_ids = item["input_ids"]
        label_mask = item["label_mask"]

        real_token_count = int(label_mask.sum().item())
        total_token_count = len(input_ids)

        # Determine which source this instance comes from.
        # Use the dataset's offsets to figure out the path index.
        path_idx = _path_index_for_instance(dataset, idx)
        source = source_for_path.get(path_idx, f"path_{path_idx}")

        s = source_stats[source]
        s["num_docs"] += 1
        s["total_tokens"] += total_token_count
        s["real_tokens"] += real_token_count
        s["doc_lengths"].append(real_token_count)

        if (idx + 1) % 10_000 == 0:
            print(f"  ... {idx + 1:,}/{num_docs:,}", flush=True)

    # Build aggregate
    agg = {
        "num_instances": sum(s["num_instances"] for s in source_stats.values()),
        "num_docs": sum(s["num_docs"] for s in source_stats.values()),
        "total_tokens": sum(s["total_tokens"] for s in source_stats.values()),
        "real_tokens": sum(s["real_tokens"] for s in source_stats.values()),
        "doc_lengths": [],
    }
    for s in source_stats.values():
        agg["doc_lengths"].extend(s["doc_lengths"])

    return dict(source_stats), agg


def _path_index_for_instance(dataset: NumpyPaddedFSLDataset, instance_idx: int) -> int:
    """Binary search over dataset.offsets to find which path owns this instance."""
    for path_idx, (start, end) in enumerate(dataset.offsets):
        if start <= instance_idx < end:
            return path_idx
    return len(dataset.offsets) - 1


def print_stats(label: str, stats: Dict, sequence_length: int):
    """Print a formatted stats block."""
    num_docs = stats["num_docs"]
    total = stats["total_tokens"]
    real = stats["real_tokens"]
    padding = total - real
    pad_pct = 100.0 * padding / total if total > 0 else 0.0

    lengths = np.array(stats["doc_lengths"])
    full_seqs = int(np.sum(lengths >= sequence_length))

    num_instances = stats["num_instances"]

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Instances:        {num_instances:>12,}")
    print(f"  Documents:        {num_docs:>12,}")
    print(f"  Sequence length:  {sequence_length:>12,}")
    print(f"  Total tokens:     {total:>12,}  (docs × seq_len)")
    print(f"  Real tokens:      {real:>12,}")
    print(f"  Padding tokens:   {padding:>12,}  ({pad_pct:.1f}%)")
    print(f"  Full sequences:   {full_seqs:>12,}  (doc_len >= seq_len)")
    if len(lengths) > 0:
        print(f"  Doc length  min:  {int(lengths.min()):>12,}")
        print(f"  Doc length  p25:  {int(np.percentile(lengths, 25)):>12,}")
        print(f"  Doc length  p50:  {int(np.percentile(lengths, 50)):>12,}")
        print(f"  Doc length  p75:  {int(np.percentile(lengths, 75)):>12,}")
        print(f"  Doc length  p95:  {int(np.percentile(lengths, 95)):>12,}")
        print(f"  Doc length  max:  {int(lengths.max()):>12,}")
        print(f"  Doc length mean:  {float(lengths.mean()):>12,.1f}")


def main():
    args = parse_args()

    tokenizer = TokenizerConfig.dolma2()
    dataset_cfg = NumpyPaddedFSLDatasetConfig.from_data_mix(
        DataMix.v3_small_ppl_validation,
        mix_base_dir=args.data_root,
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        work_dir=args.work_dir,
    )
    dataset: NumpyPaddedFSLDataset = dataset_cfg.build()  # type: ignore[assignment]
    dataset.prepare()

    print(f"Dataset: v3_small_ppl_validation")
    print(f"Data root: {args.data_root}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Paths ({len(dataset.paths)}):")
    for p in dataset.paths:
        print(f"  {p}")
    print(f"Total documents: {len(dataset):,}")

    per_source, aggregate = compute_stats(dataset, args.sequence_length)

    # Print per-source
    for source in sorted(per_source):
        print_stats(source, per_source[source], args.sequence_length)

    # Print aggregate
    print_stats("AGGREGATE", aggregate, args.sequence_length)


if __name__ == "__main__":
    main()
