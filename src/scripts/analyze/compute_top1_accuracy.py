"""Compute top-k accuracy (and mean log-prob) from .npz files produced by
collect_token_logprobs_hf.py.

Top-k accuracy: fraction of tokens where the correct next token appears
among the model top-k predictions.  k=1 uses the argmax field directly;
k>1 uses the stored ``{source}__topk_indices`` arrays (only available when
the file was produced with ``--top-k K`` where K >= k).

Reported k values: 1, 2, 4, 8, 16, 32 (whichever are available).

Usage:
    # Single file
    python src/scripts/analyze/compute_top1_accuracy.py path/to/step.npz

    # All files in a directory
    python src/scripts/analyze/compute_top1_accuracy.py path/to/dir/

    # Multiple files / globs
    python src/scripts/analyze/compute_top1_accuracy.py path/to/dir/*.npz
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

EXPECTED_FIELDS = {"correct_logit", "max_logit", "log_z", "instance_index", "token_id", "max_token_id", "position_in_seq"}
# Dolma2 / OLMo-2 vocab size (unpadded)
MAX_VALID_TOKEN_ID = 200_000  # generous upper bound; flags obvious garbage


def sanity_check(path: Path, arr: np.ndarray, src: str, ref_token_counts: Optional[dict]) -> List[str]:
    """
    Run sanity checks on a single source array.  Returns a list of warning strings
    (empty if everything looks fine).
    """
    warnings = []
    n = len(arr)

    if n == 0:
        warnings.append(f"[{src}] EMPTY array — no tokens collected")
        return warnings

    # --- Field check ---
    actual_fields = set(arr.dtype.names or [])
    missing = EXPECTED_FIELDS - actual_fields
    if missing:
        warnings.append(f"[{src}] Missing fields: {missing}")

    # --- NaN / Inf in float fields ---
    for field in ("correct_logit", "max_logit", "log_z"):
        if field not in actual_fields:
            continue
        vals = arr[field].astype(np.float32)
        n_nan = int(np.isnan(vals).sum())
        n_inf = int(np.isinf(vals).sum())
        if n_nan:
            warnings.append(f"[{src}] {n_nan} NaN values in '{field}'")
        if n_inf:
            warnings.append(f"[{src}] {n_inf} Inf values in '{field}'")

    # --- log_z >= correct_logit (partition >= any single logit) ---
    if "log_z" in actual_fields and "correct_logit" in actual_fields:
        log_z = arr["log_z"].astype(np.float32)
        correct = arr["correct_logit"].astype(np.float32)
        violations = int((log_z < correct - 0.1).sum())  # 0.1 tolerance for fp16 rounding
        if violations:
            warnings.append(f"[{src}] {violations} tokens where log_z < correct_logit (partition < individual logit)")

    # --- log_z >= max_logit ---
    if "log_z" in actual_fields and "max_logit" in actual_fields:
        log_z = arr["log_z"].astype(np.float32)
        max_l = arr["max_logit"].astype(np.float32)
        violations = int((log_z < max_l - 0.1).sum())
        if violations:
            warnings.append(f"[{src}] {violations} tokens where log_z < max_logit")

    # --- Token IDs in valid range ---
    if "token_id" in actual_fields:
        bad = int((arr["token_id"] > MAX_VALID_TOKEN_ID).sum())
        if bad:
            warnings.append(f"[{src}] {bad} token_ids exceed MAX_VALID_TOKEN_ID={MAX_VALID_TOKEN_ID}")

    # --- Log-probs in plausible range: correct_logit - log_z should be <= 0 ---
    if "correct_logit" in actual_fields and "log_z" in actual_fields:
        logprobs = arr["correct_logit"].astype(np.float32) - arr["log_z"].astype(np.float32)
        n_positive = int((logprobs > 0.1).sum())
        if n_positive:
            warnings.append(f"[{src}] {n_positive} tokens with positive log-prob (> 0.1) — likely numerical issue")
        mean_lp = float(logprobs.mean())
        if mean_lp > -0.01:
            warnings.append(f"[{src}] Suspiciously high mean log-prob: {mean_lp:.4f}")

    # --- Token count consistency across checkpoints ---
    if ref_token_counts is not None:
        ref_n = ref_token_counts.get(src)
        if ref_n is not None:
            diff_pct = abs(n - ref_n) / ref_n if ref_n > 0 else 0
            if diff_pct > 0.02:  # >2% difference is suspicious
                warnings.append(f"[{src}] Token count differs from reference by {diff_pct*100:.1f}%: got {n:,}, reference has {ref_n:,}")

    # --- Duplicate (instance, position) pairs ---
    if "instance_index" in actual_fields and "position_in_seq" in actual_fields:
        keys = np.stack([arr["instance_index"].astype(np.int64), arr["position_in_seq"].astype(np.int64)], axis=1)
        n_unique = len(np.unique(keys, axis=0))
        if n_unique != n:
            warnings.append(f"[{src}] {n - n_unique:,} duplicate (instance_index, position_in_seq) pairs")

    return warnings


TOPK_VALUES = [1, 2, 4, 8, 16, 32]


def compute_metrics(path: Path, ref_token_counts: Optional[dict] = None, run_checks: bool = True) -> dict:
    """Return per-source and aggregate top-k accuracy and mean log-prob.

    Each source entry contains:
      - ``top{k}_acc`` for each k in TOPK_VALUES that is available
      - ``mean_logprob``, ``perplexity``, ``n_tokens``

    The ``__aggregate__`` entry additionally contains ``warnings``.
    """
    data = np.load(path)
    sources = [k for k in data.keys() if not k.endswith("__topk_logits") and not k.endswith("__topk_indices")]

    results = {}
    all_warnings = []
    total_tokens = 0
    total_logprob = 0.0
    agg_topk_correct: dict = {k: 0 for k in TOPK_VALUES}
    agg_topk_available: dict = {k: False for k in TOPK_VALUES}

    for src in sorted(sources):
        arr = data[src]

        if run_checks:
            warnings = sanity_check(path, arr, src, ref_token_counts)
            all_warnings.extend(warnings)

        n = len(arr)
        logprob = float((arr["correct_logit"].astype(np.float32) - arr["log_z"].astype(np.float32)).mean())
        src_result: dict = {
            "mean_logprob": logprob,
            "perplexity": float(np.exp(-logprob)),
            "n_tokens": n,
        }

        # top-1 from argmax fields (always available)
        correct1 = int((arr["token_id"] == arr["max_token_id"]).sum())
        src_result["top1_acc"] = correct1 / n if n > 0 else 0.0
        agg_topk_correct[1] += correct1
        agg_topk_available[1] = True

        # top-k from stored topk_indices array (only if file was collected with --top-k)
        topk_key = f"{src}__topk_indices"
        if topk_key in data:
            topk_indices = data[topk_key]  # (N, K_stored)
            K_stored = topk_indices.shape[1]
            token_ids = arr["token_id"]  # (N,)
            for k in TOPK_VALUES:
                if k == 1 or k > K_stored:
                    continue
                # correct if token_id appears in topk_indices[:, :k]
                correct_k = int((topk_indices[:, :k] == token_ids[:, None]).any(axis=1).sum())
                src_result[f"top{k}_acc"] = correct_k / n if n > 0 else 0.0
                agg_topk_correct[k] += correct_k
                agg_topk_available[k] = True

        results[src] = src_result
        total_tokens += n
        total_logprob += logprob * n

    agg_logprob = total_logprob / total_tokens if total_tokens > 0 else 0.0
    agg: dict = {
        "mean_logprob": agg_logprob,
        "perplexity": float(np.exp(-agg_logprob)),
        "n_tokens": total_tokens,
        "warnings": all_warnings,
    }
    for k in TOPK_VALUES:
        if agg_topk_available[k]:
            agg[f"top{k}_acc"] = agg_topk_correct[k] / total_tokens if total_tokens > 0 else 0.0
    results["__aggregate__"] = agg
    return results


def print_results(path: Path, results: dict):
    step = path.stem
    agg = results["__aggregate__"]
    # Determine which top-k columns are available
    avail_ks = [k for k in TOPK_VALUES if f"top{k}_acc" in agg]

    topk_headers = "".join(f"  Top-{k:>2}" for k in avail_ks)
    col_width = 25 + len(avail_ks) * 9 + 10 + 10 + 10
    separator = "=" * max(60, col_width)

    print(f"\n{separator}")
    print(f"  {step}")
    print(separator)
    warnings = agg.get("warnings", [])
    if warnings:
        print(f"  *** {len(warnings)} WARNING(S) ***")
        for w in warnings:
            print(f"  ⚠  {w}")
        print()
    print(f"  {'Source':<25}{topk_headers}  {'Mean LogP':>10} {'PPL':>10} {'Tokens':>10}")
    print(f"  {'-' * (col_width - 2)}")
    for src, m in results.items():
        if src == "__aggregate__":
            continue
        topk_vals = "".join(f"  {m.get(f'top{k}_acc', float('nan')):>7.4f}" for k in avail_ks)
        print(f"  {src:<25}{topk_vals}  {m['mean_logprob']:>10.4f} {m['perplexity']:>10.2f} {m['n_tokens']:>10,}")
    print(f"  {'-' * (col_width - 2)}")
    topk_vals = "".join(f"  {agg.get(f'top{k}_acc', float('nan')):>7.4f}" for k in avail_ks)
    print(f"  {'AGGREGATE':<25}{topk_vals}  {agg['mean_logprob']:>10.4f} {agg['perplexity']:>10.2f} {agg['n_tokens']:>10,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Path(s) to .npz file(s) or a directory containing them.")
    parser.add_argument("--csv", action="store_true", help="Print aggregate results as CSV (step, top1_acc, perplexity).")
    parser.add_argument("--no-checks", action="store_true", help="Skip sanity checks (faster).")
    args = parser.parse_args()

    # Collect all npz files
    npz_files = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            npz_files.extend(sorted(path.glob("*.npz")))
        elif path.suffix == ".npz":
            npz_files.append(path)
        else:
            print(f"Warning: skipping {p} (not .npz or directory)", file=sys.stderr)

    if not npz_files:
        print("No .npz files found.", file=sys.stderr)
        sys.exit(1)

    # Sort by step number if filenames contain step info
    def step_key(p: Path) -> int:
        name = p.stem
        if "-step" in name:
            try:
                return int(name.split("-step")[1].split("-")[0])
            except (IndexError, ValueError):
                pass
        return 0

    npz_files.sort(key=step_key)

    run_checks = not args.no_checks

    # Build reference token counts from the first file — all checkpoints should
    # have identical token counts per source (same dataset, same order).
    ref_token_counts: Optional[dict] = None
    if run_checks and len(npz_files) > 1:
        first = np.load(npz_files[0])
        ref_token_counts = {
            k: len(first[k])
            for k in first.keys()
            if not k.endswith("__topk_logits") and not k.endswith("__topk_indices")
        }

    total_warnings = 0
    if args.csv:
        # Determine available top-k columns from first file
        first_results = compute_metrics(npz_files[0], ref_token_counts=ref_token_counts, run_checks=run_checks)
        avail_ks = [k for k in TOPK_VALUES if f"top{k}_acc" in first_results["__aggregate__"]]
        topk_csv_headers = ",".join(f"top{k}_acc" for k in avail_ks)
        print(f"step,{topk_csv_headers},mean_logprob,perplexity,n_tokens,n_warnings")
        # Print first file results (already computed)
        for path, results in [(npz_files[0], first_results)] + [
            (p, compute_metrics(p, ref_token_counts=ref_token_counts, run_checks=run_checks))
            for p in npz_files[1:]
        ]:
            agg = results["__aggregate__"]
            n_warn = len(agg.get("warnings", []))
            total_warnings += n_warn
            step = path.stem
            topk_csv_vals = ",".join(f"{agg.get(f'top{k}_acc', float('nan')):.6f}" for k in avail_ks)
            print(f"{step},{topk_csv_vals},{agg['mean_logprob']:.6f},{agg['perplexity']:.4f},{agg['n_tokens']},{n_warn}")
    else:
        for path in npz_files:
            results = compute_metrics(path, ref_token_counts=ref_token_counts, run_checks=run_checks)
            print_results(path, results)
            total_warnings += len(results["__aggregate__"].get("warnings", []))

    if run_checks:
        if total_warnings == 0:
            print(f"\n✓ All sanity checks passed across {len(npz_files)} file(s).")
        else:
            print(f"\n✗ {total_warnings} total warning(s) across {len(npz_files)} file(s).", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
