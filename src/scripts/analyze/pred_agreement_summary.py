"""
Compare token-level prediction agreement between different model architectures.

For each model, loads the largest-step checkpoint dump and compares the argmax
predictions (max_token_id) across every pair of models, per data source.

Alignment is done by (instance_index, position_in_seq); only the intersection
of positions present in *all* loaded models is compared.

Sanity check: for every position in the intersection, the stored correct token
(token_id) must be identical across all models — otherwise data ordering differs
and the comparison is invalid.

Per-pair statistics reported (over the shared intersection):
  - n_shared: number of token positions in the intersection
  - both_correct:        A right  ∧  B right
  - A_only_correct:      A right  ∧  B wrong
  - B_only_correct:      A wrong  ∧  B right
  - both_wrong:          A wrong  ∧  B wrong
  - pred_agree:          fraction where A and B predict the same token (right or wrong)
  - same_error:          fraction of both-wrong positions where A and B predict the same wrong token
  - jaccard_correct:     both_correct / (both_correct + A_only + B_only)  — set-similarity of correct sets
  - disagreement_bias:   (A_only_correct - B_only_correct) / n_shared     — positive => A more uniquely correct

Usage:
    python src/scripts/analyze/pred_agreement_summary.py
    python src/scripts/analyze/pred_agreement_summary.py --dump-root /path/to/dumps
    python src/scripts/analyze/pred_agreement_summary.py --models modelA modelB
"""

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DUMP_ROOT = Path("/weka/oe-training-default/yashasbls/georges-functional-analysis")
SAMPLE_TOKENS = 500_000  # max tokens per source to load (for speed)
RNG_SEED = 42


def step_from_path(p: Path) -> int:
    try:
        return int(p.stem.split("-step")[1].split("-")[0])
    except (IndexError, ValueError):
        return -1


def largest_step_file(model_dir: Path) -> Optional[Path]:
    """Return the npz file with the highest step number, ignoring tmp files."""
    files = [f for f in model_dir.glob("*.npz") if not f.name.endswith(".tmp.npz")]
    if not files:
        return None
    return max(files, key=step_from_path)


def load_source(
    npz_path: Path,
    source: str,
    rng: np.random.Generator,
    n: int = SAMPLE_TOKENS,
) -> Optional[np.ndarray]:
    """
    Load a single source from an npz file and return a structured array with
    fields (instance_index, position_in_seq, max_token_id, token_id).

    Returns None if the source is missing or has no valid data.
    """
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"  WARNING: could not load {npz_path}: {e}")
        return None
    if source not in data.files:
        return None
    arr = data[source]
    required = {"instance_index", "position_in_seq", "max_token_id", "token_id"}
    if not required.issubset(set(arr.dtype.names or [])):
        return None
    if len(arr) == 0:
        return None
    # Subsample for speed; record which indices were kept so we can align top-k
    if len(arr) > n:
        idx = rng.choice(len(arr), size=n, replace=False)
        arr = arr[idx]
    return arr


def load_topk_map(
    npz_path: Path,
    source: str,
    pos_keys: np.ndarray,
) -> Optional[Dict[int, np.ndarray]]:
    """
    Load ``{source}__topk_indices`` from *npz_path* and build a mapping
    ``composite_key -> top-k token-id array`` for the positions in
    *pos_keys* (which must be the same subsample used for the main array).

    Returns None if the top-k field is absent.

    Because the main array was subsampled, we load the full structured array
    first to get the position keys, then select the rows that correspond to
    *pos_keys*.
    """
    try:
        data = np.load(npz_path)
    except Exception:
        return None
    topk_key = f"{source}__topk_indices"
    if topk_key not in data.files or source not in data.files:
        return None
    full_arr  = data[source]
    topk_arr  = data[topk_key]  # (n_full_tokens, k)
    full_keys = (
        full_arr["instance_index"].astype(np.uint64) * 65536
        + full_arr["position_in_seq"].astype(np.uint64)
    )
    key_to_row = {int(k): i for i, k in enumerate(full_keys.tolist())}
    result: Dict[int, np.ndarray] = {}
    for pk in pos_keys.tolist():
        row = key_to_row.get(int(pk))
        if row is not None:
            result[int(pk)] = topk_arr[row]
    return result if result else None


def build_pos_arrays(
    arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return three parallel arrays indexed by position key order:
      keys  : structured array of (instance_index, position_in_seq) as uint64 composite
      preds : max_token_id (uint32)
      labels: token_id (uint32)

    The composite key is  instance_index * 2^16 + position_in_seq, which is unique
    as long as position_in_seq < 65536 (guaranteed by uint16 storage).
    """
    keys = arr["instance_index"].astype(np.uint64) * 65536 + arr["position_in_seq"].astype(np.uint64)
    preds = arr["max_token_id"].astype(np.uint32)
    labels = arr["token_id"].astype(np.uint32)
    return keys, preds, labels


def sanity_check_token_ids(
    model_names: List[str],
    shared_keys: np.ndarray,
    all_key_maps: Dict[str, Dict[int, Tuple[int, int]]],
    source: str,
) -> bool:
    """
    Verify that the correct token_id is identical across all models at every
    shared position.  Returns True if the check passes, False otherwise (and
    prints details of mismatches).
    """
    ok = True
    for k in shared_keys:
        labels = [all_key_maps[m][k][1] for m in model_names]
        if len(set(labels)) > 1:
            ok = False
            detail = ", ".join(f"{m}={l}" for m, l in zip(model_names, labels))
            print(
                f"  SANITY FAIL [{source}] key={k}: token_id mismatch across models: {detail}"
            )
            break  # report first failure only to avoid flooding output
    return ok


def compare_pair(
    name_a: str,
    name_b: str,
    shared_keys: np.ndarray,
    key_map: Dict[str, Dict[int, Tuple[int, int]]],
) -> Dict[str, float]:
    """
    Compute the full suite of comparison statistics for a single pair of models
    over the pre-computed intersection of token positions.

    ``key_map[model][composite_key]`` = (max_token_id, token_id)
    """
    ma, mb = key_map[name_a], key_map[name_b]
    n = len(shared_keys)
    if n == 0:
        nan = float("nan")
        return {
            "n_shared": 0,
            "both_correct": nan,
            "A_only_correct": nan,
            "B_only_correct": nan,
            "both_wrong": nan,
            "pred_agree": nan,
            "same_error": nan,
            "jaccard_correct": nan,
            "disagreement_bias": nan,
        }

    both_correct = 0
    a_only = 0
    b_only = 0
    both_wrong = 0
    pred_agree = 0
    same_error = 0  # count of same-pred among both-wrong tokens

    for k in shared_keys:
        pred_a, label = ma[k]
        pred_b, _ = mb[k]
        a_right = pred_a == label
        b_right = pred_b == label
        same_pred = pred_a == pred_b

        if a_right and b_right:
            both_correct += 1
        elif a_right:
            a_only += 1
        elif b_right:
            b_only += 1
        else:
            both_wrong += 1
            if same_pred:
                same_error += 1

        if same_pred:
            pred_agree += 1

    union_correct = both_correct + a_only + b_only
    jaccard = both_correct / union_correct if union_correct > 0 else float("nan")
    same_error_rate = same_error / both_wrong if both_wrong > 0 else float("nan")

    return {
        "n_shared": n,
        "both_correct": both_correct / n,
        "A_only_correct": a_only / n,
        "B_only_correct": b_only / n,
        "both_wrong": both_wrong / n,
        "pred_agree": pred_agree / n,
        "same_error": same_error_rate,
        "jaccard_correct": jaccard,
        "disagreement_bias": (a_only - b_only) / n,
    }


def compare_pair_topk(
    name_a: str,
    name_b: str,
    shared_keys: List[int],
    key_map: Dict[str, Dict[int, Tuple[int, int]]],
    topk_maps: Dict[str, Optional[Dict[int, np.ndarray]]],
) -> None:
    """
    For pairs where one model is wrong and the other is right, check whether
    the correct token appears in the wrong model's top-32 list.

    Reports:
      - A wrong, B right:  is correct token in A's top-32?
      - B wrong, A right:  is correct token in B's top-32?
      - Both wrong:        do A and B share at least one token in their top-32?

    A high "correct-in-wrong-top32" rate means the wrong model is "close" —
    it considered the right answer but ranked it below argmax.  A low rate
    means the wrong model is completely off.
    """
    tk_a = topk_maps.get(name_a)
    tk_b = topk_maps.get(name_b)
    if tk_a is None and tk_b is None:
        print(f"  Top-32 overlap: top-k data unavailable for both models.")
        return

    ma, mb = key_map[name_a], key_map[name_b]

    # Counters
    a_wrong_b_right = 0
    correct_in_a_topk = 0
    b_wrong_a_right = 0
    correct_in_b_topk = 0
    both_wrong_n = 0
    topk_overlap_both_wrong = 0  # ≥1 common token in top-32 when both wrong

    for k in shared_keys:
        pred_a, label = ma[k]
        pred_b, _     = mb[k]
        a_right = (pred_a == label)
        b_right = (pred_b == label)

        if not a_right and b_right:
            a_wrong_b_right += 1
            if tk_a is not None:
                row_a = tk_a.get(k)
                if row_a is not None and label in row_a:
                    correct_in_a_topk += 1

        elif a_right and not b_right:
            b_wrong_a_right += 1
            if tk_b is not None:
                row_b = tk_b.get(k)
                if row_b is not None and label in row_b:
                    correct_in_b_topk += 1

        elif not a_right and not b_right:
            both_wrong_n += 1
            if tk_a is not None and tk_b is not None:
                row_a = tk_a.get(k)
                row_b = tk_b.get(k)
                if row_a is not None and row_b is not None:
                    if len(set(row_a.tolist()) & set(row_b.tolist())) > 0:
                        topk_overlap_both_wrong += 1

    print(f"\n  Top-32 overlap analysis  ({name_a} vs {name_b}):")
    if a_wrong_b_right > 0 and tk_a is not None:
        r = correct_in_a_topk / a_wrong_b_right
        print(f"    A wrong, B right  (n={a_wrong_b_right:,}): "
              f"correct tok in A's top-32: {r:.1%}  "
              f"({'A is close, just ranked wrong' if r >= 0.5 else 'A is far off'})")
    if b_wrong_a_right > 0 and tk_b is not None:
        r = correct_in_b_topk / b_wrong_a_right
        print(f"    B wrong, A right  (n={b_wrong_a_right:,}): "
              f"correct tok in B's top-32: {r:.1%}  "
              f"({'B is close, just ranked wrong' if r >= 0.5 else 'B is far off'})")
    if both_wrong_n > 0 and tk_a is not None and tk_b is not None:
        r = topk_overlap_both_wrong / both_wrong_n
        print(f"    Both wrong        (n={both_wrong_n:,}): "
              f"shared token in top-32: {r:.1%}  "
              f"({'similar errors' if r >= 0.5 else 'different errors'})")


def short_name(model: str) -> str:
    """Strip common prefix/org for display."""
    return model.replace("allenai_", "").replace("allenai/", "")


def fmt(v: float) -> str:
    if np.isnan(v):
        return "    n/a"
    return f"{v:>7.3%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-root",
        type=Path,
        default=DUMP_ROOT,
        help="Root directory containing per-model subdirectories of .npz dumps.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of model directory names to compare (default: all).",
    )
    parser.add_argument(
        "--sample-tokens",
        type=int,
        default=SAMPLE_TOKENS,
        help="Max tokens to sample per source per model for speed.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(RNG_SEED)

    dump_root: Path = args.dump_root
    all_models = sorted(p.name for p in dump_root.iterdir() if p.is_dir())
    if args.models:
        all_models = [m for m in all_models if m in args.models]

    # ------------------------------------------------------------------
    # Discover largest-step file per model
    # ------------------------------------------------------------------
    model_files: Dict[str, Path] = {}
    for model in all_models:
        f = largest_step_file(dump_root / model)
        if f is None:
            print(f"Skipping {model}: no .npz dumps found")
            continue
        model_files[model] = f

    if len(model_files) < 2:
        print("Need at least 2 models with dumps to compare. Exiting.")
        return

    print("\nModels selected (largest available step):")
    for model, f in model_files.items():
        print(f"  {short_name(model):<40}  step {step_from_path(f):>10,}  ({f.name})")

    # ------------------------------------------------------------------
    # Collect all sources present across models
    # ------------------------------------------------------------------
    all_sources: set = set()
    for f in model_files.values():
        try:
            data = np.load(f)
            all_sources.update(k for k in data.files if "__topk" not in k)
        except Exception:
            pass
    all_sources_sorted = sorted(all_sources)

    # ------------------------------------------------------------------
    # Per-source comparison
    # ------------------------------------------------------------------
    print(f"\nSample tokens per model per source: {args.sample_tokens:,}")
    print(f"Alignment: (instance_index, position_in_seq) intersection\n")

    for source in all_sources_sorted:
        # Load arrays for every model that has this source
        raw: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for model, f in model_files.items():
            arr = load_source(f, source, rng, n=args.sample_tokens)
            if arr is not None:
                raw[short_name(model)] = build_pos_arrays(arr)

        if len(raw) < 2:
            continue

        # Build fast lookup: model -> {composite_key: (pred, label)}
        key_map: Dict[str, Dict[int, Tuple[int, int]]] = {}
        key_sets: List[set] = []
        for name, (keys, preds, labels) in raw.items():
            key_map[name] = dict(zip(keys.tolist(), zip(preds.tolist(), labels.tolist())))
            key_sets.append(set(keys.tolist()))

        # Intersection across ALL models (important: ensures same data order for all)
        shared_key_set = key_sets[0]
        for ks in key_sets[1:]:
            shared_key_set = shared_key_set & ks
        shared_keys = list(shared_key_set)

        model_names = list(raw.keys())

        # Load top-k maps for overlap analysis (keyed by short model name)
        topk_maps: Dict[str, Optional[Dict[int, np.ndarray]]] = {}
        for name, (keys, _, _) in raw.items():
            full_name = next((m for m in model_files if short_name(m) == name), None)
            if full_name is None:
                topk_maps[name] = None
                continue
            topk_maps[name] = load_topk_map(model_files[full_name], source, keys)

        # ------------------------------------------------------------------
        # Sanity check: token_id must agree at every shared position
        # ------------------------------------------------------------------
        passed = sanity_check_token_ids(model_names, shared_keys, key_map, source)

        print(f"{'='*70}")
        print(f"Source: {source}   (intersection: {len(shared_keys):,} tokens)")
        sanity_str = "PASS" if passed else "FAIL — token_id mismatch (data order differs!)"
        print(f"Sanity check (token_id agreement): {sanity_str}")
        print(f"{'='*70}")

        # Per-model coverage (what fraction of their tokens are in the shared set)
        print("  Coverage (fraction of each model's tokens in the shared intersection):")
        for name, (keys, _, _) in raw.items():
            pct = len(shared_key_set & set(keys.tolist())) / len(keys) if len(keys) else float("nan")
            print(f"    {name:<40}  {len(keys):>10,} tokens  {pct:>7.1%} in intersection")

        # Per-model top-1 accuracy (over each model's own full sample)
        print("  Top-1 accuracy (each model's own sample):")
        for name, (_, preds, labels) in raw.items():
            acc = (preds == labels).mean()
            print(f"    {name:<40}  {acc:>7.3%}")

        # Pairwise stats
        for name_a, name_b in combinations(model_names, 2):
            stats = compare_pair(name_a, name_b, shared_keys, key_map)
            n = stats["n_shared"]
            print(f"\n  --- {name_a}  vs  {name_b}  (n={n:,}) ---")
            print(f"    {'both_correct (A✓ ∧ B✓)':<45}  {fmt(stats['both_correct'])}")
            print(f"    {'A_only_correct (A✓ ∧ B✗)':<45}  {fmt(stats['A_only_correct'])}")
            print(f"    {'B_only_correct (A✗ ∧ B✓)':<45}  {fmt(stats['B_only_correct'])}")
            print(f"    {'both_wrong (A✗ ∧ B✗)':<45}  {fmt(stats['both_wrong'])}")
            print(f"    {'pred_agree (same prediction, right or wrong)':<45}  {fmt(stats['pred_agree'])}")
            print(f"    {'same_error (same wrong pred | both wrong)':<45}  {fmt(stats['same_error'])}")
            print(f"    {'jaccard_correct (|A∩B| / |A∪B| correct sets)':<45}  {fmt(stats['jaccard_correct'])}")
            bias = stats["disagreement_bias"]
            bias_str = f"{bias:>+8.3%}" if not np.isnan(bias) else "    n/a"
            print(f"    {'disagreement_bias (A_only - B_only, +=>A better)':<45}  {bias_str}")
            compare_pair_topk(name_a, name_b, shared_keys, key_map, topk_maps)

        print()


if __name__ == "__main__":
    main()
