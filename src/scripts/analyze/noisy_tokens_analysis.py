"""
Analyse training instability of token-level predictions across checkpoints.

For each token position tracked through a model's training trajectory, we
measure how often the model's top-1 prediction flips between correct and
incorrect.  Positions with many flips are "noisy" \u2014 the model repeatedly
acquires and forgets them, or oscillates around the decision boundary.  This
is distinct from positions that stay wrong (hard tokens) or stay correct
(trivially easy tokens).

Per-position metrics computed:
  n_acq      : wrong\u2192right transitions (acquisition events)
  n_forget   : right\u2192wrong transitions (forgetting / regression events)
  instability: n_acq + n_forget  (total correctness flips)
  ever_correct: whether the model was ever correct on this position
  ever_wrong  : whether the model was ever wrong on this position
  permanently_acquired : correct from some step onwards and never wrong again
  permanently_wrong    : never correct at any checkpoint

Categories:
  stable-correct     : permanently_acquired=True, n_forget=0
  stable-wrong       : permanently_wrong=True
  noisy              : instability >= 2  (flips at least twice)
  memorised-then-lost: n_acq >= 1 and permanently_acquired=False and
                       the last checkpoint is wrong  (acquired then regressed)
  gradual             : n_acq == 1 and n_forget == 0  (clean single acquisition,
                        may or may not be permanent depending on sustain check)

Outputs:
  1. Per-model summary: fraction of positions in each category.
  2. Top-N most unstable token IDs (by mean instability across positions
     where that token ID is the correct answer).
  3. Instability distribution histogram.
  4. Forgetting rate over training: at each checkpoint, what fraction of
     previously-correct tokens are now wrong (regression rate).

Usage:
    # Single model
    python src/scripts/analyze/noisy_tokens_analysis.py \\
        --model allenai_OLMo-2-0425-1B --source c4_en

    # All models, specific source
    python src/scripts/analyze/noisy_tokens_analysis.py --source dolma_stack

    # More checkpoints for finer resolution
    python src/scripts/analyze/noisy_tokens_analysis.py \\
        --model allenai_Olmo-3-1025-7B --max-checkpoints 80

    # Save forgetting-rate plots
    python src/scripts/analyze/noisy_tokens_analysis.py \\
        --model allenai_OLMo-2-0425-1B --plot-dir /tmp/noisy_plots
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DUMP_ROOT = Path("/weka/oe-training-default/yashasbls/georges-functional-analysis")
RNG_SEED = 42
DEFAULT_MAX_CKPTS = 50
TOP_N_TOKENS = 30          # most instable token IDs to print
MIN_OCCURRENCES = 5        # min positions with that token_id to include in top-N


# ---------------------------------------------------------------------------
# Shared helpers (mirror of phase_transition_analysis.py; kept self-contained)
# ---------------------------------------------------------------------------

def step_from_path(p: Path) -> int:
    try:
        return int(p.stem.split("-step")[1].split("-")[0])
    except (IndexError, ValueError):
        return -1


def list_checkpoints(model_dir: Path) -> List[Path]:
    return sorted(
        [f for f in model_dir.glob("*.npz") if not f.name.endswith(".tmp.npz")],
        key=step_from_path,
    )


def pick_evenly_spaced(paths: List[Path], n: int) -> List[Path]:
    if len(paths) <= n:
        return paths
    indices = [round(i * (len(paths) - 1) / (n - 1)) for i in range(n)]
    return [paths[i] for i in sorted(set(indices))]


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            "allenai/OLMo-2-0425-1B", trust_remote_code=True
        )
    except Exception:
        return None


def decode_token(tid: int, tokenizer) -> str:
    if tokenizer is None:
        return f"<id:{tid}>"
    try:
        return tokenizer.decode([tid])
    except Exception:
        return f"<id:{tid}>"


def build_trajectory(
    checkpoints: List[Path],
    source: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Build binary correctness and argmax prediction trajectories across checkpoints.

    :returns:
        ``correct_matrix`` — (n_tokens, n_steps) bool

        ``pred_matrix``    — (n_tokens, n_steps) uint32  argmax token at each step
                             (needed for confusion analysis)

        ``token_ids``       — (n_tokens,) correct token at each position

        ``pos_keys``        — (n_tokens,) uint64 composite keys

        ``steps``           — list of int step numbers
    """
    ref_data = np.load(checkpoints[0])
    if source not in ref_data.files:
        raise KeyError(f"Source '{source}' not in {checkpoints[0]}")
    ref_arr   = ref_data[source]
    pos_keys  = (
        ref_arr["instance_index"].astype(np.uint64) * 65536
        + ref_arr["position_in_seq"].astype(np.uint64)
    )
    token_ids = ref_arr["token_id"].astype(np.uint32)
    n_tokens  = len(pos_keys)
    key_to_idx = {k: i for i, k in enumerate(pos_keys.tolist())}

    n_steps = len(checkpoints)
    correct_matrix = np.zeros((n_tokens, n_steps), dtype=bool)
    pred_matrix    = np.zeros((n_tokens, n_steps), dtype=np.uint32)
    steps: List[int] = []

    for t, ckpt in enumerate(checkpoints):
        steps.append(step_from_path(ckpt))
        data = np.load(ckpt)
        if source not in data.files:
            continue
        arr = data[source]
        keys       = (
            arr["instance_index"].astype(np.uint64) * 65536
            + arr["position_in_seq"].astype(np.uint64)
        ).tolist()
        preds_col  = arr["max_token_id"].astype(np.uint32).tolist()
        labels_col = arr["token_id"].astype(np.uint32).tolist()
        for k, pred, label in zip(keys, preds_col, labels_col):
            idx = key_to_idx.get(k)
            if idx is not None:
                correct_matrix[idx, t] = (pred == label)
                pred_matrix[idx, t]    = pred
        if (t + 1) % 10 == 0 or t == n_steps - 1:
            print(f"    loaded {t+1}/{n_steps} checkpoints", end="\r", flush=True)

    print()
    return correct_matrix, pred_matrix, token_ids, pos_keys, steps


# ---------------------------------------------------------------------------
# Instability metrics
# ---------------------------------------------------------------------------

def compute_instability(
    correct_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-token instability metrics from a binary correctness matrix.

    :param correct_matrix: ``(n_tokens, n_steps)`` bool

    :returns: ``(n_acq, n_forget, instability, permanently_acquired, permanently_wrong)``

        ``n_acq``                \u2014 (n_tokens,) int: wrong\u2192right flips

        ``n_forget``             \u2014 (n_tokens,) int: right\u2192wrong flips

        ``instability``          \u2014 (n_tokens,) int: n_acq + n_forget

        ``permanently_acquired`` \u2014 (n_tokens,) bool: correct from some step t*
                                  onwards with no subsequent wrong step

        ``permanently_wrong``    \u2014 (n_tokens,) bool: never correct at any step
    """
    n_tokens, n_steps = correct_matrix.shape

    # Consecutive-step transitions
    # diff[i, t] = correct[i, t+1] - correct[i, t]  in {-1, 0, +1}
    diff = np.diff(correct_matrix.astype(np.int8), axis=1)  # (n_tokens, n_steps-1)
    n_acq    = (diff ==  1).sum(axis=1).astype(np.int32)
    n_forget = (diff == -1).sum(axis=1).astype(np.int32)
    instability = n_acq + n_forget

    # Permanently acquired: suffix from first correct step is all-correct
    # Equivalently: n_forget == 0 AND the token is correct at least once
    ever_correct = correct_matrix.any(axis=1)
    permanently_wrong     = ~ever_correct
    permanently_acquired  = ever_correct & (n_forget == 0)

    return n_acq, n_forget, instability, permanently_acquired, permanently_wrong


def forgetting_rate_over_time(
    correct_matrix: np.ndarray,
    steps: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    At each step t, compute the fraction of tokens that were correct at t-1
    but wrong at t (instantaneous forgetting / regression rate), and its
    step-gap-normalised counterpart (rate per 1,000 training steps).

    Normalising by the gap between adjacent checkpoints removes the confound
    that larger checkpoint gaps naturally produce more forgetting events.

    :returns:
        ``rates``      — (n_steps,) raw forget fraction; NaN at first step.

        ``rates_norm`` — (n_steps,) forget fraction per 1,000 training steps;
                         NaN at first step.
    """
    n_tokens, n_steps = correct_matrix.shape
    rates      = np.full(n_steps, float("nan"))
    rates_norm = np.full(n_steps, float("nan"))

    step_arr  = np.array(steps, dtype=np.float64)
    step_gaps = np.diff(step_arr, prepend=step_arr[0])  # gap[0] = 0
    step_gaps = np.maximum(step_gaps, 1.0)               # avoid div-by-zero

    for t in range(1, n_steps):
        was_correct    = correct_matrix[:, t - 1]
        now_wrong      = ~correct_matrix[:, t]
        n_prev_correct = was_correct.sum()
        if n_prev_correct > 0:
            r             = (was_correct & now_wrong).sum() / n_prev_correct
            rates[t]      = r
            rates_norm[t] = r / step_gaps[t] * 1000.0

    return rates, rates_norm


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def category_summary(
    n_acq: np.ndarray,
    n_forget: np.ndarray,
    instability: np.ndarray,
    permanently_acquired: np.ndarray,
    permanently_wrong: np.ndarray,
    correct_matrix: np.ndarray,
) -> None:
    """Print a breakdown of positions by stability category."""
    n_total = len(n_acq)

    # Categories (mutually exclusive in priority order)
    stable_correct = permanently_acquired
    stable_wrong   = permanently_wrong
    # "noisy": flips at least twice regardless of end state
    noisy          = instability >= 2
    # clean single acquisition that may or may not be permanent
    gradual        = (n_acq == 1) & (n_forget == 0) & ~permanently_acquired
    # acquired but ultimately regressed: last step is wrong
    last_wrong     = ~correct_matrix[:, -1]
    memo_lost      = (n_acq >= 1) & last_wrong & ~permanently_wrong

    cats = [
        ("stable-correct (perm. acquired, never wrong after)",  stable_correct),
        ("stable-wrong   (never correct at any step)",           stable_wrong),
        ("noisy          (instability \u2265 2 flips)",                noisy),
        ("gradual        (1 acq, 0 forget, not yet perm.)",      gradual),
        ("memorised-then-lost (acquired but last step wrong)",   memo_lost),
    ]

    print(f"\n  Category breakdown ({n_total:,} positions):")
    for label, mask in cats:
        n = int(mask.sum())
        print(f"    {label:<55}  {n:>9,}  ({n/n_total:.1%})")

    # Uncategorised
    categorised = stable_correct | stable_wrong | noisy | gradual | memo_lost
    n_other = int((~categorised).sum())
    print(f"    {'other (wrong at last step, instability=1)':<55}  "
          f"{n_other:>9,}  ({n_other/n_total:.1%})")


def top_unstable_token_ids(
    token_ids: np.ndarray,
    instability: np.ndarray,
    tokenizer,
    top_n: int = TOP_N_TOKENS,
    min_occ: int = MIN_OCCURRENCES,
) -> None:
    """
    Report token IDs with highest mean instability across all positions
    where that token is the correct answer.
    """
    # Group instability by token_id
    tid_instab: Dict[int, List[int]] = {}
    for tid, inst in zip(token_ids.tolist(), instability.tolist()):
        tid_instab.setdefault(tid, []).append(inst)

    # Filter to token IDs with enough occurrences, then rank by mean
    ranked = sorted(
        [(tid, vals) for tid, vals in tid_instab.items() if len(vals) >= min_occ],
        key=lambda x: -np.mean(x[1]),
    )[:top_n]

    print(f"\n  Top-{top_n} most unstable token IDs "
          f"(mean instability, min {min_occ} occurrences):")
    print(f"  {'token_id':>10}  {'n_pos':>7}  {'mean_inst':>10}  "
          f"{'pct_noisy':>10}  decoded")
    print(f"  {'-'*10}  {'-'*7}  {'-'*10}  {'-'*10}  -------")
    for tid, vals in ranked:
        arr    = np.array(vals)
        mean_i = arr.mean()
        pct_n  = (arr >= 2).mean()
        dec    = decode_token(tid, tokenizer)
        dec_d  = repr(dec) if len(dec.strip()) == 0 else dec
        print(f"  {tid:>10}  {len(vals):>7,}  {mean_i:>10.3f}  "
              f"{pct_n:>10.1%}  {dec_d}")


def instability_histogram(instability: np.ndarray) -> None:
    """Print an ASCII histogram of instability counts."""
    max_val = int(instability.max()) if len(instability) > 0 else 0
    counts  = np.bincount(instability, minlength=max_val + 1)
    n_total = len(instability)
    bar_w   = 40
    max_c   = counts.max() if counts.max() > 0 else 1

    print(f"\n  Instability histogram (total transitions per token position):")
    print(f"  {'flips':>6}  {'count':>9}  {'pct':>7}  bar")
    print(f"  {'-'*6}  {'-'*9}  {'-'*7}  ---")
    # Print up to value 20; group the rest
    cutoff = 20
    for v in range(min(cutoff + 1, len(counts))):
        c   = counts[v]
        bar = "#" * int(c / max_c * bar_w)
        print(f"  {v:>6}  {c:>9,}  {c/n_total:>7.2%}  |{bar}")
    if len(counts) > cutoff + 1:
        tail = counts[cutoff + 1:].sum()
        print(f"  {f'>{cutoff}':>6}  {tail:>9,}  {tail/n_total:>7.2%}")


def confusion_report(
    correct_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    token_ids: np.ndarray,
    tokenizer,
    top_n: int = 20,
    min_occ: int = 5,
) -> None:
    """
    For each noisy token position (instability >= 2), collect every
    (correct_token, predicted_token) pair at each step where the model was
    wrong.  Aggregate into a confusion-pair frequency table.

    Answers: "When the model is wrong on a noisy position, what does it
    predict instead?"  The top confusion pairs reveal systematic confusions
    (e.g. same surface form, related syntax) versus random errors.
    """
    from collections import defaultdict
    n_tokens, n_steps = correct_matrix.shape
    diff = np.abs(np.diff(correct_matrix.astype(np.int8), axis=1))
    instability = diff.sum(axis=1)

    # Only look at noisy positions
    noisy_mask = instability >= 2
    noisy_idx  = np.where(noisy_mask)[0]

    # Vectorised: collect all wrong-step (correct_tok, predicted_tok) pairs at once.
    # This avoids a slow O(n_noisy * n_steps) Python loop.
    noisy_correct = correct_matrix[noisy_idx]       # (n_noisy, n_steps)
    ni, ti        = np.where(~noisy_correct)         # indices within noisy subset
    global_idx    = noisy_idx[ni]
    flat_correct  = token_ids[global_idx]
    flat_wrong    = pred_matrix[global_idx, ti]
    valid         = flat_wrong != flat_correct       # guard: pred should differ from label
    flat_correct  = flat_correct[valid]
    flat_wrong    = flat_wrong[valid]

    pair_counts   = Counter(zip(flat_correct.tolist(), flat_wrong.tolist()))
    ranked        = pair_counts.most_common()

    # Decode
    print(f"\n  Top-{top_n} confusion pairs on noisy positions")
    print(f"  (correct → predicted, counted over all wrong steps × positions):")
    print(f"  {'correct':>12}  {'predicted':>12}  {'count':>9}  "
          f"decoded (correct → predicted)")
    print(f"  {'-'*12}  {'-'*12}  {'-'*9}  ---")

    def _dec(tid: int) -> str:
        if tokenizer is None:
            return f"<{tid}>"
        try:
            s = tokenizer.decode([tid])
            return repr(s) if len(s.strip()) == 0 else s
        except Exception:
            return f"<{tid}>"

    for (c_tid, p_tid), cnt in ranked[:top_n]:
        print(f"  {c_tid:>12}  {p_tid:>12}  {cnt:>9,}  "
              f"{_dec(c_tid)} → {_dec(p_tid)}")

    # Also report: for the top-5 most confused correct tokens,
    # what are the most common wrong predictions?
    correct_wrong: Dict[int, Counter] = {}
    for (c_tid, p_tid), cnt in pair_counts.items():
        correct_wrong.setdefault(c_tid, Counter())[p_tid] += cnt

    top_confused_correct = sorted(
        correct_wrong.items(), key=lambda x: -sum(x[1].values())
    )[:5]

    print(f"\n  Per-token confusion breakdown (top-5 most confused correct tokens):")
    for c_tid, pred_counter in top_confused_correct:
        total_wrong = sum(pred_counter.values())
        top_preds = pred_counter.most_common(3)
        preds_str = ";  ".join(
            f"{_dec(p)} ×{c}" for p, c in top_preds
        )
        print(f"    correct={_dec(c_tid)!r:>15}  total_wrong={total_wrong:>7,}  "
              f"top predictions: {preds_str}")


def forgetting_rate_report(
    rates: np.ndarray,
    rates_norm: np.ndarray,
    steps: List[int],
    bar_w: int = 40,
) -> None:
    """
    Print the per-checkpoint forgetting rate alongside the gap-normalised rate
    (per 1,000 training steps).  The normalised column removes the confound
    that coarsely-spaced checkpoints span more training steps and therefore
    accumulate more forgetting events even if the per-step rate is identical.
    """
    valid_norm = rates_norm[~np.isnan(rates_norm)]
    max_r_norm = float(valid_norm.max()) if len(valid_norm) > 0 else 1.0
    print(f"\n  Forgetting rate per checkpoint  "
          f"(raw = fraction of prev-correct tokens lost; "
          f"norm = raw / gap * 1000 steps):")
    print(f"  {'step':>12}  {'gap':>8}  {'raw':>10}  {'norm/1k':>10}  bar (norm)")
    print(f"  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*10}  ----------")
    step_arr  = np.array(steps, dtype=np.float64)
    step_gaps = np.maximum(np.diff(step_arr, prepend=step_arr[0]), 1.0)
    for t, (step, r, rn, gap) in enumerate(
        zip(steps, rates, rates_norm, step_gaps)
    ):
        if np.isnan(r):
            continue
        bar = "#" * int(rn / max_r_norm * bar_w) if not np.isnan(rn) else ""
        rn_str = f"{rn:.6f}" if not np.isnan(rn) else "    n/a"
        print(f"  {step:>12,}  {int(gap):>8,}  {r:>10.4%}  {rn_str:>10}  |{bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Single model directory name under --dump-root.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Multiple model directory names (runs each in turn).",
    )
    parser.add_argument(
        "--dump-root",
        type=Path,
        default=DUMP_ROOT,
    )
    parser.add_argument(
        "--source",
        default="c4_en",
        help="Data source to analyse (default: c4_en).",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=DEFAULT_MAX_CKPTS,
        help="Max evenly-spaced checkpoints to load per model.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_TOKENS,
        help="Number of most-unstable token IDs to print.",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip tokenizer loading (show raw token IDs only).",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="If set, save forgetting-rate and instability plots as PNG files.",
    )
    args = parser.parse_args()

    if args.model is None and not args.models:
        args.models = sorted(p.name for p in args.dump_root.iterdir() if p.is_dir())

    model_list = args.models or ([args.model] if args.model else [])
    tokenizer  = None if args.no_tokenizer else load_tokenizer()
    if tokenizer is not None:
        print("Tokenizer loaded.")
    else:
        print("Tokenizer not available \u2014 showing raw token IDs.")

    for model_name in model_list:
        model_dir = args.dump_root / model_name
        if not model_dir.is_dir():
            print(f"Skipping {model_name}: directory not found")
            continue

        all_ckpts = list_checkpoints(model_dir)
        if not all_ckpts:
            print(f"Skipping {model_name}: no checkpoints")
            continue

        selected = pick_evenly_spaced(all_ckpts, args.max_checkpoints)
        short    = model_name.replace("allenai_", "")

        print(f"\n{'='*70}")
        print(f"Model: {short}  ({len(selected)} checkpoints, source: {args.source})")
        print(f"  Step range: {step_from_path(selected[0]):,} "
              f"\u2192 {step_from_path(selected[-1]):,}")
        print(f"{'='*70}")

        try:
            print("  Building trajectories...")
            correct_matrix, pred_matrix, token_ids, pos_keys, steps = build_trajectory(
                selected, args.source
            )
        except KeyError as e:
            print(f"  Skipping: {e}")
            continue

        n_tokens, n_steps = correct_matrix.shape
        print(f"  {n_tokens:,} token positions × {n_steps} checkpoints")

        # Core metrics
        n_acq, n_forget, instability, perm_acq, perm_wrong = compute_instability(
            correct_matrix
        )

        mean_inst = instability.mean()
        pct_noisy = (instability >= 2).mean()
        print(f"  Mean instability (flips/position): {mean_inst:.3f}")
        print(f"  Fraction noisy (≥2 flips):         {pct_noisy:.1%}")

        # Category summary
        category_summary(n_acq, n_forget, instability, perm_acq, perm_wrong,
                         correct_matrix)

        # Instability histogram
        instability_histogram(instability)

        # Forgetting rate over training
        rates, rates_norm = forgetting_rate_over_time(correct_matrix, steps)
        forgetting_rate_report(rates, rates_norm, steps)

        # Top unstable token IDs
        top_unstable_token_ids(token_ids, instability, tokenizer, top_n=args.top_n)

        # Confusion analysis on noisy positions
        confusion_report(correct_matrix, pred_matrix, token_ids, tokenizer)

        # Optional plots
        if args.plot_dir:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                step_arr = np.array(steps)
                fig, axes = plt.subplots(2, 1, figsize=(14, 9))

                # Panel 1: forgetting rate over time
                valid_t    = ~np.isnan(rates)
                axes[0].plot(step_arr[valid_t], rates[valid_t] * 100, color="crimson")
                axes[0].set_xlabel("Training step")
                axes[0].set_ylabel("Forgetting rate (%)")
                axes[0].set_title(f"{short} \u2014 {args.source}: forgetting rate per step")
                axes[0].grid(True, alpha=0.3)

                # Panel 2: instability histogram
                max_inst   = min(int(instability.max()), 20)
                bins       = np.arange(max_inst + 2) - 0.5
                axes[1].hist(instability.clip(0, max_inst), bins=bins,
                             color="steelblue", alpha=0.8)
                axes[1].set_xlabel("Instability (total correctness flips)")
                axes[1].set_ylabel("# token positions")
                axes[1].set_title(
                    f"{short} \u2014 {args.source}: instability distribution"
                )
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                args.plot_dir.mkdir(parents=True, exist_ok=True)
                out = args.plot_dir / f"{model_name}_{args.source}_noisy.png"
                plt.savefig(out, dpi=120)
                plt.close()
                print(f"\n  Plot saved: {out}")
            except ImportError:
                print("  matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
