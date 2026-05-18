"""
Detect phase transitions in token-level prediction accuracy across training.

For a given model, loads evenly-spaced checkpoints and builds a binary
accuracy trajectory for every token position: correct[t] ∈ {0, 1} at each
training step t.

For each token we find its "acquisition step": the earliest training step
from which the model is correct on at least MIN_SUSTAIN_FRAC of all
subsequent checkpoints (i.e. it has durably learned the token from that
point on).

The distribution of acquisition steps across all tokens reveals **phase
transitions**: bursts of simultaneous acquisition.  The script:
  1. Plots (or prints) the acquisition-step histogram and highlights peaks.
  2. For each peak step, shows the most common acquired token IDs (decoded
     with the OLMo tokenizer if available).
  3. Optionally compares acquisition timing across multiple model families
     for the same token positions, to find cross-architecture phase
     transitions.

Usage:
    # Single model, all sources
    python src/scripts/analyze/phase_transition_analysis.py \\
        --model allenai_OLMo-2-0425-1B

    # Specific source, more checkpoints
    python src/scripts/analyze/phase_transition_analysis.py \\
        --model allenai_OLMo-2-0425-1B --source c4_en --max-checkpoints 80

    # Cross-model comparison (uses shared token positions)
    python src/scripts/analyze/phase_transition_analysis.py \\
        --models allenai_OLMo-2-0425-1B allenai_Olmo-3-1025-7B \\
        --source c4_en --cross-model

    # Save plots
    python src/scripts/analyze/phase_transition_analysis.py \\
        --model allenai_OLMo-2-0425-1B --plot-dir /tmp/phase_plots
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DUMP_ROOT = Path("/weka/oe-training-default/yashasbls/georges-functional-analysis")
RNG_SEED = 42

# Fraction of checkpoints after a candidate acquisition step that must be
# correct for the acquisition to be considered "durable".
MIN_SUSTAIN_FRAC = 0.75

# How many evenly-spaced checkpoints to load per model (memory vs resolution).
DEFAULT_MAX_CKPTS = 50

# Number of phase transition peaks to show in detail.
TOP_PEAKS = 5

# For each peak, how many sample tokens to decode/show.
TOKENS_PER_PEAK = 20

# Smoothing window (in checkpoints) for the acquisition-rate curve.
SMOOTH_WINDOW = 3

# ---------------------------------------------------------------------------
# Trivial-token mask
# ---------------------------------------------------------------------------
#
# Two orthogonal criteria for excluding "trivially predictable" positions:
#
#   Mask A — early acquisition:
#     Positions where the model is already correct at any checkpoint with
#     step ≤ EARLY_STEP_THRESHOLD.  These were essentially "solved" before
#     meaningful capability learning started and likely reflect common
#     grammatical patterns or very high-frequency vocabulary items.
#     For cross-model analysis the mask can be tightened to positions where
#     ALL model families are correct early.
#
#   Mask B — sequence-start tokens:
#     The first MIN_POSITION_IN_SEQ tokens of every instance
#     (position_in_seq < MIN_POSITION_IN_SEQ).  These appear immediately
#     after the BOS marker and are heavily constrained by document-level
#     priming; they almost always flip correct very early and inflate the
#     early part of the acquisition distribution.
#
# NOTE: The mask is COMPUTED but NOT APPLIED by default.  Pass --apply-mask
#       to filter the token set used for phase-transition analysis.
#       Mask stats are always printed so you can inspect coverage.
#       Rationale for keeping it optional: the cross-model cluster-transfer
#       analysis (point 3 in the design doc) naturally de-emphasises trivial
#       tokens because a random-looking per-model acquisition distribution
#       signals noise regardless of masking.

EARLY_STEP_THRESHOLD = 50_000   # steps; positions acquired here are "trivially easy"
MIN_POSITION_IN_SEQ  = 10       # ignore first N tokens in every instance


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def step_from_path(p: Path) -> int:
    try:
        return int(p.stem.split("-step")[1].split("-")[0])
    except (IndexError, ValueError):
        return -1


def tokens_from_path(p: Path) -> Optional[int]:
    """Extract tokens-seen from filename if present (e.g. tokens2077B)."""
    try:
        part = [x for x in p.stem.split("-") if x.startswith("tokens")][0]
        val = float(part.replace("tokens", "").replace("B", "")) * 1e9
        return int(val)
    except Exception:
        return None


def list_checkpoints(model_dir: Path) -> List[Path]:
    files = sorted(
        [f for f in model_dir.glob("*.npz") if not f.name.endswith(".tmp.npz")],
        key=step_from_path,
    )
    return files


def pick_evenly_spaced(paths: List[Path], n: int) -> List[Path]:
    if len(paths) <= n:
        return paths
    indices = [round(i * (len(paths) - 1) / (n - 1)) for i in range(n)]
    return [paths[i] for i in sorted(set(indices))]


# ---------------------------------------------------------------------------
# Tokenizer (optional)
# ---------------------------------------------------------------------------

def load_tokenizer():
    """Try to load the OLMo / GPT-NeoX tokenizer.  Returns None on failure."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "allenai/OLMo-2-0425-1B",
            trust_remote_code=True,
        )
        return tok
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        return tok
    except Exception:
        return None


def decode_tokens(token_ids: List[int], tokenizer) -> List[str]:
    if tokenizer is None:
        return [f"<id:{tid}>" for tid in token_ids]
    try:
        return [tokenizer.decode([tid]) for tid in token_ids]
    except Exception:
        return [f"<id:{tid}>" for tid in token_ids]


# ---------------------------------------------------------------------------
# Trivial-token mask helpers
# ---------------------------------------------------------------------------

def build_trivial_mask(
    correct_matrix: np.ndarray,
    pos_keys: np.ndarray,
    steps: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute trivial-token masks for a single model's trajectory.

    :param correct_matrix: ``(n_tokens, n_steps)`` bool
    :param pos_keys: ``(n_tokens,)`` uint64 composite keys
    :param steps: list of int step numbers (one per checkpoint)
    :returns: ``(mask_trivial, mask_A, mask_B)``

        ``mask_trivial`` — True where the token *should be excluded* (A ∨ B)

        ``mask_A`` — True where correct at any checkpoint ≤ EARLY_STEP_THRESHOLD

        ``mask_B`` — True where position_in_seq < MIN_POSITION_IN_SEQ
    """
    # --- Mask A: early acquisition ---
    early_indices = [i for i, s in enumerate(steps) if s <= EARLY_STEP_THRESHOLD]
    if early_indices:
        # Correct at ANY early checkpoint; generous — captures anything trivial
        # by EARLY_STEP_THRESHOLD regardless of whether it later regresses.
        mask_A = correct_matrix[:, early_indices].any(axis=1)
    else:
        # No checkpoints before threshold — fall back to the very first checkpoint.
        mask_A = correct_matrix[:, 0].copy()

    # --- Mask B: first tokens in every instance ---
    # Composite key: inst_index * 65536 + position_in_seq
    positions = (pos_keys % 65536).astype(np.uint16)
    mask_B = positions < MIN_POSITION_IN_SEQ

    mask_trivial = mask_A | mask_B
    return mask_trivial, mask_A, mask_B


def print_mask_stats(
    mask_trivial: np.ndarray,
    mask_A: np.ndarray,
    mask_B: np.ndarray,
    n_total: int,
) -> None:
    """Print coverage statistics for the trivial-token mask."""
    n_A       = int(mask_A.sum())
    n_B       = int(mask_B.sum())
    n_both    = int((mask_A & mask_B).sum())
    n_trivial = int(mask_trivial.sum())
    print(f"\n  Trivial-token mask stats (NOT applied unless --apply-mask):")
    print(f"    Mask A (correct at any step ≤ {EARLY_STEP_THRESHOLD:,}): "
          f"{n_A:>8,} / {n_total:,}  ({n_A/n_total:.1%})")
    print(f"    Mask B (position_in_seq < {MIN_POSITION_IN_SEQ}):          "
          f"{n_B:>8,} / {n_total:,}  ({n_B/n_total:.1%})")
    print(f"    Both A ∧ B:                                   "
          f"{n_both:>8,} / {n_total:,}  ({n_both/n_total:.1%})")
    print(f"    Combined (A ∨ B) — would be excluded:         "
          f"{n_trivial:>8,} / {n_total:,}  ({n_trivial/n_total:.1%})")
    print(f"    Remaining for analysis:                       "
          f"{n_total - n_trivial:>8,} / {n_total:,}  "
          f"({(n_total - n_trivial)/n_total:.1%})")


# ---------------------------------------------------------------------------
# Core trajectory builder
# ---------------------------------------------------------------------------

TOPK_ABSENT_RANK = 33  # sentinel rank when correct token not found in top-k


def build_trajectory(
    checkpoints: List[Path],
    source: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Load ``source`` from each checkpoint and build per-token accuracy,
    probability, and correct-token rank trajectories.

    When ``{source}__topk_indices`` is present in the dumps (top-32
    predictions), we also record the 1-based rank of the correct token
    within the top-32 list at each checkpoint.  If the correct token is not
    in the top-32, the rank is TOPK_ABSENT_RANK (33).

    Returns:
        correct_matrix : (n_tokens, n_steps)  bool    — argmax == correct token
        prob_matrix    : (n_tokens, n_steps)  float32  — correct-token probability
                         exp(correct_logit - log_z).  NaN where fields missing.
        rank_matrix    : (n_tokens, n_steps)  int16    — 1-based rank of correct
                         token in top-k list; TOPK_ABSENT_RANK if not in top-k;
                         0 if top-k data unavailable at this checkpoint.
        token_ids      : (n_tokens,)           uint32  — correct token at each position
        pos_keys       : (n_tokens,)           uint64 composite (inst*65536+pos)
        steps          : list of int step numbers (length n_steps)
    """
    # Use the first checkpoint to determine the full set of positions
    ref = np.load(checkpoints[0])
    if source not in ref.files:
        raise KeyError(f"Source '{source}' not found in {checkpoints[0]}")
    ref_arr = ref[source]
    pos_keys = (
        ref_arr["instance_index"].astype(np.uint64) * 65536
        + ref_arr["position_in_seq"].astype(np.uint64)
    )
    token_ids = ref_arr["token_id"].astype(np.uint32)
    n_tokens = len(pos_keys)

    # Build index: pos_key -> row index (for fast lookup in subsequent checkpoints)
    key_to_idx = {k: i for i, k in enumerate(pos_keys.tolist())}

    n_steps = len(checkpoints)
    correct_matrix = np.zeros((n_tokens, n_steps), dtype=bool)
    prob_matrix    = np.full((n_tokens, n_steps), np.nan, dtype=np.float32)
    rank_matrix    = np.zeros((n_tokens, n_steps), dtype=np.int16)

    steps: List[int] = []
    for t, ckpt in enumerate(checkpoints):
        steps.append(step_from_path(ckpt))
        data = np.load(ckpt)
        if source not in data.files:
            # Missing source at this checkpoint: leave columns as False / NaN / 0
            continue
        arr  = data[source]
        keys = (
            arr["instance_index"].astype(np.uint64) * 65536
            + arr["position_in_seq"].astype(np.uint64)
        ).tolist()
        correct_col = (arr["max_token_id"] == arr["token_id"]).tolist()

        # correct_prob = exp(correct_logit - log_z); available when both fields present
        has_prob = (
            "correct_logit" in arr.dtype.names
            and "log_z" in arr.dtype.names
        )
        if has_prob:
            prob_col = np.exp(
                arr["correct_logit"].astype(np.float32)
                - arr["log_z"].astype(np.float32)
            ).clip(0.0, 1.0).tolist()
        else:
            prob_col = [float("nan")] * len(keys)

        # Top-k rank of correct token
        topk_key = f"{source}__topk_indices"
        if topk_key in data.files:
            topk_indices = data[topk_key]  # (n_tokens_in_ckpt, k)
            k_depth = topk_indices.shape[1]
            # For each position, find rank of correct token in its top-k row.
            # Build map: key -> correct_token_id
            arr_tids = arr["token_id"]
            rank_col: List[int] = []
            for row_i, (k_val, tid) in enumerate(zip(keys, arr_tids.tolist())):
                top_row = topk_indices[row_i]
                found = False
                for r in range(k_depth):
                    if top_row[r] == tid:
                        rank_col.append(r + 1)  # 1-based
                        found = True
                        break
                if not found:
                    rank_col.append(TOPK_ABSENT_RANK)
        else:
            rank_col = [0] * len(keys)  # 0 = top-k unavailable

        for k_key, c, p, rk in zip(keys, correct_col, prob_col, rank_col):
            idx = key_to_idx.get(k_key)
            if idx is not None:
                correct_matrix[idx, t] = c
                prob_matrix[idx, t]    = p
                rank_matrix[idx, t]    = rk

        if (t + 1) % 10 == 0 or t == n_steps - 1:
            print(f"    loaded {t+1}/{n_steps} checkpoints", end="\r", flush=True)

    print()  # newline after progress
    return correct_matrix, prob_matrix, rank_matrix, token_ids, pos_keys, steps


# ---------------------------------------------------------------------------
# Acquisition step detection
# ---------------------------------------------------------------------------

def compute_acquisition_steps(
    correct_matrix: np.ndarray,
    min_sustain_frac: float = MIN_SUSTAIN_FRAC,
) -> np.ndarray:
    """
    For each token (row), find the earliest step t* such that the fraction
    of checkpoints at t >= t* where the model is correct >= min_sustain_frac.

    Returns an int array of shape (n_tokens,) with values in [0, n_steps-1]
    for acquired tokens, or -1 for tokens never durably acquired.
    """
    n_tokens, n_steps = correct_matrix.shape
    # Precompute suffix sums for fast fraction computation
    # suffix_correct[i, t] = number of correct steps in [t, n_steps)
    suffix = np.cumsum(correct_matrix[:, ::-1], axis=1)[:, ::-1]  # (n_tokens, n_steps)
    # suffix_len[t] = n_steps - t
    suffix_len = np.arange(n_steps, 0, -1, dtype=np.float32)  # (n_steps,)

    acquisition = np.full(n_tokens, -1, dtype=np.int32)

    # Vectorised: for each step t, check which tokens first meet the criterion
    already_acquired = np.zeros(n_tokens, dtype=bool)
    for t in range(n_steps):
        length = n_steps - t
        frac = suffix[:, t] / length  # (n_tokens,)
        meets = (frac >= min_sustain_frac) & ~already_acquired
        acquisition[meets] = t
        already_acquired |= meets

    return acquisition


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def smooth(arr: np.ndarray, w: int) -> np.ndarray:
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def find_peaks(counts: np.ndarray, n_peaks: int = TOP_PEAKS) -> List[int]:
    """Return indices of the top-n local maxima by count."""
    smoothed = smooth(counts.astype(float), SMOOTH_WINDOW)
    n = len(smoothed)
    is_peak = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            is_peak[i] = True
    peak_indices = np.where(is_peak)[0]
    # Sort by count descending
    peak_indices = peak_indices[np.argsort(-counts[peak_indices])]
    return peak_indices[:n_peaks].tolist()


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

# Threshold for classifying a token as LR-consolidation vs genuine new acquisition.
# If mean correct_prob at the checkpoint immediately before acquisition exceeds
# this value, the token was already near-correct and LR decay just tipped it over.
LR_CONSOLIDATION_PROB_THRESHOLD = 0.35


def describe_peak(
    peak_step_idx: int,
    steps: List[int],
    acquisition: np.ndarray,
    token_ids: np.ndarray,
    prob_matrix: np.ndarray,
    rank_matrix: np.ndarray,
    tokenizer,
    n_show: int = TOKENS_PER_PEAK,
) -> None:
    """
    Print a human-readable summary of tokens acquired at a given peak step.

    For each peak, reports the mean correct-token probability at the checkpoint
    *immediately before* acquisition.  This distinguishes two acquisition types:

    - **LR-consolidation** (pre-acquisition prob ≥ LR_CONSOLIDATION_PROB_THRESHOLD):
      The model was already assigning reasonable probability to the correct token
      before the acquisition step; the token was near the decision boundary and
      LR decay stopped the oscillation.  These dominate late-training peaks.

    - **Genuine new acquisition** (pre-acquisition prob ≈ 0):
      The model had almost no probability mass on the correct token before the
      flip — a real capability gain.  These are expected mainly in early training.

    Additionally, if rank_matrix is available (non-zero), reports the rank of
    the correct token at the checkpoint immediately before acquisition.  This
    distinguishes:

    - **Gradual promotion**: correct token was already in top-32 (rank 2–32)
      before acquisition; the model was "considering" the right answer and
      gradually promoted it to rank-1.  A smooth, progressive change.

    - **Sudden emergence**: correct token was NOT in top-32 (rank sentinel 33)
      before acquisition; the model had essentially no probability mass on
      the right answer.  A discontinuous capability jump.
    """
    mask = acquisition == peak_step_idx
    n_acquired = int(mask.sum())
    step = steps[peak_step_idx]
    print(f"\n  Peak at step {step:,}  ({n_acquired} tokens acquired)")

    if n_acquired == 0:
        return

    # --- Pre-acquisition correct_prob ---
    if peak_step_idx > 0:
        pre_probs = prob_matrix[mask, peak_step_idx - 1]  # (n_acquired,)
        valid = pre_probs[~np.isnan(pre_probs)]
        if len(valid) > 0:
            mean_pre  = float(valid.mean())
            med_pre   = float(np.median(valid))
            pct_consol = float((valid >= LR_CONSOLIDATION_PROB_THRESHOLD).mean())
            acq_type   = (
                "LR-consolidation (already near-correct)"
                if mean_pre >= LR_CONSOLIDATION_PROB_THRESHOLD
                else "genuine new acquisition (prob was near-zero)"
            )
            print(f"  Pre-acquisition correct_prob:  "
                  f"mean={mean_pre:.3f}  median={med_pre:.3f}  "
                  f"pct≥{LR_CONSOLIDATION_PROB_THRESHOLD:.2f}: {pct_consol:.1%}")
            print(f"  Classification: {acq_type}")
        else:
            print(f"  Pre-acquisition correct_prob: n/a (prob fields missing in dumps)")

        # --- Pre-acquisition rank distribution (from top-k data) ---
        pre_ranks = rank_matrix[mask, peak_step_idx - 1]  # (n_acquired,) int16
        has_rank = (pre_ranks != 0).any()  # 0 = top-k unavailable
        if has_rank:
            valid_ranks = pre_ranks[pre_ranks != 0]
            n_rank1     = int((valid_ranks == 1).sum())
            n_rank2_5   = int(((valid_ranks >= 2) & (valid_ranks <= 5)).sum())
            n_rank6_32  = int(((valid_ranks >= 6) & (valid_ranks <= 32)).sum())
            n_absent    = int((valid_ranks == TOPK_ABSENT_RANK).sum())
            n_total_rk  = len(valid_ranks)
            print(f"  Pre-acquisition rank of correct token in top-32:")
            print(f"    rank=1  (already top-1 = no real change):  "
                  f"{n_rank1:>6,}  ({n_rank1/n_total_rk:.1%})")
            print(f"    rank 2–5   (almost correct, near miss):    "
                  f"{n_rank2_5:>6,}  ({n_rank2_5/n_total_rk:.1%})")
            print(f"    rank 6–32  (considered but not prominent): "
                  f"{n_rank6_32:>6,}  ({n_rank6_32/n_total_rk:.1%})")
            print(f"    absent (>32, sudden emergence):            "
                  f"{n_absent:>6,}  ({n_absent/n_total_rk:.1%})")
            # Classify acquisition type using rank
            if n_absent / n_total_rk >= 0.5:
                rank_class = "SUDDEN EMERGENCE (>50% were absent from top-32 before acquisition)"
            elif (n_rank2_5 + n_rank6_32) / n_total_rk >= 0.5:
                rank_class = "GRADUAL PROMOTION (>50% were already in top-32 before acquisition)"
            else:
                rank_class = "MIXED (split between gradual and sudden)"
            print(f"  Rank-based classification: {rank_class}")
    else:
        print(f"  Pre-acquisition correct_prob: n/a (first checkpoint)")

    tids = token_ids[mask]
    counts = Counter(tids.tolist())
    top = counts.most_common(n_show)

    print(f"  {'token_id':>10}  {'count':>8}  {'pct':>7}  decoded")
    print(f"  {'-'*10}  {'-'*8}  {'-'*7}  -------")
    for tid, cnt in top:
        pct = cnt / n_acquired * 100
        decoded = decode_tokens([tid], tokenizer)[0]
        # Escape whitespace for display
        decoded_display = repr(decoded) if len(decoded.strip()) == 0 else decoded
        print(f"  {tid:>10}  {cnt:>8,}  {pct:>6.1f}%  {decoded_display}")


def print_accuracy_trajectory(
    correct_matrix: np.ndarray,
    steps: List[int],
) -> None:
    """Print mean accuracy at each training step."""
    mean_acc = correct_matrix.mean(axis=0)
    print(f"\n  {'step':>12}  {'mean_acc':>10}  {'delta':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}")
    prev = None
    for t, (step, acc) in enumerate(zip(steps, mean_acc)):
        delta = acc - prev if prev is not None else float("nan")
        delta_str = f"{delta:>+10.4f}" if not np.isnan(delta) else "         -"
        print(f"  {step:>12,}  {acc:>10.4f}  {delta_str}")
        prev = acc


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------

def _find_norm_peaks(
    acq_frac: np.ndarray,
    n_bins: int = 20,
    n_peaks: int = TOP_PEAKS,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Bin acquisition fractions (normalised to [0, 1]) and find density peaks.

    :returns: ``(counts, bin_centers, peak_bin_indices)``
    """
    valid = acq_frac[~np.isnan(acq_frac)]
    counts, bin_edges = np.histogram(valid, bins=n_bins, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak_idxs = find_peaks(counts, n_peaks)
    return counts, bin_centers, peak_idxs


def cross_model_phase_comparison(
    model_trajectories: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]],
    source: str,
    tokenizer,
    n_bins: int = 20,
) -> None:
    """
    Find acquisition clusters independently per model (normalised training
    fraction), then perform cluster-transfer analysis: for each model's top
    acquisition peak, show where *those same token positions* fall in every
    other model's acquisition distribution.

    A concentrated peak in the receiving model means both architectures
    acquired that capability cluster at a similar *relative* point in
    training — strong evidence for a real capability phase transition.
    A flat / spread distribution means the cluster is model-specific or
    noise.

    **Important:** step numbers are NOT compared directly across models.
    Each model is normalised to its own training fraction [0, 1] because
    total compute differs between model families.
    """
    model_names = list(model_trajectories.keys())
    N_CLUSTER_TRANSFER = 500   # token positions to transfer per cluster

    print(f"\n{'='*70}")
    print(f"Cross-model phase comparison — source: {source}")
    print(f"Step numbers are NOT compared directly; each model is normalised")
    print(f"to its own training fraction [0, 1].")
    print(f"{'='*70}")

    # --- Step 1: per-model acquisition fractions and per-model clusters ---
    acq_maps:    Dict[str, Dict[int, Tuple[float, int]]] = {}
    acq_frac_arr: Dict[str, np.ndarray] = {}
    pos_key_lists: Dict[str, List[int]] = {}

    for name, (mat, tids, pos_keys, steps) in model_trajectories.items():
        acq = compute_acquisition_steps(mat)
        n_steps = len(steps)
        frac = np.where(acq >= 0, acq / max(n_steps - 1, 1), np.nan)
        acq_frac_arr[name] = frac
        acq_maps[name] = {
            k: (float(f), int(t))
            for k, f, t in zip(pos_keys.tolist(), frac.tolist(), tids.tolist())
        }
        pos_key_lists[name] = pos_keys.tolist()

        counts, bin_centers, peak_idxs = _find_norm_peaks(frac, n_bins)
        n_acq = int((acq >= 0).sum())
        print(f"\n  {name}:  {n_acq:,}/{len(acq):,} tokens durably acquired")
        print(f"  Acquisition distribution (fraction of own training):")
        max_c = counts.max() if counts.max() > 0 else 1
        bar_w = 30
        for b, c in enumerate(counts):
            bar = "#" * int(c / max_c * bar_w)
            pk = " <-- PEAK" if b in peak_idxs else ""
            print(f"    [{bin_centers[b]:.2f}]  {c:>7,}  |{bar}{pk}")

    # --- Step 2: cluster-transfer analysis ---
    # For each model's biggest peak, take the token positions in that cluster
    # and show their acquisition-fraction distribution in every other model.
    # High concentration at a single bin in the receiving model = shared
    # capability cluster.  Flat = noise or model-specific phenomenon.
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers_all = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Shared positions (present in ALL models)
    shared_keys: set = set(pos_key_lists[model_names[0]])
    for name in model_names[1:]:
        shared_keys &= set(pos_key_lists[name])
    print(f"\n  Shared positions (present in all models): {len(shared_keys):,}")

    print(f"\n  Cluster-transfer analysis:")
    print(f"  For each model's top acquisition peak, show where those exact")
    print(f"  token positions fall in every other model's distribution.")
    print(f"  conc = fraction of transferred tokens in the receiving model's")
    print(f"  single busiest bin; high conc = shared capability cluster.")

    rng = np.random.default_rng(RNG_SEED)

    for src_name in model_names:
        frac = acq_frac_arr[src_name]
        pos_keys_src = pos_key_lists[src_name]
        counts, bin_centers, peak_idxs = _find_norm_peaks(frac, n_bins)
        if not peak_idxs:
            continue

        top_peak = peak_idxs[0]   # biggest density peak for this model
        peak_lo  = bin_edges[top_peak]
        peak_hi  = bin_edges[top_peak + 1]

        # Token positions in the peak cluster that are also shared across models
        cluster_keys = [
            k for k, f in zip(pos_keys_src, frac.tolist())
            if not np.isnan(f) and peak_lo <= f < peak_hi and k in shared_keys
        ]
        if not cluster_keys:
            continue
        if len(cluster_keys) > N_CLUSTER_TRANSFER:
            cluster_keys = rng.choice(
                cluster_keys, N_CLUSTER_TRANSFER, replace=False
            ).tolist()

        # Characterise the cluster: top token IDs
        cluster_tids = [
            acq_maps[src_name][k][1] for k in cluster_keys
            if k in acq_maps[src_name]
        ]
        top_tids = Counter(cluster_tids).most_common(5)
        decoded_top = ", ".join(
            f"{decode_tokens([t], tokenizer)[0]!r}×{c}" for t, c in top_tids
        )

        print(f"\n  Source: {src_name}")
        print(f"    Top peak: frac [{peak_lo:.2f}, {peak_hi:.2f})  "
              f"n={len(cluster_keys)} tokens")
        print(f"    Top token IDs in cluster: {decoded_top}")
        print(f"    {'model':<35} {'acq':>5} {'never':>5}  "
              f"{'conc@peak':>10}  histogram")

        for dst_name in model_names:
            dst_fracs = [
                acq_maps[dst_name][k][0]
                for k in cluster_keys
                if k in acq_maps[dst_name] and not np.isnan(acq_maps[dst_name][k][0])
            ]
            n_acq  = len(dst_fracs)
            n_miss = len(cluster_keys) - n_acq

            if dst_name == src_name:
                print(f"    {dst_name:<35} {'(source)':>5}")
                continue

            if not dst_fracs:
                print(f"    {dst_name:<35} {'0':>5} {len(cluster_keys):>5}  "
                      f"{'n/a':>10}")
                continue

            dst_arr = np.array(dst_fracs)
            dst_counts, _ = np.histogram(dst_arr, bins=n_bins, range=(0.0, 1.0))
            top_bin    = int(np.argmax(dst_counts))
            conc       = dst_counts[top_bin] / n_acq
            max_dc     = dst_counts.max() if dst_counts.max() > 0 else 1
            bar_w      = 20
            hist_str   = "".join(
                "#" * max(1, int(c / max_dc * bar_w)) if c > 0 else "."
                for c in dst_counts
            )
            print(f"    {dst_name:<35} {n_acq:>5} {n_miss:>5}  "
                  f"conc={conc:.0%}@[{bin_centers_all[top_bin]:.2f}]  "
                  f"|{hist_str}|")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Single model directory name (for single-model analysis).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Multiple model directory names (for cross-model comparison).",
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
        "--min-sustain-frac",
        type=float,
        default=MIN_SUSTAIN_FRAC,
        help="Min fraction of subsequent checkpoints that must be correct to "
             "count as 'durably acquired' (default: 0.75).",
    )
    parser.add_argument(
        "--cross-model",
        action="store_true",
        help="Run cross-model phase comparison (requires --models or all models).",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="If set, save acquisition histogram plots as PNG files here.",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip tokenizer loading (show raw token IDs only).",
    )
    parser.add_argument(
        "--apply-mask",
        action="store_true",
        default=False,
        help=(
            "Apply the trivial-token mask before acquisition analysis. "
            "Excludes positions correct at any step ≤ EARLY_STEP_THRESHOLD "
            f"({EARLY_STEP_THRESHOLD:,}) and positions with "
            f"position_in_seq < MIN_POSITION_IN_SEQ ({MIN_POSITION_IN_SEQ}). "
            "Mask stats are always printed regardless of this flag."
        ),
    )
    args = parser.parse_args()

    if args.model is None and not args.models:
        # Default: all models in the dump root
        args.models = sorted(p.name for p in args.dump_root.iterdir() if p.is_dir())

    model_list = args.models or ([args.model] if args.model else [])

    # Load tokenizer
    tokenizer = None if args.no_tokenizer else load_tokenizer()
    if tokenizer is not None:
        print("Tokenizer loaded.")
    else:
        print("Tokenizer not available — showing raw token IDs.")

    cross_model_data: Dict[str, Tuple] = {}

    for model_name in model_list:
        model_dir = args.dump_root / model_name
        if not model_dir.is_dir():
            print(f"Skipping {model_name}: directory not found")
            continue

        all_ckpts = list_checkpoints(model_dir)
        if not all_ckpts:
            print(f"Skipping {model_name}: no checkpoints found")
            continue

        selected = pick_evenly_spaced(all_ckpts, args.max_checkpoints)
        short = model_name.replace("allenai_", "")

        print(f"\n{'='*70}")
        print(f"Model: {short}  ({len(selected)} checkpoints, source: {args.source})")
        print(f"  Step range: {step_from_path(selected[0]):,} → {step_from_path(selected[-1]):,}")
        print(f"{'='*70}")

        try:
            print(f"  Building trajectories...")
            correct_matrix, prob_matrix, rank_matrix, token_ids, pos_keys, steps = build_trajectory(
                selected, args.source
            )
        except KeyError as e:
            print(f"  Skipping: {e}")
            continue

        n_tokens, n_steps = correct_matrix.shape
        print(f"  {n_tokens:,} token positions × {n_steps} checkpoints")

        # --- Trivial-token mask (always computed; optionally applied) ---
        mask_trivial, mask_A, mask_B = build_trivial_mask(correct_matrix, pos_keys, steps)
        print_mask_stats(mask_trivial, mask_A, mask_B, n_tokens)

        if args.apply_mask:
            keep = ~mask_trivial
            correct_matrix = correct_matrix[keep]
            prob_matrix    = prob_matrix[keep]
            rank_matrix    = rank_matrix[keep]
            token_ids      = token_ids[keep]
            pos_keys       = pos_keys[keep]
            n_tokens       = int(keep.sum())
            n_steps        = correct_matrix.shape[1]
            print(f"  --apply-mask active: {n_tokens:,} tokens retained.")

        # Accuracy trajectory
        print_accuracy_trajectory(correct_matrix, steps)

        # Acquisition steps
        print(f"\n  Computing acquisition steps (min_sustain_frac={args.min_sustain_frac})...")
        acquisition = compute_acquisition_steps(correct_matrix, args.min_sustain_frac)
        n_acquired = int((acquisition >= 0).sum())
        n_never = int((acquisition < 0).sum())
        print(f"  Durably acquired: {n_acquired:,} / {n_tokens:,} ({n_acquired/n_tokens:.1%})")
        print(f"  Never acquired:   {n_never:,} / {n_tokens:,} ({n_never/n_tokens:.1%})")

        # Acquisition histogram
        acq_valid = acquisition[acquisition >= 0]
        step_counts = np.zeros(n_steps, dtype=int)
        for idx in acq_valid:
            step_counts[idx] += 1

        # Normalise acquisition counts by the training-step gap to the previous
        # checkpoint so that coarsely-spaced checkpoint pairs don't dominate.
        # Units: acquisitions per 1,000 training steps.
        step_arr  = np.array(steps, dtype=np.float64)
        step_gaps = np.diff(step_arr, prepend=step_arr[0])  # gap[0] = 0 (first ckpt)
        step_gaps[0] = step_gaps[1] if n_steps > 1 else 1.0  # fallback for single ckpt
        step_gaps = np.maximum(step_gaps, 1.0)               # avoid div-by-zero
        norm_counts = step_counts.astype(np.float64) / step_gaps * 1000.0

        peak_indices = find_peaks(norm_counts, TOP_PEAKS)

        # Print histogram with ASCII bar chart
        print(f"\n  Acquisition histogram (acquisitions per 1k training steps):")
        max_norm  = norm_counts.max() if norm_counts.max() > 0 else 1.0
        bar_width = 40
        smoothed  = smooth(norm_counts, SMOOTH_WINDOW)
        for t in range(n_steps):
            bar_len     = int(smoothed[t] / max_norm * bar_width)
            peak_marker = " <-- PEAK" if t in peak_indices else ""
            bar         = "#" * bar_len
            print(f"  step {steps[t]:>10,}  "
                  f"raw={step_counts[t]:>6,}  "
                  f"norm={norm_counts[t]:>7.2f}/1k  "
                  f"|{bar}{peak_marker}")

        # Detailed peak analysis
        print(f"\n  Top {len(peak_indices)} acquisition peaks:")
        for peak_idx in peak_indices:
            describe_peak(peak_idx, steps, acquisition, token_ids, prob_matrix, rank_matrix, tokenizer)

        if args.plot_dir:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(2, 1, figsize=(14, 8))
                step_arr = np.array(steps)
                # Accuracy curve
                axes[0].plot(step_arr, correct_matrix.mean(axis=0) * 100)
                axes[0].set_xlabel("Training step")
                axes[0].set_ylabel("Mean top-1 accuracy (%)")
                axes[0].set_title(f"{short} — {args.source} accuracy")
                axes[0].grid(True, alpha=0.3)
                # Acquisition histogram
                axes[1].bar(range(n_steps), step_counts, color="steelblue", alpha=0.7)
                axes[1].plot(range(n_steps), smooth(step_counts.astype(float), SMOOTH_WINDOW),
                             color="red", linewidth=2, label="smoothed")
                for peak_idx in peak_indices:
                    axes[1].axvline(peak_idx, color="orange", linestyle="--", alpha=0.8)
                axes[1].set_xticks(range(0, n_steps, max(1, n_steps // 10)))
                axes[1].set_xticklabels(
                    [f"{steps[i]:,}" for i in range(0, n_steps, max(1, n_steps // 10))],
                    rotation=45, ha="right"
                )
                axes[1].set_xlabel("Training step")
                axes[1].set_ylabel("Tokens acquired")
                axes[1].set_title(f"{short} — {args.source} acquisition rate")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                plt.tight_layout()
                args.plot_dir.mkdir(parents=True, exist_ok=True)
                out = args.plot_dir / f"{model_name}_{args.source}.png"
                plt.savefig(out, dpi=120)
                plt.close()
                print(f"\n  Plot saved: {out}")
            except ImportError:
                print("  matplotlib not available; skipping plot.")

        # Store for cross-model comparison (prob_matrix not needed downstream)
        if args.cross_model:
            cross_model_data[short] = (correct_matrix, token_ids, pos_keys, steps)

    # Cross-model phase comparison
    if args.cross_model and len(cross_model_data) >= 2:
        cross_model_phase_comparison(cross_model_data, args.source, tokenizer)


if __name__ == "__main__":
    main()
