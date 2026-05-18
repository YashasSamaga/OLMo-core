"""
Frequency-domain analysis of per-token correct_prob trajectories over training.

Checkpoints are non-uniformly spaced in step-time, so a standard FFT is
incorrect.  This script uses the **Lomb-Scargle periodogram** (scipy
implementation) which is designed for irregularly-sampled signals.  An
interpolated-FFT fallback is provided if scipy is unavailable.

Three analyses are run for each model / source:

1. **Mean-trajectory spectrum**
   The mean correct_prob across all token positions at each checkpoint is a
   single 1-D signal.  Its power spectrum reveals global oscillation modes in
   the training dynamics (e.g. a periodic cycle in the data mix would appear
   as a spike at the corresponding frequency).

2. **Per-token dominant frequency vs instability**
   For each token position in the noisy subset (instability >= 2), we compute
   the Lomb-Scargle spectrum and record the dominant frequency.  Reported as:
   - Distribution of dominant frequencies across noisy tokens
   - Scatter of dominant-frequency vs instability count
   - Comparison of dominant-frequency distributions: noisy vs stable-correct

3. **Spectral clustering of the noisy population** (optional, --cluster)
   Embed each noisy token's power spectrum as a vector, reduce to 2-D with
   PCA, and find clusters with k-means.  Tokens in the same cluster have
   similar oscillation patterns.  Each cluster is characterised by its
   centroid spectrum shape and the top token IDs it contains.

Key design decisions:
- Frequencies are expressed in units of *cycles per 1,000 training steps* so
  they are comparable across models with different total training lengths.
- The time axis is the actual step number (not checkpoint index) so gaps are
  handled correctly.
- correct_prob = exp(correct_logit - log_z) is used as the signal; it is
  real-valued and lives in (0, 1), making it much more informative than binary
  correct/wrong for spectral analysis.
- Only the noisy population (instability >= 2) is analysed per-token;
  stable-correct and stable-wrong tokens have trivial spectra (DC + noise).

Usage:
    # Single model, mean-trajectory spectrum only
    python src/scripts/analyze/spectral_analysis.py \\
        --model allenai_OLMo-2-0425-1B --source c4_en

    # Full per-token analysis (slower)
    python src/scripts/analyze/spectral_analysis.py \\
        --model allenai_OLMo-2-0425-1B --source c4_en --per-token

    # With spectral clustering (requires scikit-learn)
    python src/scripts/analyze/spectral_analysis.py \\
        --model allenai_OLMo-2-0425-1B --source c4_en --per-token --cluster

    # Save plots
    python src/scripts/analyze/spectral_analysis.py \\
        --model allenai_OLMo-2-0425-1B --plot-dir /tmp/spectral_plots

    # All models
    python src/scripts/analyze/spectral_analysis.py --per-token
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DUMP_ROOT = Path("/weka/oe-training-default/yashasbls/georges-functional-analysis")
RNG_SEED = 42
DEFAULT_MAX_CKPTS = 50

# Max noisy tokens to run per-token Lomb-Scargle on (subsampled for speed).
MAX_PER_TOKEN = 10_000

# Number of frequency grid points for Lomb-Scargle.
N_FREQS = 256

# Minimum instability count to be included in the noisy population.
NOISY_THRESHOLD = 2

# Number of k-means clusters for spectral clustering.
N_CLUSTERS = 6


# ---------------------------------------------------------------------------
# Shared helpers
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


# ---------------------------------------------------------------------------
# Trajectory builder — loads correct_prob matrix
# ---------------------------------------------------------------------------

def build_prob_trajectory(
    checkpoints: List[Path],
    source: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Build per-token correct_prob trajectories from a sequence of checkpoints.

    correct_prob[i, t] = exp(correct_logit[i,t] - log_z[i,t]) ∈ (0, 1).

    :returns:
        ``prob_matrix``   — (n_tokens, n_steps) float32, NaN where unavailable

        ``correct_matrix``— (n_tokens, n_steps) bool (argmax == correct token)

        ``token_ids``     — (n_tokens,) uint32 correct token at each position

        ``pos_keys``      — (n_tokens,) uint64 composite keys

        ``steps``         — list of int step numbers
    """
    ref_data  = np.load(checkpoints[0])
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

    n_steps       = len(checkpoints)
    prob_matrix   = np.full((n_tokens, n_steps), np.nan, dtype=np.float32)
    correct_matrix = np.zeros((n_tokens, n_steps), dtype=bool)
    steps: List[int] = []

    for t, ckpt in enumerate(checkpoints):
        steps.append(step_from_path(ckpt))
        data = np.load(ckpt)
        if source not in data.files:
            continue
        arr = data[source]
        keys = (
            arr["instance_index"].astype(np.uint64) * 65536
            + arr["position_in_seq"].astype(np.uint64)
        ).tolist()
        correct_col = (arr["max_token_id"] == arr["token_id"]).tolist()
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

        for k, c, p in zip(keys, correct_col, prob_col):
            idx = key_to_idx.get(k)
            if idx is not None:
                correct_matrix[idx, t] = c
                prob_matrix[idx, t]    = p

        if (t + 1) % 10 == 0 or t == n_steps - 1:
            print(f"    loaded {t+1}/{n_steps} checkpoints", end="\r", flush=True)

    print()
    return prob_matrix, correct_matrix, token_ids, pos_keys, steps


# ---------------------------------------------------------------------------
# Lomb-Scargle / FFT helpers
# ---------------------------------------------------------------------------

def _freq_grid(steps: List[int], n_freqs: int = N_FREQS) -> np.ndarray:
    """
    Build a frequency grid in units of *cycles per 1,000 training steps*.

    Minimum frequency: 1 cycle over the full training run.
    Maximum frequency (Nyquist-like): 1 cycle over 2× the median step gap.
    """
    step_arr  = np.array(steps, dtype=np.float64)
    total     = step_arr[-1] - step_arr[0]
    gaps      = np.diff(step_arr)
    med_gap   = float(np.median(gaps)) if len(gaps) > 0 else 1.0
    f_min     = 1.0 / total * 1000.0          # cycles per 1k steps
    f_max     = 1.0 / (2.0 * med_gap) * 1000.0
    return np.linspace(f_min, f_max, n_freqs)


def lomb_scargle_power(
    steps: List[int],
    signal: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """
    Compute the Lomb-Scargle periodogram for a non-uniformly sampled signal.

    :param steps:  list of step numbers (time axis)
    :param signal: 1-D float array, same length as steps (NaNs dropped)
    :param freqs:  frequency grid in cycles per 1,000 training steps
    :returns:      normalised power spectrum, same length as freqs
    """
    step_arr = np.array(steps, dtype=np.float64)
    valid    = ~np.isnan(signal)
    t        = step_arr[valid] / 1000.0   # convert to "kilo-steps" for unit match
    y        = signal[valid].astype(np.float64)
    if len(y) < 4:
        return np.zeros(len(freqs))

    # Linear detrend: remove the best-fit line through the trajectory.
    # Mean-centering alone leaves slow trends that dominate the spectrum;
    # detrending exposes oscillatory components on top of the trend.
    coeffs = np.polyfit(t, y, deg=1)
    y = y - np.polyval(coeffs, t)

    try:
        from scipy.signal import lombscargle
        # lombscargle expects angular frequencies
        omega = 2.0 * np.pi * freqs
        power = lombscargle(t, y, omega, normalize=True)
    except ImportError:
        # Fallback: interpolate to uniform grid then use FFT
        t_uniform = np.linspace(t[0], t[-1], len(t))
        y_uniform = np.interp(t_uniform, t, y)
        fft_out   = np.abs(np.fft.rfft(y_uniform)) ** 2
        fft_freqs = np.fft.rfftfreq(len(y_uniform), d=(t_uniform[1] - t_uniform[0]))
        power     = np.interp(freqs, fft_freqs, fft_out / (fft_out.max() + 1e-12))

    return power.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-token instability (minimal, self-contained)
# ---------------------------------------------------------------------------

def instability_from_correct(correct_matrix: np.ndarray) -> np.ndarray:
    """Return per-token instability (total correctness flips)."""
    diff = np.diff(correct_matrix.astype(np.int8), axis=1)
    return (np.abs(diff)).sum(axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# Analysis 1: mean-trajectory spectrum
# ---------------------------------------------------------------------------

def mean_trajectory_spectrum(
    prob_matrix: np.ndarray,
    steps: List[int],
    freqs: np.ndarray,
    bar_w: int = 50,
) -> np.ndarray:
    """
    Compute and print the power spectrum of the mean correct_prob trajectory.

    :returns: power spectrum array (len = n_freqs)
    """
    # NaN-safe mean across tokens at each step
    mean_traj = np.nanmean(prob_matrix, axis=0)  # (n_steps,)
    power     = lomb_scargle_power(steps, mean_traj, freqs)

    print(f"\n  Mean correct_prob trajectory spectrum")
    print(f"  (Lomb-Scargle; freq in cycles per 1,000 training steps)")
    print(f"  Total training span: {steps[-1] - steps[0]:,} steps  "
          f"  n_checkpoints={len(steps)}")

    # Report top-5 dominant frequencies
    top_idx = np.argsort(-power)[:5]
    print(f"\n  Top-5 dominant frequencies:")
    print(f"  {'rank':>4}  {'freq (cyc/1k)':>15}  {'period (k steps)':>18}  "
          f"{'norm_power':>12}")
    print(f"  {'-'*4}  {'-'*15}  {'-'*18}  {'-'*12}")
    for rank, idx in enumerate(top_idx, 1):
        f    = freqs[idx]
        pwr  = power[idx]
        prd  = 1.0 / f if f > 0 else float("inf")
        print(f"  {rank:>4}  {f:>15.6f}  {prd:>18.2f}  {pwr:>12.4f}")

    # ASCII bar chart
    max_p = power.max() if power.max() > 0 else 1.0
    print(f"\n  Power spectrum (ASCII):")
    print(f"  {'freq':>10}  {'period(k)':>10}  bar")
    print(f"  {'-'*10}  {'-'*10}  ---")
    # Print at most 40 lines (downsample visually)
    step_v = max(1, len(freqs) // 40)
    for i in range(0, len(freqs), step_v):
        f   = freqs[i]
        p   = power[i]
        bar = "#" * int(p / max_p * bar_w)
        pk  = " <--" if i in top_idx else ""
        print(f"  {f:>10.5f}  {1/f:>10.2f}  |{bar}{pk}")

    return power


# ---------------------------------------------------------------------------
# Analysis 2: per-token dominant frequency
# ---------------------------------------------------------------------------

def per_token_spectrum_analysis(
    prob_matrix: np.ndarray,
    correct_matrix: np.ndarray,
    token_ids: np.ndarray,
    steps: List[int],
    freqs: np.ndarray,
    tokenizer,
    rng: np.random.Generator,
    max_tokens: int = MAX_PER_TOKEN,
) -> None:
    """
    For the noisy and stable-correct token subsets, compute per-token
    Lomb-Scargle spectra and report dominant frequencies.

    Compares the dominant-frequency distribution of noisy vs stable-correct
    tokens to show whether oscillating tokens have higher-frequency structure.
    """
    instab = instability_from_correct(correct_matrix)
    ever_correct = correct_matrix.any(axis=1)

    noisy_idx   = np.where(instab >= NOISY_THRESHOLD)[0]
    stable_idx  = np.where((instab < 2) & ever_correct)[0]

    # Subsample for speed
    if len(noisy_idx) > max_tokens:
        noisy_idx = rng.choice(noisy_idx, max_tokens, replace=False)
    if len(stable_idx) > max_tokens:
        stable_idx = rng.choice(stable_idx, max_tokens, replace=False)

    print(f"\n  Per-token spectrum analysis")
    print(f"  Noisy population (instability ≥ {NOISY_THRESHOLD}): "
          f"{len(noisy_idx):,} tokens analysed  "
          f"(total noisy: {int((instab >= NOISY_THRESHOLD).sum()):,})")
    print(f"  Stable-correct population: {len(stable_idx):,} tokens analysed")

    def _dominant_freqs(indices: np.ndarray) -> np.ndarray:
        dom_freqs = np.full(len(indices), np.nan, dtype=np.float32)
        for j, i in enumerate(indices):
            sig = prob_matrix[i]  # (n_steps,)
            pwr = lomb_scargle_power(steps, sig, freqs)
            if pwr.max() > 0:
                dom_freqs[j] = freqs[np.argmax(pwr)]
            if (j + 1) % 500 == 0 or j == len(indices) - 1:
                print(f"    {j+1}/{len(indices)}", end="\r", flush=True)
        print()
        return dom_freqs

    print(f"  Computing spectra for noisy tokens...")
    dom_noisy  = _dominant_freqs(noisy_idx)
    print(f"  Computing spectra for stable-correct tokens...")
    dom_stable = _dominant_freqs(stable_idx)

    def _freq_hist(dom_freqs: np.ndarray, label: str) -> None:
        valid = dom_freqs[~np.isnan(dom_freqs)]
        if len(valid) == 0:
            print(f"  {label}: no valid dominant frequencies")
            return
        n_bins   = 20
        counts, edges = np.histogram(valid, bins=n_bins)
        centers  = (edges[:-1] + edges[1:]) / 2
        max_c    = counts.max() if counts.max() > 0 else 1
        bar_w    = 40
        print(f"\n  {label} — dominant frequency distribution:")
        print(f"  mean={valid.mean():.5f}  median={np.median(valid):.5f}  "
              f"p75={np.percentile(valid, 75):.5f}  p90={np.percentile(valid, 90):.5f}")
        print(f"  {'freq':>10}  {'period(k)':>10}  {'count':>7}  bar")
        print(f"  {'-'*10}  {'-'*10}  {'-'*7}  ---")
        for c_val, f_val, cnt in zip(centers, edges, counts):
            period = 1.0 / c_val if c_val > 0 else float("inf")
            bar    = "#" * int(cnt / max_c * bar_w)
            print(f"  {c_val:>10.5f}  {period:>10.2f}  {cnt:>7,}  |{bar}")

    _freq_hist(dom_noisy,  "Noisy tokens")
    _freq_hist(dom_stable, "Stable-correct tokens")

    # Comparison: are noisy tokens dominated by higher frequencies?
    valid_noisy  = dom_noisy[~np.isnan(dom_noisy)]
    valid_stable = dom_stable[~np.isnan(dom_stable)]
    if len(valid_noisy) > 0 and len(valid_stable) > 0:
        print(f"\n  Frequency comparison (noisy vs stable-correct):")
        print(f"  {'':30}  {'noisy':>12}  {'stable-correct':>14}")
        print(f"  {'mean dom. freq (cyc/1k)':30}  {valid_noisy.mean():>12.5f}  "
              f"{valid_stable.mean():>14.5f}")
        print(f"  {'median dom. freq (cyc/1k)':30}  {np.median(valid_noisy):>12.5f}  "
              f"{np.median(valid_stable):>14.5f}")
        ratio = valid_noisy.mean() / valid_stable.mean() if valid_stable.mean() > 0 else float("nan")
        print(f"  {'ratio noisy/stable':30}  {ratio:>12.3f}")
        print(f"  (>1 = noisy tokens oscillate at higher frequencies)")

    # Top token IDs among highest-frequency noisy tokens
    if len(valid_noisy) > 0:
        high_freq_thresh = np.percentile(valid_noisy, 75)
        high_freq_mask   = dom_noisy >= high_freq_thresh
        high_freq_tids   = token_ids[noisy_idx[high_freq_mask & ~np.isnan(dom_noisy)]].tolist()
        top_tids         = Counter(high_freq_tids).most_common(10)
        print(f"\n  Top token IDs in high-frequency noisy tokens "
              f"(dom_freq ≥ p75={high_freq_thresh:.5f}):")
        print(f"  {'token_id':>10}  {'count':>7}  decoded")
        print(f"  {'-'*10}  {'-'*7}  -------")
        for tid, cnt in top_tids:
            dec = decode_token(tid, tokenizer)
            dec_d = repr(dec) if len(dec.strip()) == 0 else dec
            print(f"  {tid:>10}  {cnt:>7,}  {dec_d}")

    return dom_noisy, dom_stable, noisy_idx, stable_idx


# ---------------------------------------------------------------------------
# Analysis 3: spectral clustering
# ---------------------------------------------------------------------------

def spectral_clustering(
    prob_matrix: np.ndarray,
    correct_matrix: np.ndarray,
    token_ids: np.ndarray,
    steps: List[int],
    freqs: np.ndarray,
    tokenizer,
    rng: np.random.Generator,
    n_clusters: int = N_CLUSTERS,
    max_tokens: int = MAX_PER_TOKEN,
) -> None:
    """
    Embed each noisy token's power spectrum as a vector and cluster with k-means
    (after PCA dimensionality reduction).  Each cluster is characterised by its
    centroid spectrum and the token IDs it contains.

    Requires scikit-learn.
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
    except ImportError:
        print("  scikit-learn not available; skipping spectral clustering.")
        return

    instab    = instability_from_correct(correct_matrix)
    noisy_idx = np.where(instab >= NOISY_THRESHOLD)[0]
    if len(noisy_idx) > max_tokens:
        noisy_idx = rng.choice(noisy_idx, max_tokens, replace=False)

    print(f"\n  Spectral clustering: {len(noisy_idx):,} noisy tokens → "
          f"{n_clusters} clusters")

    # Build power spectrum matrix (n_tokens_noisy, n_freqs)
    spectra = np.zeros((len(noisy_idx), len(freqs)), dtype=np.float32)
    for j, i in enumerate(noisy_idx):
        pwr = lomb_scargle_power(steps, prob_matrix[i], freqs)
        spectra[j] = pwr
        if (j + 1) % 500 == 0 or j == len(noisy_idx) - 1:
            print(f"    {j+1}/{len(noisy_idx)}", end="\r", flush=True)
    print()

    # L2-normalise each spectrum so clustering is based on shape, not amplitude
    spectra_norm = normalize(spectra, norm="l2")

    # PCA to reduce dimensionality before k-means (speeds up + stabilises)
    n_components = min(32, spectra_norm.shape[1], spectra_norm.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=RNG_SEED)
    embedded = pca.fit_transform(spectra_norm)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {n_components} components, {var_explained:.1%} variance explained")

    km = KMeans(n_clusters=n_clusters, random_state=RNG_SEED, n_init=10)
    labels = km.fit_predict(embedded)

    print(f"\n  Clusters (sorted by size):")
    for cl in sorted(range(n_clusters), key=lambda c: -(labels == c).sum()):
        mask_cl = labels == cl
        n_cl    = int(mask_cl.sum())
        # Centroid spectrum in original space (mean of member spectra)
        centroid = spectra[mask_cl].mean(axis=0)
        dom_freq = freqs[np.argmax(centroid)]
        period   = 1.0 / dom_freq if dom_freq > 0 else float("inf")
        # Instability stats for this cluster
        cl_instab = instab[noisy_idx[mask_cl]]
        # Top token IDs
        cl_tids   = token_ids[noisy_idx[mask_cl]].tolist()
        top_tids  = Counter(cl_tids).most_common(5)
        decoded   = ", ".join(
            f"{decode_token(t, tokenizer)!r}×{c}" for t, c in top_tids
        )
        print(f"\n  Cluster {cl}  (n={n_cl:,})")
        print(f"    Centroid dominant freq: {dom_freq:.5f} cyc/1k  "
              f"(period ≈ {period:.1f}k steps)")
        print(f"    Instability: mean={cl_instab.mean():.1f}  "
              f"median={np.median(cl_instab):.1f}  "
              f"max={cl_instab.max()}")
        print(f"    Top token IDs: {decoded}")

        # Print centroid spectrum shape (ASCII)
        bar_w  = 40
        max_cp = centroid.max() if centroid.max() > 0 else 1.0
        step_v = max(1, len(freqs) // 20)
        centroid_str = "|" + "".join(
            "#" if centroid[i] >= 0.5 * max_cp else "."
            for i in range(0, len(freqs), step_v)
        ) + "|"
        print(f"    Centroid shape (low→high freq): {centroid_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--dump-root", type=Path, default=DUMP_ROOT)
    parser.add_argument(
        "--source", default="c4_en",
        help="Data source to analyse (default: c4_en).",
    )
    parser.add_argument(
        "--max-checkpoints", type=int, default=DEFAULT_MAX_CKPTS,
        help="Max evenly-spaced checkpoints to load per model.",
    )
    parser.add_argument(
        "--per-token", action="store_true", default=False,
        help="Run per-token dominant-frequency analysis (slower).",
    )
    parser.add_argument(
        "--cluster", action="store_true", default=False,
        help="Run spectral clustering of the noisy population "
             "(requires --per-token and scikit-learn).",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=N_CLUSTERS,
        help=f"Number of spectral clusters (default: {N_CLUSTERS}).",
    )
    parser.add_argument(
        "--max-per-token", type=int, default=MAX_PER_TOKEN,
        help="Max noisy tokens to run per-token spectra on "
             f"(subsampled; default: {MAX_PER_TOKEN:,}).",
    )
    parser.add_argument(
        "--n-freqs", type=int, default=N_FREQS,
        help=f"Number of frequency grid points (default: {N_FREQS}).",
    )
    parser.add_argument(
        "--no-tokenizer", action="store_true",
        help="Skip tokenizer loading (show raw token IDs only).",
    )
    parser.add_argument(
        "--plot-dir", type=Path, default=None,
        help="If set, save plots as PNG files here.",
    )
    args = parser.parse_args()

    if args.model is None and not args.models:
        args.models = sorted(
            p.name for p in args.dump_root.iterdir() if p.is_dir()
        )
    model_list = args.models or ([args.model] if args.model else [])
    tokenizer  = None if args.no_tokenizer else load_tokenizer()
    rng        = np.random.default_rng(RNG_SEED)

    if tokenizer:
        print("Tokenizer loaded.")
    else:
        print("Tokenizer not available — showing raw token IDs.")

    for model_name in model_list:
        model_dir = args.dump_root / model_name
        if not model_dir.is_dir():
            print(f"Skipping {model_name}: directory not found")
            continue
        all_ckpts = sorted(
            [f for f in model_dir.glob("*.npz") if not f.name.endswith(".tmp.npz")],
            key=step_from_path,
        )
        if not all_ckpts:
            print(f"Skipping {model_name}: no checkpoints")
            continue

        selected = pick_evenly_spaced(all_ckpts, args.max_checkpoints)
        short    = model_name.replace("allenai_", "")

        print(f"\n{'='*70}")
        print(f"Model: {short}  ({len(selected)} checkpoints, source: {args.source})")
        print(f"  Step range: {step_from_path(selected[0]):,} "
              f"→ {step_from_path(selected[-1]):,}")
        print(f"{'='*70}")

        try:
            print("  Building prob trajectories...")
            prob_matrix, correct_matrix, token_ids, pos_keys, steps = \
                build_prob_trajectory(selected, args.source)
        except KeyError as e:
            print(f"  Skipping: {e}")
            continue

        n_tokens, n_steps = prob_matrix.shape
        nan_frac = np.isnan(prob_matrix).mean()
        print(f"  {n_tokens:,} tokens × {n_steps} checkpoints  "
              f"(NaN fraction: {nan_frac:.1%})")

        if nan_frac > 0.9:
            print("  WARNING: >90% NaN — correct_logit/log_z fields may be missing "
                  "in these dumps.  Mean-trajectory spectrum will be unreliable.")

        freqs = _freq_grid(steps, args.n_freqs)
        print(f"  Frequency grid: {freqs[0]:.5f} – {freqs[-1]:.5f} cyc/1k steps  "
              f"(periods: {1/freqs[-1]:.1f}k – {1/freqs[0]:.1f}k steps)")

        # --- Analysis 1: mean trajectory ---
        mean_power = mean_trajectory_spectrum(prob_matrix, steps, freqs)

        # --- Analysis 2: per-token ---
        dom_noisy = dom_stable = noisy_idx_out = stable_idx_out = None
        if args.per_token:
            result = per_token_spectrum_analysis(
                prob_matrix, correct_matrix, token_ids, steps, freqs,
                tokenizer, rng, max_tokens=args.max_per_token,
            )
            if result is not None:
                dom_noisy, dom_stable, noisy_idx_out, stable_idx_out = result

        # --- Analysis 3: clustering ---
        if args.cluster and args.per_token:
            spectral_clustering(
                prob_matrix, correct_matrix, token_ids, steps, freqs,
                tokenizer, rng,
                n_clusters=args.n_clusters,
                max_tokens=args.max_per_token,
            )

        # --- Plots ---
        if args.plot_dir:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                args.plot_dir.mkdir(parents=True, exist_ok=True)
                step_arr = np.array(steps)

                # Plot 1: mean trajectory + its spectrum
                fig, axes = plt.subplots(2, 1, figsize=(14, 9))
                mean_traj = np.nanmean(prob_matrix, axis=0)
                axes[0].plot(step_arr, mean_traj)
                axes[0].set_xlabel("Training step")
                axes[0].set_ylabel("Mean correct_prob")
                axes[0].set_title(f"{short} — {args.source}: mean prob trajectory")
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(freqs, mean_power)
                axes[1].set_xlabel("Frequency (cycles / 1k training steps)")
                axes[1].set_ylabel("Normalised power")
                axes[1].set_title(f"{short} — {args.source}: mean trajectory spectrum")
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                out = args.plot_dir / f"{model_name}_{args.source}_mean_spectrum.png"
                plt.savefig(out, dpi=120)
                plt.close()
                print(f"\n  Plot saved: {out}")

                # Plot 2: per-token dominant frequency comparison
                if dom_noisy is not None and dom_stable is not None:
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
                    for ax, dom, label, color in [
                        (axes[0], dom_noisy,  "Noisy tokens",        "crimson"),
                        (axes[1], dom_stable, "Stable-correct tokens", "steelblue"),
                    ]:
                        valid = dom[~np.isnan(dom)]
                        ax.hist(valid, bins=30, color=color, alpha=0.8)
                        ax.set_xlabel("Dominant frequency (cyc/1k steps)")
                        ax.set_ylabel("# tokens")
                        ax.set_title(f"{short} — {label}")
                        ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    out2 = args.plot_dir / \
                        f"{model_name}_{args.source}_per_token_freq.png"
                    plt.savefig(out2, dpi=120)
                    plt.close()
                    print(f"  Plot saved: {out2}")

            except ImportError:
                print("  matplotlib not available; skipping plots.")


if __name__ == "__main__":
    main()
