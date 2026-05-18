"""
Compute top-1 accuracy across training checkpoints for all available model dumps.

For each model, picks 4-6 evenly-spaced checkpoint files and reports per-source
top-1 accuracy (fraction of positions where the model's argmax == the correct token).

Top-1 accuracy is simply: token_id == max_token_id
"""

import random
from pathlib import Path

import numpy as np

DUMP_ROOT = Path("/weka/oe-training-default/yashasbls/georges-functional-analysis")
MODELS = sorted(p.name for p in DUMP_ROOT.iterdir() if p.is_dir())
N_SAMPLES = 5  # checkpoints to pick per model
SAMPLE_TOKENS = 200_000  # tokens to sample per checkpoint for speed
RNG_SEED = 42


def top1_acc(arr: np.ndarray, rng: np.random.Generator) -> float:
    idx = rng.choice(len(arr), size=min(SAMPLE_TOKENS, len(arr)), replace=False)
    s = arr[idx]
    return (s["token_id"] == s["max_token_id"]).mean()


def pick_checkpoints(paths: list[Path], n: int) -> list[Path]:
    """Pick n evenly-spaced files, always including the first and last."""
    if len(paths) <= n:
        return paths
    indices = [round(i * (len(paths) - 1) / (n - 1)) for i in range(n)]
    return [paths[i] for i in sorted(set(indices))]


def step_from_path(p: Path) -> int:
    try:
        return int(p.stem.split("-step")[1].split("-")[0])
    except (IndexError, ValueError):
        return -1


rng = np.random.default_rng(RNG_SEED)

for model in MODELS:
    model_dir = DUMP_ROOT / model
    npz_files = sorted(
        [f for f in model_dir.glob("*.npz") if not f.name.endswith(".tmp.npz")],
        key=step_from_path,
    )
    if not npz_files:
        print(f"\n{model}: no dumps found")
        continue

    selected = pick_checkpoints(npz_files, N_SAMPLES)
    print(f"\n{'='*60}")
    print(f"Model: {model}  ({len(npz_files)} total checkpoints, showing {len(selected)})")
    print(f"{'checkpoint':<45} {'step':>10}  {'c4_en acc':>10}  {'avg acc':>10}")
    print(f"{'-'*45} {'-'*10}  {'-'*10}  {'-'*10}")

    for npz_path in selected:
        step = step_from_path(npz_path)
        try:
            data = np.load(npz_path)
            sources = [k for k in data.files if "__topk" not in k]
            accs = {}
            for src in sources:
                arr = data[src]
                if "token_id" not in arr.dtype.names or "max_token_id" not in arr.dtype.names:
                    continue
                accs[src] = top1_acc(arr, rng)
            c4_acc = accs.get("c4_en", float("nan"))
            avg_acc = np.mean(list(accs.values())) if accs else float("nan")
            print(f"{npz_path.name:<45} {step:>10,}  {c4_acc:>10.3%}  {avg_acc:>10.3%}")
        except Exception as e:
            print(f"{npz_path.name:<45} {step:>10,}  ERROR: {e}")
