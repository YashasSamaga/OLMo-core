"""
Collect per-token log-prob statistics from HuggingFace model checkpoints.

This is the HF-native variant of ``collect_token_logprobs.py``.  Instead of
loading OLMo-core-format checkpoints, it pulls weights directly from
HuggingFace Hub revisions (branches) using ``AutoModelForCausalLM``.

For each revision this script runs a forward pass over a fixed validation
corpus and saves a compressed ``.npz`` file with one structured array per
data source (e.g. ``c4_en``, ``dolma_wiki``, ``pile``, etc.).  Each array
contains only non-padding tokens with fields:

    [correct_logit, max_logit, log_z,
     instance_index, token_id, max_token_id, position_in_seq]

per token (float16 for logit fields, uint32 for instance_index / token_id /
max_token_id, uint16 for position_in_seq).  20 bytes per token.

When ``--top-k K`` is given (K > 0), two additional arrays are stored per
source: ``{source}__topk_logits`` of shape ``(N, K)`` (float16) and
``{source}__topk_indices`` of shape ``(N, K)`` (uint32), containing the
logits and token IDs of the K highest-scoring tokens at each position.

The (instance_index, position_in_seq) pair uniquely identifies every
token across all checkpoint dumps, making outputs directly comparable
to those from ``collect_token_logprobs.py``.

Loading a specific source from a checkpoint::

    data = np.load("stage1-step10000-tokens21B.npz")
    c4 = data["c4_en"]
    log_prob = c4["correct_logit"] - c4["log_z"]
    topk_logits = data["c4_en__topk_logits"]    # (N, K) float16, if --top-k was used
    topk_indices = data["c4_en__topk_indices"]  # (N, K) uint32

Usage (single GPU)::

    python src/scripts/analyze/collect_token_logprobs_hf.py \\
        --model allenai/OLMo-2-0425-1B \\
        --revisions stage1-step{0..990000..10000} \\
        --output-dir /weka/oe-training-default/ai2-llm/checkpoints/trajectory-logprobs/olmo2-1b \\
        --target-tokens 50_000_000

Usage (multi-GPU, one revision per GPU)::

    torchrun --nproc-per-node=8 src/scripts/analyze/collect_token_logprobs_hf.py \\
        --model allenai/OLMo-2-0425-1B \\
        --revisions stage1-step{0..990000..10000} \\
        --output-dir <dir> \\
        --target-tokens 50_000_000

Each GPU processes a disjoint slice of the revision list.  All GPUs use the
same dataset so outputs are byte-for-byte comparable across revisions.
"""

import argparse
import gc
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from olmo_core.data import (
    DataCollator,
    DataMix,
    NumpyFSLDataLoader,
    NumpyPaddedFSLDataset,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.collator import PaddingDirection
from olmo_core.distributed.utils import (
    get_fs_local_rank,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
)
from olmo_core.utils import gc_cuda, prepare_cli_environment, seed_all

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output dtype — identical to collect_token_logprobs.py for compatibility
# ---------------------------------------------------------------------------
OUTPUT_DTYPE = np.dtype(
    [
        ("correct_logit", np.float16),
        ("max_logit", np.float16),
        ("log_z", np.float16),
        ("instance_index", np.uint32),   # dataset instance (≈ document) index
        ("token_id", np.uint32),          # correct next token id
        ("max_token_id", np.uint32),      # model's top prediction token id
        ("position_in_seq", np.uint16),   # 0-based position within the sequence
    ]
)

# ---------------------------------------------------------------------------
# Data root — Weka if available, otherwise GCS public endpoint
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = "/weka/oe-training-default/ai2-llm"


def build_source_map(dataset: NumpyPaddedFSLDataset) -> Dict[int, str]:
    """
    Build a mapping from instance index to source name.

    Returns a dict ``{instance_idx: source_name}`` covering every instance in
    the dataset.  Source names are derived from the directory structure
    (e.g. ``.../c4_en/val/part-0-00000.npy`` → ``c4_en``).
    """
    source_for_path: Dict[int, str] = {}
    for path_idx, path_str in enumerate(dataset.paths):
        parts = Path(path_str).parts
        source_for_path[path_idx] = parts[-3] if len(parts) >= 3 else Path(path_str).stem

    instance_to_source: Dict[int, str] = {}
    for path_idx, (start, end) in enumerate(dataset.offsets):
        source = source_for_path[path_idx]
        for idx in range(start, end):
            instance_to_source[idx] = source
    return instance_to_source


def build_data_loader(
    data_root: str,
    tokenizer: TokenizerConfig,
    sequence_length: int,
    global_batch_size: int,
    work_dir: str,
    target_tokens: int,
) -> Tuple[NumpyFSLDataLoader, Dict[int, str]]:
    """Build a fixed data loader over v3_small_ppl_validation.

    Returns (data_loader, instance_to_source) where instance_to_source maps
    each dataset instance index to its source name (e.g. 'c4_en').
    """
    dataset_cfg = NumpyPaddedFSLDatasetConfig.from_data_mix(
        DataMix.v3_small_ppl_validation,
        mix_base_dir=data_root,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
        work_dir=work_dir,
    )
    dataset: NumpyPaddedFSLDataset = dataset_cfg.build()  # type: ignore[assignment]
    dataset.prepare()

    collator = DataCollator(
        pad_token_id=tokenizer.pad_token_id,
        pad_direction=PaddingDirection.right,
        label_ignore_index=-100,
    )

    data_loader = NumpyFSLDataLoader(
        dataset,
        global_batch_size=global_batch_size,
        collator=collator,
        work_dir=work_dir,
        seed=0,
        dp_world_size=1,
        dp_rank=0,
        fs_local_rank=get_fs_local_rank(),
        target_device_type="cuda",
        num_workers=4,
    )

    sample_size = min(500, len(dataset))
    real_token_count = 0
    for i in range(sample_size):
        item = dataset[i]
        if "label_mask" in item:
            real_token_count += int(item["label_mask"].sum().item())
        else:
            real_token_count += len(item["input_ids"])
    avg_real_per_instance = real_token_count / sample_size
    estimated_real_tokens = int(avg_real_per_instance * len(dataset))

    log.info(
        f"Dataset: {len(dataset):,} instances, ~{estimated_real_tokens:,} real tokens "
        f"(~{100 * avg_real_per_instance / sequence_length:.0f}% non-padding per instance)"
    )
    if estimated_real_tokens < target_tokens:
        log.warning(
            f"Dataset has only ~{estimated_real_tokens:,} real (non-padding) tokens but target is "
            f"{target_tokens:,}. Will use all available tokens."
        )

    instance_to_source = build_source_map(dataset)
    return data_loader, instance_to_source


@torch.no_grad()
def collect_stats(
    model: torch.nn.Module,
    data_loader: NumpyFSLDataLoader,
    target_tokens: int,
    device: torch.device,
    instance_to_source: Dict[int, str],
    label_ignore_index: int = -100,
    top_k: int = 0,
    compute_batch_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Run forward passes and accumulate per-token statistics.

    Returns a dict mapping source name → structured numpy array of shape (N,)
    with OUTPUT_DTYPE, containing only non-padding tokens.

    When *top_k* > 0, two additional keys per source are included:
    ``{source}__topk_logits`` (N, K) float16 and ``{source}__topk_indices``
    (N, K) uint32.
    """
    model.eval()
    source_buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
    source_topk_logits: Dict[str, List[np.ndarray]] = defaultdict(list)
    source_topk_indices: Dict[str, List[np.ndarray]] = defaultdict(list)
    total_tokens = 0
    t0 = time.monotonic()

    pbar = tqdm(desc="Tokens", total=target_tokens, unit="tok", unit_scale=True)

    for batch in data_loader:
        full_input_ids = batch["input_ids"]  # (B, T) on CPU
        full_indices = batch["index"]        # (B,) on CPU
        full_label_mask = batch.get("label_mask")  # (B, T) or None
        B, T = full_input_ids.shape

        step = compute_batch_size or B
        assert B % step == 0, f"data batch size {B} must be divisible by compute_batch_size {step}"
        sub_recs: List[np.ndarray] = []
        sub_topk_logits: List[np.ndarray] = []
        sub_topk_ids: List[np.ndarray] = []

        for sb_start in range(0, B, step):
            input_ids = full_input_ids[sb_start:sb_start + step].to(device)
            sb_indices = full_indices[sb_start:sb_start + step]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, use_cache=False)
                pred_logits = outputs.logits[:, :-1, :].float()  # (step, T-1, V)  fp32

            target_ids = input_ids[:, 1:]  # (step, T-1)

            if full_label_mask is not None:
                mask = full_label_mask[sb_start:sb_start + step, 1:].to(device).to(torch.int8)
            else:
                mask = (target_ids != label_ignore_index).to(torch.int8)

            log_z         = torch.logsumexp(pred_logits, dim=-1)
            correct_logit = pred_logits.gather(
                -1, target_ids.clamp(min=0).unsqueeze(-1)
            ).squeeze(-1)
            max_logit, max_token_id = pred_logits.max(dim=-1)

            if top_k > 0:
                topk_vals, topk_ids = torch.topk(pred_logits, top_k, dim=-1)  # (step, T-1, K)

            del pred_logits

            inst_idx = sb_indices.unsqueeze(1).expand(step, T - 1)
            pos_in_seq = torch.arange(T - 1, device="cpu").unsqueeze(0).expand(step, T - 1)

            mask_np = mask.cpu().numpy().ravel()
            real_mask = mask_np == 1

            n_sub = int(real_mask.sum())
            rec = np.empty(n_sub, dtype=OUTPUT_DTYPE)
            rec["correct_logit"]  = correct_logit.cpu().half().numpy().ravel()[real_mask]
            rec["max_logit"]      = max_logit.cpu().half().numpy().ravel()[real_mask]
            rec["log_z"]          = log_z.cpu().half().numpy().ravel()[real_mask]
            rec["instance_index"] = inst_idx.numpy().ravel().astype(np.uint32)[real_mask]
            rec["token_id"]       = target_ids.cpu().numpy().ravel().astype(np.uint32)[real_mask]
            rec["max_token_id"]   = max_token_id.cpu().numpy().ravel().astype(np.uint32)[real_mask]
            rec["position_in_seq"] = pos_in_seq.numpy().ravel().astype(np.uint16)[real_mask]
            sub_recs.append(rec)

            if top_k > 0:
                sub_topk_logits.append(topk_vals.cpu().half().numpy().reshape(-1, top_k)[real_mask])
                sub_topk_ids.append(topk_ids.cpu().numpy().reshape(-1, top_k).astype(np.uint32)[real_mask])

        rec = np.concatenate(sub_recs)
        if top_k > 0:
            topk_logits_np = np.concatenate(sub_topk_logits)
            topk_ids_np = np.concatenate(sub_topk_ids)

        n = len(rec)
        total_tokens += n
        pbar.update(n)

        inst_indices = rec["instance_index"]
        unique_insts = np.unique(inst_indices)
        for uid in unique_insts:
            source = instance_to_source.get(int(uid), "unknown")
            uid_mask = inst_indices == uid
            source_buffers[source].append(rec[uid_mask])
            if top_k > 0:
                source_topk_logits[source].append(topk_logits_np[uid_mask])
                source_topk_indices[source].append(topk_ids_np[uid_mask])

        if total_tokens >= target_tokens:
            break

    pbar.close()
    elapsed = time.monotonic() - t0
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else float("inf")
    log.info(f"Forward passes: {elapsed:.1f}s, {tok_per_sec:,.0f} tok/s")

    result: Dict[str, np.ndarray] = {}
    for source, bufs in source_buffers.items():
        arr = np.concatenate(bufs)
        sort_key = np.lexsort((arr["position_in_seq"], arr["instance_index"]))
        result[source] = arr[sort_key]
        if top_k > 0:
            result[f"{source}__topk_logits"] = np.concatenate(
                source_topk_logits[source]
            )[sort_key]
            result[f"{source}__topk_indices"] = np.concatenate(
                source_topk_indices[source]
            )[sort_key]
    total_per_source = {s: len(a) for s, a in result.items() if "__topk" not in s}
    log.info(f"Collected {total_tokens:,} non-padding tokens across {len(total_per_source)} sources: {total_per_source}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g. 'allenai/OLMo-2-0425-1B').",
    )
    parser.add_argument(
        "--revisions",
        nargs="+",
        required=True,
        help="List of HF revisions (branches) to evaluate. "
             "Use bash brace expansion, e.g. stage1-step{0..990000..10000}.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write per-revision .npz files into.",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=50_000_000,
        help="Number of non-padding tokens to collect per revision (default: 50M).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=4096,
        help="Sequence length for the evaluation dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="If > 0, also store logits and token IDs for the top K predictions "
             "at each position (as {source}__topk_logits and {source}__topk_indices "
             "arrays in the .npz).",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Global batch size in tokens (controls dataset truncation). Defaults to 16 * sequence_length.",
    )
    parser.add_argument(
        "--compute-batch-size",
        type=int,
        default=None,
        help="Number of sequences per forward pass. Defaults to global_batch_size // sequence_length. "
             "Set this lower than --global-batch-size to reduce GPU memory usage without changing "
             "which tokens are collected.",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Root directory for eval data (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp/collect_token_logprobs",
        help="Local working directory for dataset preprocessing cache.",
    )
    return parser.parse_args()


def main():
    prepare_cli_environment()
    args = parse_args()

    # ---------------------------------------------------------------------------
    # Distributed setup
    # ---------------------------------------------------------------------------
    init_distributed()
    local_rank = get_local_rank()
    rank = get_rank()
    world_size = get_world_size()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    seed_all(42)

    # ---------------------------------------------------------------------------
    # Shard revision list across ranks
    # ---------------------------------------------------------------------------
    all_revisions = args.revisions
    my_revisions = all_revisions[rank::world_size]
    log.info(
        f"Rank {rank}/{world_size}: processing {len(my_revisions)}/{len(all_revisions)} revisions"
    )

    # ---------------------------------------------------------------------------
    # Output directory — nest under model ID (e.g. allenai_OLMo-2-0425-1B/)
    # ---------------------------------------------------------------------------
    safe_model = args.model.replace("/", "_")
    output_dir = Path(args.output_dir) / safe_model
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Build data loader (fixed across all revisions)
    # ---------------------------------------------------------------------------
    tokenizer = TokenizerConfig.dolma2()
    global_batch_size = args.global_batch_size or (16 * args.sequence_length)
    compute_batch_seqs = args.compute_batch_size  # None means use full loader batch
    if compute_batch_seqs is not None:
        data_batch_seqs = global_batch_size // args.sequence_length
        assert data_batch_seqs % compute_batch_seqs == 0, (
            f"--compute-batch-size {compute_batch_seqs} must divide "
            f"--global-batch-size // --sequence-length = {data_batch_seqs}"
        )
    data_loader, instance_to_source = build_data_loader(
        data_root=args.data_root,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        global_batch_size=global_batch_size,
        work_dir=args.work_dir,
        target_tokens=args.target_tokens,
    )

    # ---------------------------------------------------------------------------
    # Main loop — load a fresh model per revision from HF Hub
    # ---------------------------------------------------------------------------
    for rev_i, revision in enumerate(my_revisions, 1):
        safe_name = revision.replace("/", "_")
        out_path = output_dir / f"{safe_name}.npz"

        if out_path.exists():
            log.info(f"[{rev_i}/{len(my_revisions)}] [{revision}] Already exists, skipping.")
            continue

        rev_t0 = time.monotonic()
        log.info(f"[{rev_i}/{len(my_revisions)}] [{revision}] Loading {args.model} from HF Hub ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)
        # Sanity-check: model vocab must be compatible with the dolma2 tokenizer.
        model_vocab = model.config.vocab_size
        base_vocab = tokenizer.vocab_size
        if model_vocab < base_vocab:
            raise ValueError(
                f"Model vocab_size={model_vocab} is smaller than dolma2 "
                f"vocab_size={base_vocab}. The pre-tokenized eval data "
                f"is only valid for dolma2-tokenizer models."
            )
        model = torch.compile(model)
        log.info(f"[{rev_i}/{len(my_revisions)}] [{revision}] Model loaded and compiled. Running forward passes ...")

        data_loader.reshuffle(epoch=1, in_memory=True)

        result = collect_stats(
            model=model,
            data_loader=data_loader,
            target_tokens=args.target_tokens,
            device=device,
            instance_to_source=instance_to_source,
            top_k=args.top_k,
            compute_batch_size=compute_batch_seqs,
        )

        tmp_path = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp_path, **result)
        tmp_path.rename(out_path)
        total_records = sum(len(a) for k, a in result.items() if "__topk" not in k)
        rev_elapsed = time.monotonic() - rev_t0
        log.info(
            f"[{rev_i}/{len(my_revisions)}] [{revision}] Saved {total_records:,} records across "
            f"{len(result)} sources to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, "
            f"{rev_elapsed:.1f}s total)"
        )

        # Free the model before loading the next revision.
        del model
        gc.collect()
        data_loader.reset()
        gc_cuda()

    log.info("Done.")


if __name__ == "__main__":
    main()
