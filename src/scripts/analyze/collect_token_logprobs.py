"""
Collect per-token log-prob statistics from a sequence of model checkpoints.

For each checkpoint this script runs a forward pass over a fixed validation
corpus and saves a compressed ``.npz`` file with one structured array per
data source (e.g. ``c4_en``, ``dolma_wiki``, ``pile``, etc.).  Each array
contains only non-padding tokens with fields:

    [correct_logit, max_logit, log_z,
     instance_index, token_id, max_token_id, position_in_seq]

per token (float16 for logit fields, uint32 for instance_index / token_id /
max_token_id, uint16 for position_in_seq).  20 bytes per token.

The (instance_index, position_in_seq) pair uniquely identifies every
token across all checkpoint dumps.

Loading a specific source from a checkpoint::

    data = np.load("step001000.npz")
    c4 = data["c4_en"]          # structured array for c4_en only
    print(c4["correct_prob"])   # per-token correct-token probabilities

Usage (interactive A100 session, single GPU):
    python src/scripts/analyze/collect_token_logprobs.py \\
        --checkpoint-dirs gs://ai2-llm/checkpoints/shanea/OLMo-medium/peteish7/step{1000..928000..1000} \\
        --output-dir /weka/oe-training-default/ai2-llm/checkpoints/trajectory-logprobs/peteish7 \\
        --model olmo2_7B \\
        --target-tokens 50_000_000

Usage (multi-GPU to shard checkpoints across GPUs):
    torchrun --nproc-per-node=8 src/scripts/analyze/collect_token_logprobs.py \\
        --checkpoint-dirs <dirs> \\
        --output-dir <dir> \\
        --model olmo2_7B \\
        --target-tokens 50_000_000

Each GPU processes a disjoint slice of the checkpoint list.  All GPUs use the
same dataset so outputs are byte-for-byte comparable across checkpoints.
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from olmo_core.data import (
    DataCollator,
    DataMix,
    NumpyFSLDataLoader,
    NumpyPaddedFSLDataset,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.collator import PaddingDirection


def build_source_map(dataset: NumpyPaddedFSLDataset) -> Dict[int, str]:
    """
    Build a mapping from instance index to source name.

    Returns a dict ``{instance_idx: source_name}`` covering every instance in
    the dataset.  Source names are derived from the directory structure
    (e.g. ``.../c4_en/val/part-0-00000.npy`` → ``c4_en``).
    """
    # path_idx -> source name
    source_for_path: Dict[int, str] = {}
    for path_idx, path_str in enumerate(dataset.paths):
        parts = Path(path_str).parts
        source_for_path[path_idx] = parts[-3] if len(parts) >= 3 else Path(path_str).stem

    # instance_idx -> source name  (store range boundaries for fast lookup)
    instance_to_source: Dict[int, str] = {}
    for path_idx, (start, end) in enumerate(dataset.offsets):
        source = source_for_path[path_idx]
        for idx in range(start, end):
            instance_to_source[idx] = source
    return instance_to_source
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.distributed.utils import (
    get_fs_local_rank,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
)
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig
from olmo_core.utils import gc_cuda, prepare_cli_environment, seed_all

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output dtype
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
# Supported model configs
# ---------------------------------------------------------------------------

_HYBRID_7B_REMOVE_HEADS = 2


def _olmo_hybrid_7B(vocab_size: int, **kwargs) -> TransformerConfig:
    """OLMo 3.2 7B hybrid: 3 GDN layers + 1 attention layer, repeating."""
    config = TransformerConfig.olmo3_7B(vocab_size=vocab_size, **kwargs)
    assert isinstance(config.block, TransformerBlockConfig)
    assert isinstance(config.block.sequence_mixer, AttentionConfig)

    config.d_model -= _HYBRID_7B_REMOVE_HEADS * 128
    num_heads = config.block.sequence_mixer.n_heads - _HYBRID_7B_REMOVE_HEADS
    config.block.sequence_mixer.n_heads = num_heads

    attn_block = config.block
    gdn_block = attn_block.replace(
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=num_heads,
            head_dim=int(0.75 * config.d_model / num_heads),
            allow_neg_eigval=True,
        ),
    )
    config.block = {"gdn": gdn_block, "attn": attn_block}
    config.block_pattern = ["gdn", "gdn", "gdn", "attn"]
    return config


MODEL_CONFIGS = {
    "olmo2_1B": TransformerConfig.olmo2_1B_v2,
    "olmo2_7B": TransformerConfig.olmo2_7B,
    "olmo2_13B": TransformerConfig.olmo2_13B,
    "olmo2_32B": TransformerConfig.olmo2_32B,
    "olmo3_7B": TransformerConfig.olmo3_7B,
    "olmo3_32B": TransformerConfig.olmo3_32B,
    "olmo_hybrid_7B": _olmo_hybrid_7B,
}

# ---------------------------------------------------------------------------
# Data root — Weka if available, otherwise GCS public endpoint
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = "/weka/oe-training-default/ai2-llm"


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

    # Estimate real (non-padding) token count.
    # NumpyPaddedFSLDataset pads short documents, so len(dataset) * seq_len vastly
    # overcounts.  Sample a few instances to estimate the actual non-padding fraction.
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
) -> Dict[str, np.ndarray]:
    """
    Run forward passes and accumulate per-token statistics.

    Returns a dict mapping source name → structured numpy array of shape (N,)
    with OUTPUT_DTYPE, containing only non-padding tokens.
    """
    model.eval()
    source_buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
    total_tokens = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)  # (B, T)
        # The standard convention: label at position t is input_ids[t+1].
        # NumpyPaddedFSLDataset already packs sequences so we can treat every
        # token except the last one as a prediction target.
        B, T = input_ids.shape

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)  # (B, T, V)

        logits = logits.float()  # upcast before numerical ops

        # Prediction: token t predicts token t+1
        pred_logits = logits[:, :-1, :]       # (B, T-1, V)
        target_ids  = input_ids[:, 1:]        # (B, T-1)

        # Build mask: 1 where target is a real token, 0 at padding
        # label_mask may be present in the batch; fall back to all-ones
        if "label_mask" in batch:
            mask = batch["label_mask"][:, 1:].to(device).to(torch.int8)  # (B, T-1)
        else:
            mask = (target_ids != label_ignore_index).to(torch.int8)     # (B, T-1)

        # Compute per-token stats — gather before logsumexp to keep peak memory low
        log_z         = torch.logsumexp(pred_logits, dim=-1)              # (B, T-1)
        correct_logit = pred_logits.gather(
            -1, target_ids.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)                                                      # (B, T-1)
        max_logit, max_token_id = pred_logits.max(dim=-1)                 # (B, T-1) each

        del pred_logits  # free V-dim tensor immediately

        # Build per-token identifiers:
        #   instance_index: (B,) -> broadcast to (B, T-1)
        #   token_id:       target_ids already (B, T-1)
        #   position_in_seq: 0..T-2 for each row
        inst_idx = batch["index"].unsqueeze(1).expand(B, T - 1)   # (B, T-1)
        pos_in_seq = torch.arange(T - 1, device="cpu").unsqueeze(0).expand(B, T - 1)  # (B, T-1)

        # Flatten and filter out padding positions.
        mask_np = mask.cpu().numpy().ravel()  # 1 = real, 0 = padding
        real_mask = mask_np == 1

        n = int(real_mask.sum())
        rec = np.empty(n, dtype=OUTPUT_DTYPE)
        rec["correct_logit"]  = correct_logit.cpu().half().numpy().ravel()[real_mask]
        rec["max_logit"]      = max_logit.cpu().half().numpy().ravel()[real_mask]
        rec["log_z"]          = log_z.cpu().half().numpy().ravel()[real_mask]
        rec["instance_index"] = inst_idx.numpy().ravel().astype(np.uint32)[real_mask]
        rec["token_id"]       = target_ids.cpu().numpy().ravel().astype(np.uint32)[real_mask]
        rec["max_token_id"]   = max_token_id.cpu().numpy().ravel().astype(np.uint32)[real_mask]
        rec["position_in_seq"] = pos_in_seq.numpy().ravel().astype(np.uint16)[real_mask]

        total_tokens += n

        # Group by source using instance_index -> source mapping.
        inst_indices = rec["instance_index"]
        # Vectorized lookup: get unique instance indices in this batch.
        unique_insts = np.unique(inst_indices)
        for uid in unique_insts:
            source = instance_to_source.get(int(uid), "unknown")
            source_buffers[source].append(rec[inst_indices == uid])

        if total_tokens >= target_tokens:
            break

    result: Dict[str, np.ndarray] = {}
    for source, bufs in source_buffers.items():
        arr = np.concatenate(bufs)
        # Sort by (instance_index, position_in_seq) to get a canonical token
        # ordering that is independent of batch size or loader details.
        sort_key = np.lexsort((arr["position_in_seq"], arr["instance_index"]))
        result[source] = arr[sort_key]
    total_per_source = {s: len(a) for s, a in result.items()}
    log.info(f"Collected {total_tokens:,} non-padding tokens across {len(result)} sources: {total_per_source}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint-dirs",
        nargs="+",
        required=True,
        help="List of checkpoint directories (local or gs:// / weka paths). "
             "Each must point to a directory containing model_and_optim/.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write per-checkpoint .npy files into.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        required=True,
        help="Model architecture.",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=50_000_000,
        help="Number of non-padding tokens to collect per checkpoint (default: 50M).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=4096,
        help="Sequence length for the evaluation dataset.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Global batch size in tokens. Defaults to 8 * sequence_length.",
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
    parser.add_argument(
        "--step-tag",
        default=None,
        help="Optional tag to embed in output filenames alongside the checkpoint path hash. "
             "If omitted, the last path component is used (e.g. 'step001000').",
    )
    return parser.parse_args()


def checkpoint_tag(ckpt_dir: str) -> str:
    """Extract a short tag from a checkpoint path, e.g. 'step001000'."""
    return Path(ckpt_dir.rstrip("/")).name


def main():
    prepare_cli_environment()
    args = parse_args()

    # ---------------------------------------------------------------------------
    # Distributed setup — always initialize (single-GPU gets world_size=1).
    # load_model_and_optim_state requires an initialized process group.
    # ---------------------------------------------------------------------------
    init_distributed()
    local_rank = get_local_rank()
    rank = get_rank()
    world_size = get_world_size()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    seed_all(42)

    # ---------------------------------------------------------------------------
    # Shard checkpoint list across ranks (each rank processes a disjoint slice)
    # ---------------------------------------------------------------------------
    all_checkpoints = args.checkpoint_dirs
    my_checkpoints = all_checkpoints[rank::world_size]
    log.info(f"Rank {rank}/{world_size}: processing {len(my_checkpoints)}/{len(all_checkpoints)} checkpoints")

    # ---------------------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Build model (once — weights are reloaded per checkpoint)
    # ---------------------------------------------------------------------------
    tokenizer = TokenizerConfig.dolma2()
    model_cfg_fn = MODEL_CONFIGS[args.model]
    model_cfg = model_cfg_fn(vocab_size=tokenizer.padded_vocab_size())
    model = model_cfg.build(init_device=str(device))
    model = model.to(dtype=torch.bfloat16)

    # ---------------------------------------------------------------------------
    # Build data loader (fixed across all checkpoints)
    # ---------------------------------------------------------------------------
    global_batch_size = args.global_batch_size or (8 * args.sequence_length)
    data_loader, instance_to_source = build_data_loader(
        data_root=args.data_root,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        global_batch_size=global_batch_size,
        work_dir=args.work_dir,
        target_tokens=args.target_tokens,
    )

    # ---------------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------------
    for ckpt_dir in my_checkpoints:
        tag = args.step_tag if args.step_tag else checkpoint_tag(ckpt_dir)
        out_path = output_dir / f"{tag}.npz"

        if out_path.exists():
            log.info(f"[{tag}] Already exists, skipping.")
            continue

        log.info(f"[{tag}] Loading checkpoint from {ckpt_dir} ...")
        ckpt_model_path = ckpt_dir.rstrip("/") + "/model_and_optim"
        load_model_and_optim_state(
            ckpt_model_path,
            model,
            optim=None,
            strict=True,
        )
        log.info(f"[{tag}] Checkpoint loaded. Running forward passes ...")

        # Pin epoch=1 so every checkpoint sees the identical data order.
        data_loader.reshuffle(epoch=1, in_memory=True)

        result = collect_stats(
            model=model,
            data_loader=data_loader,
            target_tokens=args.target_tokens,
            device=device,
            instance_to_source=instance_to_source,
        )

        np.savez_compressed(out_path, **result)
        total_records = sum(len(a) for a in result.values())
        log.info(
            f"[{tag}] Saved {total_records:,} records across {len(result)} sources "
            f"to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)"
        )

        # Reset bookkeeping after iteration (required by DataLoaderBase protocol).
        data_loader.reset()
        gc_cuda()

    log.info("Done.")


if __name__ == "__main__":
    main()
