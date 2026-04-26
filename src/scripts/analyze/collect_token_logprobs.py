"""
Collect per-token log-prob statistics from a sequence of model checkpoints.

For each checkpoint this script runs a forward pass over a fixed validation
corpus and saves a structured numpy array with:

    [correct_logit, log_z, correct_prob, top1_prob, mask]

per token (all float16 except mask which is int8).

Storage: ~450 MB per checkpoint (50 M tokens).

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
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.distributed as dist

from olmo_core.data import (
    DataCollator,
    DataMix,
    NumpyFSLDataLoader,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.collator import PaddingDirection
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.distributed.utils import get_fs_local_rank, get_rank, get_world_size
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
        ("log_z", np.float16),
        ("correct_prob", np.float16),
        ("top1_prob", np.float16),
        ("mask", np.int8),
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
) -> NumpyFSLDataLoader:
    """Build a fixed data loader over the dolma_wiki validation split."""
    dataset_cfg = NumpyPaddedFSLDatasetConfig.from_data_mix(
        DataMix.v3_small_ppl_validation,
        mix_base_dir=data_root,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
        work_dir=work_dir,
    )
    dataset: NumpyPaddedFSLDataset = dataset_cfg.build()
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

    total_dataset_tokens = len(dataset) * sequence_length
    if total_dataset_tokens < target_tokens:
        log.warning(
            f"Dataset only has {total_dataset_tokens:,} tokens but target is "
            f"{target_tokens:,}. Will use all available tokens."
        )

    return data_loader


@torch.no_grad()
def collect_stats(
    model: torch.nn.Module,
    data_loader: NumpyFSLDataLoader,
    target_tokens: int,
    device: torch.device,
    label_ignore_index: int = -100,
) -> np.ndarray:
    """
    Run forward passes and accumulate per-token statistics.

    Returns a structured numpy array of shape (N,) with OUTPUT_DTYPE, where N
    is the number of non-padding tokens (up to target_tokens).
    """
    model.eval()
    buffers: List[np.ndarray] = []
    total_tokens = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)  # (B, T)
        labels = batch["input_ids"].to(device)      # labels = input_ids shifted inside model
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
        correct_prob  = (correct_logit - log_z).exp()                     # (B, T-1)
        top1_prob     = (pred_logits.max(dim=-1).values - log_z).exp()   # (B, T-1)

        del pred_logits  # free V-dim tensor immediately

        # Flatten and convert to numpy
        n = B * (T - 1)
        rec = np.empty(n, dtype=OUTPUT_DTYPE)
        rec["correct_logit"] = correct_logit.cpu().half().numpy().ravel()
        rec["log_z"]         = log_z.cpu().half().numpy().ravel()
        rec["correct_prob"]  = correct_prob.cpu().half().numpy().ravel()
        rec["top1_prob"]     = top1_prob.cpu().half().numpy().ravel()
        rec["mask"]          = mask.cpu().numpy().ravel()

        buffers.append(rec)
        total_tokens += int(mask.sum().item())

        if total_tokens >= target_tokens:
            break

    result = np.concatenate(buffers)
    log.info(f"Collected {total_tokens:,} non-padding tokens ({len(result):,} total positions)")
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
    # Distributed setup (optional — works fine single-GPU too)
    # ---------------------------------------------------------------------------
    is_dist = dist.is_available() and dist.is_initialized()
    rank = get_rank() if is_dist else 0
    world_size = get_world_size() if is_dist else 1

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
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
    data_loader = build_data_loader(
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
        tag = checkpoint_tag(ckpt_dir)
        out_path = output_dir / f"{tag}.npy"

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

        data_loader.reshuffle(in_memory=True)
        data_loader.reset()

        result = collect_stats(
            model=model,
            data_loader=data_loader,
            target_tokens=args.target_tokens,
            device=device,
        )

        np.save(out_path, result)
        log.info(f"[{tag}] Saved {len(result):,} records to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

        gc_cuda()

    log.info("Done.")


if __name__ == "__main__":
    main()
