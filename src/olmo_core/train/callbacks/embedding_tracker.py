"""Callback to track embedding diagnostics (token usage, coverage)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from olmo_core.distributed.utils import get_full_tensor

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class EmbeddingDiagnosticsCallback(Callback):
    """
    Tracks embedding-related diagnostics during training.

    When ``track_unique_tokens`` is enabled, this callback counts the number of distinct
    token IDs that appear at least once in the **global** batch at each logged step.
    Local per-rank counts are accumulated via :func:`torch.bincount` and then
    :func:`dist.all_reduce` (SUM) so the result reflects the full data-parallel batch,
    not a single shard.
    """

    enabled: bool = False
    log_interval: Optional[int] = None

    track_unique_tokens: bool = True
    """Log the number of unique token IDs (with at least one occurrence) in the global batch."""

    track_weight_stats: bool = True
    """Log embedding weight RMS, mean, and stddev (computed over the full, unsharded weight)."""

    namespace: str = "embedding_diagnostics"
    """Metric namespace prefix."""

    # ---- internal state (not serialised) ----
    _handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list, repr=False)
    _embedding_token_ids: List[torch.Tensor] = field(default_factory=list, repr=False)
    _embedding_vocab_size: Optional[int] = field(default=None, repr=False)
    _embedding_module: Optional[nn.Embedding] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def post_attach(self):
        if not self.enabled:
            return

        model = getattr(self.trainer.train_module, "model", None)
        if model is None or not isinstance(model, nn.Module):
            return

        embeddings = getattr(model, "embeddings", None)
        if embeddings is not None and isinstance(embeddings, nn.Embedding):
            self._embedding_vocab_size = embeddings.num_embeddings
            if self.track_weight_stats:
                self._embedding_module = embeddings
            if self.track_unique_tokens:
                self._handles.append(
                    embeddings.register_forward_hook(self._make_embedding_forward_hook())
                )

    def post_train(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._embedding_token_ids.clear()
        self._embedding_module = None

    def post_step(self):
        if not self._should_log():
            self._embedding_token_ids.clear()
            return

        if self.track_unique_tokens:
            self._log_unique_token_metrics()
        if self.track_weight_stats:
            self._log_weight_stats()

        self._embedding_token_ids.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_log(self) -> bool:
        if not self.enabled:
            return False
        interval = self.log_interval or self.trainer.metrics_collect_interval
        if interval <= 0:
            return False
        return self.step % interval == 0

    def _log_unique_token_metrics(self):
        """Count the number of token IDs activated at least once across all ranks."""
        device = self.trainer.device
        vocab_size = self._embedding_vocab_size
        if vocab_size is None:
            return

        # Build local per-token activation counts as a [vocab_size] tensor.
        if self._embedding_token_ids:
            all_ids = torch.cat([t.view(-1) for t in self._embedding_token_ids]).to(
                device=device, dtype=torch.long
            )
            counts = torch.bincount(all_ids, minlength=vocab_size).float()
        else:
            counts = torch.zeros(vocab_size, dtype=torch.float32, device=device)

        # Reduce across all ranks so the counts reflect the full global batch.
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        # Number of token IDs with at least one occurrence globally.
        num_unique = (counts > 0).sum().float()
        self.trainer.record_metric(
            f"{self.namespace}/unique_tokens_in_step",
            num_unique,
            reduce_type=None,  # already globally reduced
        )

        # Median occurrence count over the tokens that appeared at least once.
        activated_counts = counts[counts > 0]
        if activated_counts.numel() > 0:
            median_count = torch.median(activated_counts)
        else:
            median_count = torch.tensor(0.0, device=device)
        self.trainer.record_metric(
            f"{self.namespace}/median_token_occurrences",
            median_count,
            reduce_type=None,  # already globally reduced
        )

        # How many top tokens account for X% of total occurrences.
        # Sort counts descending, compute cumulative share, find cutoff.
        if activated_counts.numel() > 0:
            sorted_counts, _ = activated_counts.sort(descending=True)
            cumsum = sorted_counts.cumsum(dim=0)
            total = cumsum[-1]
            for pct in (0.50, 0.75, 0.90):
                n_tokens = (cumsum < total * pct).sum().item() + 1
                self.trainer.record_metric(
                    f"{self.namespace}/tokens_for_{int(pct * 100)}pct_mass",
                    torch.tensor(float(n_tokens), device=device),
                    reduce_type=None,
                )
        else:
            for pct in (0.50, 0.75, 0.90):
                self.trainer.record_metric(
                    f"{self.namespace}/tokens_for_{int(pct * 100)}pct_mass",
                    torch.tensor(0.0, device=device),
                    reduce_type=None,
                )

    def _log_weight_stats(self):
        """Log RMS, mean, and stddev of the full embedding weight matrix."""
        if self._embedding_module is None:
            return
        w = get_full_tensor(self._embedding_module.weight.detach()).float()
        self.trainer.record_metric(
            f"{self.namespace}/weight_rms",
            w.pow(2).mean().sqrt(),
            reduce_type=None,
        )
        self.trainer.record_metric(
            f"{self.namespace}/weight_mean",
            w.mean(),
            reduce_type=None,
        )
        self.trainer.record_metric(
            f"{self.namespace}/weight_std",
            w.std(),
            reduce_type=None,
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _make_embedding_forward_hook(self):
        """Capture token IDs fed to the embedding layer."""

        def hook(
            module: nn.Module,
            inputs: Tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ):
            del module, output
            if not self._should_log():
                return
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            self._embedding_token_ids.append(inputs[0].detach())

        return hook
