"""Callback to track optimization diagnostics (residuals, gradients, params, norms)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from olmo_core.nn.layer_norm import LayerNorm, RMSNorm
from olmo_core.nn.residual_stream import ResidualStream
from olmo_core.distributed.utils import get_local_tensor

from ..common import MetricMergeStrategy, ReduceType
from .callback import Callback


@dataclass
class OptimizationDiagnosticsCallback(Callback):
    """
    Tracks optimization-related diagnostics.

    IMPORTANT: Metrics are computed on each rank's local shard only. We call
    :func:`get_local_tensor()` and then log with reduce/merge = mean, so the
    final logged value is the mean of per-rank local-shard statistics, NOT the
    true global (full-parameter/full-activation) statistic.

    Ratios computed from local norms can differ from ratios computed on the full
    tensors, especially under tensor/model parallelism.
    """

    enabled: bool = False
    log_interval: Optional[int] = None
    eps: float = 1e-8  # for ratios and norms.

    track_residual_updates: bool = False  # Log residual norm, update norm, and ratio per ResidualStream.

    eps_hit_tolerance: float = 0.1  # Extra fractional slack above eps to count as a hit.
    track_layer_norm_eps: bool = False  # Log eps-hit indicators for LayerNorm/RMSNorm.

    track_param_grad_rmse: bool = False  # Log per-parameter grad RMSE.
    track_param_grad_meanvar: bool = False  # Log per-parameter grad mean/stddev.

    track_param_meanvar: bool = False  # Log per-parameter mean and stddev.
    track_update_param_ratio: bool = False  # Log per-parameter update/param ratio.

    track_activation_rmse: bool = False  # Log activation RMSE over all modules.
    track_activation_meanvar: bool = False  # Log activation mean/stddev over all modules.
    track_activation_norm: bool = False  # Log mean activation norm over batch/tokens for all modules.

    track_activation_grad_rmse: bool = False  # Log activation grad RMSE over all modules.
    track_activation_grad_meanvar: bool = False  # Log activation grad mean/stddev over all modules.
    track_activation_grad_norm: bool = False  # Log mean activation grad norm over batch/tokens for all modules.

    track_update_rmse: bool = False  # Log per-parameter update RMSE.

    track_optimizer_state_rmse_meanvar: bool = False  # Log optimizer state RMSE/mean/stddev.

    track_lm_head: bool = False  # Log LM head metrics (top-1 mass, max/avg/min logits).

    track_param_movement: bool = False  # Log fraction of parameters moving beyond threshold.
    param_movement_threshold: float = 0.01  # Threshold for "significant" parameter movement (0.1% by default).
    
    track_embedding_usage: bool = False  # Log embedding usage statistics (num activated, avg grad, activation counts).
    
    track_gradient_outliers: bool = False  # Log when gradients exceed mu ± k*sigma (uses AdamW exp_avg/exp_avg_sq)
    gradient_outlier_k: float = 6.0  # Number of standard deviations for outlier detection
    
    namespace: str = "optim_diagnostics"  # Metric namespace prefix.

    _handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list, repr=False)
    _prev_params: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)
    _param_name_map: Optional[Dict[int, str]] = field(default=None, repr=False)

    _embedding_token_ids: List[torch.Tensor] = field(default_factory=list, repr=False)  # Token IDs seen in forward pass

    def post_attach(self):
        if not self.enabled:
            return

        model = getattr(self.trainer.train_module, "model", None)
        if model is None or not isinstance(model, nn.Module):
            return

        self._param_name_map = {id(p): n for n, p in model.named_parameters()}

        if self.track_residual_updates:
            for name, module in model.named_modules():
                if isinstance(module, ResidualStream):
                    self._handles.append(
                        module.register_forward_hook(self._make_residual_stream_forward_hook(name))
                    )
                    self._handles.append(
                        module.register_full_backward_hook(
                            self._make_residual_stream_backward_hook(name)
                        )
                    )

        if (
            self.track_activation_rmse
            or self.track_activation_meanvar
            or self.track_activation_norm
            or self.track_activation_grad_rmse
            or self.track_activation_grad_meanvar
            or self.track_activation_grad_norm
        ):
            for name, module in model.named_modules():
                self._handles.append(
                    module.register_forward_hook(self._make_activation_forward_hook(name))
                )
                self._handles.append(
                    module.register_full_backward_hook(self._make_activation_backward_hook(name))
                )

        if self.track_layer_norm_eps:
            for name, module in model.named_modules():
                if isinstance(module, LayerNorm):
                    self._handles.append(
                        module.register_forward_hook(self._make_layer_norm_hook(name))
                    )

        if self.track_lm_head:
            lm_head = getattr(model, "lm_head", None)
            if lm_head is not None and isinstance(lm_head, nn.Module):
                self._handles.append(
                    lm_head.register_forward_hook(self._make_lm_head_hook())
                )

        if self.track_embedding_usage:
            embeddings = getattr(model, "embeddings", None)
            if embeddings is not None and isinstance(embeddings, nn.Embedding):
                self._handles.append(
                    embeddings.register_forward_hook(self._make_embedding_forward_hook())
                )

    def post_train(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._prev_params = None
        self._embedding_token_ids.clear()

    def pre_step(self, batch):
        del batch
        if not self._should_log():
            return

    def pre_optim_step(self):
        if not self._should_log():
            return

        model = getattr(self.trainer.train_module, "model", None)
        if model is None or not isinstance(model, nn.Module):
            return

        optim = getattr(self.trainer.train_module, "optim", None)
        if self.track_gradient_outliers:
            self._check_gradient_outliers(model, optim)

        if self.track_param_grad_rmse or self.track_param_grad_meanvar:
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                grad = get_local_tensor(param.grad.detach()).float()
                if self.track_param_grad_rmse:
                    grad_rmse = self._rmse(grad)
                    self._log_metric(f"grads/{name}/rmse", grad_rmse)
                if self.track_param_grad_meanvar:
                    grad_mean, grad_std = self._meanstd(grad)
                    self._log_metric(f"grads/{name}/mean", grad_mean)
                    self._log_metric(f"grads/{name}/stddev", grad_std)

        if self.track_update_param_ratio or self.track_param_movement:
            self._prev_params = {
                name: get_local_tensor(p.detach()).float().clone()
                for name, p in model.named_parameters()
                if p.requires_grad
            }

    def _should_log(self) -> bool:
        if not self.enabled:
            return False
        interval = self.log_interval or self.trainer.metrics_collect_interval
        if interval <= 0:
            return False
        return self.step % interval == 0

    def _log_metric(
        self,
        path: str,
        value: torch.Tensor,
        *,
        reduce_type: ReduceType = ReduceType.mean,
        merge_strategy: MetricMergeStrategy = MetricMergeStrategy.mean,
    ) -> None:
        self.trainer.record_metric(
            f"{self.namespace}/{path}",
            value,
            reduce_type=reduce_type,
            merge_strategy=merge_strategy,
        )

    def _rmse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute RMSE over all elements in the tensor."""
        return tensor.pow(2).mean().sqrt()

    def _meanstd(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean/stddev over all elements in the tensor."""
        mean = tensor.mean()
        std = tensor.var(unbiased=False).sqrt()
        return mean, std

    def _check_gradient_outliers(self, model: nn.Module, optim):
        """Check for gradient outliers (values beyond mu ± k*sigma) per parameter using AdamW state."""
        if optim is None or not hasattr(optim, 'state'):
            return
        
        # Check if optimizer is AdamW-like (has exp_avg and exp_avg_sq)
        # Works with torch.optim.AdamW and custom AdamW implementations
        if not any('exp_avg' in state and 'exp_avg_sq' in state for state in optim.state.values()):
            return
        
        for param in optim.state.keys():
            name = None
            if self._param_name_map is not None:
                name = self._param_name_map.get(id(param))
            if name is None:
                name = f"param_{id(param)}"
            
            if param.grad is None:
                continue
            
            state = optim.state[param]
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            
            # Need both exp_avg and exp_avg_sq to compute mean and std
            if exp_avg is None or exp_avg_sq is None:
                continue
            
            grad = get_local_tensor(param.grad.detach()).float()
            
            # Use AdamW's exponential moving averages to estimate mean and variance
            # exp_avg is the first moment (mean estimate)
            # exp_avg_sq is the second moment (used to compute variance)
            mean_est = get_local_tensor(exp_avg.detach()).float()
            second_moment = get_local_tensor(exp_avg_sq.detach()).float()
            
            # Variance = E[X^2] - E[X]^2
            variance = second_moment - mean_est.pow(2)
            variance = torch.clamp(variance, min=0)  # Ensure non-negative due to numerical issues
            std_est = variance.sqrt()
            
            # Check for outliers at multiple k-sigma thresholds (4 and 6)
            for k in [4.0, self.gradient_outlier_k]:
                lower_bound = mean_est - k * std_est
                upper_bound = mean_est + k * std_est
                
                # Count outliers
                outlier_mask = (grad < lower_bound) | (grad > upper_bound)
                has_outlier = outlier_mask.any()
                
                # Log binary indicator
                outlier_indicator = torch.tensor(1.0 if has_outlier else 0.0, device=grad.device)
                self._log_metric(
                    f"gradient_outliers/{name}/{int(k)}sigma_event",
                    outlier_indicator,
                )
                
                # Optionally log fraction of outliers
                if has_outlier:
                    outlier_frac = outlier_mask.float().mean()
                    self._log_metric(
                        f"gradient_outliers/{name}/{int(k)}sigma_fraction",
                        outlier_frac,
                    )

    def _mean_norm_over_batch_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute mean L2 norm over batch/tokens (all dims except the last)."""
        return tensor.norm(dim=-1).mean()

    def _vector_rmse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute RMSE per vector (last dim), then average over batch/tokens."""
        return tensor.pow(2).mean(dim=-1).sqrt().mean()

    def _vector_meanstd(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean/stddev per vector (last dim), then average over batch/tokens."""
        mean = tensor.mean(dim=-1).mean()
        std = tensor.var(dim=-1, unbiased=False).sqrt().mean()
        return mean, std

    def _log_embedding_metrics(self):
        if not self._embedding_token_ids:
            return

        # Collect all token IDs from all forward passes in this step
        all_token_ids = []
        for batch_token_ids in self._embedding_token_ids:
            all_token_ids.extend(batch_token_ids.view(-1).tolist())

        if not all_token_ids:
            return

        # Count activations per token
        activation_counts = {}
        for token_id in all_token_ids:
            token_id_int = int(token_id)
            if token_id_int not in activation_counts:
                activation_counts[token_id_int] = 0
            activation_counts[token_id_int] += 1

        # Number of embeddings activated (unique token IDs used)
        num_activated = len(activation_counts)
        self._log_metric("embeddings/num_activated", torch.tensor(float(num_activated)))

        # Median count of activations per embedding token
        counts = list(activation_counts.values())
        if counts:
            counts_tensor = torch.tensor(counts, dtype=torch.float32)
            median_count = torch.median(counts_tensor)
            self._log_metric("embeddings/median_activation_count", median_count)

    def _make_residual_stream_forward_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, torch.Tensor], output: torch.Tensor):
            del module
            if not self._should_log():
                return
            if not isinstance(output, torch.Tensor) or len(inputs) < 2:
                return
            residual = inputs[0]
            if not isinstance(residual, torch.Tensor):
                return
            # Use output - residual to capture the actual applied update (after dropout/alpha).
            # inputs[1] is the pre-dropout/pre-alpha update.
            update = output - residual
            res_norm = residual.norm(dim=-1).mean()
            upd_norm = update.norm(dim=-1).mean()
            ratio = (update.norm(dim=-1) / (residual.norm(dim=-1) + self.eps)).mean()

            self._log_metric(
                f"residual_update/activations/{name}/residual_norm",
                res_norm,
            )
            self._log_metric(
                f"residual_update/activations/{name}/update_norm",
                upd_norm,
            )
            self._log_metric(
                f"residual_update/activations/{name}/update_residual_ratio",
                ratio,
            )

        return hook

    def _make_residual_stream_backward_hook(self, name: str):
        def hook(
            module: nn.Module,
            grad_input: Tuple[Optional[torch.Tensor], ...],
            grad_output: Tuple[Optional[torch.Tensor], ...],
        ):
            del module, grad_output
            if not self._should_log():
                return
            if len(grad_input) < 2:
                return
            grad_residual = grad_input[0]
            grad_update = grad_input[1]
            if grad_residual is None or grad_update is None:
                return

            res_grad_norm = grad_residual.norm(dim=-1).mean()
            self._log_metric(
                f"residual_update/grads/{name}/residual_norm",
                res_grad_norm,
            )

        return hook

    def _make_activation_forward_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            del module, inputs
            if not self._should_log():
                return
            if not isinstance(output, torch.Tensor):
                return

            if (
                self.track_activation_rmse
                or self.track_activation_meanvar
                or self.track_activation_norm
            ):
                act = get_local_tensor(output.detach()).float()
                if self.track_activation_rmse:
                    act_rmse = self._vector_rmse(act)
                    self._log_metric(f"activations/{name}/rmse", act_rmse)
                if self.track_activation_meanvar:
                    act_mean, act_std = self._vector_meanstd(act)
                    self._log_metric(f"activations/{name}/mean", act_mean)
                    self._log_metric(f"activations/{name}/stddev", act_std)
                if self.track_activation_norm:
                    act_norm = self._mean_norm_over_batch_tokens(act)
                    self._log_metric(f"activations/{name}/norm", act_norm)

        return hook

    def _make_activation_backward_hook(self, name: str):
        def hook(
            module: nn.Module,
            grad_input: Tuple[Optional[torch.Tensor], ...],
            grad_output: Tuple[Optional[torch.Tensor], ...],
        ):
            del module, grad_input
            if not self._should_log():
                return
            if not grad_output or grad_output[0] is None or not isinstance(grad_output[0], torch.Tensor):
                return

            if (
                self.track_activation_grad_rmse
                or self.track_activation_grad_meanvar
                or self.track_activation_grad_norm
            ):
                act_grad = get_local_tensor(grad_output[0].detach()).float()
                if self.track_activation_grad_rmse:
                    act_grad_rmse = self._vector_rmse(act_grad)
                    self._log_metric(f"activation_grads/{name}/rmse", act_grad_rmse)
                if self.track_activation_grad_meanvar:
                    act_grad_mean, act_grad_std = self._vector_meanstd(act_grad)
                    self._log_metric(f"activation_grads/{name}/mean", act_grad_mean)
                    self._log_metric(f"activation_grads/{name}/stddev", act_grad_std)
                if self.track_activation_grad_norm:
                    act_grad_norm = self._mean_norm_over_batch_tokens(act_grad)
                    self._log_metric(f"activation_grads/{name}/norm", act_grad_norm)

        return hook

    def _make_layer_norm_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            del output
            if not self._should_log():
                return
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return

            if isinstance(module, RMSNorm):
                variance = x.pow(2).mean(-1)
            else:
                mean = x.mean(-1, keepdim=True)
                variance = (x - mean).pow(2).mean(-1)

            eps = getattr(module, "eps", None)
            if eps is None:
                return

            threshold = eps * (1.0 + self.eps_hit_tolerance)
            hit = (variance <= threshold).float().sum()

            self._log_metric(
                f"layernorm_eps/{name}/eps_hit",
                hit,
                reduce_type=ReduceType.sum,
                merge_strategy=MetricMergeStrategy.sum,
            )

        return hook

    def post_step(self):
        if not self._should_log():
            self._prev_params = None
            return

        model = getattr(self.trainer.train_module, "model", None)
        if model is None or not isinstance(model, nn.Module):
            self._prev_params = None
            return

        optim = getattr(self.trainer.train_module, "optim", None)
        if (
            self.track_update_param_ratio
            or self.track_update_rmse
            or self.track_param_movement
        ) and self._prev_params is not None:
            step_factor = 1.0
            if optim is not None and hasattr(optim, "step_skipped"):
                try:
                    step_factor = 1.0 - float(optim.step_skipped().item())
                except Exception:
                    step_factor = 1.0

            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                prev = self._prev_params.get(name)
                if prev is None:
                    continue
                current = get_local_tensor(p.detach()).float()
                update = current - prev

                if self.track_update_param_ratio:
                    denom = current.abs().clamp(min=self.eps)
                    ratio = (update.abs() / denom).mean()
                    self._log_metric(f"params/{name}/update_param_ratio", ratio)

                if self.track_update_rmse:
                    update_rmse = self._rmse(update)
                    self._log_metric(f"params/{name}/update_rmse", update_rmse)
                
                if self.track_param_movement:
                    # Count parameters that moved more than threshold relative to their magnitude
                    param_abs = prev.abs()
                    rel_change = update.abs() / (param_abs + self.eps)
                    moving_count = (rel_change > self.param_movement_threshold).float().sum()
                    total_count = torch.tensor(float(rel_change.numel()), device=rel_change.device)
                    moving_frac = moving_count / total_count
                    self._log_metric(
                        f"params/{name}/moving_fraction_gt_rel_{self.param_movement_threshold}",
                        moving_frac,
                    )
                
        if self.track_param_meanvar:
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                mean = get_local_tensor(p.detach()).float().mean()
                self._log_metric(f"params/{name}/mean", mean)
                std = get_local_tensor(p.detach()).float().var(unbiased=False).sqrt()
                self._log_metric(f"params/{name}/stddev", std)

        if self.track_optimizer_state_rmse_meanvar and optim is not None:
            for param, state in optim.state.items():
                name = None
                if self._param_name_map is not None:
                    name = self._param_name_map.get(id(param))
                if name is None:
                    name = f"param_{id(param)}"

                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg is not None:
                    exp_avg_t = get_local_tensor(exp_avg.detach()).float()
                    rmse = self._rmse(exp_avg_t)
                    mean, std = self._meanstd(exp_avg_t)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_rmse", rmse)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_mean", mean)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_stddev", std)
                if exp_avg_sq is not None:
                    exp_avg_sq_t = get_local_tensor(exp_avg_sq.detach()).float()
                    rmse = self._rmse(exp_avg_sq_t)
                    mean, std = self._meanstd(exp_avg_sq_t)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_sq_rmse", rmse)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_sq_mean", mean)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_sq_stddev", std)

        if self.track_embedding_usage:
            self._log_embedding_metrics()

        self._prev_params = None
        self._embedding_token_ids.clear()

    def _make_lm_head_hook(self):
        """Hook to track LM head logits statistics and probability mass during training."""
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output):
            del module, inputs
            if not self._should_log():
                return

            # Only track during training (when output is LMOutputWithLoss with a loss field).
            # During inference, output is raw logits tensor.
            if not hasattr(output, "loss"):
                return

            logits = None
            if hasattr(output, "logits"):
                logits = output.logits
            else:
                return

            if logits is None or not isinstance(logits, torch.Tensor):
                return

            # Use full logits tensor (don't shard) for correct softmax and statistics
            logits_tensor = logits.detach().float()
            logits_flat = logits_tensor.view(-1, logits_tensor.shape[-1])
            probs = torch.softmax(logits_flat, dim=-1)

            top1_mass = probs.max(dim=-1)[0].mean()
            self._log_metric("lm_head/top1_mass", top1_mass)

            entropy = -(probs * torch.log(probs + self.eps)).sum(dim=-1).mean()
            self._log_metric("lm_head/entropy", entropy)

            max_logit = logits_flat.max(dim=-1)[0].mean()
            min_logit = logits_flat.min(dim=-1)[0].mean()
            avg_logit = logits_flat.mean(dim=-1).mean()
            logit_std = logits_flat.std()

            self._log_metric("lm_head/logit_max", max_logit)
            self._log_metric("lm_head/logit_min", min_logit)
            self._log_metric("lm_head/logit_avg", avg_logit)
            self._log_metric("lm_head/logit_std", logit_std)

        return hook

    def _make_embedding_forward_hook(self):
        """Hook to track embedding usage (which token IDs are used in forward pass)."""
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            del module, output
            if not self._should_log():
                return

            # The input to nn.Embedding is the token IDs tensor
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return

            token_ids = inputs[0]
            # Store token IDs for this batch
            self._embedding_token_ids.append(token_ids.detach().cpu())

        return hook
