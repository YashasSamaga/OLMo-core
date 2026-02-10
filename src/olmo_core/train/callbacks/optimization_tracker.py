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
    track_param_grad_meanvar: bool = False  # Log per-parameter grad mean/variance.

    track_param_meanvar: bool = False  # Log per-parameter mean and variance.
    track_update_param_ratio: bool = False  # Log per-parameter update/param ratio.

    track_activation_rmse: bool = False  # Log activation RMSE over all modules.
    track_activation_meanvar: bool = False  # Log activation mean/variance over all modules.
    track_activation_norm: bool = False  # Log mean activation norm over batch/tokens for all modules.

    track_activation_grad_rmse: bool = False  # Log activation grad RMSE over all modules.
    track_activation_grad_meanvar: bool = False  # Log activation grad mean/variance over all modules.
    track_activation_grad_norm: bool = False  # Log mean activation grad norm over batch/tokens for all modules.

    track_update_rmse: bool = False  # Log per-parameter update RMSE.

    track_optimizer_state_rmse_meanvar: bool = False  # Log optimizer state RMSE/mean/variance.
    namespace: str = "optim_diagnostics"  # Metric namespace prefix.

    _handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list, repr=False)
    _prev_params: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)
    _param_name_map: Optional[Dict[int, str]] = field(default=None, repr=False)

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

    def post_train(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._prev_params = None

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

        if self.track_param_grad_rmse or self.track_param_grad_meanvar:
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                grad = get_local_tensor(param.grad.detach()).float()
                if self.track_param_grad_rmse:
                    grad_rmse = self._rmse(grad)
                    self._log_metric(f"grads/{name}/rmse", grad_rmse)
                if self.track_param_grad_meanvar:
                    grad_mean, grad_var = self._meanvar(grad)
                    self._log_metric(f"grads/{name}/mean", grad_mean)
                    self._log_metric(f"grads/{name}/var", grad_var)

        if self.track_update_param_ratio:
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

    def _meanvar(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean/variance over all elements in the tensor."""
        mean = tensor.mean()
        var = tensor.var(unbiased=False)
        return mean, var

    def _mean_norm_over_batch_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute mean L2 norm over batch/tokens (all dims except the last)."""
        return tensor.norm(dim=-1).mean()

    def _vector_rmse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute RMSE per vector (last dim), then average over batch/tokens."""
        return tensor.pow(2).mean(dim=-1).sqrt().mean()

    def _vector_meanvar(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean/variance per vector (last dim), then average over batch/tokens."""
        mean = tensor.mean(dim=-1).mean()
        var = tensor.var(dim=-1, unbiased=False).mean()
        return mean, var

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
            upd_grad_norm = grad_update.norm(dim=-1).mean()
            grad_ratio = (
                grad_update.norm(dim=-1) / (grad_residual.norm(dim=-1) + self.eps)
            ).mean()

            self._log_metric(
                f"residual_update/grads/{name}/residual_norm",
                res_grad_norm,
            )
            self._log_metric(
                f"residual_update/grads/{name}/update_norm",
                upd_grad_norm,
            )
            self._log_metric(
                f"residual_update/grads/{name}/update_residual_ratio",
                grad_ratio,
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
                    act_mean, act_var = self._vector_meanvar(act)
                    self._log_metric(f"activations/{name}/mean", act_mean)
                    self._log_metric(f"activations/{name}/var", act_var)
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
                    act_grad_mean, act_grad_var = self._vector_meanvar(act_grad)
                    self._log_metric(f"activation_grads/{name}/mean", act_grad_mean)
                    self._log_metric(f"activation_grads/{name}/var", act_grad_var)
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
                    denom = current.abs()
                    mask = denom > 0
                    if mask.any():
                        ratio = (update.abs()[mask] / denom[mask]).mean()
                    else:
                        ratio = torch.tensor(0.0, device=current.device)
                    self._log_metric(f"params/{name}/update_param_ratio", ratio)

                if self.track_update_rmse:
                    update_rmse = self._rmse(update)
                    self._log_metric(f"params/{name}/update_rmse", update_rmse)


        if self.track_param_meanvar:
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                mean = get_local_tensor(p.detach()).float().mean()
                self._log_metric(f"params/{name}/mean", mean)
                var = get_local_tensor(p.detach()).float().var(unbiased=False)
                self._log_metric(f"params/{name}/var", var)

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
                    mean, var = self._meanvar(exp_avg_t)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_rmse", rmse)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_mean", mean)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_var", var)
                if exp_avg_sq is not None:
                    exp_avg_sq_t = get_local_tensor(exp_avg_sq.detach()).float()
                    rmse = self._rmse(exp_avg_sq_t)
                    mean, var = self._meanvar(exp_avg_sq_t)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_sq_rmse", rmse)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_sq_mean", mean)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_sq_var", var)

        self._prev_params = None
