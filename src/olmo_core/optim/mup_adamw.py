import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from olmo_core.nn.transformer import Transformer

from .config import MatrixAwareOptimConfig, OptimConfig, OptimGroupOverride

log = logging.getLogger(__name__)


@OptimConfig.register("mup_adamw")
@dataclass
class MuPAdamWConfig(MatrixAwareOptimConfig):
    """
    AdamW with muP learning rate scaling.

    Per-group LR:

    - Embedding: base ``lr`` (no scaling), no weight decay
    - Hidden (matrix) weights: ``lr * (base_d_model / d_model)``
    - Vector params (biases, LayerNorm): base ``lr`` (no scaling)
    - Output head: ``lr * (base_d_model / d_model)``

    :param lr: Base learning rate.
    :param betas: AdamW betas.
    :param eps: AdamW epsilon.
    :param weight_decay: Weight decay.
    :param foreach: Use foreach implementation.
    :param fused: Use fused implementation.
    :param base_d_model: The ``d_model`` of the proxy model used for HP tuning.
    :param d_model: The ``d_model`` of the target model.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    foreach: Optional[bool] = None
    fused: Optional[bool] = None

    base_d_model: int = 256
    d_model: int = 256

    @classmethod
    def optimizer(cls) -> type:
        return torch.optim.AdamW

    def default_group_overrides(self, model: torch.nn.Module) -> list[OptimGroupOverride]:
        assert isinstance(model, Transformer)
        params = self.categorize_parameters(model)
        lr_scale = self.base_d_model / self.d_model

        return [
            OptimGroupOverride(params=params["embed"], opts=dict(weight_decay=0.0)),
            OptimGroupOverride(params=params["matrix"], opts=dict(lr=self.lr * lr_scale)),
            OptimGroupOverride(params=params["vector"], opts=dict()),
            OptimGroupOverride(params=params["lm_head"], opts=dict(lr=self.lr * lr_scale)),
        ]

    def create_optimizer(
        self, model: torch.nn.Module, strict: bool = True, **kwargs
    ) -> torch.optim.AdamW:
        kwargs.pop("base_d_model", None)
        kwargs.pop("d_model", None)
        kwargs.pop("foreach", None)
        kwargs.pop("fused", None)

        return torch.optim.AdamW(
            self.build_groups(model, strict=strict),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            foreach=self.foreach,
            fused=self.fused,
        )
