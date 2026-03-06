"""
Coordinate check utility for validating muP correctness.

Runs a few steps of training at multiple widths and checks that per-layer activation
and gradient statistics remain O(1) — the hallmark of correct muP scaling.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import torch

from olmo_core.config import Config

log = logging.getLogger(__name__)


@dataclass
class CoordCheckConfig(Config):
    """
    Configuration for the muP coordinate check.

    :param base_d_model: The base width for the muP proxy model.
    :param widths: List of ``d_model`` values to sweep.
    :param n_steps: Number of training steps per width.
    :param vocab_size: Vocabulary size for the test models.
    :param n_layers: Number of transformer layers.
    :param seq_len: Sequence length for random input.
    :param lr: Learning rate for the optimizer.
    :param batch_size: Batch size for random input.
    """

    base_d_model: int = 128
    widths: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    n_steps: int = 5
    vocab_size: int = 1000
    n_layers: int = 4
    seq_len: int = 128
    lr: float = 1e-3
    batch_size: int = 4


def run_coord_check(
    config: CoordCheckConfig,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Run muP coordinate check across multiple model widths.

    For each width, builds a model with muP enabled, runs a few training steps on
    random data, and records per-layer activation/gradient mean and std.

    Correct muP: all statistics should be O(1) across widths (i.e., they should not
    grow or shrink systematically with width).

    :param config: The coordinate check configuration.

    :returns: ``{width: {layer_name: {"act_mean": ..., "act_std": ..., "grad_mean": ..., "grad_std": ...}}}``
    """
    from olmo_core.nn.attention import AttentionConfig
    from olmo_core.nn.feed_forward import FeedForwardConfig
    from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
    from olmo_core.nn.lm_head import LMHeadConfig
    from olmo_core.nn.transformer.config import (
        TransformerBlockConfig,
        TransformerConfig,
    )
    from olmo_core.nn.transformer.init import InitMethod
    from olmo_core.nn.transformer.mup import MuPConfig
    from olmo_core.optim.mup_adamw import MuPAdamWConfig

    results: Dict[int, Dict[str, Dict[str, float]]] = {}

    for width in config.widths:
        log.info(f"Running coord check for width={width}")

        n_heads = max(1, width // 64)
        hidden_size = width * 4

        transformer_config = TransformerConfig(
            d_model=width,
            vocab_size=config.vocab_size,
            n_layers=config.n_layers,
            block=TransformerBlockConfig(
                sequence_mixer=AttentionConfig(n_heads=n_heads, bias=False),
                feed_forward=FeedForwardConfig(hidden_size=hidden_size, bias=False),
                layer_norm=LayerNormConfig(name=LayerNormType.rms, bias=False),
            ),
            lm_head=LMHeadConfig(bias=False),
            init_method=InitMethod.fan_in,
            mup=MuPConfig(base_d_model=config.base_d_model),
        )

        model = transformer_config.build(init_device="cpu")
        model.init_weights(device=torch.device("cpu"))

        optim_config = MuPAdamWConfig(
            lr=config.lr,
            base_d_model=config.base_d_model,
            d_model=width,
        )
        optimizer = optim_config.build(model)

        # Hooks to record activations.
        activations: Dict[str, torch.Tensor] = {}

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()

            return hook

        hooks = []
        for name, module in model.named_modules():
            if name and ("blocks" in name or name == "lm_head"):
                hooks.append(module.register_forward_hook(make_hook(name)))

        # Training loop.
        for step in range(config.n_steps):
            input_ids = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))
            labels = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))

            output = model(input_ids, labels=labels)
            loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Remove hooks.
        for h in hooks:
            h.remove()

        # Record statistics from the last forward pass.
        layer_stats: Dict[str, Dict[str, float]] = {}
        for name, act in activations.items():
            layer_stats[name] = {
                "act_mean": act.float().mean().item(),
                "act_std": act.float().std().item(),
            }

        # Record gradient statistics.
        for name, param in model.named_parameters():
            if param.grad is not None:
                layer_stats.setdefault(name, {})
                layer_stats[name]["grad_mean"] = param.grad.float().mean().item()
                layer_stats[name]["grad_std"] = param.grad.float().std().item()

        results[width] = layer_stats

    return results
