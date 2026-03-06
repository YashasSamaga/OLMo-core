"""
Tests for muP with gemma_like_ladder v2 models.

Verifies that muP integrates correctly with the v2 architecture:
- PeriNorm blocks with sliding window (local) + full (global) attention
- GQA with 8 KV heads, head_dim=128
- Elementwise gating, QK norm
- embed_scale = sqrt(d_model)
"""

import math
import sys

import pytest
import torch

# The ladder script lives under src/scripts/ which isn't a normal package,
# so we import it via path manipulation.
sys.path.insert(0, "src/scripts/train/ladder")
from gemma_like_ladder import GemmaLikeTransformerConfig  # type: ignore[import-untyped]

from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.nn.transformer.mup import MuPConfig
from olmo_core.optim.mup_adamw import MuPAdamWConfig

VOCAB_SIZE = 1000  # small vocab for fast tests


def _make_v2_mup_config(
    d_model: int,
    n_layers: int,
    n_heads: int,
    hidden_size: int,
    base_d_model: int = 640,
) -> "GemmaLikeTransformerConfig":
    """Build a v2 config with muP enabled."""
    return GemmaLikeTransformerConfig.v2(
        d_model=d_model,
        hidden_size=hidden_size,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=VOCAB_SIZE,
        attn_backend=None,  # use torch backend for CPU tests
        init_method=InitMethod.fan_in,
        mup=MuPConfig(base_d_model=base_d_model),
    )


# ---- v2 model size specs: (d_model, hidden_size, n_layers, n_heads) ----
V2_260M = (640, 640 * 8, 10, 8)
V2_709M = (1024, 1024 * 8, 15, 16)


class TestV2MuPConfig:
    """Tests that muP config fields are correctly set on v2 models."""

    def test_logit_scale_set(self):
        """logit_scale should be base_d_model / d_model."""
        base = 640
        d_model, hidden, n_layers, n_heads = V2_260M
        config = _make_v2_mup_config(d_model, n_layers, n_heads, hidden, base_d_model=base)
        expected = base / d_model
        assert config.lm_head.logit_scale == pytest.approx(expected)

    def test_logit_scale_varies_with_width(self):
        """Wider model should have smaller logit_scale."""
        base = 640
        config_small = _make_v2_mup_config(*V2_260M, base_d_model=base)
        config_large = _make_v2_mup_config(*V2_709M, base_d_model=base)
        assert config_small.lm_head.logit_scale > config_large.lm_head.logit_scale

    def test_softmax_scale_on_local_block(self):
        """Local (SWA) attention blocks should get muP softmax_scale = 1/head_dim."""
        config = _make_v2_mup_config(*V2_260M, base_d_model=640)
        head_dim = 128  # v2 uses explicit head_dim=128
        expected = 1.0 / head_dim

        # The base block is the local block.
        assert isinstance(config.block, type(config.block))  # single block, not dict
        assert isinstance(config.block.sequence_mixer, AttentionConfig)
        assert config.block.sequence_mixer.softmax_scale == pytest.approx(expected)

    def test_softmax_scale_on_global_blocks(self):
        """Global attention override blocks should also get muP softmax_scale."""
        config = _make_v2_mup_config(*V2_260M, base_d_model=640)
        head_dim = 128
        expected = 1.0 / head_dim

        assert config.block_overrides is not None
        for layer_idx, override in config.block_overrides.items():
            assert isinstance(override.sequence_mixer, AttentionConfig)
            assert override.sequence_mixer.softmax_scale == pytest.approx(expected), (
                f"Global block at layer {layer_idx} missing muP softmax_scale"
            )

    def test_embed_scale_preserved(self):
        """v2 uses embed_scale=sqrt(d_model); muP should not override it."""
        d_model = V2_260M[0]
        config = _make_v2_mup_config(*V2_260M, base_d_model=640)
        assert config.embed_scale == pytest.approx(math.sqrt(d_model))


class TestV2MuPBuild:
    """Tests that v2 models build and initialize correctly with muP."""

    def test_build_and_init(self):
        """Model should build and init_weights without error."""
        config = _make_v2_mup_config(*V2_260M, base_d_model=640)
        model = config.build(init_device="meta")
        model.init_weights(device=torch.device("cpu"))

    def test_logit_scale_on_built_model(self):
        """Built model's LM head should have the correct logit_scale."""
        base = 640
        d_model = V2_260M[0]
        config = _make_v2_mup_config(*V2_260M, base_d_model=base)
        model = config.build(init_device="cpu")
        assert model.lm_head._logit_scale == pytest.approx(base / d_model)

    def test_attention_scale_on_built_model(self):
        """All attention backends should have muP scale = 1/head_dim."""
        config = _make_v2_mup_config(*V2_260M, base_d_model=640)
        model = config.build(init_device="cpu")

        head_dim = 128
        expected = 1.0 / head_dim

        for block_key, block in model.blocks.items():
            if hasattr(block.attention, "backend"):
                scale = block.attention.backend.scale
                assert scale == pytest.approx(expected), (
                    f"Block {block_key} attention scale: expected {expected}, got {scale}"
                )

    @pytest.mark.parametrize(
        "init_device, device",
        [
            pytest.param("cpu", "cpu", id="cpu->cpu"),
            pytest.param("meta", "cpu", id="meta->cpu"),
        ],
    )
    def test_output_head_init_scaling(self, init_device, device):
        """Output head init should have extra 1/sqrt(m) muP scaling."""
        base = 640
        d_model, hidden, n_layers, n_heads = V2_709M
        config = _make_v2_mup_config(d_model, n_layers, n_heads, hidden, base_d_model=base)
        model = config.build(init_device=init_device)
        model.init_weights(device=torch.device(device))

        m_ratio = d_model / base
        # fan_in init: std = 1/sqrt(d_model), muP adds 1/sqrt(m)
        expected_std = (d_model**-0.5) / math.sqrt(m_ratio)
        actual_std = model.lm_head.w_out.weight.std().item()
        assert actual_std == pytest.approx(expected_std, rel=0.25), (
            f"LM head std: expected ~{expected_std:.6f}, got {actual_std:.6f}"
        )


class TestV2MuPForwardBackward:
    """Tests that v2 models run forward/backward correctly with muP."""

    def test_forward_produces_finite_logits(self):
        config = _make_v2_mup_config(*V2_260M, base_d_model=640)
        model = config.build(init_device="cpu")
        model.init_weights(device=torch.device("cpu"))

        input_ids = torch.randint(0, VOCAB_SIZE, (2, 64))
        logits = model(input_ids)
        assert torch.isfinite(logits).all(), "Non-finite logits detected"

    def test_forward_backward_with_loss(self):
        config = _make_v2_mup_config(*V2_260M, base_d_model=640)
        model = config.build(init_device="cpu")
        model.init_weights(device=torch.device("cpu"))

        input_ids = torch.randint(0, VOCAB_SIZE, (2, 64))
        labels = torch.randint(0, VOCAB_SIZE, (2, 64))
        output = model(input_ids, labels=labels)
        assert torch.isfinite(output.loss), f"Loss is not finite: {output.loss.item()}"

        output.loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"

    def test_optimizer_step(self):
        """Full train step: forward, backward, optimizer.step()."""
        d_model = V2_260M[0]
        base = 640
        config = _make_v2_mup_config(*V2_260M, base_d_model=base)
        model = config.build(init_device="cpu")
        model.init_weights(device=torch.device("cpu"))

        optim_config = MuPAdamWConfig(lr=1e-3, base_d_model=base, d_model=d_model)
        optimizer = optim_config.build(model)

        input_ids = torch.randint(0, VOCAB_SIZE, (2, 64))
        labels = torch.randint(0, VOCAB_SIZE, (2, 64))
        output = model(input_ids, labels=labels)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()


class TestV2MuPOptimizer:
    """Tests that MuPAdamWConfig assigns correct LR groups for v2 models."""

    def test_lr_groups(self):
        d_model = V2_709M[0]
        base = 640
        base_lr = 1e-3
        lr_scale = base / d_model

        config = _make_v2_mup_config(*V2_709M, base_d_model=base)
        model = config.build(init_device="cpu")
        model.init_weights(device=torch.device("cpu"))

        optim_config = MuPAdamWConfig(lr=base_lr, base_d_model=base, d_model=d_model)
        optimizer = optim_config.build(model)

        embed_ids = {id(p) for p in model.embeddings.parameters()}
        block_2d_ids = {id(p) for _, p in model.blocks.named_parameters() if p.ndim == 2}
        lm_head_2d_ids = {id(p) for _, p in model.lm_head.named_parameters() if p.ndim == 2}

        for group in optimizer.param_groups:
            param_ids = {id(p) for p in group["params"]}
            lr = group["lr"]

            if param_ids & embed_ids:
                assert group["weight_decay"] == 0.0
            if param_ids & block_2d_ids:
                assert lr == pytest.approx(base_lr * lr_scale)
            if param_ids & lm_head_2d_ids:
                assert lr == pytest.approx(base_lr * lr_scale)

    def test_width_transfer_lr_decreases(self):
        """Wider model should get smaller LR for matrix params."""
        base = 640
        base_lr = 1e-3

        configs = []
        for spec in [V2_260M, V2_709M]:
            d_model = spec[0]
            config = _make_v2_mup_config(*spec, base_d_model=base)
            model = config.build(init_device="cpu")
            model.init_weights(device=torch.device("cpu"))
            optim_config = MuPAdamWConfig(lr=base_lr, base_d_model=base, d_model=d_model)
            optimizer = optim_config.build(model)

            block_2d_ids = {id(p) for _, p in model.blocks.named_parameters() if p.ndim == 2}
            for group in optimizer.param_groups:
                if {id(p) for p in group["params"]} & block_2d_ids:
                    configs.append(group["lr"])
                    break

        # 260M (d=640) should have higher matrix LR than 709M (d=1024)
        assert configs[0] > configs[1]


class TestV2WithoutMuP:
    """Sanity: v2 models without muP should have default scaling."""

    def test_no_logit_scale(self):
        config = GemmaLikeTransformerConfig.v2_260M(VOCAB_SIZE, attn_backend=None)
        model = config.build(init_device="cpu")
        assert model.lm_head._logit_scale is None

    def test_default_attention_scale(self):
        config = GemmaLikeTransformerConfig.v2_260M(VOCAB_SIZE, attn_backend=None)
        model = config.build(init_device="cpu")
        # Without muP, scale should be None (backend defaults to 1/sqrt(head_dim))
        block = model.blocks["0"]
        assert block.attention.backend.scale is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
