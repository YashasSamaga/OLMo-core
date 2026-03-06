"""
Tests for muP (Maximal Update Parameterization).
"""

import math

import pytest
import torch

from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig
from olmo_core.nn.transformer.coord_check import CoordCheckConfig, run_coord_check
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.nn.transformer.mup import MuPConfig
from olmo_core.optim.mup_adamw import MuPAdamWConfig


def _build_mup_config(d_model, base_d_model=128, n_layers=2, vocab_size=1000):
    """Helper to build a TransformerConfig with muP enabled."""
    n_heads = max(1, d_model // 64)
    hidden_size = d_model * 4
    return TransformerConfig(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        block=TransformerBlockConfig(
            sequence_mixer=AttentionConfig(n_heads=n_heads, bias=False),
            feed_forward=FeedForwardConfig(hidden_size=hidden_size, bias=False),
            layer_norm=LayerNormConfig(name=LayerNormType.rms, bias=False),
        ),
        lm_head=LMHeadConfig(bias=False),
        init_method=InitMethod.fan_in,
        mup=MuPConfig(base_d_model=base_d_model),
    )


@pytest.mark.parametrize("d_model", [128, 256, 512])
@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cpu", id="cpu->cpu"),
        pytest.param("meta", "cpu", id="meta->cpu"),
    ],
)
def test_mup_init(d_model, init_device, device):
    """Test that muP init uses correct std for output head with extra 1/sqrt(m) scaling."""
    base_d_model = 128
    config = _build_mup_config(d_model, base_d_model=base_d_model)
    model = config.build(init_device=init_device)
    model.init_weights(device=torch.device(device))

    m_ratio = d_model / base_d_model

    # Embedding init: std ~1.0 for fan_in
    embedding_std = model.embeddings.weight.std().item()
    assert abs(embedding_std - 1.0) < 0.15, f"Expected embedding std ~1.0, got {embedding_std}"

    # Hidden weights: std ~ 1/sqrt(fan_in) (fan_in init, no muP scaling)
    block = model.blocks["0"]
    expected_d = 1.0 / math.sqrt(d_model)
    tol_d = expected_d * 0.25
    for name in ("w_q", "w_k", "w_v", "w_out"):
        actual = getattr(block.attention, name).weight.std().item()
        assert (
            abs(actual - expected_d) < tol_d
        ), f"attention.{name} std: expected ~{expected_d:.5f}, got {actual:.5f}"

    # Output head: std ~ 1/sqrt(d_model) * 1/sqrt(m)
    expected_lm = (d_model**-0.5) / math.sqrt(m_ratio)
    tol_lm = expected_lm * 0.25
    lm_head_std = model.lm_head.w_out.weight.std().item()
    assert (
        abs(lm_head_std - expected_lm) < tol_lm
    ), f"LM head std: expected ~{expected_lm:.5f}, got {lm_head_std:.5f}"


@pytest.mark.parametrize("d_model", [128, 256])
def test_mup_attention_scale(d_model):
    """Verify attention softmax_scale = 1/head_dim when muP enabled."""
    base_d_model = 128
    config = _build_mup_config(d_model, base_d_model=base_d_model)
    model = config.build(init_device="cpu")

    n_heads = max(1, d_model // 64)
    head_dim = d_model // n_heads
    expected_scale = 1.0 / head_dim

    block = model.blocks["0"]
    backend_scale = block.attention.backend.scale
    assert (
        abs(backend_scale - expected_scale) < 1e-6
    ), f"Expected attention scale {expected_scale}, got {backend_scale}"


@pytest.mark.parametrize("d_model", [128, 256, 512])
def test_mup_logit_scale(d_model):
    """Verify output logits are scaled by base_d_model/d_model."""
    base_d_model = 128
    config = _build_mup_config(d_model, base_d_model=base_d_model)
    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))

    expected_logit_scale = base_d_model / d_model
    assert model.lm_head._logit_scale is not None
    assert (
        abs(model.lm_head._logit_scale - expected_logit_scale) < 1e-6
    ), f"Expected logit_scale {expected_logit_scale}, got {model.lm_head._logit_scale}"


def test_mup_optimizer_lr_groups():
    """Test that MuPAdamWConfig assigns correct per-group LRs."""
    d_model = 512
    base_d_model = 128
    base_lr = 1e-3
    lr_scale = base_d_model / d_model

    config = _build_mup_config(d_model, base_d_model=base_d_model)
    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))

    optim_config = MuPAdamWConfig(
        lr=base_lr,
        base_d_model=base_d_model,
        d_model=d_model,
    )
    optimizer = optim_config.build(model)

    # Collect param groups
    groups = optimizer.param_groups

    # We expect 4 groups: embed, matrix, vector, lm_head
    # Embed group: base lr, no weight decay
    # Matrix group: lr * lr_scale
    # Vector group: base lr
    # LM head group: lr * lr_scale
    found_embed = False
    found_matrix = False
    found_vector = False
    found_lm_head = False

    for group in groups:
        lr = group.get("lr", base_lr)
        wd = group.get("weight_decay", optim_config.weight_decay)

        # Identify groups by their params
        param_ids = {id(p) for p in group["params"]}
        model_embed_ids = {id(p) for p in model.embeddings.parameters()}
        model_lm_head_2d_ids = {id(p) for n, p in model.lm_head.named_parameters() if p.ndim == 2}
        model_block_2d_ids = {id(p) for n, p in model.blocks.named_parameters() if p.ndim == 2}
        model_block_1d_ids = {id(p) for n, p in model.blocks.named_parameters() if p.ndim < 2}
        model_lm_head_1d_ids = {id(p) for n, p in model.lm_head.named_parameters() if p.ndim < 2}

        if param_ids & model_embed_ids:
            found_embed = True
            assert wd == 0.0, f"Embed group should have weight_decay=0.0, got {wd}"

        if param_ids & model_block_2d_ids:
            found_matrix = True
            expected_lr = base_lr * lr_scale
            assert (
                abs(lr - expected_lr) < 1e-10
            ), f"Matrix group lr: expected {expected_lr}, got {lr}"

        if param_ids & (model_block_1d_ids | model_lm_head_1d_ids):
            found_vector = True
            assert abs(lr - base_lr) < 1e-10, f"Vector group lr: expected {base_lr}, got {lr}"

        if param_ids & model_lm_head_2d_ids:
            found_lm_head = True
            expected_lr = base_lr * lr_scale
            assert (
                abs(lr - expected_lr) < 1e-10
            ), f"LM head group lr: expected {expected_lr}, got {lr}"

    assert found_embed, "Did not find embed param group"
    assert found_matrix, "Did not find matrix param group"
    assert found_vector, "Did not find vector param group"
    assert found_lm_head, "Did not find lm_head param group"


def test_mup_forward_backward():
    """End-to-end test: model forward+backward runs without error with muP enabled."""
    d_model = 256
    base_d_model = 128
    config = _build_mup_config(d_model, base_d_model=base_d_model)
    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))

    optim_config = MuPAdamWConfig(
        lr=1e-3,
        base_d_model=base_d_model,
        d_model=d_model,
    )
    optimizer = optim_config.build(model)

    # Forward + backward
    input_ids = torch.randint(0, 1000, (2, 64))
    labels = torch.randint(0, 1000, (2, 64))
    output = model(input_ids, labels=labels)
    loss = output.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Verify loss is finite
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


def test_mup_no_mup_default_behavior():
    """Verify that without muP config, behavior is unchanged."""
    config = TransformerConfig(
        d_model=256,
        vocab_size=1000,
        n_layers=2,
        block=TransformerBlockConfig(
            sequence_mixer=AttentionConfig(n_heads=4, bias=False),
            feed_forward=FeedForwardConfig(hidden_size=1024, bias=False),
            layer_norm=LayerNormConfig(name=LayerNormType.rms, bias=False),
        ),
        lm_head=LMHeadConfig(bias=False),
        init_method=InitMethod.fan_in,
    )
    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))

    # No logit scaling
    assert model.lm_head._logit_scale is None

    # Default attention scale should be None (backend uses 1/sqrt(head_dim) internally)
    backend_scale = model.blocks["0"].attention.backend.scale
    assert backend_scale is None, f"Expected default scale to be None, got {backend_scale}"


def test_coord_check():
    """Verify activation/gradient stats are O(1) across widths with muP."""
    config = CoordCheckConfig(
        base_d_model=64,
        widths=[64, 128, 256],
        n_steps=3,
        n_layers=2,
        vocab_size=500,
        seq_len=32,
        batch_size=2,
    )
    results = run_coord_check(config)

    # Collect activation stds across widths for a few representative layers.
    # For correct muP, these should not grow/shrink systematically with width.
    assert len(results) == 3

    # Check that each width has recorded some statistics.
    for width in config.widths:
        assert width in results
        assert len(results[width]) > 0, f"No stats recorded for width={width}"

    # Check that the ratio of act_std between max and min width is bounded.
    # With correct muP, stats should be O(1), so the ratio shouldn't be large.
    common_layers = set(results[config.widths[0]].keys())
    for width in config.widths[1:]:
        common_layers &= set(results[width].keys())

    act_std_layers = [
        layer for layer in common_layers if "act_std" in results[config.widths[0]].get(layer, {})
    ]

    for layer in act_std_layers:
        stds = [results[w][layer]["act_std"] for w in config.widths]
        if all(s > 0 for s in stds):
            ratio = max(stds) / min(stds)
            # With muP, we expect the ratio to be modest (< 10x).
            # Without muP, it could grow as sqrt(width_ratio) or worse.
            assert ratio < 10.0, (
                f"Layer {layer}: act_std ratio across widths = {ratio:.2f}, "
                f"stds = {stds}. This suggests muP scaling may be incorrect."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
