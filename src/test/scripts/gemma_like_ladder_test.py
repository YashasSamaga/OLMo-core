"""
Tests for gemma_like_ladder.py — specifically the _get_norm_wd_overrides helper.
"""

import importlib.util
from pathlib import Path

import pytest

from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerBlockType, TransformerConfig
from olmo_core.optim import SkipStepAdamWConfig, OptimGroupOverride


def _load_ladder():
    path = Path(__file__).parents[2] / "scripts" / "train" / "ladder" / "gemma_like_ladder.py"
    spec = importlib.util.spec_from_file_location("gemma_like_ladder", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore
    return mod


ladder = _load_ladder()
_get_norm_wd_overrides = ladder._get_norm_wd_overrides


def _build_tiny_peri_norm_model() -> TransformerConfig:
    """Minimal peri_norm transformer with QK norms, matching the v2 architecture."""
    layer_norm = LayerNormConfig(name=LayerNormType.rms, bias=False, one_plus_gamma=True)
    block = TransformerBlockConfig(
        name=TransformerBlockType.peri_norm,
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            bias=False,
            rope=RoPEConfig(name=RoPEType.default, theta=10_000),
            qk_norm=layer_norm,
            use_head_qk_norm=True,
        ),
        feed_forward=FeedForwardConfig(
            hidden_size=128, bias=False, activation=ActivationFunction.silu
        ),
        layer_norm=layer_norm,
    )
    return TransformerConfig(
        d_model=64,
        vocab_size=1024,
        n_layers=2,
        block=block,
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False),
    )


def _wd_by_param(model, lnwd_mode: str, base_wd: float = 0.1):
    """Return {param_name: weight_decay} for all trainable params under the given mode."""
    overrides = [
        OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0)),
        *_get_norm_wd_overrides(lnwd_mode),
    ]
    config = SkipStepAdamWConfig(lr=1e-3, weight_decay=base_wd, group_overrides=overrides)
    groups = config.build_groups(model)
    result = {}
    for group in groups:
        wd = group.get("weight_decay", base_wd)
        for p in group["params"]:
            # find name
            for name, param in model.named_parameters():
                if param is p:
                    result[name] = wd
                    break
    return result


@pytest.fixture(scope="module")
def model():
    cfg = _build_tiny_peri_norm_model()
    return cfg.build(init_device="cpu")


QK_NORM_SUFFIX = (".q_norm.weight", ".k_norm.weight")
BLOCK_NORM_SUFFIX = (
    ".attention_norm.weight",
    ".feed_forward_norm.weight",
    ".post_attention_norm.weight",
    ".post_feed_forward_norm.weight",
)
LM_HEAD_NORM = "lm_head.norm.weight"


def test_all_keeps_wd_on_all_norms(model):
    wd = _wd_by_param(model, "all")
    for name, val in wd.items():
        if name == "embeddings.weight":
            assert val == 0.0
        else:
            assert val == 0.1, f"Expected WD=0.1 for {name} in all mode, got {val}"


def test_only_qk_lnwd_keeps_wd_on_qk_removes_from_block(model):
    # only QK norms keep WD; block/LM-head norms have WD removed
    wd = _wd_by_param(model, "only_qk_lnwd")
    for name, val in wd.items():
        if any(name.endswith(s) for s in BLOCK_NORM_SUFFIX) or name == LM_HEAD_NORM:
            assert val == 0.0, f"Expected WD=0 for block norm {name} in only_qk_lnwd, got {val}"
        elif any(name.endswith(s) for s in QK_NORM_SUFFIX):
            assert val == 0.1, f"Expected WD=0.1 for QK norm {name} in only_qk_lnwd, got {val}"
        elif name == "embeddings.weight":
            assert val == 0.0
        else:
            assert val == 0.1


def test_only_block_lnwd_keeps_wd_on_block_removes_from_qk(model):
    # only block/LM-head norms keep WD; QK norms have WD removed
    wd = _wd_by_param(model, "only_block_lnwd")
    for name, val in wd.items():
        if any(name.endswith(s) for s in QK_NORM_SUFFIX):
            assert val == 0.0, f"Expected WD=0 for QK norm {name} in only_block_lnwd, got {val}"
        elif any(name.endswith(s) for s in BLOCK_NORM_SUFFIX) or name == LM_HEAD_NORM:
            assert val == 0.1, f"Expected WD=0.1 for block norm {name} in only_block_lnwd, got {val}"
        elif name == "embeddings.weight":
            assert val == 0.0
        else:
            assert val == 0.1


def test_disabled_removes_wd_from_all_norms(model):
    wd = _wd_by_param(model, "disabled")
    for name, val in wd.items():
        if any(name.endswith(s) for s in QK_NORM_SUFFIX + BLOCK_NORM_SUFFIX) or name == LM_HEAD_NORM:
            assert val == 0.0, f"Expected WD=0 for {name} in disabled mode, got {val}"
        elif name == "embeddings.weight":
            assert val == 0.0
        else:
            assert val == 0.1, f"Expected WD=0.1 for {name} in disabled mode, got {val}"


def test_disabled_equals_only_qk_lnwd_union_only_block_lnwd(model):
    """disabled mode should remove WD from exactly the union of only_qk_lnwd and only_block_lnwd."""
    wd_disabled = _wd_by_param(model, "disabled")
    wd_qk = _wd_by_param(model, "only_qk_lnwd")
    wd_block = _wd_by_param(model, "only_block_lnwd")

    zero_disabled = {n for n, v in wd_disabled.items() if v == 0.0}
    zero_qk = {n for n, v in wd_qk.items() if v == 0.0}
    zero_block = {n for n, v in wd_block.items() if v == 0.0}

    assert zero_disabled == zero_qk | zero_block
    assert zero_qk & zero_block == {"embeddings.weight"}  # only overlap is embeddings
