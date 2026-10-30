"""Check selected frequencies against HF's actual GLM rotary output layout."""
from pathlib import Path
import runpy

import pytest
import torch
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    apply_rotary_pos_emb_interleave,
)


def test_tail_selects_both_components_of_low_frequency_pairs():
    helpers = runpy.run_path(str(Path(__file__).parents[1] /
                                "exp/bench/test_glm47_d256_gates.py"))
    indexer = helpers["make_indexer"](256)
    # HF consumes interleaved pairs but returns cat(real, imag). Tag each
    # frequency independently so the oracle does not duplicate index offsets.
    q = torch.zeros(1, 1, 1, 64)
    q[..., ::2] = 1
    angles = torch.linspace(0.05, 1.55, 32)
    cos = angles.cos().repeat(2).view(1, 1, 64)
    sin = angles.sin().repeat(2).view(1, 1, 64)
    rotated, _ = apply_rotary_pos_emb_interleave(q, q, cos, sin)
    key = torch.cat((torch.full((1, 1, 1, 192), -99.), rotated), -1)
    expected = torch.cat((angles[-16:].cos(), angles[-16:].sin()))
    indices = indexer._tail_indices(torch.device("cpu"))
    selected = key[0, 0, 0, indices]
    torch.testing.assert_close(selected, expected)
    torch.testing.assert_close(selected[:16].square() + selected[16:].square(),
                               torch.ones(16))


def test_production_selector_keeps_nope_content_for_l1_and_l2():
    """GLM 选择器不得再把 192 维 NoPE 内容全部替换成 RoPE tail。"""
    helpers = runpy.run_path(str(Path(__file__).parents[1] /
                                "exp/bench/test_glm47_d256_gates.py"))
    indexer = helpers["make_indexer"](256)
    expected = torch.arange(64)
    torch.testing.assert_close(
        indexer._subspace_indices(torch.device("cpu")), expected)

    # 不只检查 L1 索引：实际 prepare_index 必须让 L2 fine features 与
    # L1 使用同一份 NoPE 内容，防止配置名变了但消费端仍读 RoPE tail。
    keys = torch.randn(1, 65, 2, 256)
    index = indexer.prepare_index(
        keys, torch.tensor([0, 65], dtype=torch.int32))
    assert index["k_min"].shape[-1] == 64
    assert index["k_max"].shape[-1] == 64
    assert index["k_qat"].shape[-1] == 64
    torch.testing.assert_close(index["k_qat"], keys[..., :64])


def test_rope_only_tail_is_rejected_for_glm_d256():
    helpers = runpy.run_path(str(Path(__file__).parents[1] /
                                "exp/bench/test_glm47_d256_gates.py"))
    args = helpers["make_prod_args"](256)
    args.tli_subspace = "tail"
    indexer = helpers["make_indexer"](256, args)
    with pytest.raises(ValueError, match="RoPE-only tail32"):
        indexer.prepare_index(
            torch.randn(1, 65, 2, 256),
            torch.tensor([0, 65], dtype=torch.int32))
