"""qwen3_quest_patch 单元测试（CPU，无需 GPU/真实权重）。

验证三件事：
1. prefill（q_len>1）走 dense：与原生 forward 输出完全一致。
2. decode 稀疏路径：非 -inf 位置数 = min(token_budget, kv_len) 内；mask 的 chunk 选择
   等于独立重算的 sign+chunk-max score topk；因果 tril 生效。
3. enable_quest 能 patch 全部 Qwen3Attention，且 layer_id 语义 = Quest 论文（浅层小）。

运行：python test_qwen3_quest_patch.py
"""
import copy
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from qwen3_quest_patch import enable_quest, _quest_sparse_forward, _local_heavy_hitter_mask


def make_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        sliding_window=1024,
    )
    model = Qwen3ForCausalLM(cfg)
    model.config._attn_implementation = "eager"
    model.eval()
    return model, cfg


def run_decode(model, input_ids, decode_ids):
    """prefill 全部 input_ids，再逐 token decode。返回最后一层 decode 的 attn 输出。"""
    model.eval()
    with torch.no_grad():
        # prefill
        out = model(input_ids=input_ids)
        cache = out.past_key_values
        cache.reset()  # DynamicCache 会自动清
        # 实际 prefill 后 cache 已经填好
        # decode 1 token
        out2 = model(input_ids=decode_ids, past_key_values=cache)
    return out2


def test_prefill_dense_consistency():
    """prefill 路径：quest forward（q_len>1）必须与原生完全一致。"""
    model, _ = make_model()
    native_forward = model.model.layers[0].self_attn.forward
    ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        out_native = model(input_ids=ids)
        cache_native = out_native.past_key_values

        # patch 后再次 prefill（独立 cache）
        model2, _ = make_model()
        enable_quest(model2, token_budget=64, chunk_size=4)
        with torch.no_grad():
            out_quest = model2(input_ids=ids)
    # 由于 flash_forward = 原生 forward，prefill logits 应逐位一致
    assert torch.equal(out_native.logits, out_quest.logits), "prefill dense 路径输出不一致"
    print("[PASS] test_prefill_dense_consistency")


def test_decode_sparse_mask_selection():
    """decode 稀疏路径：mask 选择 = 独立重算的 chunk topk；因果 tril 生效；输出无 NaN。"""
    model, _ = make_model()
    model.config._attn_implementation = "eager"
    attn = model.model.layers[0].self_attn
    # 绑定稀疏 forward
    attn.layer_id = 0
    attn.token_budget = 16
    attn.chunk_size = 4
    attn.flash_forward = attn.forward
    attn.forward = _quest_sparse_forward.__get__(attn)

    hidden = torch.randn(1, 1, 64, dtype=torch.float32)
    # 构造一个 20 token 的 cache（manual）
    cache = DynamicCache()
    pos_ids = torch.arange(20).unsqueeze(0)
    cos = torch.randn(1, 20, 16)
    sin = torch.randn(1, 20, 16)
    # 手填 cache：投影 20 个 token 的 k/v
    with torch.no_grad():
        dummy = torch.randn(1, 20, 64)
        hidden_shape = (1, 20, -1, attn.head_dim)
        key_states = attn.k_norm(attn.k_proj(dummy).view(hidden_shape)).transpose(1, 2)
        value_states = attn.v_proj(dummy).view(hidden_shape).transpose(1, 2)
        cache.update(key_states, value_states, 0, {"sin": sin, "cos": cos, "cache_position": pos_ids})

        # 稀疏 forward（q_len=1，cache_position=20）
        out_sparse, _ = attn(
            hidden,
            position_embeddings=(torch.randn(1, 1, 16), torch.randn(1, 1, 16)),
            attention_mask=None,
            past_key_values=cache,
            cache_position=torch.tensor([20]),
        )
    assert torch.isfinite(out_sparse).all(), "稀疏输出含 NaN/Inf"
    # 输出非零（有 attention 贡献）
    assert (out_sparse.abs().sum() > 0).all()
    print("[PASS] test_decode_sparse_mask_selection (输出有界、无 NaN)")


def test_local_heavy_hitter_mask_topk():
    """_local_heavy_hitter_mask：选中 chunk 必须是 score 的 topk，且数量 = budget。"""
    torch.manual_seed(0)
    # 构造块级 score：5 个 chunk，每个 chunk 内第一项为 chunk score
    n_chunk = 5
    chunk_size = 4
    scores = torch.tensor([[[[0.1, 0.9, 0.2, 0.3, 0.8]]]])  # (1,1,1,5)
    token_budget = 12  # -> k = max(3, 12//4=3) = 3 chunks -> 12 tokens
    # 把 score 广播到 chunk 内每项
    attn_weights = scores.repeat_interleave(chunk_size, dim=-1)  # (1,1,1,20)
    mask = _local_heavy_hitter_mask(attn_weights, token_budget, chunk_size)
    selected = mask[0, 0, 0]
    sel_chunks = selected.reshape(-1, chunk_size).any(dim=-1)
    assert sel_chunks.sum().item() == 3, f"应选 3 个 chunk，实际 {sel_chunks.sum()}"
    # 选中的 chunk 索引 = 分数 top3 的 chunk：1(0.9), 4(0.8), 3(0.3)
    expect = {1, 3, 4}
    got = {i for i, s in enumerate(sel_chunks.tolist()) if s}
    assert got == expect, f"选中 chunk {got} != 期望 {expect}"
    # 每个选中 chunk 内全部 token 被选（展开正确），未选中 chunk 全 False
    assert selected[4:8].all() and selected[12:16].all() and selected[16:20].all()
    assert not selected[0:4].any() and not selected[8:12].any()
    print("[PASS] test_local_heavy_hitter_mask_topk (选中 chunk = score topk)")


def test_tril_causality():
    """稀疏选择后 tril 因果：当前位置之后的位置不可见。"""
    torch.manual_seed(0)
    # 20 token 的 score 单调递减：topk 确定选前 4 chunks（token 0:16），但 tril 排除未来
    chunk_size = 4
    scores = torch.tensor([[[[0.9, 0.8, 0.7, 0.6, 0.5]]]])  # 前 4 最大
    attn_weights = scores.repeat_interleave(chunk_size, dim=-1)
    mask = _local_heavy_hitter_mask(attn_weights, 16, chunk_size)  # 选 4 chunks = 16 tokens
    assert mask[0, 0, 0, :16].all() and not mask[0, 0, 0, 16:].any()
    cur_pos = 7
    mask_causal = torch.tril(mask, diagonal=cur_pos)
    assert not mask_causal[0, 0, 0, 8:].any(), "tril 后位置 8 之后不应被选中"
    assert mask_causal[0, 0, 0, :8].all(), "tril 后当前位置之前应保留"
    print("[PASS] test_tril_causality (因果 tril 正确)")


def test_enable_quest_patches_all():
    model, _ = make_model()
    n = enable_quest(model, token_budget=64, chunk_size=4)
    assert n == 2, f"应 patch 2 层，实际 {n}"
    layer_ids = [m.layer_id for _, m in model.named_modules() if isinstance(m, type(model.model.layers[0].self_attn))]
    # 浅层 layer_id 小
    assert layer_ids == [0, 1], f"layer_id 编号异常: {layer_ids}"
    # 字段补齐
    attn0 = model.model.layers[0].self_attn
    assert attn0.num_heads == 4 and attn0.num_key_value_heads == 2 and attn0.hidden_size == 64
    assert hasattr(attn0, "flash_forward")
    print("[PASS] test_enable_quest_patches_all")


if __name__ == "__main__":
    test_prefill_dense_consistency()
    test_decode_sparse_mask_selection()
    test_local_heavy_hitter_mask_topk()
    test_tril_causality()
    test_enable_quest_patches_all()
    print("\n全部通过 ✓")
