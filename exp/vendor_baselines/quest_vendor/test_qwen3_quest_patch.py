"""qwen3_quest_patch 单元测试（CPU，官方 Quest forward 绑定验证）。

验证三件事：
1. enable_quest 能 patch 全部 Qwen3Attention，字段补齐（num_heads/kv_heads/
   hidden_size/rotary_emb/layer_id/token_budget/chunk_size）。
2. prefill（q_len>1）走 dense：与未 patch 模型的输出完全一致。
3. decode 走官方 Quest 稀疏路径：输出有界无 NaN，且与 dense 输出不同
   （稀疏生效）。

运行：python test_qwen3_quest_patch.py
"""
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen3_quest_patch import enable_quest


def make_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,  # 前 2 层 dense（layer_id<2），后 2 层稀疏
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


def run(model, input_ids, num_new=3):
    """prefill input_ids，再逐 token decode num_new 个。返回 (prefill_logits, 各步 decode logits)。"""
    out = model(input_ids=input_ids)
    pre_logits = out.logits
    cache = out.past_key_values
    dec_logits = []
    for _ in range(num_new):
        out = model(input_ids=torch.tensor([[input_ids.shape[1]]], device=input_ids.device), past_key_values=cache)
        cache = out.past_key_values
        dec_logits.append(out.logits)
    return pre_logits, dec_logits


def test_enable_patches_and_fields():
    model, _ = make_model()
    n = enable_quest(model, token_budget=16, chunk_size=4)
    assert n == 4, f"应 patch 4 层，实际 {n}"
    attn0 = model.model.layers[0].self_attn
    assert attn0.num_heads == 4 and attn0.num_key_value_heads == 2 and attn0.hidden_size == 64
    assert hasattr(attn0, "flash_forward") and hasattr(attn0, "_official_forward")
    assert attn0.rotary_emb is not None, "rotary_emb 未注入"
    layer_ids = [m.layer_id for _, m in model.named_modules()
                 if m.__class__.__name__ == "Qwen3Attention"]
    assert layer_ids == [0, 1, 2, 3], f"layer_id 编号异常: {layer_ids}"
    print("[PASS] test_enable_patches_and_fields")


def test_prefill_dense_consistency():
    model, _ = make_model()
    ids = torch.randint(0, 128, (1, 8))
    pre_native, dec_native = run(model, ids)

    model2, _ = make_model()
    enable_quest(model2, token_budget=16, chunk_size=4)
    pre_quest, _ = run(model2, ids)

    # prefill 走 dense（flash_forward），应完全一致
    assert torch.equal(pre_native, pre_quest), "prefill dense 输出不一致"
    print("[PASS] test_prefill_dense_consistency")


def test_full_budget_dense_equivalence():
    """【GPT 审计 2026-10-11 门禁】token_budget 全预算时，quest_vendor 的
    prefill+decode 输出必须与 dense 完全一致（torch.equal）。

    全保留的实现前提（官方语义）：local_heavy_hitter_mask 的 topk 用
    k = min(max(3, token_budget // chunk_size), num_chunks)，而 quest forward
    里 token_budget = min(kv_seq_len, self.token_budget)。当 kv_seq_len 不是
    chunk_size 整数倍时，kv_seq_len // chunk_size < num_chunks → 最后含新
    token 的 chunk 会被丢弃 → 稀疏必生效，无法用「全预算」验证 dense 等价。
    因此本测试用 chunk_size=1（每 token 一个 chunk），使 kv_seq_len // 1 =
    num_chunks，真正全保留。此时若 K/V 前处理（q_norm/k_norm）缺失，decode
    新 K 与 prefill 缓存 K 不在同一空间，输出会确定性不一致（暴露 bug）。"""
    model, _ = make_model()
    ids = torch.randint(0, 128, (1, 40))
    pre_native, dec_native = run(model, ids, num_new=5)

    model2, _ = make_model()
    enable_quest(model2, token_budget=100000, chunk_size=1)  # 全预算（chunk=1 保证全保留）
    pre_quest, dec_quest = run(model2, ids, num_new=5)

    assert torch.equal(pre_native, pre_quest), "全预算 prefill 输出不一致"
    for i, (dn, dq) in enumerate(zip(dec_native, dec_quest)):
        assert torch.equal(dn, dq), (
            f"全预算 decode step {i} 输出不一致——K/V 前处理（q_norm/k_norm）"
            f"未对齐（prefill 缓存与 decode 新 K 不在同一空间）"
        )
    print("[PASS] test_full_budget_dense_equivalence")


def test_decode_sparse_active_and_bounded():
    model, _ = make_model()
    # 序列（100）须超 token_budget（16）——短序列 + 大预算时稀疏零删减，
    # 要求输出不同反而掩盖 norm bug（GPT 审计 2026-10-11 指正）。
    ids = torch.randint(0, 128, (1, 100))
    _, dec_native = run(model, ids, num_new=5)

    model2, _ = make_model()
    enable_quest(model2, token_budget=16, chunk_size=4)
    _, dec_quest = run(model2, ids, num_new=5)

    for i, (dn, dq) in enumerate(zip(dec_native, dec_quest)):
        assert torch.isfinite(dq).all(), f"decode step {i} 输出含 NaN/Inf"
        assert torch.isfinite(dn).all(), f"dense step {i} 输出含 NaN/Inf"
        # 序列超预算 → 稀疏生效 → 输出与 dense 不同
        diff = (dq - dn).abs().max().item()
        assert diff > 0, f"decode step {i} 稀疏与 dense 输出相同（稀疏未生效）"
    print("[PASS] test_decode_sparse_active_and_bounded")


if __name__ == "__main__":
    test_enable_patches_and_fields()
    test_prefill_dense_consistency()
    test_full_budget_dense_equivalence()
    test_decode_sparse_active_and_bounded()
    print("\n全部通过 ✓")
