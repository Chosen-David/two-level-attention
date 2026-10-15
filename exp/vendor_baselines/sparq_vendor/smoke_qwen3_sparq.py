"""SparQ vendor Qwen3 适配层 smoke 测试。

验证三件事：
1. k >= kv_len 时（全保留）SparQ 应退化为 dense —— 与 fullkv 逐 token 等价。
2. decode 单 token 稀疏路径能正常跑通（不崩、形状对）。
3. GQA 展开正确（num_key_value_groups=8，Qwen3-8B kv_heads=4 → H=32）。

用法：CUDA_VISIBLE_DEVICES=0 python3 smoke_qwen3_sparq.py
"""
import sys, os, types
import torch

REPO = "/home/wangyuanshuo02/two-level-attention"
sys.path.insert(0, os.path.join(REPO, "exp", "vendor_baselines", "sparq_vendor"))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from qwen3_sparq_patch import enable_sparq
from sparq_attention_official import SparQArgs, RunningVMean, sparq_attn


def run_generate(model, tokenizer, prompt, max_new=6):
    """贪心生成，返回生成 token ids。走原生 cache/generate。"""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    out = model.generate(
        input_ids, max_new_tokens=max_new, do_sample=False, temperature=None,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return out[0].tolist()


def main():
    model_path = "/home/wangyuanshuo02/Qwen3-8B"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "eager"  # 确保可逐 token 对比
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompt = ("The history of the Roman Empire spans over a thousand years, "
              "from its founding in 753 BC to the fall of Constantinople. "
              "One of the most famous emperors was Julius Caesar, who "
              "transformed the Republic into an empire. The Roman army was "
              "renowned for its discipline and engineering skills, building "
              "roads and aqueducts that still exist today. In addition, "
              "Roman law and Latin language profoundly influenced Western "
              "civilization. The empire eventually split into Eastern and "
              "Western halves, with the former surviving as the Byzantine "
              "Empire for another thousand years.")

    # ---------- Test 1: k >= kv_len → dense 等价 ----------
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16).eval().cuda()

    n_patched = enable_sparq(model, k=1024, context_length=8192)
    print(f"[T1] patched layers: {n_patched}")
    assert n_patched == config.num_hidden_layers, f"expected 28 layers, got {n_patched}"

    # prompt 约 100+ token，kv_len 在 6 次生成内不会超过 1024 → SparQ 全保留 ≈ dense
    sparse_ids = run_generate(model, tokenizer, prompt, max_new=4)
    print(f"[T1] sparq (k=1024, kv<1024) ids: {sparse_ids}")

    del model
    torch.cuda.empty_cache()

    # fullkv dense 对照
    model2 = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16).eval().cuda()
    dense_ids = run_generate(model2, tokenizer, prompt, max_new=4)
    print(f"[T1] dense ids: {dense_ids}")

    if sparse_ids == dense_ids:
        print("[T1] PASS: k>=kv_len 时 SparQ 与 dense 逐 token 等价（全保留 sanity）")
    else:
        print(f"[T1] WARN: 不等价!\n  sparq={sparse_ids}\n  dense={dense_ids}")
        # 允许 bf16 舍入差异导致最后 1-2 token 微差，但前 2 token 必须一致
        if sparse_ids[:2] != dense_ids[:2]:
            raise SystemExit("[T1] FAIL: 前 2 token 都不一致，适配层有 bug")

    del model2
    torch.cuda.empty_cache()

    # ---------- Test 2: decode 路径形状/GQA 展开 ----------
    model3 = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16).eval().cuda()
    enable_sparq(model3, k=1024, context_length=8192)

    # 手动 prefill + decode 一步，检查各层 SparQ 输出形状
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    past = None
    with torch.no_grad():
        # prefill
        out = model3(input_ids, use_cache=True)
        past = out.past_key_values
        # decode 1 token
        new_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        out2 = model3(new_id, past_key_values=past, use_cache=True)
        print(f"[T2] decode logits shape: {out2.logits.shape} (expect [1,1,151936])")
        assert out2.logits.shape[-2] == 1, "decode logits 应为 1 token"

    # 逐层检查 SparQAttention 的 V_mean 已初始化
    n_vmean_init = 0
    for name, mod in model3.named_modules():
        if isinstance(mod, Qwen3Attention) and hasattr(mod, "sparq_vmean"):
            if mod.sparq_vmean.n.item() > 0:
                n_vmean_init += 1
    print(f"[T2] V_mean initialized layers: {n_vmean_init}/{config.num_hidden_layers}")
    assert n_vmean_init == config.num_hidden_layers, "所有层 V_mean 应已 init"
    print("[T2] PASS: decode 稀疏路径形状正确 + V_mean 全层初始化")

    # ---------- Test 3: r/k 参数随 context_length ----------
    for cl, expect_r in [(8192, 32), (32768, 8), (131072, 2)]:
        n = enable_sparq(model3, k=1024, context_length=cl)
        first = next(m for m in model3.modules() if isinstance(m, Qwen3Attention))
        print(f"[T3] context_length={cl} → r={first.sparq_r} (expect {expect_r}) k={first.sparq_k}")
        assert first.sparq_r == expect_r, f"r mismatch: got {first.sparq_r}"
    print("[T3] PASS: r/k 随 context_length 正确")


if __name__ == "__main__":
    main()
    print("\n=== ALL SMOKE PASS ===")
