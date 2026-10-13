"""FASA Qwen3 Phase 2 smoke：短序列 + 大 budget = 无剪枝 = dense 等价。

验证逻辑：decode 剪枝分支在 seq_len < budget 时 preceding_token_num=seq_len
（core_module_with_padding_old 的 min 语义）→ 剪枝退化为全量 → 输出应与
dense（未 patch）逐 token 一致。这是适配正确性的硬门禁。
"""
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

sys.path.insert(0, ".")
from qwen3_fasa_patch import enable_fasa  # noqa: E402

MODEL = "/home/wangyuanshuo02/Qwen3-8B"


def greedy(model, tok, prompt, max_new=8):
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        pid = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        gen = [pid.item()]
        for _ in range(max_new - 1):
            out = model(input_ids=pid, past_key_values=past, use_cache=True)
            past = out.past_key_values
            pid = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            gen.append(pid.item())
    return tok.decode(gen, skip_special_tokens=True)


def main():
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    # dense 基线
    dense = AutoModelForCausalLM.from_pretrained(
        MODEL, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto").eval()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt = ("The quick brown fox jumps over the lazy dog. "
              "In a remote village there lived a wise old owl. "
              "It watched the stars every night and shared stories.")
    ref = greedy(dense, tok, prompt)
    del dense
    torch.cuda.empty_cache()

    # FASA 大 budget（> seq_len → 无剪枝）
    fasa = AutoModelForCausalLM.from_pretrained(
        MODEL, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto").eval()
    n = enable_fasa(fasa, agreement_dir="dimension_rankings",
                    model_name="Qwen3-8B", dataset_name="qasper",
                    budget=2048, threshold_ratio=0.8, min_k=16, max_k=16)
    print(f"[smoke] patched {n} layers")
    got = greedy(fasa, tok, prompt)
    print(f"[smoke] dense: {ref!r}")
    print(f"[smoke] fasa : {got!r}")
    if ref == got:
        print("[smoke] PASS: fasa(no-prune) == dense")
    else:
        print("[smoke] FAIL: mismatch")
        sys.exit(1)


if __name__ == "__main__":
    main()
