# -*- coding: utf-8 -*-
"""Qwen3 + MInference 在线搜索脚本：生成 best_pattern JSON（32 层×40 头）。

MInference 官方 is_search 流程：对每层每 head 用 search_pattern（vertical_and_slash /
stream_llm / block_sparse 三种候选取 mass 最优）搜出 (ty, v_size, s_size, score)。
本脚本在 Qwen3 上跑一次长 prefill（须 >3500 token，search_pattern 的 vs_list 上限），
逐层写入 config-out JSON，正式推理（pred.py method=minference_vendor, is_search=False）
复用该 JSON。

用法（fluentllmenv，py3.11/transformers 4.56.2）：
  /home/fluentllmenv/bin/python3 qwen3_minference_search.py \
      --model /home/wangyuanshuo02/Qwen3-8B \
      --config-out exp/vendor_baselines/minference_vendor/best_pattern_qwen3_8b.json \
      --seq-len 8192

断点续搜：config-out 若已存在部分层（官方按 len(config_list) 从断点继续），
完整 32 层后再跑会 raise "Search already completed"（预期）。
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen3_minference_patch import enable_minference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/wangyuanshuo02/Qwen3-8B")
    ap.add_argument("--config-out", required=True)
    ap.add_argument("--seq-len", type=int, default=8192)
    args = ap.parse_args()

    out_dir = os.path.dirname(args.config_out) or "."
    os.makedirs(out_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    n = enable_minference(
        model, is_search=True, config_path=args.config_out, starting_layer=-1
    )
    print(f"patched {n} Qwen3Attention layers")

    # 须 >3500 token（search_pattern vs_list vertical_size 上限），默认 8192。
    # 用自然多样文本（避免重复句子导致部分 head score≤0 搜出空 pattern）。
    para = (
        "The capital of France is Paris, a city known for its art, fashion, "
        "gastronomy, and rich history. The Eiffel Tower, built in 1889, stands "
        "as a global icon of engineering and beauty. Along the Seine River, "
        "museums like the Louvre house masterpieces spanning centuries. "
        "French cuisine, from croissants to coq au vin, reflects regional "
        "traditions and the pursuit of culinary excellence. Paris remains a "
        "cultural beacon, attracting millions of visitors every year. "
        "Meanwhile, scientific research advances rapidly across many fields. "
        "Researchers in machine learning develop new methods for processing "
        "long sequences of text, balancing accuracy with computational cost. "
        "Sparse attention mechanisms aim to approximate full attention while "
        "reducing memory and latency. "
    )
    n_rep = max(1, args.seq_len // 60 + 1)
    text = para * n_rep
    ids = tok(text, return_tensors="pt", truncation=True, max_length=args.seq_len).input_ids.cuda()
    print("search prompt len:", ids.shape[1])

    with torch.no_grad():
        out = model(ids, use_cache=False)

    # sanitize：搜索文本下个别 head 可能 score≤0 搜出空 pattern ['',0,0,0]。
    # 官方 minference_prefill_kernel 对"不存在的 head"默认兜底
    # ('vertical_and_slash',1000,6096,1)，但空 pattern 存在于文件时 .get 返回空
    # → KeyError ''。替换为默认（pattern 数据预处理，不改官方 kernel 源码）。
    data = json.load(open(args.config_out))
    changed = 0
    for layer in data:
        for hid, pat in layer.items():
            if isinstance(pat, list) and len(pat) >= 1 and pat[0] == "":
                layer[hid] = ["vertical_and_slash", 1000, 6096, 1]
                changed += 1
    if changed:
        json.dump(data, open(args.config_out, "w"), indent=2)
    print(f"search DONE layers={len(data)} logits={tuple(out.logits.shape)} "
          f"sanitized_empty={changed}")
    print("layer0 head0:", data[0].get("0"))


if __name__ == "__main__":
    main()
