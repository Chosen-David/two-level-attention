# -*- coding: utf-8 -*-
"""AsyncTLS GLM-4.7-Flash channel 校准脚本（Double-Sparsity 口径）。

与 identify_channel.py 同一校准语义：held-out 样本上按 token 取
|q_h[i]| 与 |k_h[i]| 的 max，逐 KV group 计算 s_i = qmax·kmax，
取 top-dc=32 维。GLM MLA 下 num_attention_heads == num_key_value_heads，
每个 KV group 仅 1 个 query head。

用法（在装有 transformers 5.12.1 的环境）：
  PYTHONPATH=/home/wangyuanshuo02/.local/pylibs_tf512:$REPO \
  CUDA_VISIBLE_DEVICES=0 \
  /home/fluentllmenv/bin/python -m exp.vendor_baselines.asynctls_vendor.glm4_moe_lite_identify_channel \
      --model_path /home/wangyuanshuo02/models/huggingface.co/zai-org/GLM-4.7-Flash \
      --data_path /home/wangyuanshuo02/datasets/LongBench \
      --dataset qasper --num_samples 8 --max_len 4096 --dc 32 \
      --output exp/vendor_baselines/asynctls_vendor/calib/glm-4.7-flash_qasper_channels.json
"""
import argparse
import json
import os
import types

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteAttention,
)

from glm4_moe_lite_asynctls_patch import _glm_project_qkv


_collector = {}


def _collect_forward(self, hidden_states, position_embeddings,
                     attention_mask=None, past_key_values=None, **kwargs):
    """Glm4MoeLiteAttention prefill 收集 q/k（rotary 后），随后走原生 forward。"""
    query_states, key_states, _ = _glm_project_qkv(
        self, hidden_states, position_embeddings)
    query_max = query_states[0].detach().abs().amax(dim=1)
    key_max = key_states[0].detach().abs().amax(dim=1)
    previous = _collector[self.layer_idx]
    if previous is None:
        _collector[self.layer_idx] = (query_max, key_max)
    else:
        _collector[self.layer_idx] = (
            torch.maximum(previous[0], query_max),
            torch.maximum(previous[1], key_max),
        )
    return self.flash_forward(
        hidden_states, position_embeddings, attention_mask,
        past_key_values, **kwargs)


def _load_samples(data_path, dataset, num_samples, max_len, tokenizer, device):
    samples = []
    path = os.path.join(data_path, dataset + ".jsonl")
    with open(path, "r") as f:
        for line in f:
            if len(samples) >= num_samples:
                break
            samples.append(json.loads(line))
    tokenized = []
    for s in samples:
        text = s.get("context", "")
        toks = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=max_len).input_ids
        tokenized.append(toks.to(device))
    return tokenized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--dataset", default="qasper")
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--dc", type=int, default=32)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)

    n_layers = config.num_hidden_layers
    for li in range(n_layers):
        _collector[li] = None

    n_patched = 0
    for module in model.modules():
        if not isinstance(module, Glm4MoeLiteAttention):
            continue
        module.flash_forward = module.forward
        module.forward = types.MethodType(_collect_forward, module)
        n_patched += 1
    assert n_patched == n_layers, f"{n_patched} != {n_layers}"

    samples = _load_samples(args.data_path, args.dataset, args.num_samples,
                            args.max_len, tokenizer, device)
    with torch.no_grad():
        for toks in samples:
            model(toks)

    channels = []
    for li in range(n_layers):
        collected = _collector[li]
        if collected is None:
            raise RuntimeError(f"layer {li} 未收集到校准 Q/K")
        qmax_abs, kmax_abs = collected
        n_qh, head_dim = qmax_abs.shape
        kv_heads = kmax_abs.shape[0]
        groups = n_qh // kv_heads
        query_grouped = qmax_abs.reshape(
            kv_heads, groups, head_dim).mean(dim=1)
        scores = query_grouped * kmax_abs
        top = torch.topk(scores, k=args.dc, dim=1).indices.tolist()
        channels.append([sorted(group) for group in top])
        print(f"layer {li}: {kv_heads} KV groups; group 0 channels {top[0][:8]}...")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "schema": "asynctls-per-kv-group-channels-v1",
            "model_path": args.model_path,
            "dc": args.dc,
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "max_len": args.max_len,
            "channels": channels,
        }, f, indent=1)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
