# -*- coding: utf-8 -*-
"""AsyncTLS 语义复现的 channel 校准脚本（Double-Sparsity 口径）。

论文 §4 Stage-2：校准数据集 Dcal 上计算 channel 重要性
  s_i = (1/G) Σ_h max_(q,k)∈Dcal |q_h[i]| · max_(q,k)∈Dcal |k_h[i]|，
取每层 top-dc=32 维作为代表 channel 集 C。

本脚本对齐 FASA 校准口径：held-out 样本 = LongBench qasper 前 N 条，
每条截前 max_len 个 token（控制校准开销）。Qwen3-8B 全层 36。

用法：
  PYTHONPATH=$HOME/local_pylibs CUDA_VISIBLE_DEVICES=0 \
  $PYTHON -m exp.vendor_baselines.asynctls_vendor.identify_channel \
      --model_path $HOME/Qwen3-8B \
      --data_path $HOME/sparse-bench/third_party/KVCache-Factory/data/LongBench \
      --dataset qasper --num_samples 8 --max_len 4096 --dc 32 \
      --output dimension_channels/qwen3-8b/qasper_channels.json
"""
import argparse
import json
import os
import types

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, apply_rotary_pos_emb,
)


def _collect_forward(self, hidden_states, position_embeddings, attention_mask=None,
                     past_key_values=None, cache_position=None, **kwargs):
    """Qwen3Attention prefill 收集 q/k（rotary 后），随后走原生 forward。"""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
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


_collector = {}


def _load_samples(data_path, dataset, num_samples, max_len, tokenizer, device):
    """读 LongBench jsonl 样本，tokenize 并截断。返回 token id 张量列表。"""
    import json
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    n_layers = config.num_hidden_layers
    for li in range(n_layers):
        _collector[li] = None

    layer_id = 0
    def _patch(container):
        nonlocal layer_id
        for name, module in list(container._modules.items()):
            if isinstance(module, Qwen3Attention):
                module.layer_idx = layer_id
                module.flash_forward = module.forward
                module.forward = types.MethodType(_collect_forward, module)
                layer_id += 1
            elif len(list(module._modules.items())) > 0:
                _patch(module)
    _patch(model)
    assert layer_id == n_layers

    samples = _load_samples(args.data_path, args.dataset, args.num_samples,
                            args.max_len, tokenizer, device)
    with torch.no_grad():
        for toks in samples:
            model(toks)

    # 每个 KV group 独立计算 s_i，仅在共享同一 key 的 G 个 query head 内平均。
    channels = []
    for li in range(n_layers):
        collected = _collector[li]
        if collected is None:
            raise RuntimeError(f"layer {li} 未收集到校准 Q/K")
        qmax_abs, kmax_abs = collected
        n_qh, head_dim = qmax_abs.shape
        kv_heads = kmax_abs.shape[0]
        groups = n_qh // kv_heads
        query_grouped = qmax_abs.reshape(kv_heads, groups, head_dim).mean(dim=1)
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
