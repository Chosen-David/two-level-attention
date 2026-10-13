# FASA Phase 1 Qwen3 校准（deepseek 2026-10-13）
# 官方 identify_important_fre.py 仅支持 Llama/Qwen2（transformers 4.43 flash 类）。
# 本文件把 partial 统计逻辑移植到 transformers 4.56 Qwen3Attention：
#   - Qwen3 单类统一（eager/sdpa/flash），q_norm/k_norm 在投影后、RoPE 前；
#   - kv 不做预展开（attention_interface 内部 repeat_kv），故 partial 调用前手动展开；
#   - 其余语义（partial vs full top-k 交集）逐字节保持官方
#     partial_frequency_against_full_frequency 实现。
# 产出：dimension_rankings/<model>/<dataset>_<n>.pkl（dill，res_dict 结构）。
# 消费：convert_agreement.py 聚合为 <dataset>_agreement.pkl 供 Phase 2。
import argparse
import json
import os
import sys
from collections import defaultdict

import dill
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, apply_rotary_pos_emb, eager_attention_forward, repeat_kv,
)

# ---- 官方 partial_frequency_against_full_frequency（模型无关，逐字节保持）----
res_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
sparsity_list = [128, 256, 512, 1024, 2048, 4096]


def partial_frequency_against_full_frequency(
        query_states, key_states, value_states, layer_idx, num_hidden_layers):
    bs, q_heads, _, head_dim = query_states.shape
    bs, k_heads, seq_len, _ = key_states.shape
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    sorted_indices_full = torch.argsort(attn_weights, dim=-1, descending=True)
    with torch.no_grad():
        for frequency_group_idx in range(0, head_dim, 2):
            query_states_partial = query_states[:, :, :,
                                                frequency_group_idx:(frequency_group_idx + 2)]
            key_states_partial = key_states[:, :, :,
                                            frequency_group_idx:(frequency_group_idx + 2)]
            attn_weights_partial = torch.matmul(
                query_states_partial, key_states_partial.transpose(2, 3))
            sorted_indices_partial = torch.argsort(
                attn_weights_partial, dim=-1, descending=True)
            del attn_weights_partial
            for sparsity in sparsity_list:
                preceding_token_num = int(seq_len * sparsity) if sparsity < 1 else sparsity
                preceding_token_num = min(seq_len, preceding_token_num)

                sorted_indices_p = sorted_indices_partial[:, :, :, :preceding_token_num]
                sorted_indices_f = sorted_indices_full[:, :, :, :preceding_token_num]

                a = sorted_indices_f.squeeze(2)
                b = sorted_indices_p.squeeze(2)

                intersection_mask = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1)
                intersection_count = intersection_mask.sum(-1)
                temporal_res = intersection_count.squeeze(dim=0) / preceding_token_num
                temporal_res = temporal_res.tolist()

                res_dict[sparsity][frequency_group_idx][layer_idx].append(temporal_res)


# ---- Qwen3Attention partial forward（基于 4.56 原 forward，仅插入统计）----
def qwen3_attention_forward_partial(
        self, hidden_states, position_embeddings, attention_mask,
        past_key_values=None, cache_position=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        try:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        except RuntimeError as e:
            print(f"[DBG] L{self.layer_idx} hidden={tuple(hidden_states.shape)} "
                  f"qlen={query_states.shape[2]} "
                  f"key_in={tuple(key_states.shape)} "
                  f"cache_keys={tuple(past_key_values.layers[self.layer_idx].keys.shape)}",
                  flush=True)
            raise

    # ---- FASA 统计插入点：decode（q_len==1）时，手动展开 kv + 统计 ----
    if query_states.shape[2] == 1:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        partial_frequency_against_full_frequency(
            query_states, key_states, value_states,
            self.layer_idx, self.config.num_hidden_layers)

    # ---- 原逻辑走 attention_interface（校准阶段无关紧要，保持等价）----
    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


# ---- LongBench prompt 模板（官方口径，仅取 qasper 一条）----
LB_PROMPTS = {
    "qasper": ("You are given a scientific article and a question. Answer the question as "
               "concisely as you can, using a single phrase or sentence if possible. If the "
               "question cannot be answered based on the information in the article, write "
               '"unanswerable". If the question is a yes/no question, answer "yes", "no", or '
               '"unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n '
               "Answer the question based on the above article as concisely as you can, using "
               "a single phrase or sentence if possible. If the question cannot be answered "
               'based on the information in the article, write "unanswerable". If the question '
               'is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide '
               "any explanation.\n\nQuestion: {input}\n\nAnswer:"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/wangyuanshuo02/Qwen3-8B")
    parser.add_argument("--model_name", type=str, default="Qwen3-8B")
    parser.add_argument("--dataset_name", type=str, default="qasper")
    parser.add_argument("--data_root", type=str,
                        default="/home/wangyuanshuo02/sparse-bench/third_party/"
                                "PyramidKV/data/LongBench")
    parser.add_argument("--evaluate_num", type=int, default=8)
    parser.add_argument("--max_new", type=int, default=32,
                        help="每样本 decode 步数（触发统计的 token 位置数）")
    parser.add_argument("--out_root", type=str, default="dimension_rankings")
    args = parser.parse_args()

    model_name = args.model_name
    ranks_dir = os.path.join(args.out_root, model_name)
    os.makedirs(ranks_dir, exist_ok=True)
    ranks_file_name = os.path.join(ranks_dir, f"{args.dataset_name}_{args.evaluate_num}.pkl")

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 挂 partial forward（Qwen3Attention 单类统一，直接替换 forward）
    patched = 0
    for m in model.modules():
        if isinstance(m, Qwen3Attention):
            m.forward = qwen3_attention_forward_partial.__get__(m, Qwen3Attention)
            patched += 1
    if patched == 0:
        raise RuntimeError("没有挂载任何 Qwen3Attention 模块")
    print(f"[fasa-phase1] patched {patched} Qwen3Attention layers")

    data_file = os.path.join(args.data_root, f"{args.dataset_name}.jsonl")
    rows = [json.loads(l) for l in open(data_file)][:args.evaluate_num]
    prompt_tmpl = LB_PROMPTS.get(args.dataset_name)
    if prompt_tmpl is None:
        prompt_tmpl = LB_PROMPTS["qasper"]
    print(f"[fasa-phase1] dataset={args.dataset_name} n={len(rows)} "
          f"max_new={args.max_new} -> {ranks_file_name}")

    total_decode = 0
    for r_i, row in enumerate(rows):
        prompt = prompt_tmpl.format(context=row["context"], input=row["input"])
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            # 注意：绝不传 logits_to_keep=None（会导致 hidden_states[:, None, :] →
            # logits 4 维 → argmax 后 input_ids 3 维 → 后续 embed/attention 4 维
            # 崩溃）。与 pred_ruler.py 同口径：超 32K 档传 logits_to_keep=1，
            # ≤32K 档不传（默认 0）。
            output = model(
                input_ids=input_ids, use_cache=True,
                **( {"logits_to_keep": 1} if input_ids.shape[-1] > 32768 else {}))
            past = output.past_key_values
            pred_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            for _ in range(args.max_new):
                out = model(input_ids=pred_idx, past_key_values=past, use_cache=True)
                past = out.past_key_values
                pred_idx = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                total_decode += 1
        torch.cuda.empty_cache()
        print(f"[fasa-phase1] sample {r_i + 1}/{len(rows)} done "
              f"(decode_steps_so_far={total_decode})")
    print(f"[fasa-phase1] total decode steps = {total_decode}")

    with open(ranks_file_name, "wb") as f:
        dill.dump(res_dict, f)
    print(f"[fasa-phase1] saved res_dict -> {ranks_file_name}")


if __name__ == "__main__":
    main()
