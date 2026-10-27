"""E178: 完整 e2e — prefill + decode (TTFT + TPOT + throughput)。

两臂对比：
- dense: Qwen3-8B 原生 SDPA
- tli:   Qwen3-8B + TLI champion flags (K1=64, iter3)

度量：prefill_ms, ttft_ms, tpot_ms, total_e2e_ms, throughput_tok/s, peak_mem_gib
"""
import argparse
import json
import os
import sys
import time

import torch

# 确保项目根在 path 上
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_dir)

from transformers import Qwen3ForCausalLM


def load_model(model_path, device):
    print(f"加载模型: {model_path}")
    model = Qwen3ForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation='sdpa')
    model.eval()
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"参数量: {total_params:.2f}B, 层数: {model.config.num_hidden_layers}")
    return model


def install_tli(model):
    """给模型安装 TLI patch（champion flags）。"""
    from sparse_attn.patches import register_patch
    from sparse_attn.arguments import add_sparse_attn_args
    parser = argparse.ArgumentParser()
    add_sparse_attn_args(parser)
    args = parser.parse_args([
        '--method', 'tli',
        '--tli_far_method', 'minmax', '--tli_near_method', 'avg',
        '--tli_near_pages', '32', '--tli_near_tokens', 'off',
        '--tli_near_len', '4096', '--tli_subspace', 'tail',
        '--tia_level2_cmp_ratio', '4', '--tli_sink_tokens', '4',
        '--tli_swa_size', '128', '--tia_level1_topk', '64',
        '--tia_level2_topk', '1024',
    ])
    register_patch(model, args)
    return model


@torch.inference_mode()
def run_e2e(model, L, max_gen=128, warmup=3):
    """单次 e2e：prefill → decode。返回各阶段耗时。"""
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # 构造输入
    input_ids = torch.randint(1, 1000, (1, L), device=dev)
    cache_position = torch.arange(L, device=dev)
    attention_mask = torch.ones(1, L, dtype=dtype, device=dev)

    # ── Prefill ──
    if dev.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                cache_position=cache_position, use_cache=True)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    prefill_end = time.perf_counter()
    prefill_ms = (prefill_end - t0) * 1e3

    first_token = out.logits[0, -1, :].argmax(dim=-1).item()
    past_kv = out.past_key_values

    # ── Warmup decode ──
    decode_ids = torch.tensor([[first_token]], device=dev)
    for step in range(warmup):
        cache_pos = torch.tensor([L + step], device=dev)
        out = model(input_ids=decode_ids, cache_position=cache_pos,
                    past_key_values=past_kv, use_cache=True)
        pred = out.logits[0, -1, :].argmax(dim=-1).item()
        decode_ids = torch.tensor([[pred]], device=dev)
    if dev.type == 'cuda':
        torch.cuda.synchronize()

    # ── Measured decode ──
    step_ms = []
    decode_ids = torch.tensor([[first_token]], device=dev)
    decode_t0 = time.perf_counter()
    for step in range(max_gen):
        cache_pos = torch.tensor([L + step], device=dev)
        t1 = time.perf_counter()
        out = model(input_ids=decode_ids, cache_position=cache_pos,
                    past_key_values=past_kv, use_cache=True)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - t1) * 1e3)
        pred = out.logits[0, -1, :].argmax(dim=-1).item()
        decode_ids = torch.tensor([[pred]], device=dev)
    decode_end = time.perf_counter()

    decode_total_ms = (decode_end - decode_t0) * 1e3
    tpot_ms = decode_total_ms / len(step_ms)
    total_ms = (decode_end - t0) * 1e3
    throughput = max_gen / (total_ms / 1e3)
    peak_mem = (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == 'cuda' else 0

    return {
        'prefill_ms': prefill_ms,
        'ttft_ms': prefill_ms,
        'tpot_ms': tpot_ms,
        'decode_total_ms': decode_total_ms,
        'total_e2e_ms': total_ms,
        'throughput_tok_s': throughput,
        'peak_mem_gib': peak_mem,
        'first_token': first_token,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/home/wangyuanshuo02/Qwen3-8B')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--lengths', nargs='+', type=int, default=[32768, 65536, 131072])
    parser.add_argument('--max_gen', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', default='exp/trace/results/e178_full_e2e.json')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda:0'

    print(f"=== E178 完整 e2e (prefill+decode) ===")
    print(f"GPU: {args.gpu}, repeats: {args.repeats}, max_gen: {args.max_gen}")

    all_results = {}

    # ── Dense 臂 ──
    print(f"\n{'#'*60}")
    print(f"# DENSE 臂")
    print(f"{'#'*60}")
    model_dense = load_model(args.model_path, device)
    dense_results = {}
    for L in args.lengths:
        print(f"\nL={L:,}")
        samples = []
        for r in range(args.repeats):
            res = run_e2e(model_dense, L, args.max_gen)
            samples.append(res)
            print(f"  [{r+1}/{args.repeats}] prefill={res['prefill_ms']:.1f}ms "
                  f"tpot={res['tpot_ms']:.3f}ms total={res['total_e2e_ms']:.1f}ms "
                  f"thr={res['throughput_tok_s']:.1f}tok/s")
        # 取中位数
        tpot_sorted = sorted(s['tpot_ms'] for s in samples)
        prefill_sorted = sorted(s['prefill_ms'] for s in samples)
        total_sorted = sorted(s['total_e2e_ms'] for s in samples)
        thr_sorted = sorted(s['throughput_tok_s'] for s in samples)
        mem_sorted = sorted(s['peak_mem_gib'] for s in samples)
        mid = len(tpot_sorted) // 2
        dense_results[str(L)] = {
            'prefill_ms': prefill_sorted[mid],
            'tpot_ms': tpot_sorted[mid],
            'total_e2e_ms': total_sorted[mid],
            'throughput_tok_s': thr_sorted[mid],
            'peak_mem_gib': mem_sorted[mid],
        }
        print(f"  → median: prefill={prefill_sorted[mid]:.1f}ms, "
              f"tpot={tpot_sorted[mid]:.3f}ms, "
              f"total={total_sorted[mid]:.1f}ms, "
              f"thr={thr_sorted[mid]:.1f}tok/s, "
              f"mem={mem_sorted[mid]:.2f}GiB")
    del model_dense
    torch.cuda.empty_cache()

    # ── TLI 臂 ──
    print(f"\n{'#'*60}")
    print(f"# TLI 臂")
    print(f"{'#'*60}")
    model_tli = load_model(args.model_path, device)
    install_tli(model_tli)
    tli_results = {}
    for L in args.lengths:
        print(f"\nL={L:,}")
        samples = []
        for r in range(args.repeats):
            res = run_e2e(model_tli, L, args.max_gen)
            samples.append(res)
            print(f"  [{r+1}/{args.repeats}] prefill={res['prefill_ms']:.1f}ms "
                  f"tpot={res['tpot_ms']:.3f}ms total={res['total_e2e_ms']:.1f}ms "
                  f"thr={res['throughput_tok_s']:.1f}tok/s")
        tpot_sorted = sorted(s['tpot_ms'] for s in samples)
        prefill_sorted = sorted(s['prefill_ms'] for s in samples)
        total_sorted = sorted(s['total_e2e_ms'] for s in samples)
        thr_sorted = sorted(s['throughput_tok_s'] for s in samples)
        mem_sorted = sorted(s['peak_mem_gib'] for s in samples)
        mid = len(tpot_sorted) // 2
        tli_results[str(L)] = {
            'prefill_ms': prefill_sorted[mid],
            'tpot_ms': tpot_sorted[mid],
            'total_e2e_ms': total_sorted[mid],
            'throughput_tok_s': thr_sorted[mid],
            'peak_mem_gib': mem_sorted[mid],
        }
        print(f"  → median: prefill={prefill_sorted[mid]:.1f}ms, "
              f"tpot={tpot_sorted[mid]:.3f}ms, "
              f"total={total_sorted[mid]:.1f}ms, "
              f"thr={thr_sorted[mid]:.1f}tok/s, "
              f"mem={mem_sorted[mid]:.2f}GiB")

    # ── 汇总 ──
    print(f"\n{'='*70}")
    print(f"汇总：完整 e2e 对比")
    print(f"{'='*70}")
    print(f"{'长度':<10} {'Dense TPOT':<15} {'TLI TPOT':<15} {'TPOT加速比':<12} "
          f"{'Dense Total':<15} {'TLI Total':<15} {'Total加速比':<12}")
    print(f"{'-'*82}")
    for L in args.lengths:
        dk = dense_results[str(L)]
        tk = tli_results[str(L)]
        tpot_speedup = dk['tpot_ms'] / tk['tpot_ms']
        total_speedup = dk['total_e2e_ms'] / tk['total_e2e_ms']
        print(f"{L:<10} {dk['tpot_ms']:<15.3f} {tk['tpot_ms']:<15.3f} {tpot_speedup:<12.2f} "
              f"{dk['total_e2e_ms']:<15.1f} {tk['total_e2e_ms']:<15.1f} {total_speedup:<12.2f}")

    # 保存
    all_results = {
        'config': vars(args),
        'dense': dense_results,
        'tli': tli_results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
