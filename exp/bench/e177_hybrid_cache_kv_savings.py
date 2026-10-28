"""E177: HybridCache KV 容量估算（不分配缓存，不测 CUDA allocator）。

对比三种配置的每层 KV cache 存储量（GiB）：
1. dense（全宽线性，无 far-skip）
2. TLI linear（全宽线性 + far-skip 但不蒸发）
3. TLI hybrid（ring buffer 紧凑层 + linear 非紧凑层）

输出：每层 KV bytes → 全模型总 GiB + 节省比例。
"""
import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Qwen3Config, Qwen3ForCausalLM


def giB(n_bytes):
    return n_bytes / (1024 ** 3)


def measure_kv(config_label, model, lengths, sink=4, near_len=4096,
               window=128, block_size=64, skip_ratio=None):
    """计算给定配置下全模型 KV 总存储。

    skip_ratio: far-skip 层占比（0.0=全 linear, 0.7=70% 层用 ring）
    """
    num_layers = model.config.num_hidden_layers
    num_kv = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    bs = len(lengths)
    dtype_bytes = 2  # bf16

    per_token_per_layer = num_kv * head_dim * dtype_bytes * 2  # K+V

    if skip_ratio is None or skip_ratio == 0.0:
        # 全宽线性：每层 L_max * per_token
        max_len = max(lengths)
        per_layer = bs * max_len * per_token_per_layer
        total = num_layers * per_layer
        return total, per_layer, per_layer
    else:
        # hybrid: skip 层用 ring buffer，其余用全宽
        num_skip = int(num_layers * skip_ratio)
        num_linear = num_layers - num_skip
        max_len = max(lengths)

        # linear 层
        linear_per_layer = bs * max_len * per_token_per_layer

        # ring 层: capacity = sink + near + window + block - 1
        ring_cap = sink + near_len + window + block_size - 1
        ring_per_layer = bs * ring_cap * per_token_per_layer

        total = num_linear * linear_per_layer + num_skip * ring_per_layer
        return total, linear_per_layer, ring_per_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='Qwen/Qwen3-8B')
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--lengths', nargs='+', type=int, default=[32768, 65536, 131072])
    parser.add_argument('--skip_ratios', nargs='+', type=float, default=[0.0, 0.3, 0.5, 0.7, 0.9])
    args = parser.parse_args()

    print(f"=== E177 HybridCache KV 容量估算（非显存实测） ===")
    print(f"模型: {args.model_name}, bs={args.bs}")

    # 本地构建 config（避免联网）
    model_configs = {
        'Qwen3-8B': dict(num_hidden_layers=36, num_key_value_heads=8, head_dim=128),
        'Qwen3-32B': dict(num_hidden_layers=64, num_key_value_heads=8, head_dim=128),
        'Llama-3.1-8B': dict(num_hidden_layers=32, num_key_value_heads=8, head_dim=128),
    }
    for short, full in [('8B', 'Qwen3-8B'), ('32B', 'Qwen3-32B'), ('Llama', 'Llama-3.1-8B')]:
        if short in args.model_name or args.model_name == full:
            mc = model_configs[full]
            break
    else:
        mc = model_configs['Qwen3-8B']

    cfg = type('Cfg', (), {
        'num_hidden_layers': mc['num_hidden_layers'],
        'num_key_value_heads': mc['num_key_value_heads'],
        'head_dim': mc['head_dim'],
    })
    num_layers = cfg.num_hidden_layers
    num_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    print(f"模型: {args.model_name} → 层数={num_layers}, Hkv={num_kv}, D={head_dim}")

    per_token_per_layer = num_kv * head_dim * 2 * 2  # bf16 K+V
    print(f"每层每 token: {per_token_per_layer} bytes = {per_token_per_layer/1024:.1f} KiB")
    print()

    for L in args.lengths:
        print(f"\n{'='*60}")
        print(f"序列长度 L={L:,}")
        print(f"{'='*60}")
        lengths = [L] * args.bs

        # Dense 基线
        dense_total, _, _ = measure_kv("dense", type('M', (), {'config': cfg})(),
                                       lengths, skip_ratio=0.0)
        print(f"\n{'配置':<25} {'单层 (MiB)':<15} {'全模型 (GiB)':<15} {'vs dense':<10}")
        print(f"{'-'*65}")
        print(f"{'dense (full linear)':<25} {dense_total/num_layers/1024**2:<15.2f} "
              f"{giB(dense_total):<15.3f} {'1.00×':<10}")

        for ratio in args.skip_ratios:
            if ratio == 0.0:
                continue
            total, linear_pl, ring_pl = measure_kv(
                f"hybrid-{ratio:.0%}", type('M', (), {'config': cfg})(),
                lengths, skip_ratio=ratio)
            num_skip = int(num_layers * ratio)
            num_linear = num_layers - num_skip
            speedup = dense_total / total if total > 0 else float('inf')
            print(f"{'hybrid skip=' + f'{ratio:.0%}':<25} "
                  f"L:{linear_pl/1024**2:.1f}/R:{ring_pl/1024**2:.1f}  "
                  f"{giB(total):<15.3f} {speedup:.2f}x "
                  f"({num_linear}lin+{num_skip}ring)")

    # 论文视角汇总表
    print(f"\n\n{'='*60}")
    print(f"容量估算表：KV Cache 理论字节数 vs 序列长度")
    print(f"{'='*60}")
    print(f"\n{'长度':<10} {'Dense (GiB)':<15} {'TLI-linear':<15} {'TLI-hybrid(70%)':<18} {'节省':<10}")
    print(f"{'-'*68}")

    for L in [8192, 16384, 32768, 65536, 131072, 262144]:
        lengths = [L] * args.bs
        dense, _, _ = measure_kv("d", type('M', (), {'config': cfg})(), lengths, skip_ratio=0.0)
        # TLI linear: 与 dense 相同（全宽）
        tli_lin = dense
        # TLI hybrid 70% skip
        tli_hyb, _, _ = measure_kv("h", type('M', (), {'config': cfg})(), lengths, skip_ratio=0.7)
        saving = (1 - tli_hyb / dense) * 100
        print(f"{L:<10} {giB(dense):<15.3f} {giB(tli_lin):<15.3f} "
              f"{giB(tli_hyb):<18.3f} {saving:.1f}%")


if __name__ == '__main__':
    main()
