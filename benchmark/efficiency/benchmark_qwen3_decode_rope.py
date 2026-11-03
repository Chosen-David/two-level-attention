#!/usr/bin/env python3
"""iter89：Qwen3-8B B1 decode Q/K RoPE 独立逐位对拍与微基准。

由用户后续在独占 GPU 上运行；不加载真实模型，不测试 prefill。比较
transformers 4.56.2 原始 apply_rotary_pos_emb 与 iter89 单 launch 实现，
随后用 triton.testing.do_bench 分别测量完整 Python wrapper 的延迟。
"""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from triton.testing import do_bench  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import (  # noqa: E402
    apply_rotary_pos_emb,
)

from sparse_attn.ops.qwen3_rope_triton import qwen3_decode_rope  # noqa: E402


def _make_inputs(seed, scale, device):
    """构造与 q_norm/k_norm(...).transpose(1, 2) 同布局的输入。"""
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        (1, 1, 32, 128), generator=generator,
        dtype=torch.float32, device=device,
    )
    k = torch.randn(
        (1, 1, 8, 128), generator=generator,
        dtype=torch.float32, device=device,
    )
    q = (q * scale).to(torch.bfloat16).transpose(1, 2)
    k = (k * scale).to(torch.bfloat16).transpose(1, 2)

    angles = torch.rand(
        (1, 1, 128), generator=generator,
        dtype=torch.float32, device=device,
    ) * (2.0 * torch.pi)
    cos = angles.cos().to(torch.bfloat16)
    sin = angles.sin().to(torch.bfloat16)
    return q, k, cos, sin


def _assert_bitwise(name, actual, expected):
    """按 BF16 位型比较，避免 allclose 掩盖单 ULP 或舍入点差异。"""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f'{name} 元数据不一致: actual={actual.shape}/{actual.dtype}, '
            f'expected={expected.shape}/{expected.dtype}')
    actual_bits = actual.view(torch.int16)
    expected_bits = expected.view(torch.int16)
    if not torch.equal(actual_bits, expected_bits):
        mismatches = (actual_bits != expected_bits).sum().item()
        raise AssertionError(f'{name} 非逐位一致: {mismatches} 个元素不同')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checks', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=25)
    parser.add_argument('--rep', type=int, default=100)
    args = parser.parse_args()
    if args.checks < 1:
        raise ValueError('--checks 必须 >= 1')

    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('该微基准仅支持 CUDA 设备')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用')

    scales = (1.0, 2.0 ** -8, 2.0 ** 8, 2.0 ** -16)
    with torch.inference_mode():
        for index in range(args.checks):
            q, k, cos, sin = _make_inputs(
                seed=890000 + index,
                scale=scales[index % len(scales)],
                device=device,
            )
            expected_q, expected_k = apply_rotary_pos_emb(q, k, cos, sin)
            owner = SimpleNamespace()
            actual_q, actual_k = qwen3_decode_rope(
                q, k, cos, sin, owner=owner)
            _assert_bitwise('Q', actual_q, expected_q)
            _assert_bitwise('K', actual_k, expected_k)
        print(f'[parity] {args.checks} 组 Q/K BF16 位型一致', flush=True)

        # 独立固定输入做计时；owner 与生产 attention 模块一样持久持有输出。
        q, k, cos, sin = _make_inputs(891000, 1.0, device)
        owner = SimpleNamespace()

        def run_hf():
            return apply_rotary_pos_emb(q, k, cos, sin)

        def run_triton():
            return qwen3_decode_rope(q, k, cos, sin, owner=owner)

        hf_ms = do_bench(
            run_hf, warmup=args.warmup, rep=args.rep, return_mode='median')
        triton_ms = do_bench(
            run_triton, warmup=args.warmup, rep=args.rep, return_mode='median')

    print(f'[shape] Q={tuple(q.shape)} K={tuple(k.shape)} dtype={q.dtype}',
          flush=True)
    print(f'[do_bench median] HF={hf_ms:.6f} ms  Triton={triton_ms:.6f} ms',
          flush=True)
    print(f'[ratio] HF / Triton = {hf_ms / triton_ms:.4f}x', flush=True)


if __name__ == '__main__':
    main()
