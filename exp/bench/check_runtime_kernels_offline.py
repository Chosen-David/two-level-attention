"""Compile decode attention specializations without a GPU. Not a speed test.

Run from the repo root with PYTHONPATH=. and Triton 3.4 installed.
"""
import json

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from sparse_attn.ops.eager_decoding import _eager_attn_kernel_b
from sparse_attn.ops.hybrid_attention_triton import _attend


def main():
    records = []
    for sm in (80, 90):
        for dtype in ('fp32', 'fp16', 'bf16'):
            for dim in (16, 128):
                ragged = ASTSource(_attend, signature={
                    'Q': '*'+dtype, 'META': '*i64', 'IDS': '*i64',
                    'VALID': '*i1', 'ALIVE': '*i1', 'OUT': '*'+dtype},
                    constexprs=dict(QB=4*dim, QH=dim, H=2, G=2, D=dim,
                                    W=1156, SCALE=dim**-.5, TILE=64))
                direct = ASTSource(_eager_attn_kernel_b, signature={
                    'IDX': '*i32', 'COUNT': '*i32', 'Q': '*fp32',
                    'K': '*'+dtype, 'V': '*'+dtype, 'OUT': '*'+dtype},
                    constexprs=dict(R=2, stride_qb=4*dim, stride_kb=301*2*dim,
                        stride_kt=2*dim, stride_kh=dim, stride_qh=dim,
                        stride_ob=4*dim, G=2, D=dim, PER_QH=False, BW=256, W=277))
                for name, source in (('ragged', ragged), ('direct', direct)):
                    kernel = triton.compile(source, target=GPUTarget('cuda', sm, 32),
                        options={'num_warps': 4, 'enable_fp_fusion': False})
                    records.append(dict(kernel=name, sm=sm, dtype=dtype, dim=dim,
                                        compiled=True, shared_bytes=kernel.metadata.shared))
    print(json.dumps(dict(triton=triton.__version__, executions=0, records=records), indent=2))


if __name__ == '__main__':
    main()
