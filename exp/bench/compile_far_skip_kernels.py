"""Compile representative CUDA kernels without claiming GPU execution."""
import argparse
import hashlib
import json
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from sparse_attn.ops import far_skip_kernels as kernels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--arch', type=int, nargs='+', default=[80, 90])
    opts = parser.parse_args()
    rows = []
    for arch in opts.arch:
        cases = []
        for position_type in ('i32', 'i64', '*i64'):
            for ring, masked in ((True, False), (True, True), (False, True)):
                cases.append((kernels.write_pair,
                    {'K': '*bf16', 'V': '*bf16', 'NewK': '*bf16', 'NewV': '*bf16',
                     'Position': position_type, 'Alive': '*i1' if masked else '*bf16'},
                    dict(H=8, D=128, K_H=4291*128, K_S=128, K_D=1,
                         V_H=4291*128, V_S=128, V_D=1, NK_H=128, NK_D=1,
                         NV_H=128, NV_D=1, SINK=4 if ring else 0,
                         TAIL=4287 if ring else 0, HAS_ALIVE=masked,
                         TENSOR_POSITION=position_type.startswith('*'), BLOCK=256)))
        for dtype in ('fp32', 'bf16'):
            cases.append((kernels.order_keys,
                          {'Scores': '*'+dtype, 'Valid': '*i1', 'Keys': '*i64'},
                          dict(N=2334, S0=2334, S1=1, V0=2334, V1=1, HAS_VALID=True, BLOCK=256)))
        for projection, pair in ((False, True), (True, False)):
            for ring in (False, True):
                cases.append((kernels.gather_rows,
                    {'X': '*bf16', 'Y': '*bf16', 'Positions': '*i64', 'Valid': '*i1',
                     'Dims': '*i64', 'OutX': '*bf16', 'OutY': '*bf16'},
                    dict(N=2334, D=32 if projection else 128,
                         X_S=128, X_H=4291*128, X_D=1, Y_S=128, Y_H=4291*128, Y_D=1,
                         P_H=2334, P_N=1, V_H=2334, V_N=1, DIM_STRIDE=1,
                         SINK=4 if ring else 0, TAIL=4287 if ring else 0,
                         HAS_VALID=True, HAS_DIMS=projection, PAIR=pair,
                         BLOCK_N=4, BLOCK_D=32 if projection else 128)))
        for function, signature, constants in cases:
            # Length/stride inputs change during eager decode. Their runtime
            # signatures must not become constexpr/JIT-per-token variants.
            for name in function.do_not_specialize:
                if name in constants:
                    constants.pop(name)
                    signature[name] = 'i32'
            compiled = triton.compile(ASTSource(function, signature, constexprs=constants),
                                      target=GPUTarget('cuda', arch, 32),
                                      options={'num_warps': 4})
            rows.append(dict(kernel=function.__name__, arch=arch, signature=signature,
                             constants=constants, shared_bytes=compiled.metadata.shared,
                             ptx_sha256=hashlib.sha256(compiled.asm['ptx'].encode()).hexdigest(),
                             cubin_bytes=len(compiled.asm['cubin'])))
    source = Path(kernels.__file__)
    receipt = dict(status='compiled only; no GPU execution or timing', triton=triton.__version__,
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cases=rows)
    Path(opts.out).write_text(json.dumps(receipt, indent=2)+'\n')
    print(f'{len(rows)} CUDA specializations compiled; execution remains unverified')


if __name__ == '__main__':
    main()
