"""Offline production-tile CUDA compilation; not GPU execution or timing."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from baseline_kernel import _score_fine_cand_kernel as baseline
from sparse_attn.indexer import tli_indexer as module


def main():
    rows = []
    ptxas = Path(triton.__file__).parent / 'backends/nvidia/bin/ptxas'
    for arch in (80, 90):
        for dtype in ('fp32', 'fp16', 'bf16'):
            for d in (32, 64, 128):
                bt, nw = module.TLIIndexer._TILE_FINE[d]
                for g in (1, 4):
                    sig = dict(KQ='*' + dtype, Q='*fp32', CAND='*i64', CLEN='*i64', OUT='*' + dtype)
                    sig.update({name: 'i32' for name in ('R', 'stride_kt', 'stride_kh', 'stride_qh',
                                                        'stride_ch', 'stride_cr', 'stride_oh', 'stride_or')})
                    const = dict(Hkv=3, G=g, D=d, BT=bt)
                    for arm, fn in (('baseline', baseline), ('candidate', module._score_fine_cand_kernel)):
                        obj = triton.compile(ASTSource(fn, sig, constexprs=const),
                                             target=GPUTarget('cuda', arch, 32), options=dict(num_warps=nw))
                        ptx = obj.asm['ptx']
                        with tempfile.TemporaryDirectory() as tmp:
                            path = Path(tmp) / 'kernel.ptx'
                            path.write_text(ptx)
                            target = re.search(r'\.target\s+(sm_\w+)', ptx)[1]
                            result = subprocess.run([str(ptxas), '-v', '--gpu-name=' + target,
                                str(path), '-o', str(Path(tmp) / 'kernel.cubin')],
                                capture_output=True, text=True, check=True)
                        rows.append(dict(arm=arm, arch=arch, dtype=dtype, d=d, g=g, bt=bt,
                            num_warps=nw, shared_bytes=obj.metadata.shared, ptxas=result.stderr,
                            ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
                            global_load_instruction_count=sum('ld.global' in line for line in ptx.splitlines()),
                            predicated_global_load_instruction_count=sum('ld.global' in line and '@' in line for line in ptx.splitlines()),
                            registers=int(re.search(r'Used (\d+) registers', result.stderr)[1])))
                        if arch == 90 and dtype == 'bf16' and d == 32 and g == 4:
                            Path(__file__).with_name(arm + '-sm90-d32-g4.ptx').write_text(ptx)
    result = dict(scope='offline compilation only; dynamic strides without runtime alignment hints',
                  triton=triton.__version__, candidate_source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                  baseline_sha256=hashlib.sha256(Path(__file__).with_name('baseline_kernel.py').read_bytes()).hexdigest(),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=rows)
    Path(__file__).with_name('production-compile.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'{len(rows)} specializations compiled; not GPU execution')


if __name__ == '__main__':
    main()
