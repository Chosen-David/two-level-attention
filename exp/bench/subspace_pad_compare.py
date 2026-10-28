"""Cold/full index preparation paired CUDA benchmark, not production TPOT."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from sparse_attn.arguments import add_sparse_attn_args
from sparse_attn.indexer.tli_indexer import TLIIndexer
from exp.bench.subspace_pad_reference import _prepare_index_base as reference


def make_indexer(mode='tail', device='cpu'):
    parser = argparse.ArgumentParser()
    add_sparse_attn_args(parser)
    args = parser.parse_args([
        '--method', 'tli', '--tia_block_size', '64',
        '--tli_near_pages', '32', '--tli_near_tokens', 'off',
        '--tli_near_len', '4096', '--tli_sink_tokens', '4',
        '--tli_swa_size', '128', '--tia_level1_topk', '64',
        '--tia_level2_topk', '1024', '--tli_subspace',
        'mla_rope' if mode == 'mla_rope' else ('full' if mode == 'full' else 'tail'),
        '--tia_level2_cmp_ratio', '4', '--tli_enable_layer_skip', 'false'])
    ix = TLIIndexer(args)
    ix.layer_idx = 0
    ix.enable_kmeans = False
    if mode == 'moba': ix.moba_gate = True
    if mode == 'quant': ix.l2_quant = True
    if mode == 'l2full': ix.l2_subspace = 'full'
    if mode in ('projection', 'lazy_projection'):
        basis = torch.arange(2*128*8, dtype=torch.float32).reshape(2,128,8) / 2048
        if mode == 'projection':
            ix._basis = basis[:, ix._subspace_indices(torch.device(device))].to(device)
        else:
            ix._basis_all = basis[None]
    if mode == 'static_pair':
        ix.static_pair = True
        ix._pair_idx = torch.arange(0, 128, 4, device=device)
    return ix


def equal_dict(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], torch.Tensor):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        else:
            assert a[key] == b[key], key


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lengths', type=int, nargs='+', default=[32769, 131073])
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--dtype', choices=['float16', 'bfloat16'], default='bfloat16')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required; CPU timing is not a GPU performance result')
    import sparse_attn.indexer.tli_indexer as source
    assert source._HAS_TRITON, 'benchmark requires the existing fused statistic kernel'
    torch.manual_seed(271027)
    records = []
    for mode in ('tail', 'mla_rope', 'full', 'moba'):
        for length in a.lengths:
            dim = 192 if mode == 'mla_rope' else 128
            k = torch.randn(1, 8, length, dim, device='cuda', dtype=getattr(torch,a.dtype)).transpose(1,2)
            cu = torch.tensor([0,length], dtype=torch.int32, device='cuda')
            arms = [make_indexer(mode, 'cuda') for _ in range(2)]
            funcs = [reference, TLIIndexer._prepare_index_base]
            for _ in range(3):
                for ix, fn in zip(arms, funcs): fn(ix,k,cu)
            assert all(ix._stat_ok and ix._triton_ok for ix in arms), 'fused path fell back'
            outputs = [fn(ix,k,cu) for ix,fn in zip(arms,funcs)]
            equal_dict(*outputs)
            del outputs
            for repeat in range(a.repeats):
                for arm in ([0,1] if repeat%2 == 0 else [1,0]):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    allocated = torch.cuda.memory_allocated()
                    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    wall = time.perf_counter()
                    start.record()
                    out = funcs[arm](arms[arm], k, cu)
                    end.record(); end.synchronize()
                    records.append(dict(mode=mode,length=length,arm=arm,repeat=repeat,
                        cuda_ms=start.elapsed_time(end),wall_ms=(time.perf_counter()-wall)*1000,
                        peak_allocated_delta=torch.cuda.max_memory_allocated()-allocated))
                    assert arms[arm]._stat_ok and arms[arm]._triton_ok, 'timed arm fell back'
                    del out
    a.out.write_text(json.dumps(dict(device=torch.cuda.get_device_name(), dtype=a.dtype,
        torch=torch.__version__,source_sha256=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
        scope='full preparation only; includes CPU padding metadata synchronization; no TPOT claim',
        records=records),indent=2)+'\n')


if __name__ == '__main__': main()
