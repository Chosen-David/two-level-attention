"""Compare true-length eager selection with capacity-width Graph selection.

Diagnostic only: random BF16 Q/K, frozen gate, no model accuracy or timing claim.
Unlike sparse-selector integration tests, the reference uses compute_score/mask
on a genuinely shorter cache, not compute_mask_graph on the same capacity.
"""
import json
import os
from pathlib import Path
import runpy
import sys

import torch

from request_gpu_monitor import GPU0Monitor
from sparse_attn.indexer.tli_indexer import TLIIndexer


def main(output):
    output = Path(output)
    args = runpy.run_path(str(Path(__file__).with_name('test_tli_graph_gates.py')))['_make_prod_args']
    results = []
    with GPU0Monitor(output.with_suffix('.gpu.json')):
        for seed in (887, 888):
            torch.manual_seed(seed)
            capacity = 32676
            k = torch.randn(1, capacity, 8, 128, device='cuda', dtype=torch.bfloat16)
            q = torch.randn(1, 1, 64, 128, device='cuda', dtype=k.dtype)
            for length in (32613, 32627, 32643, capacity):
                for skip_far in (False, True):
                    indexes = [TLIIndexer(args()) for _ in range(3)]
                    for idx in indexes:
                        idx.layer_idx = 2
                        idx.clear()
                        idx.skip_far = skip_far
                        idx._dyn_frozen = True
                    eager, full, sparse = indexes
                    actual_k = k[:, :length].contiguous()
                    cu = torch.tensor([0, length], device='cuda', dtype=torch.int32)
                    pos = torch.tensor([length - 1], device='cuda')
                    index = eager.prepare_index(actual_k, cu)
                    scores = eager.compute_score(q, pos, index, 128**-.5)
                    expected = eager.compute_mask(pos, scores)
                    padded = k.clone()
                    padded[:, length:] = 0
                    cap_cu = torch.tensor([0, capacity], device='cuda', dtype=torch.int32)
                    vl = torch.tensor([length], device='cuda')
                    masks = []
                    for idx, policy in ((full, '0'), (sparse, '1')):
                        os.environ['TLI_SPARSE_SELECTION'] = policy
                        index = idx.prepare_index(padded, cap_cu)
                        # Exercise actual capture/replay with fixed buffer addresses.
                        for _ in range(2):
                            idx.compute_mask_graph(q, index, 128**-.5, vl)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            mask = idx.compute_mask_graph(q, index, 128**-.5, vl)
                        graph.replay()
                        torch.cuda.synchronize()
                        masks.append(mask.clone())
                    full_mask, sparse_mask = masks
                    results.append(dict(seed=seed, length=length, capacity=capacity,
                        skip_far=skip_far,
                        eager_full_diff=int((expected != full_mask[..., :length]).sum()),
                        eager_sparse_diff=int((expected != sparse_mask[..., :length]).sum()),
                        full_sparse_diff=int((full_mask != sparse_mask).sum()),
                        invalid_full=int(full_mask[..., length:].sum()),
                        invalid_sparse=int(sparse_mask[..., length:].sum()),
                        full_route=full._graph_normalization_route,
                        sparse_route=sparse._graph_normalization_route))
                    output.write_text(json.dumps(results, indent=2) + '\n')
                    print(json.dumps(results[-1]), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
