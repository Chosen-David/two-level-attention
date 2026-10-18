"""CPU glue reproduction; CUDA scoring is substituted, not GPU acceptance."""
import os
import hashlib
from types import SimpleNamespace
from unittest.mock import patch
import torch
from sparse_attn.ops import batch_select as b

class CudaMetadata:
    shape = (1, 256, 2, 128)
    is_cuda = True
    def dim(self):
        return 4

def indexer():
    return SimpleNamespace(
        args=SimpleNamespace(tia_block_size=64, tia_level1_topk=4,
                             tia_level2_topk=65),
        _triton_ok=True, far_select='4bit', near_select='4bit',
        alpha=.25, beta=.25, gamma=0., per_q_head=False,
        sigma_select='none', moba_gate=False, sinkswa_only=False,
        enable_async=False, _basis=None, _basis_all=None,
        l2_subspace='same', enable_subspace=True, subspace='tail',
        static_pair=False, sink_tokens=0, sliding_window_size=64,
        _sink_pool_lo_blk=0, far_method='minmax', near_method='avg')

if __name__ == '__main__':
    torch.set_num_threads(1)
    idx = indexer()
    q = torch.zeros(1, 256, 2, 128)
    k = torch.zeros(1, 256, 1, 128)
    k[0, :, 0, 0] = -torch.arange(256, dtype=torch.float32) / 128
    print('synthetic q+k sha256:', hashlib.sha256(q.numpy().tobytes() + k.numpy().tobytes()).hexdigest())
    coarse = k[:, ::64, :, :32]
    idx.prepare_index = lambda *a: dict(k_min=coarse, k_max=coarse,
                                      k_avg=coarse, k_qat=k[..., :32])
    idx._subspace_indices = lambda dev: torch.arange(32, device=dev)
    def score(keys, query, dtype):
        return keys[:, 0, 0].to(dtype)[None].expand(query.shape[0], -1)
    idx._fast_score_fine = score
    idx._fast_score_coarse = lambda lo, hi, query, dtype: score(lo, query, dtype)
    print('metadata gate admits sink0:', b._gate_ok(idx, CudaMetadata(), CudaMetadata()))
    # Bypass CUDA-only gate explicitly to inspect the unchanged real CPU glue.
    with patch.object(b, '_gate_ok', return_value=True), patch.dict(os.environ, {
            'TLI_PREFILL_SELECT_GRAPH': '0', 'TLI_PREFILL_L2_TOPK_CAND': '0',
            'TLI_PREFILL_SHARED_GQA_INDICES': '0'}):
        result = b.batched_group_sels(idx, q, k, 1., [256], 128)
    assert result is not None
    expected = [0] + list(range(192, 256))
    selected = result[0][0].nonzero().flatten().tolist()
    print('expected far token 0 plus SWA192:256; actual far:', [x for x in selected if x < 192])
    print('count', len(selected), 'missing', sorted(set(expected)-set(selected)),
          'extra', sorted(set(selected)-set(expected)))
    # Minimal scatter subcase isolates collision from scoring/softmax/topk.
    buf = torch.full((1, 1, 2, 4), -torch.inf)
    b._scatter_candidate_scores(buf, torch.tensor([[8., 7.], [8., 7.]]),
                                torch.tensor([[0, 3]]), torch.tensor([[True, False]]))
    print('valid token0 score expected [8,8], actual', buf[0, 0, :, 0].tolist())
