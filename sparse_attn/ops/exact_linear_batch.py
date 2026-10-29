"""Opt-in linear batch consumer with exact per-row candidate producers.

L1/index maintenance still loops over request owners. Fine scoring, candidate
normalization/ranking, protected/selected attention and merge each consume the
whole batch once. Only warmup copies full feature storage into the arena.

Scope: immutable greedy no-skip BF16 batch, pinned sparse normalization/ranking
geometry, serial graph-state route. Refill requires indexer.clear() and warmup;
reorder is unsupported (as with TLIStaticLayer). Clear poisons captured ready
checks until warmup. Keep the owner alive as long as any captured graph uses it.
"""
import copy
import os
from contextlib import contextmanager
import torch
from .cuda_graph_state import graph_state
from .batch_candidate_score import score_candidates_batched
from .sparse_normalization import candidate_probabilities_batched, supported_geometry
from .unique_ranking import rank_unique_batched
from .partial_attention import partial_attention, merge_partials


def _config_signature(idx):
    return tuple(getattr(idx, key, None) for key in (
        'head_dim', 'alpha', 'beta', 'gamma', 'nt_near_pages', 'nt_near_len',
        'sink_tokens', 'sliding_window_size', 'enable_subspace', 'subspace',
        'cmp_ratio', 'l2_subspace', 'l2_quant', 'far_method', 'near_method',
        'far_select', 'near_select', 'boundary_bypass', 'per_q_head',
        'enable_async', 'moba_gate', 'sigma_select', 'static_pair', 'dyn_dgate',
        'skip_far', 'far_skip_backend')) + tuple(getattr(idx.args, key) for key in (
            'tia_block_size', 'tia_level1_topk', 'tia_level2_topk'))


def _validate_metadata(idx, q, k, v, gs):
    if (q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or q.shape[0] < 2
            or q.shape[1] != 1 or q.shape[0] != k.shape[0]
            or q.shape[-1] != 128 or k.shape[-1] != 128 or k.shape[2] < 1
            or q.shape[2] % k.shape[2] or k.shape[1] >= 65536
            or any(t.dtype != torch.bfloat16 or t.device != q.device
                   or t.stride(-1) != 1 or t.requires_grad for t in (q,k,v))):
        raise ValueError('exact batch requires BF16 Q[B,1,Hq,128], K/V[B,S,Hkv,128]')
    b = q.shape[0]
    if (not gs.enabled or gs.bs != b
            or any(t is None or t.shape != (b,) or t.device != q.device
                   or not t.is_contiguous()
                   for t in (gs.pos_buf, gs.valid_len_buf, gs.alive_buf))
            or gs.pos_buf.dtype != torch.int64 or gs.valid_len_buf.dtype != torch.int64
            or gs.alive_buf.dtype != torch.bool):
        raise ValueError('exact batch requires enabled fixed-address graph inputs[B]')
    if gs is not graph_state():
        raise ValueError('exact batch must bind the shared graph state')
    if (idx.skip_far or idx.dyn_dgate or idx.static_pair or idx._basis is not None
            or idx._basis_all is not None or idx.moba_gate or idx.per_q_head
            or idx.gamma is not None or idx.enable_async or idx.sigma_select != 'none'
            or idx.far_skip_backend != 'legacy' or not idx.boundary_bypass
            or idx.args.tia_level2_topk != 1024 or idx.alpha <= 0 or idx.beta <= 0
            or idx.nt_near_pages is None or idx.nt_near_len is None
            or idx.far_select != '4bit' or idx.near_select != '4bit'
            or (idx.enable_kmeans and (idx.far_select != '4bit' or idx.near_select != '4bit'))):
        raise ValueError('exact batch requires fixed no-skip/no-projection production geometry')
    bs = idx.args.tia_block_size
    if type(bs) is not int or bs < 1 or bs & (bs-1):
        raise ValueError('exact batch requires power-of-two page geometry')
    pages = (k.shape[1]+bs-1)//bs
    nfar = min(max(0,idx.args.tia_level1_topk-int(idx.nt_near_pages)),pages)
    nnear = min(max(0,int(idx.nt_near_pages)),pages)
    if (nfar+nnear+2)*bs < 1024:
        raise ValueError('exact batch requires candidate width >= ranking width')
    if (os.environ.get('TLI_OPERATOR_MODE', 'off') != 'serial'
            or os.environ.get('TLI_OPERATOR_BACKEND', 'simt') != 'simt'
            or os.environ.get('TLI_OPERATOR_SELECTED_STAGES', '1') != '1'
            or os.environ.get('TLI_L2_BYPASS', '0') not in ('', '0')
            or os.environ.get('TLI_GRAPH_INC', '1') != '1'
            or os.environ.get('TLI_SPARSE_SELECTION', '0') != '1'):
        raise ValueError('exact batch requires serial SIMT stage1, incremental L2 and TLI_SPARSE_SELECTION=1')


def _validate_supported(q, k):
    if not q.is_cuda or not supported_geometry(q.dtype, q.device, k.shape[1],
                                              q.shape[2]//k.shape[2], q.shape[0]*k.shape[2]):
        raise ValueError('exact batch requires supported pinned CUDA sparse normalization')


def _storage_signature(q, k, v, gs):
    return (tuple((t.device, t.dtype, tuple(t.shape), tuple(t.stride()), t.data_ptr())
                  for t in (k, v, gs.pos_buf, gs.valid_len_buf, gs.alive_buf)),
            tuple(q.shape), q.dtype, q.device)


@contextmanager
def _row_scope(gs, b):
    saved = gs.enabled, gs.bs, gs.pos_buf, gs.valid_len_buf, gs.alive_buf
    gs.enabled, gs.bs = True, 1
    gs.pos_buf, gs.valid_len_buf, gs.alive_buf = (t[b:b+1] for t in saved[2:])
    try:
        yield
    finally:
        gs.enabled, gs.bs, gs.pos_buf, gs.valid_len_buf, gs.alive_buf = saved


def _clone_owner(source):
    owner = copy.copy(source)
    owner.args = copy.deepcopy(source.args)
    # Reset before clear so a shallow copy cannot invalidate the source owner.
    owner._exact_linear_batch = None
    owner._retired_exact_linear_batches = []
    owner._ginc = None
    owner.clear()
    owner._graph_const_cache = None
    owner._graph_const_entries = {}
    owner._skip_ids = None if source._skip_ids is None else frozenset(source._skip_ids)
    owner.skip_far = source.skip_far
    return owner


class ExactLinearBatch:
    def __init__(self, source, q, k, v, gs):
        self.signature = _storage_signature(q,k,v,gs)
        # Keep raw-address inputs alive even if the caller drops a cache object.
        self.storage = k,v,gs.pos_buf,gs.valid_len_buf,gs.alive_buf
        self.config = _config_signature(source)
        self.rows = [_clone_owner(source) for _ in range(q.shape[0])]
        self.feature_arena = None
        self.cu = torch.tensor([0,k.shape[1]],device=k.device,dtype=torch.int32)
        self.ready = torch.zeros((),device=k.device,dtype=torch.bool)
        self.previous_lengths = torch.empty_like(gs.valid_len_buf)
        self.previous_alive = torch.empty_like(gs.alive_buf)
        self.needs_warmup = True

    def invalidate(self):
        # Preserve arena/summary references retained by older graphs. A replay
        # without warmup fails its device ready check instead of using stale KV.
        self.ready.fill_(False)
        self.needs_warmup = True
        for row in self.rows:
            row.clear()

    def warmup(self, k, gs):
        for b, row in enumerate(self.rows):
            with _row_scope(gs,b):
                index = row.prepare_index(k[b:b+1], self.cu)
            if row._ginc is None or row._ginc.get('stale',False):
                raise RuntimeError('exact batch needs persistent incremental index storage')
            fine = index['k_qat']
            if self.feature_arena is None:
                self.feature_arena = torch.empty((len(self.rows),*fine.shape[1:]),
                                                device=fine.device,dtype=fine.dtype)
            if fine.shape[-1] not in (16,32,64,128,256):
                raise ValueError('unsupported exact batch fine feature dimension')
            target = self.feature_arena[b:b+1]
            if fine.shape != target.shape or fine.dtype != target.dtype:
                raise RuntimeError('exact batch feature geometry changed; recapture required')
            if fine.data_ptr() != target.data_ptr():
                target.copy_(fine)  # Full feature copy occurs only outside capture.
            row._ginc['k_qat'] = target
        self.previous_lengths.copy_(gs.valid_len_buf)
        self.previous_alive.copy_(gs.alive_buf)
        self.ready.fill_(True)
        self.needs_warmup = False

    def produce(self, q, k, gs, scale):
        torch._assert_async(self.ready, 'exact batch was cleared; warm up before replay')
        lengths, alive = gs.valid_len_buf, gs.alive_buf
        torch._assert_async(((lengths >= 1) & (lengths <= k.shape[1])
                             & (gs.pos_buf == lengths-1)).all(), 'invalid exact batch positions')
        torch._assert_async((((lengths == self.previous_lengths)
                             | (lengths == self.previous_lengths+alive.to(torch.int64)))
                            & (~alive | self.previous_alive)).all(),
                            'exact batch refill/reorder requires clear and warmup')
        produced = []
        for b, row in enumerate(self.rows):
            with _row_scope(gs,b):
                index = row.prepare_index(k[b:b+1], self.cu)
                if index['k_qat'].data_ptr() != self.feature_arena[b:b+1].data_ptr():
                    raise RuntimeError('exact batch feature arena binding lost')
                produced.append(row.compute_candidate_inputs_graph(
                    q[b:b+1], index, scale, gs.valid_len_buf))
        self.previous_lengths.copy_(lengths)
        self.previous_alive.copy_(alive)
        return produced


def _consume(candidates, arena, q, k, v, lengths, scale):
    if not candidates or len(candidates) != q.shape[0] or arena.shape[0] != len(candidates):
        raise ValueError("exact batch needs one producer output per arena row")
    first = candidates[0]
    geometry = (first.capacity, first.group_size, first.output_dtype,
                first.rank_width, first.sink, first.swa, first.ids.shape)
    for b, c in enumerate(candidates):
        if ((c.capacity,c.group_size,c.output_dtype,c.rank_width,c.sink,c.swa,c.ids.shape) != geometry
                or c.rank_width != 1024 or c.ids.shape[-1] < 1024
                or c.k_fine.data_ptr() != arena[b:b+1].data_ptr()):
            raise ValueError('incompatible exact batch producer outputs')
    # These are bounded candidate/query tensors, never full-history K copies.
    ids = torch.stack([c.ids for c in candidates])
    keep = torch.stack([c.keep for c in candidates])
    counts = torch.stack([c.counts for c in candidates])
    query = torch.stack([c.q_fine for c in candidates])
    scores = score_candidates_batched(arena,query,ids,counts,out_dtype=first.output_dtype)
    p, unique = candidate_probabilities_batched(scores,ids,keep,first.capacity,
                                               first.group_size,return_unique=True)
    selection = rank_unique_batched(ids,p,unique,
        torch.stack([c.lower for c in candidates]), torch.stack([c.upper for c in candidates]),
        torch.stack([c.quota for c in candidates]),capacity=first.capacity,width=1024,sink=first.sink)
    kw = dict(sink=first.sink,swa=first.swa,scale=scale,splits=4,backend='simt')
    packed = selection.indices,selection.counts
    protected = partial_attention(q,k,v,packed,lengths,protected=True,**kw)
    selected = partial_attention(q,k,v,packed,lengths,protected=False,**kw)
    return merge_partials(protected,selected,q.dtype,backend='simt')


def exact_linear_batch_decode(indexer, q, k, v, gs, scale):
    """Explicit exact route: unsupported cases raise, never use legacy parallel.

    Refill/clear requires a new warmup before capture/replay. Storage replacement
    additionally requires clear and graph recapture. TLIStaticLayer rejects
    reorder; external raw tensor permutation is outside this fixed-owner API.
    """
    indexer._resolve_skip()
    _validate_metadata(indexer,q,k,v,gs)
    _validate_supported(q,k)
    capturing = torch.cuda.is_current_stream_capturing()
    state = getattr(indexer,'_exact_linear_batch',None)
    signature = _storage_signature(q,k,v,gs)
    config = _config_signature(indexer)
    if state is not None and (state.signature != signature or state.config != config):
        if not state.needs_warmup or capturing:
            raise ValueError('exact batch storage/config changed; clear and recapture')
        # Poisoned old captures retain their own false ready flag and buffers.
        retired = getattr(indexer,'_retired_exact_linear_batches',[])
        retired.append(state)
        indexer._retired_exact_linear_batches = retired
        state = None
    if state is None:
        if capturing:
            raise RuntimeError('warm up exact linear batch before graph capture')
        state = ExactLinearBatch(indexer,q,k,v,gs)
        indexer._exact_linear_batch = state
    if state.needs_warmup:
        if capturing:
            raise RuntimeError('cleared exact batch requires warmup before capture')
        state.warmup(k,gs)
    candidates = state.produce(q,k,gs,scale)
    return _consume(candidates,state.feature_arena,q,k,v,gs.valid_len_buf,scale)
