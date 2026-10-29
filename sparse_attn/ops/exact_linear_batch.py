"""Opt-in linear batch consumer with exact per-row candidate producers.

Production tail32 maintenance and L1/candidate production, fine scoring, candidate
normalization/ranking, protected/selected attention and merge each consume the
whole batch once. Other maintenance geometries keep an explicit row fallback.
Only warmup copies full feature/summary storage into arenas.

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
from .batch_candidate_producer import BatchedCandidateProducer, BatchedCandidateInputs
from .batch_index_maintenance import maintain_batched, maintenance_route


def _config_signature(idx):
    return tuple(getattr(idx, key, None) for key in (
        'head_dim', 'alpha', 'beta', 'gamma', 'nt_near_pages', 'nt_near_len',
        'sink_tokens', 'sliding_window_size', 'enable_subspace', 'subspace',
        'cmp_ratio', 'l2_subspace', 'l2_quant', 'far_method', 'near_method',
        'far_select', 'near_select', 'boundary_bypass', 'per_q_head',
        'enable_async', 'moba_gate', 'sigma_select', 'static_pair', 'dyn_dgate',
        'skip_far', 'far_skip_backend', '_triton_ok')) + tuple(getattr(idx.args, key) for key in (
            'tia_block_size', 'tia_level1_topk', 'tia_level2_topk'))


def _validate_metadata(idx, q, k, v, gs):
    from .sparse_policy import live_normalization_enabled
    if not live_normalization_enabled():
        raise ValueError('exact batch requires live-length normalization')
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
        self.maintenance_route = "uninitialized"
        self.maintenance_calls = {}
        self.rows = [_clone_owner(source) for _ in range(q.shape[0])]
        self.feature_arena = None
        self.summary_arenas = {}
        self.producer = BatchedCandidateProducer(source,k)
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
            # Summary storage is also bound once, not stacked each step.
            for name in ('k_min','k_max','k_avg'):
                summary=index.get(name)
                if name not in self.summary_arenas:
                    self.summary_arenas[name]=(None if summary is None else torch.empty(
                        (len(self.rows),*summary.shape[1:]),device=summary.device,dtype=summary.dtype))
                arena=self.summary_arenas[name]
                if summary is None:
                    if arena is not None:raise RuntimeError('inconsistent exact batch summary presence')
                    continue
                if arena is None or summary.shape!=arena[b:b+1].shape:
                    raise RuntimeError('exact batch summary geometry changed; recapture required')
                view=arena[b:b+1]
                if summary.data_ptr()!=view.data_ptr():view.copy_(summary)
                row._ginc[name]=view
        if any(not getattr(row,'_triton_ok',False) for row in self.rows):
            self.maintenance_route='row_fallback:owner_kernel'
        else:
            self.maintenance_route=maintenance_route(self.rows[0],k,gs.pos_buf,gs.alive_buf,
                self.producer.coarse_dims,self.summary_arenas,self.feature_arena)
        if self.maintenance_route=='batched_tail32':
            indices=self.producer.coarse_dims
            torch._assert_async(((indices>=0)&(indices<128)).all(), 'maintenance feature index out of bounds')
            # The old producer maintained the current block even for initially
            # dead rows. Canonicalize once with that same row path before the
            # batched steady path skips dead rows; preserve reduction semantics.
            for b,row in enumerate(self.rows):
                with _row_scope(gs,b):row.prepare_index(k[b:b+1],self.cu)
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
        route=self.maintenance_route
        self.maintenance_calls[route]=self.maintenance_calls.get(route,0)+1
        if route=='batched_tail32':
            maintain_batched(k,gs.pos_buf,alive,self.producer.coarse_dims,
                self.summary_arenas['k_min'],self.summary_arenas['k_max'],
                self.summary_arenas.get('k_avg'),self.feature_arena,check_values=False)
        else:
            for b, row in enumerate(self.rows):
                with _row_scope(gs,b):
                    index = row.prepare_index(k[b:b+1], self.cu)
                    if index['k_qat'].data_ptr() != self.feature_arena[b:b+1].data_ptr():
                        raise RuntimeError('exact batch feature arena binding lost')
                    for name,arena in self.summary_arenas.items():
                        if arena is not None and index[name].data_ptr()!=arena[b:b+1].data_ptr():
                            raise RuntimeError('exact batch summary arena binding lost')
        # Only explicitly unsupported maintenance geometries keep a row loop.
        produced=self.producer(q,self.feature_arena,self.summary_arenas,lengths,scale)
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
    inputs=BatchedCandidateInputs(arena,query,ids,keep,counts,lengths,
        torch.stack([c.lower for c in candidates]),torch.stack([c.upper for c in candidates]),
        torch.stack([c.quota for c in candidates]),first.capacity,first.group_size,
        first.output_dtype,first.rank_width,first.sink,first.swa)
    return _consume_batched(inputs,q,k,v,lengths,scale)


def _consume_batched(inputs,q,k,v,lengths,scale):
    if (inputs.k_fine.shape[0]!=q.shape[0] or inputs.ids.shape[0]!=q.shape[0]
            or inputs.rank_width!=1024 or inputs.ids.shape[-1]<1024):
        raise ValueError('incompatible batched exact producer output')
    scores=score_candidates_batched(inputs.k_fine,inputs.q_fine,inputs.ids,inputs.counts,
                                    out_dtype=inputs.output_dtype)
    p,unique=candidate_probabilities_batched(scores,inputs.ids,inputs.keep,inputs.capacity,
                                            inputs.group_size,return_unique=True,valid_lengths=lengths)
    selection=rank_unique_batched(inputs.ids,p,unique,inputs.lower,inputs.upper,inputs.quota,
        capacity=inputs.capacity,width=1024,sink=inputs.sink)
    kw=dict(sink=inputs.sink,swa=inputs.swa,scale=scale,splits=4,backend='simt')
    packed=selection.indices,selection.counts
    protected=partial_attention(q,k,v,packed,lengths,protected=True,**kw)
    selected=partial_attention(q,k,v,packed,lengths,protected=False,**kw)
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
    if capturing:
        from .cuda_graph_state import current_capture_graph, retain_capture_state
        if current_capture_graph() is None:
            raise RuntimeError('exact batch capture requires capture_graph() owner scope')
    state = getattr(indexer,'_exact_linear_batch',None)
    if q.is_cuda and not capturing and (state is None or state.needs_warmup):
        # Request admission, before arena allocation/mutation or capture. Once
        # admitted, the device monotonic-length contract preserves this bound.
        if not bool(((gs.valid_len_buf > 9216) & (gs.valid_len_buf <= k.shape[1])).all().item()):
            raise ValueError('exact batch requires every initial live length in [9217, capacity]')
    signature = _storage_signature(q,k,v,gs)
    config = _config_signature(indexer)
    if state is not None and (state.signature != signature or state.config != config):
        if not state.needs_warmup or capturing:
            raise ValueError('exact batch storage/config changed; clear and recapture')
        # Each live Graph pins its own poisoned state and raw addresses.
        # Dropping that Graph releases obsolete KV instead of retiring forever.
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
    if capturing:
        retain_capture_state(state)
    candidates = state.produce(q,k,gs,scale)
    indexer._exact_batch_maintenance_route=state.maintenance_route
    indexer._exact_batch_maintenance_calls=state.maintenance_calls
    return _consume_batched(candidates,q,k,v,gs.valid_len_buf,scale)
