"""Physical-arena TLI selection without full-length projected/fine-score tensors.

L1 reuses ``TLIIndexer._select_l1_blocks``. Fine scores cover only each KV
head's L1/bypass/protected candidates. Probabilities are restored to absolute
logical positions before the original free-competition top-k so zero-probability
fillers and torch's tie behavior remain unchanged.
"""
import torch
from einops import einsum as reference_einsum

from .eager_decoding import _eager_expand_kernel, _err_flag


def _geometry(idx, length):
    bs = idx.args.tia_block_size
    kt = (length+bs-1)//bs
    near_base = length-idx.sliding_window_size
    near_len = max(bs, int(idx.nt_near_len))
    near_blks = max(idx._sink_pool_lo_blk, (near_base-near_len)//bs)
    swa_lo_blk = max(near_blks, kt-max(1, idx.sliding_window_size//bs))
    return bs, kt, near_blks, swa_lo_blk


def _candidate_ids(candidate, width, cu):
    """有序展开 [Hkv,S] bool；GPU 使用既有 kernel，CPU 仅作测试回退。"""
    heads, length = candidate.shape
    if candidate.is_cuda:
        ids = torch.full((heads,width), -1, device=candidate.device, dtype=torch.int32)
        counts = torch.empty(heads, device=candidate.device, dtype=torch.int32)
        _eager_expand_kernel[(heads,)](
            candidate, ids, counts, _err_flag(candidate.device), cu,
            length, candidate.stride(0), W=width, TP=1024, num_warps=8)
        valid = torch.arange(width,device=candidate.device)[None] < counts[:,None]
        return ids.long(), valid, counts
    logical = torch.arange(length,device=candidate.device)
    packed = torch.where(candidate,logical[None],length).sort(-1).values[...,:width]
    valid = packed < length
    return packed, valid, valid.sum(-1,dtype=torch.int32)


def _protected_mask(positions, heads, length, sink, swa):
    selected = ((positions < sink) |
                (positions >= max(0,length-swa)))[None].expand(heads,-1)
    return selected[None,None]


def _physical_selection(mask, positions, layer, width):
    """可见域 mask → 排序物理槽位；仅物化最终 K2 个 ID，不 gather K/V。"""
    visible = positions.numel()
    cu = getattr(layer,'_attention_cu',None)
    if cu is None or cu.device != positions.device:
        cu = torch.tensor([0,visible],device=positions.device,dtype=torch.int32)
        layer._attention_cu = cu
    else:
        cu[1].fill_(visible)
    offsets,valid,counts = _candidate_ids(mask[0,0].contiguous(),width,cu)
    safe = offsets.clamp(0,max(0,visible-1))
    logical = positions[safe]
    physical = layer.physical_positions(logical)
    return physical.to(torch.int32).contiguous(),counts


def region_mask_physical(indexer, q, layer, scale, intermediates=None):
    """返回 [1,1,Hkv,visible] mask，K/V 始终从物理 arena 按需读取。"""
    idx = indexer
    length = layer.logical_length
    positions = layer.logical_positions(q.device)
    heads = q.shape[2]
    kv_heads = layer.arena_k.shape[1]
    if heads % kv_heads:
        raise ValueError('frozen physical selection requires integral GQA groups')
    sink, swa = idx.sink_tokens, idx.sliding_window_size
    if layer.skip_near and layer.skip_far:
        mask = _protected_mask(positions,kv_heads,length,sink,swa)
        receipt = dict(logical_tokens=length,visible_tokens=positions.numel(),
            selected_per_kv_head=mask.sum(-1,dtype=torch.int32)[0,0],
            scorer='protected_only',selector='protected_only',
            l1_calls=0,l2_topk_calls=0)
        if intermediates is not None:
            physical_ids,counts = _physical_selection(
                mask,positions,layer,min(length,idx.args.tia_level2_topk))
            intermediates.update(l1=None,candidate_ids=None,probabilities=None,
                l2_indices=None,physical_ids=physical_ids,physical_counts=counts)
        return mask,receipt

    if (idx.subspace != 'tail' or idx.l2_subspace != 'same' or idx.gamma is not None
            or idx._basis_all is not None or idx.far_select != '4bit'
            or idx.near_select != '4bit' or idx.per_q_head or idx.enable_async
            or idx.l2_quant or idx.far_method != 'minmax' or idx.near_method != 'avg'):
        raise ValueError('physical frozen selection requires the validated production geometry')
    bs,kt,near_blks,swa_lo_blk = _geometry(idx,length)
    far_lo = idx._sink_pool_lo_blk
    dims = idx._subspace_indices(q.device)
    page_parts = []
    if not layer.skip_far and far_lo < near_blks:
        page_parts.append(torch.arange(far_lo,near_blks,device=q.device))
    if not layer.skip_near and near_blks < swa_lo_blk:
        page_parts.append(torch.arange(near_blks,swa_lo_blk,device=q.device))
    pages = (torch.cat(page_parts) if page_parts else
             torch.empty(0,device=q.device,dtype=torch.int64))
    score_coarse = q.new_zeros((1,heads,kt))
    score_avg = q.new_zeros((1,heads,kt))
    group = heads//kv_heads
    idx.group_size = group
    if pages.numel():
        logical = pages[:,None]*bs + torch.arange(bs,device=q.device)[None]
        valid = (logical < length) & layer.visible_mask(logical.clamp(max=max(0,length-1)))
        safe = logical.clamp(0,length-1)
        physical = layer.physical_positions(safe)
        keys = layer.arena_k[physical].index_select(-1,dims)
        keys = torch.where(valid[...,None,None],keys,0)
        k_min,k_max,k_avg = keys.amin(1),keys.amax(1),keys.mean(1)
        bq = (q[0,0]*scale).to(torch.float32).index_select(-1,dims)
        # 保持 reference 的 BF16 块统计 dtype 与连续 [page,KV-head,D] 布局；
        # 即使数值相同，FP32 输入也会让 Triton 产生不同的 BF16 舍入边界。
        mn,mx,av = (x.contiguous() for x in (k_min,k_max,k_avg))
        coarse = avg = None
        if not layer.skip_far and q.is_cuda and getattr(idx,'_triton_ok',False):
            try:
                # keep/near 的 reference 粗筛与 avg 均走全宽 Triton。
                coarse = idx._fast_score_coarse(mn,mx,bq,q.dtype)
                avg = idx._fast_score_fine(av,bq,q.dtype)
            except Exception:
                coarse = avg = None
        if coarse is None:
            # far 及 Triton 回退的 reference 都使用展开 GQA 的 einops；
            # 紧凑 torch.einsum 的归约布局不同，可能改变 L1 并列顺序。
            mn32 = mn.repeat_interleave(group,dim=1).to(torch.float32)
            mx32 = mx.repeat_interleave(group,dim=1).to(torch.float32)
            av32 = av.repeat_interleave(group,dim=1).to(torch.float32)
            coarse = (reference_einsum(
                bq.clamp(max=0),mn32,'h d, kt h d -> h kt') +
                      reference_einsum(
                bq.clamp(min=0),mx32,'h d, kt h d -> h kt'))
            avg = reference_einsum(bq,av32,'h d, kt h d -> h kt')
        score_coarse[0].index_copy_(-1,pages,coarse.to(q.dtype))
        score_avg[0].index_copy_(-1,pages,avg.to(q.dtype))
    if layer.skip_far:
        score_coarse[...,far_lo:near_blks] = float('-inf')
        score_avg[...,far_lo:near_blks] = float('-inf')
    l1 = idx._select_l1_blocks(score_coarse,score_avg,length)
    topk_pages = l1['topk_mask_block'][0,0]

    logical = torch.arange(length,device=q.device)
    protected = (logical < sink) | (logical >= max(0,length-swa))
    candidate = ((topk_pages[:,logical//bs] | protected[None])
                 & layer.visible_mask(logical)[None])
    if layer.skip_far and far_lo < near_blks:
        # dgate_short 的参考实现把整个 far 前缀（含 sink）留为 -inf；
        # sink 仍由最终 mask 强制保留，但不进入 L2 softmax 分母。
        candidate[..., :near_blks*bs] = False
    candidate = candidate.contiguous()
    bypass = len(l1['bypass_blks'])
    width = min(length,(l1['i_f'].shape[-1]+l1['i_n'].shape[-1]+bypass)*bs+sink+swa)
    cu = getattr(layer,'_selection_cu',None)
    if cu is None or cu.device != q.device:
        cu = torch.tensor([0,length],device=q.device,dtype=torch.int32)
        layer._selection_cu = cu
    else:
        cu[1].fill_(length)
    cand_ids,cand_valid,_ = _candidate_ids(candidate,width,cu)
    safe = cand_ids.clamp(0,length-1)
    physical = layer.physical_positions(safe)
    head_ids = torch.arange(kv_heads,device=q.device)[:,None]
    bq = (q[0,0]*scale).to(torch.float32).index_select(-1,dims)
    fine = None
    if layer.skip_far:
        # far reference 使用展开 GQA 的 einops；紧凑归约或 Triton 的累加
        # 布局不同，在 32K top-k 边界会改变选中集合。
        keys = layer.arena_k[physical,head_ids].index_select(-1,dims).to(torch.float32)
        keys = keys.repeat_interleave(group,dim=0).permute(1,0,2).contiguous()
        fine = reference_einsum(
            bq,keys,'h d, kt h d -> h kt').to(q.dtype)
        fine = fine.reshape(kv_heads,group,-1)
    else:
        if q.is_cuda and getattr(idx,'_triton_ok',False):
            try:
                # keep/near 与 reference 全宽 scorer 共用 Triton 归约；
                # 异常时沿用原 torch 路径。
                kq = layer.arena_k.index_select(-1,dims)
                cand_len = cand_valid.sum(1).to(torch.int32)
                fine = idx._fast_score_fine_cand(
                    kq,bq,physical,cand_len,cand_ids.shape[1],q.dtype)
                fine = fine.reshape(kv_heads,group,-1)
            except Exception:
                fine = None
        if fine is None:
            keys = layer.arena_k[physical,head_ids].index_select(-1,dims).to(torch.float32)
            bq = bq.reshape(kv_heads,group,-1)
            fine = torch.einsum('hgd,hrd->hgr',bq,keys).to(q.dtype)
    fine = fine.masked_fill(~cand_valid[:,None],float('-inf'))
    probability = fine.softmax(-1).nan_to_num(nan=0).mean(1)
    probability = probability.masked_fill(
        cand_valid & (cand_ids >= max(0,length-swa)),1.0)
    padded = probability.new_zeros((kv_heads,length+1))
    padded.scatter_(1,torch.where(cand_valid,cand_ids,length),probability)
    p = padded[:,:length]

    k2 = min(length,idx.args.tia_level2_topk)
    far_tok_lo = far_lo*bs
    far_tok_hi = min(near_blks*bs,length)
    swa_lo = max(0,length-swa)
    k2_mid = max(0,k2-sink-(length-swa_lo))
    mid = p[:,far_tok_lo:swa_lo]
    l2 = torch.topk(mid,min(k2_mid,mid.shape[-1]),dim=-1).indices + far_tok_lo
    selected = torch.zeros((kv_heads,length),device=q.device,dtype=torch.bool)
    selected.scatter_(-1,l2,True)
    selected[:,:min(sink,length)] = True
    selected[:,swa_lo:length] = True
    mask = selected.index_select(-1,positions)[None,None]
    receipt = dict(logical_tokens=length,visible_tokens=positions.numel(),
        selected_per_kv_head=mask.sum(-1,dtype=torch.int32)[0,0],
        scorer='physical_candidate_score',selector='physical_l1_original_free_topk',
        l1_calls=1,l2_topk_calls=1)
    if intermediates is not None:
        physical_ids,counts = _physical_selection(mask,positions,layer,k2)
        intermediates.update(l1=l1,candidate_ids=cand_ids,candidate_valid=cand_valid,
            probabilities=p,l2_indices=l2,score_coarse=score_coarse,
            score_coarse_avg=score_avg,far_tok_hi=far_tok_hi,
            physical_ids=physical_ids,physical_counts=counts)
    return mask,receipt
