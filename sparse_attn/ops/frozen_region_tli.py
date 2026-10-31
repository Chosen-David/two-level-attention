"""Eager TLI reference adapter for physically segmented region caches.

Reuse TLI's actual compute_score/compute_mask, including the configured tail
or GLM NoPE dimensions and L1/L2 budgets. Only compact projected key scratch uses logical addressing;
full-width K/V remain in surviving nodes. This is a quality reference, not a
latency or peak-memory optimization. No scoring reads near_staging.
"""
import argparse
import os
import torch
from sparse_attn.arguments import add_sparse_attn_args
from .eager_decoding import eager_decoding_attn, eager_decoding_attn_physical


def tli_args(subspace='tail'):
    parser = argparse.ArgumentParser()
    add_sparse_attn_args(parser)
    return parser.parse_args(['--method','tli', '--tli_far_method','minmax',
        '--tli_near_method','avg', '--tli_near_pages','32',
        '--tli_near_tokens','off', '--tli_near_len','4096',
        '--tli_subspace',subspace, '--tia_level2_cmp_ratio','4',
        '--tli_sink_tokens','4', '--tli_swa_size','128',
        '--tia_level1_topk','128', '--tia_level2_topk','1024',
        '--tli_enable_layer_skip','false', '--tli_far_skip_backend','legacy',
        '--tli_far_skip_cache','retain'])


def region_mask(indexer, q, k, positions, length, scale):
    """Q/K [1,S,H,D], positions enumerate visible physical K in order."""
    if os.environ.get('TLI_FINE_GATHER', '1') != '0':
        raise ValueError('frozen reference requires TLI_FINE_GATHER=0')
    idx = indexer
    supported_subspace = (idx.subspace == 'tail' or
                          (idx.subspace == 'nope' and idx.head_dim == 256))
    if (not supported_subspace or idx.l2_subspace != 'same' or idx.gamma is not None
            or idx._basis_all is not None or idx.far_select != '4bit'
            or idx.near_select != '4bit' or idx.per_q_head or idx.enable_async):
        raise ValueError(
            'frozen reference supports shared-head tail TLI or GLM D=256 NoPE64 only')
    dims = idx._subspace_indices(k.device)
    ids = (positions.to(device=k.device, dtype=torch.long)
           if torch.is_tensor(positions) else
           torch.tensor(positions, device=k.device, dtype=torch.long))
    bs = idx.args.tia_block_size
    padded = (length+bs-1)//bs*bs
    projected = k.new_zeros((1,padded,k.shape[2],dims.numel()))
    projected[:,ids] = k.index_select(-1,dims)
    pages = projected.reshape(1,-1,bs,k.shape[2],dims.numel())
    fine = projected[:,:length]
    if idx.l2_quant:
        fine = idx.min_max_per_token_quant(fine)
    cu = torch.tensor([0,length],device=k.device,dtype=torch.int32)
    dictionary = dict(k_min=pages.amin(2), k_max=pages.amax(2),
        k_avg=pages.mean(2), k_qat=fine,
        cu_seqlens_k_coarse=torch.tensor([0,padded//bs],device=k.device,dtype=torch.int32),
        cu_seqlens_k_fine=cu, valid_length=length)
    qids = torch.tensor([length-1], device=k.device, dtype=torch.int32)
    scores = idx.compute_score(q,qids,dictionary,scale)
    if scores.get('cand_idx') is not None:
        raise ValueError('unexpected candidate-only scorer in reference')
    visible = torch.zeros(length,device=k.device,dtype=torch.bool)
    visible[ids] = True
    scores['score_fine'].masked_fill_(~visible, float('-inf'))
    mask = idx.compute_mask(qids,scores)
    # L1 bypass pages and zero-probability quota fillers must never restore
    # deleted/staged tokens. Project mask onto physically visible positions.
    compact_mask = mask.index_select(-1,ids)
    return compact_mask, dict(subspace=idx.subspace,dimensions=dims.tolist(),
        logical_tokens=length,visible_tokens=len(positions),
        selected_per_kv_head=compact_mask.sum(-1).flatten().tolist(),
        scorer='TLIIndexer.compute_score',selector='TLIIndexer.compute_mask')


def _capture_decode_trace(cache, layer, receipt, mask, positions, width,
                          physical_ids=None):
    """仅在 correctness 模式保留当步实际支持集；默认路径不做额外张量运算。"""
    if not getattr(cache, 'decode_trace_enabled', False):
        return
    from .frozen_region_selector import _candidate_ids
    ids = (positions.to(device=mask.device, dtype=torch.long)
           if torch.is_tensor(positions) else
           torch.tensor(positions, device=mask.device, dtype=torch.long))
    cu = torch.tensor([0,ids.numel()],device=mask.device,dtype=torch.int32)
    offsets,valid,_ = _candidate_ids(mask[0,0].contiguous(),width,cu)
    safe = offsets.clamp(0,max(0,ids.numel()-1))
    logical = torch.where(valid,ids[safe],torch.full_like(ids[safe],-1))
    entry = dict(receipt)
    entry.update(decode_step=layer.logical_length-layer.prompt_length-1,
                 selected_logical_ids=logical)
    if physical_ids is not None:
        expected = layer.physical_positions(logical.clamp(min=0))
        mapping_valid = ((expected == physical_ids.long()) | ~valid).all()
        entry.update(selected_physical_ids=physical_ids,
                     physical_mapping_matches=mapping_valid)
    entries = getattr(layer, 'tli_decode_trace', None)
    if entries is None:
        entries = layer.tli_decode_trace = []
    entries.append(entry)


def decode(module,q,k,v,cache,cache_position):
    layer = cache.layers[module.layer_idx]
    if q.shape[:1] != (1,) or q.shape[-2] != 1 or not layer.frozen:
        raise ValueError('frozen TLI requires frozen B1 single-token decode')
    idx = module.indexer
    geometry = layer.geometry
    if (idx.sink_tokens != geometry.sink or idx.sliding_window_size != geometry.window
            or idx.nt_near_len != geometry.near or idx.args.tia_block_size != geometry.page):
        raise ValueError('frozen TLI geometry and physical cache geometry differ')
    if layer.arena_k is None:
        k,v = cache.update(k,v,module.layer_idx,dict(cache_position=cache_position))
        positions = layer.gather()[2]
        q,k,v = (x.transpose(1,2).contiguous() for x in (q,k,v))
    else:
        layer.append_decode(k,v)
        q = q.transpose(1,2)
    idx.skip_far = layer.skip_far
    if layer.arena_k is None:
        mask,receipt = region_mask(idx,q,k,positions,layer.logical_length,module.scaling)
    else:
        from .frozen_region_selector import region_mask_physical
        selection = {}
        mask,receipt = region_mask_physical(idx,q,layer,module.scaling,selection)
        receipt.update(skip_near=layer.skip_near,skip_far=layer.skip_far,
                       physical_arena=True)
        layer.tli_receipt = receipt
        _capture_decode_trace(
            cache,layer,receipt,mask,layer.logical_positions(q.device),
            min(layer.logical_length,idx.args.tia_level2_topk),
            selection['physical_ids'])
        return eager_decoding_attn_physical(
            q,layer.arena_k,layer.arena_v,
            selection['physical_ids'],selection['physical_counts'],
            softmax_scale=module.scaling)
    receipt.update(skip_near=layer.skip_near,skip_far=layer.skip_far,
                   physical_arena=False)
    layer.tli_receipt = receipt
    _capture_decode_trace(
        cache,layer,receipt,mask,positions,
        min(layer.logical_length,idx.args.tia_level2_topk))
    cu = torch.tensor([0,len(positions)],device=k.device,dtype=torch.int32)
    return eager_decoding_attn(q,k,v,mask,1,cu,softmax_scale=module.scaling)
