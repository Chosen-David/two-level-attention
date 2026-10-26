"""Measure one-time static compaction and actual storage/allocator residency.

Synthetic frozen gate decisions, not a model/dgate quality experiment. GPU
measurements require --device cuda; CPU emits storage checks only. No empty_cache
is used: reserved memory is reported separately from live allocated memory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import weakref
from types import SimpleNamespace
import torch
from transformers.cache_utils import DynamicCache
from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.ops.tli_static_cache import TLIStaticCache, fill_static_from_dynamic
from sparse_attn.ops.far_skip_static import FarSkipStaticLayer


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--length',type=int,default=32768)
    p.add_argument('--layers',type=int,default=6)
    p.add_argument('--skip-layers',type=int,nargs='*',default=[1])
    p.add_argument('--steps',type=int,default=256)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    if a.device=='cuda' and not torch.cuda.is_available():p.error('real CUDA required')
    if min(a.length,a.layers,a.steps)<1 or any(li<0 or li>=a.layers for li in a.skip_layers):p.error('invalid dimensions/layers')
    cuda=a.device=='cuda';dtype=torch.bfloat16 if cuda else torch.float32
    sync=torch.cuda.synchronize if cuda else lambda:None
    def memory():
        sync()
        return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated()) if cuda else None
    modules=[]
    for li in range(a.layers):
        args=SimpleNamespace(tia_block_size=64,tia_level1_topk=64,tia_level2_topk=1024,
            tia_level2_cmp_ratio=4,tia_enable_async_topk=False,tli_enable_layer_skip=False,
            tli_subspace='tail',tli_sink_tokens=4,tli_swa_size=128,tli_near_pages=32,
            tli_near_tokens='off',tli_near_len=4096,tli_far_skip_backend='compact',
            tli_far_skip_cache='evict',tli_dyn_dgate=True)
        idx=TLIIndexer(args);idx.layer_idx=li
        idx._dyn_skip_pending=li in a.skip_layers
        modules.append(SimpleNamespace(indexer=idx,layer_idx=li,sliding_window=None))
    model=SimpleNamespace(training=False,modules=lambda:iter(modules))
    baseline=memory();dyn=DynamicCache()
    for li in range(a.layers):
        dyn.update(torch.randn(1,8,a.length,128,device=a.device,dtype=dtype),
                   torch.randn(1,8,a.length,128,device=a.device,dtype=dtype),li)
    old_refs=[weakref.ref(t) for layer in dyn.layers for t in (layer.keys,layer.values)]
    before_bytes=sum(t.untyped_storage().nbytes() for layer in dyn.layers for t in (layer.keys,layer.values))
    before=memory()
    static=TLIStaticCache(SimpleNamespace(num_hidden_layers=a.layers),a.length+a.steps+1,1)
    if cuda:torch.cuda.reset_peak_memory_stats()
    sync();begin=time.perf_counter()
    fill_static_from_dynamic(static,dyn,[a.length],model=model)
    sync();setup_ms=(time.perf_counter()-begin)*1000
    before_release=memory();del dyn
    after=memory()
    assert all(ref() is None for ref in old_refs),'prefill tensor reference retained'
    layer_receipts=[]
    for li,layer in enumerate(static.layers):
        layer_receipts.append(dict(layer=li,skip=modules[li].indexer.skip_far,
            capacity=layer.keys.shape[-2],kv_storage_bytes=sum(t.untyped_storage().nbytes() for t in (layer.keys,layer.values))))
    new_k=torch.randn(1,8,1,128,device=a.device,dtype=dtype);new_v=torch.randn_like(new_k)
    pos=torch.tensor([a.length],device=a.device);times=[]
    pointers=[(l.keys.data_ptr(),l.values.data_ptr()) for l in static.layers]
    for step in range(a.steps):
        pos.fill_(a.length+step)
        sync();begin=time.perf_counter()
        for layer in static.layers:
            if isinstance(layer,FarSkipStaticLayer):layer.update(new_k,new_v,{'cache_position':pos})
        sync();times.append((time.perf_counter()-begin)*1000)
    assert pointers==[(l.keys.data_ptr(),l.values.data_ptr()) for l in static.layers]
    result=dict(args=vars(a),cuda_executed=cuda,torch=torch.__version__,
        gpu=torch.cuda.get_device_name() if cuda else None,
        gate_source='synthetic per-prompt pending decisions; freeze_dyn_dgate executed; no quality claim',
        memory_before_prefill=baseline,memory_after_prefill=before,
        memory_after_setup_before_releasing_source=before_release,memory_after_release=after,
        source_references_released=True,kv_storage_before_bytes=before_bytes,
        kv_storage_after_bytes=sum(r['kv_storage_bytes'] for r in layer_receipts),layers=layer_receipts,
        setup_ms=setup_ms,ring_append_wall_ms=times if cuda else None,
        timing_scope='setup separately; append-only loop across skipped layers; no attention/indexer',
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            [__file__,'sparse_attn/ops/far_skip_cache.py','sparse_attn/ops/far_skip_static.py',
             'sparse_attn/ops/tli_static_cache.py','sparse_attn/indexer/tli_indexer.py']})
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('cuda_executed','source_references_released','kv_storage_before_bytes','kv_storage_after_bytes')}))


if __name__=='__main__':main()
