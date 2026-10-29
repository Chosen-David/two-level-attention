"""First-step eager/exact batch attention trace; observations invalidate timing.

Copies only Q, selected-token metadata/masks and merged attention output. Never
modifies an input/output, and restores all observer hooks on exceptions. Trace
consumers must verify tokens against the corresponding uninstrumented requests.
"""
from contextlib import contextmanager
from pathlib import Path
import torch


@contextmanager
def record_batch_decode_trace(directory, max_steps=1):
    from sparse_attn.patches import qwen3_attn_patch as patch
    from sparse_attn.ops import exact_linear_batch as exact
    from sparse_attn.ops.cuda_graph_state import graph_state
    if max_steps<1:raise ValueError('positive trace steps required')
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    outer,attention,partial,replay=(patch._tl_decode_batched,patch.eager_decoding_attn,
                                    exact.partial_attention,torch.cuda.CUDAGraph.replay)
    current={'layer':None,'selection':None}
    captured=[];steps={};graph_steps={}

    def copy(t):return t.detach().clone()
    def save(name,records):
        rows=[{k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in r.items()}
              for r in records]
        with (root/name).open('xb') as stream:torch.save(rows,stream)

    def observed_eager(q,k,v,mask,*args,**kw):
        out=attention(q,k,v,mask,*args,**kw)
        layer=current['layer']
        if layer is not None and steps.get(layer,0)<max_steps:
            current['selection']={'mask':copy(mask),'capacity':k.shape[1]}
        return out

    def observed_partial(q,k,v,selection,lengths,*args,**kw):
        out=partial(q,k,v,selection,lengths,*args,**kw)
        if current['layer'] is not None and torch.cuda.is_current_stream_capturing() and not kw.get('protected'):
            ids,counts=selection
            current['selection']=dict(ids=copy(ids),counts=copy(counts),lengths=copy(lengths),
                capacity=k.shape[1],sink=kw['sink'],swa=kw['swa'])
        return out

    def observed_outer(module,indexer,q,k,v,ctx):
        layer=int(module.layer_idx);previous=dict(current)
        current.update(layer=layer,selection=None)
        try:
            out=outer(module,indexer,q,k,v,ctx)
            capturing=graph_state().enabled and torch.cuda.is_current_stream_capturing()
            eager=not graph_state().enabled
            step=steps.get(layer,0)
            if capturing or (eager and step<max_steps):
                if current['selection'] is None:raise RuntimeError('trace missed attention selection')
                row=dict(layer=layer,q=copy(q.transpose(1,2)),output=copy(out),**current['selection'])
                if capturing:captured.append(row)
                else:save(f'eager_layer{layer:03d}_step{step:03d}.pt',[row])
            if eager:steps[layer]=step+1
            return out
        finally:current.clear();current.update(previous)

    def observed_replay(graph):
        result=replay(graph)
        if graph not in graph_steps:
            graph_steps[graph]=[len(graph_steps),0,list(captured)];captured.clear()
        request,step,records=graph_steps[graph]
        if step<max_steps:save(f'graph_request{request:03d}_step{step:03d}.pt',records)
        graph_steps[graph][1]+=1
        return result

    patch._tl_decode_batched=observed_outer
    patch.eager_decoding_attn=observed_eager
    exact.partial_attention=observed_partial
    torch.cuda.CUDAGraph.replay=observed_replay
    try:yield
    finally:
        patch._tl_decode_batched=outer
        patch.eager_decoding_attn=attention
        exact.partial_attention=partial
        torch.cuda.CUDAGraph.replay=replay
