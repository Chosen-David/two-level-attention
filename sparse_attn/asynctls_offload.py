"""Paper-derived AsyncTLS §4.2 candidate; CUDA execution is opt-in, unvalidated.

Immutable resident generations consume M[t-1]; M[t] is prepared for next step.
Assumptions: current token is an extra candidate (not forced into final top-k),
complete tail block, §4.1 symmetric fake-int4 reference. No packed-index claim.
"""
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import weakref
import types
import torch


class LaggedKV:
    """One layer/request, batch=1. CPU oracle and explicit CUDA transport backend.

    Host storage never reallocates while DMA is pending. Resident blocks are
    immutable; reuse shares references, growing tails create new device tensors.
    close() waits on this object's stream before releasing pinned host buffers.
    """
    def __init__(self, q, k, v, channels, *, blocks=128, tokens=1024,
                 block_size=64, capacity=131200, backend='cpu'):
        self.stream = None
        self.pending_event = None
        self.closed = False
        self.failed = False
        self.resident = None
        self.pending = None
        self.stats = dict(new_blocks=0, reused_blocks=0, tail_updates=0,
                          h2d_bytes=0, d2h_bytes=0, decode_steps=0)
        if backend not in ('cpu', 'cuda'):
            raise ValueError('backend must be explicitly cpu or cuda')
        if any(type(x) is not int or x <= 0 for x in (blocks,tokens,block_size,capacity)):
            raise ValueError('positive integer capacities/budgets required')
        self.backend,self.blocks,self.tokens,self.block_size,self.capacity=backend,blocks,tokens,block_size,capacity
        self.device,self.dtype=k.device,k.dtype
        if k.device.type != backend:
            raise ValueError('backend/device mismatch; no automatic CUDA activation')
        self._validate(q,k,v,initial=True)
        self.hkv,self.length,self.dim=k.shape[1:]
        self.hq=q.shape[1]
        self.groups=self.hq//self.hkv
        if (channels.dtype!=torch.long or channels.ndim!=2 or channels.shape[0]!=self.hkv
                or channels.shape[1]<1 or any(len(set(r))!=len(r) or any(x<0 or x>=self.dim for x in r) for r in channels.tolist())):
            raise ValueError('invalid calibrated channels')
        self.channels=channels.detach().cpu().clone()
        # Normal tensors retain version counters even under outer inference_mode.
        with torch.inference_mode(False):
            self.host_k=torch.empty((self.hkv,capacity,self.dim),dtype=self.dtype,pin_memory=backend=='cuda')
            self.host_v=torch.empty_like(self.host_k,pin_memory=backend=='cuda')
        # Native prefill is a deliberate initialization barrier; no decode timing claim.
        self.host_k[:,:self.length].copy_(k[0].detach().cpu())
        self.host_v[:,:self.length].copy_(v[0].detach().cpu())
        self.stats['d2h_bytes']=2*k.numel()*k.element_size() if backend=='cuda' else 0
        nblocks=(capacity+block_size-1)//block_size
        self.upper=torch.empty((self.hkv,nblocks,self.dim),device=self.device,dtype=torch.float32)
        self.lower=torch.empty_like(self.upper)
        for b,start in enumerate(range(0,self.length,block_size)):
            part=k[0,:,start:start+block_size].float()
            self.upper[:,b]=part.amax(1);self.lower[:,b]=part.amin(1)
        if backend=='cuda':
            if torch.cuda.is_current_stream_capturing():
                raise ValueError('offload candidate does not support CUDA graph capture')
            self.stream=torch.cuda.Stream(device=self.device)
        self._schedule(q, {}, None, self._ready_event())

    def _validate(self,q,k,v,initial=False):
        if (not all(isinstance(x,torch.Tensor) for x in (q,k,v)) or q.ndim!=4 or k.ndim!=4
                or v.shape!=k.shape or q.shape[0]!=1 or k.shape[0]!=1 or q.shape[2]!=1
                or k.shape[1]<1 or q.shape[1]%k.shape[1] or q.shape[-1]!=k.shape[-1]
                or k.shape[2]<1 or (not initial and k.shape[2]!=1)):
            raise ValueError('requires q[1,Hq,1,D], k/v[1,Hkv,N,D], N=1 on decode')
        if any(x.requires_grad for x in (q,k,v)):
            raise ValueError('offload candidate requires inference/no_grad tensors')
        if not q.is_floating_point() or any(x.dtype!=self.dtype or x.device!=self.device for x in (q,k,v)):
            raise ValueError('Q/K/V dtype/device mismatch')
        if initial and k.shape[2]>self.capacity:raise ValueError('prefill exceeds capacity')
        if not initial and (k.shape[1]!=self.hkv or k.shape[-1]!=self.dim or q.shape[1]!=self.hq):
            raise ValueError('head layout changed')
        if not all(bool(torch.isfinite(x).all()) for x in (q,k,v)):
            raise ValueError('nonfinite Q/K/V')

    def _ready_event(self):
        if self.backend=='cpu':return None
        event=torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        return event

    def _select_blocks(self,q):
        count=(self.length+self.block_size-1)//self.block_size
        qq=q[0,:,0].float().reshape(self.hkv,self.groups,self.dim)
        scores=(torch.einsum('hgd,hbd->hgb',qq.clamp_min(0),self.upper[:,:count])+
                torch.einsum('hgd,hbd->hgb',qq.clamp_max(0),self.lower[:,:count])).sum(1)
        # Host list is deliberate reference overhead; no CUDA-graph/performance claim.
        return [sorted(row) for row in scores.topk(min(self.blocks,count),dim=-1).indices.tolist()]

    def _schedule(self,q,previous,append,ready):
        context=torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        with context:
            if self.stream is not None:
                self.stream.wait_event(ready)
                q.record_stream(self.stream)
            if append is not None:
                k,v,pos=append
                if self.stream is not None:
                    k.record_stream(self.stream);v.record_stream(self.stream)
                self.host_k[:,pos:pos+1].copy_(k[0],non_blocking=self.backend=='cuda')
                self.host_v[:,pos:pos+1].copy_(v[0],non_blocking=self.backend=='cuda')
                self.stats['d2h_bytes']+=2*k.numel()*k.element_size() if self.backend=='cuda' else 0
                b=pos//self.block_size
                if pos%self.block_size==0:
                    self.upper[:,b]=k[0,:,0].float();self.lower[:,b]=k[0,:,0].float()
                else:
                    self.upper[:,b]=torch.maximum(self.upper[:,b],k[0,:,0].float())
                    self.lower[:,b]=torch.minimum(self.lower[:,b],k[0,:,0].float())
            chosen=self._select_blocks(q)
            target={}
            for h,blocks in enumerate(chosen):
                for b in blocks:
                    key=(h,b);start=b*self.block_size;end=min(start+self.block_size,self.length)
                    old=previous.get(key)
                    if old is not None and old[0].shape[0]==end-start:
                        target[key]=old;self.stats['reused_blocks']+=1
                    elif old is not None and append is not None and old[0].shape[0]==end-start-1 and end==self.length:
                        # Append comes from the current device token, not a full host recopy.
                        if self.stream is not None:
                            old[0].record_stream(self.stream);old[1].record_stream(self.stream)
                        target[key]=(torch.cat((old[0],append[0][0,h]),0),torch.cat((old[1],append[1][0,h]),0))
                        self.stats['tail_updates']+=1
                    else:
                        target[key]=(self.host_k[h,start:end].to(self.device,copy=True,non_blocking=self.backend=='cuda'),
                                     self.host_v[h,start:end].to(self.device,copy=True,non_blocking=self.backend=='cuda'))
                        self.stats['new_blocks']+=1
                        self.stats['h2d_bytes']+=2*(end-start)*self.dim*self.host_k.element_size() if self.backend=='cuda' else 0
            self.pending=(target,chosen,self.length)
            self.host_versions=(self.host_k._version,self.host_v._version)
            if self.stream is not None:
                self.pending_event=torch.cuda.Event();self.pending_event.record(self.stream)

    def _acquire(self):
        if self.pending is None:raise RuntimeError('missing next resident generation')
        if self.pending_event is not None:
            compute=torch.cuda.current_stream(self.device)
            compute.wait_event(self.pending_event)
            for k,v in self.pending[0].values():
                k.record_stream(compute);v.record_stream(compute)
        self.resident=self.pending;self.pending=None
        return self.resident

    def _attend(self,q,resident,k,v,pos):
        outputs=[];support=[]
        blocks,chosen,_=resident
        for h,ids in enumerate(chosen):
            keys=[blocks[(h,b)][0] for b in ids];values=[blocks[(h,b)][1] for b in ids]
            positions=[torch.arange(b*self.block_size,b*self.block_size+blocks[(h,b)][0].shape[0],device=self.device) for b in ids]
            # Explicit paper-underspecified policy: newest token is a candidate.
            keys.append(k[0,h]);values.append(v[0,h]);positions.append(torch.tensor([pos],device=self.device))
            kk=torch.cat(keys).float();vv=torch.cat(values).float();positions=torch.cat(positions)
            qq=q[0,h*self.groups:(h+1)*self.groups,0].float()
            channels=self.channels[h].to(self.device)
            low=kk[:,channels];scale=low.abs().amax(-1,keepdim=True)/7
            low=(low/scale.clamp_min(torch.finfo(torch.float32).tiny)).round().clamp(-7,7)*scale
            approx=(qq[:,channels]@low.T/math.sqrt(self.dim)).softmax(-1).mean(0)
            selected=approx.topk(min(self.tokens,len(kk))).indices
            output=(qq@kk[selected].T/math.sqrt(self.dim)).softmax(-1)@vv[selected]
            outputs.append(output.to(self.dtype));support.append(positions[selected])
        return torch.cat(outputs)[None,:,None,:],support

    def step(self,q,k,v,position):
        if self.closed or self.failed:raise RuntimeError('closed/failed offload cache')
        try:
            if (self.host_k._version,self.host_v._version)!=self.host_versions:
                raise ValueError('host KV modified outside offload owner')
            self._validate(q,k,v)
            if type(position)is not int or position!=self.length or self.length>=self.capacity:
                raise ValueError('noncontiguous position or exhausted capacity')
            if self.backend=='cuda' and torch.cuda.is_current_stream_capturing():
                raise ValueError('CUDA graph capture unsupported')
            resident=self._acquire()
            ready=self._ready_event()  # Before attention launches; prefetch need not wait on them.
            out,support=self._attend(q,resident,k,v,position)
            self.length+=1
            self._schedule(q,resident[0],(k,v,position),ready)
            self.stats['decode_steps']+=1
            return out,support
        except Exception:
            self.failed=True
            self.close()
            raise

    def host_views(self):
        return self.host_k[:,:self.length].unsqueeze(0),self.host_v[:,:self.length].unsqueeze(0)

    def close(self):
        if not self.closed:
            # Also covers exceptions before a final event was recorded.
            if self.stream is not None:self.stream.synchronize()
            self.pending=None;self.resident=None;self.closed=True

    def __del__(self):
        try:self.close()
        except Exception:pass


def system_identity(args):
    from sparse_attn.asynctls_component import frozen_spec
    _,base=frozen_spec(args)
    backend=getattr(args,'asynctls_backend','cpu');capacity=getattr(args,'asynctls_capacity',131200)
    if backend not in ('cpu','cuda') or type(capacity)is not int or capacity<1:
        raise ValueError('explicit valid backend/capacity required')
    return dict(base,method='asynctls_paper',scope='paper-derived lagged/offload reproduction candidate; not accepted full-system reproduction',
                offload=backend=='cuda',backend=backend,capacity=capacity,selection='previous-step blocks + current-token candidate',
                system_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gpu_validated=False,
                timing='includes host selection barriers, transfers and request completion; no CUDA graph')


def enable_system(model,args):
    from sparse_attn.asynctls_component import enable_component,frozen_spec
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention,apply_rotary_pos_emb
    from transformers.cache_utils import DynamicCache,DynamicLayer
    if hasattr(model, '_baseline_identity'):
        raise ValueError('model already has a baseline adapter')
    identity=system_identity(args)
    if any(p.device.type!=identity['backend'] for p in model.parameters()):
        raise ValueError('model/backend mismatch; CUDA must be explicitly selected')
    count=enable_component(model,args)  # Reuse checkpoint, RoPE, mask and poisoned-cache gates.
    data,_=frozen_spec(args);stats=model._baseline_stats
    engines=weakref.WeakKeyDictionary()
    def clear(cache=None):
        for owner,layers in list(engines.items()):
            if cache is None or owner is cache:
                for engine in layers.values():engine.close()
                del engines[owner]
    def before(module,positional,kw):
        cache=kw.get('past_key_values')
        if cache is not None and (type(cache) is not DynamicCache or getattr(cache,'offloading',False)
                or any(type(layer) is not DynamicLayer for layer in cache.layers)):
            raise ValueError('paper reproduction requires plain non-offloading DynamicCache/DynamicLayer')
        if kw.get('return_dict',True) is not True:raise ValueError('paper reproduction requires return_dict=True')
        if cache is None:clear()
        elif cache in engines:
            for i,engine in engines[cache].items():
                if i>=len(cache.layers):raise ValueError('cache layer removed')
                layer=cache.layers[i];hk,hv=engine.host_views()
                if layer.keys is None or layer.values is None or layer.keys.data_ptr()!=hk.data_ptr() or layer.values.data_ptr()!=hv.data_ptr() or layer.keys.shape!=hk.shape or layer.values.shape!=hv.shape:
                    raise ValueError('cache host storage replaced/cropped')
    def count_stats(engine,previous=None):
        for key,value in engine.stats.items():
            name='offload_'+key
            stats[name]=stats.get(name,0)+value-(previous or {}).get(key,0)
    def after(module,positional,kw,out):
        if out is None:clear(kw.get('past_key_values'))
    model.register_forward_pre_hook(before,with_kwargs=True)
    model.register_forward_hook(after,with_kwargs=True,always_call=True)
    def forward(self,hidden_states,position_embeddings,attention_mask=None,past_key_values=None,cache_position=None,**kwargs):
        if past_key_values is None:raise ValueError('paper reproduction requires DynamicCache')
        layers=engines.setdefault(past_key_values,{})
        old=past_key_values.get_seq_length(self.layer_idx)
        n=hidden_states.shape[1]
        cos,sin=position_embeddings
        if old==0:
            if n>identity['capacity']:raise ValueError('prefill exceeds host capacity')
            result=self._component_native(hidden_states,position_embeddings,attention_mask,past_key_values,cache_position,**kwargs)
            q=self.q_norm(self.q_proj(hidden_states[:,-1:]).view(1,1,-1,self.head_dim)).transpose(1,2)
            # Only the final prefill query initializes M[0]; full K already contains RoPE.
            dummy=torch.zeros(1,self.config.num_key_value_heads,1,self.head_dim,device=q.device,dtype=q.dtype)
            q,_=apply_rotary_pos_emb(q,dummy,cos[:,-1:],sin[:,-1:])
            layer=past_key_values.layers[self.layer_idx]
            engine=LaggedKV(q,layer.keys,layer.values,torch.tensor(data['channels'][str(self.layer_idx)],dtype=torch.long),
                            blocks=identity['blocks'],tokens=identity['tokens'],capacity=identity['capacity'],backend=identity['backend'])
            layers[self.layer_idx]=engine
            count_stats(engine)
            layer.keys,layer.values=engine.host_views()
            stats['native_prefill_calls']=stats.get('native_prefill_calls',0)+1
            return result
        if n!=1 or self.layer_idx not in layers:raise ValueError('missing layer state or chunked continuation')
        engine=layers[self.layer_idx]
        if engine.length!=old:raise ValueError('host cache length modified externally')
        dims=(1,1,-1,self.head_dim)
        q=self.q_norm(self.q_proj(hidden_states).view(dims)).transpose(1,2)
        k=self.k_norm(self.k_proj(hidden_states).view(dims)).transpose(1,2)
        v=self.v_proj(hidden_states).view(dims).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,cos,sin)
        previous=dict(engine.stats)
        out,support=engine.step(q,k,v,old)
        count_stats(engine,previous)
        layer=past_key_values.layers[self.layer_idx];layer.keys,layer.values=engine.host_views()
        stats['paper_decode_calls']=stats.get('paper_decode_calls',0)+1
        stats['selected_tokens_sum']=stats.get('selected_tokens_sum',0)+sum(x.numel() for x in support)
        return self.o_proj(out.transpose(1,2).reshape(1,1,-1)),None
    for module in model.modules():
        if isinstance(module,Qwen3Attention):module.forward=types.MethodType(forward,module)
    model._baseline_identity=identity
    model._asynctls_close=clear
    model._asynctls_engines=engines
    return count


def plan():
    return dict(method='asynctls_paper',status='dry-run only',gpu_started=False,
                quality_lengths=[32768],quality_suite='existing LongBench',
                kernel_lengths=[32768,65536,131072],e2e_lengths=[32768,65536,98304],
                calibration='required: frozen held-out checkpoint/RoPE identity',
                limitations=['paper-derived assumptions','GPU unvalidated','not packed int4 index','batch1 candidate'])

if __name__=='__main__':print(json.dumps(plan(),indent=2))
