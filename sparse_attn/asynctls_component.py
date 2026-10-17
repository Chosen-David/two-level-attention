"""Paper-derived synchronous AsyncTLS §4.1 component, NOT author code/system.

Reproduction assumptions: current-step blocks, complete partial last block,
per-token symmetric int4 [-7,7] round-to-even fake quantization, softmax over
candidate tokens. No offload, prior-step blocks, packed storage, or speed claim.
"""
import hashlib
import json
import math
from pathlib import Path
import types
import weakref
import torch


def select_attention(q, k, v, channels, blocks=128, tokens=1024, block_size=64):
    # One unpadded decode query; group-level selection and head-level softmax.
    if (q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or q.shape[0] != 1
            or k.shape[0] != 1 or q.shape[2] != 1 or q.shape[-1] != k.shape[-1]
            or k.shape[1] == 0 or q.shape[1] % k.shape[1] or k.shape[2] == 0):
        raise ValueError('component requires q[1,Hq,1,D], k/v[1,Hkv,S,D]')
    if any(type(x) is not int or x <= 0 for x in (blocks,tokens,block_size)):
        raise ValueError('positive integer budgets required')
    if any(x.device != q.device or x.dtype != q.dtype for x in (k,v)) or not q.is_floating_point():
        raise ValueError('Q/K/V must share floating dtype and device')
    if not all(bool(torch.isfinite(x).all()) for x in (q,k,v)):
        raise ValueError('nonfinite Q/K/V')
    hkv,n,d=k.shape[1:]; g=q.shape[1]//hkv
    if (channels.ndim != 2 or channels.shape[0] != hkv or channels.shape[1] == 0
            or channels.dtype != torch.long or bool((channels < 0).any())
            or bool((channels >= d).any()) or any(len(set(row)) != len(row) for row in channels.tolist())):
        raise ValueError('unique in-range calibrated channels per KV head required')
    output=[]; supports=[]
    for h in range(hkv):
        qq=q[0,h*g:(h+1)*g,0].float(); kk=k[0,h].float()
        parts=kk.split(block_size)
        upper=torch.stack([x.amax(0) for x in parts]);lower=torch.stack([x.amin(0) for x in parts])
        scores=(qq.clamp_min(0)@upper.T+qq.clamp_max(0)@lower.T).sum(0)
        chosen=scores.topk(min(blocks,len(parts))).indices.tolist()
        ids=torch.cat([torch.arange(b*block_size,min((b+1)*block_size,n),device=q.device) for b in chosen]).sort().values
        c=channels[h].to(q.device)
        low=kk[ids][:,c];scale=low.abs().amax(-1,keepdim=True)/7
        safe=scale.clamp_min(torch.finfo(torch.float32).tiny)
        low=(low/safe).round().clamp(-7,7)*scale
        approx=(qq[:,c]@low.T/math.sqrt(d)).softmax(-1).mean(0)
        selected=ids[approx.topk(min(tokens,len(ids))).indices].sort().values
        probs=(qq@kk[selected].T/math.sqrt(d)).softmax(-1)
        output.append((probs@v[0,h,selected].float()).to(q.dtype));supports.append(selected)
    return torch.cat(output,0)[None,:,None,:],supports


def checkpoint_files(directory):
    """Hash local checkpoint bytes once at install; no downloads or model loading."""
    root=Path(directory)
    files=sorted(root.glob('*.safetensors'))+sorted(root.glob('pytorch_model*.bin'))
    if not root.is_dir() or not files or not (root/'config.json').is_file():
        raise ValueError('component requires a local checkpoint with config and weights')
    hashes={}
    for path in [root/'config.json',*files]:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        hashes[path.name]=h.hexdigest()
    return hashes


def frozen_spec(args):
    if hasattr(args,'_asynctls_spec'):return args._asynctls_spec
    path=getattr(args,'asynctls_calibration',None)
    if not path:raise ValueError('component requires --asynctls_calibration; no synthetic default calibration')
    raw=Path(path).read_bytes();data=json.loads(raw)
    provenance=data.get('provenance',{})
    if (data.get('schema')!='asynctls-component-calibration-v1' or not isinstance(provenance,dict)
            or any(not isinstance(provenance.get(k),str) or not provenance[k].strip() for k in ('producer','held_out_input_sha256','score_rule'))
            or len(provenance.get('held_out_input_sha256',''))!=64
            or any(c not in '0123456789abcdef' for c in provenance.get('held_out_input_sha256',''))
            or provenance.get('score_rule')!='mean_qheads(max_abs_q * max_abs_k); post-RoPE; held-out'):
        raise ValueError('calibration schema and held-out provenance required')
    for key in ('asynctls_blocks','asynctls_tokens'):
        if type(getattr(args,key,None)) is not int or getattr(args,key)<=0:raise ValueError('explicit positive component budgets required')
    identity=dict(method='asynctls_component',paper='2604.07815v1',scope='synchronous section 4.1 component reproduction; NOT full AsyncTLS',
        calibration_sha256=hashlib.sha256(raw).hexdigest(),blocks=args.asynctls_blocks,tokens=args.asynctls_tokens,block_size=64,
        quantization='per-token symmetric fake-int4 [-7,7], round-to-even; assumed',partial_block='include all valid tokens; assumed',
        token_softmax='within selected blocks; assumed',implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gpu_quality_validated=False,offload=False)
    args._asynctls_spec=(data,identity)
    return args._asynctls_spec


def enable_component(model,args):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention,apply_rotary_pos_emb
    from transformers.cache_utils import DynamicCache
    data,identity=frozen_spec(args)
    config=model.config
    shape=dict(model_type=config.model_type,num_hidden_layers=config.num_hidden_layers,num_attention_heads=config.num_attention_heads,num_key_value_heads=config.num_key_value_heads,head_dim=config.head_dim)
    if config.model_type!='qwen3' or data.get('model_shape')!=shape:
        raise ValueError('calibration model shape mismatch; Qwen3 only')
    if getattr(config,'use_sliding_window',False) or getattr(config,'sliding_window',None) is not None:
        raise ValueError('component sliding-window configurations unsupported')
    effective_rope={key:getattr(config,key,None) for key in ('rope_theta','rope_scaling')}
    if data.get('effective_rope') != effective_rope:
        raise ValueError('post-RoPE calibration effective RoPE mismatch')
    if data.get('checkpoint_files') != checkpoint_files(config._name_or_path):
        raise ValueError('calibration checkpoint bytes mismatch')
    modules=[m for m in model.modules() if isinstance(m,Qwen3Attention)]
    if len(modules)!=config.num_hidden_layers:raise ValueError('incomplete Qwen3 attention coverage')
    if set(data.get('channels',{}))!={str(m.layer_idx) for m in modules}:raise ValueError('exact calibration layer coverage required')
    # Validate the entire calibration before any binding.
    calibrated={}
    for m in modules:
        rows=data['channels'][str(m.layer_idx)]
        if (not isinstance(rows,list) or len(rows)!=config.num_key_value_heads or
            any(not isinstance(r,list) or not r or len(r)!=len(rows[0]) or len(set(r))!=len(r)
                or any(type(i)is not int or not 0<=i<m.head_dim for i in r) for r in rows)):
            raise ValueError('invalid calibration channel rows')
        calibrated[m.layer_idx]=torch.tensor(rows,dtype=torch.long)
    stats={}
    poisoned=weakref.WeakSet()
    owned=weakref.WeakSet()
    def before(module,positional,kw):
        ids=kw.get('input_ids');cache=kw.get('past_key_values')
        if positional or not isinstance(ids,torch.Tensor) or ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]<1:
            raise ValueError('component requires keyword input_ids[1,N]')
        if kw.get('use_cache',True) is not True:raise ValueError('component requires cache')
        if cache is not None and (not isinstance(cache,DynamicCache) or cache in poisoned):
            raise ValueError('component unsupported or failed cache')
        old=0 if cache is None else cache.get_seq_length()
        if old and cache not in owned:raise ValueError('component external populated cache rejected')
        if old and ids.shape[1]!=1:raise ValueError('component chunked continuation unsupported')
        expected=torch.arange(old,old+ids.shape[1],device=ids.device)
        for name,shape in (('cache_position',(ids.shape[1],)),('position_ids',(1,ids.shape[1]))):
            pos=kw.get(name)
            if pos is not None and (not isinstance(pos,torch.Tensor) or pos.dtype!=torch.long or pos.device!=ids.device
                    or pos.shape!=shape or not torch.equal(pos.reshape(-1),expected)):
                raise ValueError('component requires contiguous append positions')
        mask=kw.get('attention_mask')
        if mask is not None and (not isinstance(mask,torch.Tensor) or mask.shape!=(1,old+ids.shape[1])
                or mask.device!=ids.device or not bool((mask==1).all())):
            raise ValueError('component requires unpadded full prompts')
    def after(module,positional,kw,out):
        cache=kw.get('past_key_values')
        if out is None:
            if isinstance(cache,DynamicCache):poisoned.add(cache)
        elif isinstance(getattr(out,'past_key_values',None),DynamicCache):
            owned.add(out.past_key_values)
    model.register_forward_pre_hook(before,with_kwargs=True)
    model.register_forward_hook(after,with_kwargs=True,always_call=True)
    def forward(self,hidden_states,position_embeddings,attention_mask=None,past_key_values=None,cache_position=None,**kwargs):
        n=hidden_states.shape[1]
        old=0 if past_key_values is None else past_key_values.get_seq_length(self.layer_idx)
        if n>1 or old==0:
            if old:raise ValueError('component chunked continuation unsupported')
            stats['native_prefill_calls']=stats.get('native_prefill_calls',0)+1
            return self._component_native(hidden_states,position_embeddings,attention_mask,past_key_values,cache_position,**kwargs)
        if hidden_states.shape[0]!=1 or not isinstance(past_key_values,DynamicCache):raise ValueError('component requires batch1 DynamicCache')
        if cache_position is None or cache_position.shape!=(1,) or cache_position.dtype!=torch.long or int(cache_position[0])!=old:
            raise ValueError('component requires contiguous append position')
        if attention_mask is not None:
            valid=(attention_mask.shape==(1,1,1,old+1) and
                   bool((attention_mask if attention_mask.dtype==torch.bool else attention_mask==0).all()))
            if not valid:raise ValueError('component padding/custom masks unsupported')
        dims=(*hidden_states.shape[:-1],-1,self.head_dim)
        q=self.q_norm(self.q_proj(hidden_states).view(dims)).transpose(1,2)
        k=self.k_norm(self.k_proj(hidden_states).view(dims)).transpose(1,2)
        v=self.v_proj(hidden_states).view(dims).transpose(1,2)
        cos,sin=position_embeddings;q,k=apply_rotary_pos_emb(q,k,cos,sin)
        if not all(bool(torch.isfinite(x).all()) for x in (q,k,v)):
            raise ValueError('component nonfinite projected Q/K/V')
        k,v=past_key_values.update(k,v,self.layer_idx,{'sin':sin,'cos':cos,'cache_position':cache_position})
        out,support=select_attention(q,k,v,calibrated[self.layer_idx],identity['blocks'],identity['tokens'])
        stats['component_decode_calls']=stats.get('component_decode_calls',0)+1
        stats['selected_tokens_sum']=stats.get('selected_tokens_sum',0)+sum(x.numel() for x in support)
        return self.o_proj(out.transpose(1,2).reshape(*hidden_states.shape[:-1],-1)),None
    for m in modules:
        m._component_native=m.forward;m.forward=types.MethodType(forward,m)
    model._baseline_identity=identity;model._baseline_stats=stats
    return len(modules)
