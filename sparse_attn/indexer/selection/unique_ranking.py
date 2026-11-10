"""Fused BF16/FP16 unique-candidate ranking with 16/32-bit token packing."""
import torch
import triton
import triton.language as tl
from .selector import TokenSelection


def _workspace_empty(workspace,name,shape,dtype,device):
    if workspace is None:
        return torch.empty(shape,device=device,dtype=dtype)
    key=(name,tuple(shape),dtype,device)
    out=workspace.get(key)
    if out is None:
        out=workspace[key]=torch.empty(shape,device=device,dtype=dtype)
    return out


def _launch_value_signature(values):
    """按 tensor 指针/shape/stride/dtype 与标量原值构造完整重放签名。"""
    sig=[]
    for x in values:
        if isinstance(x,torch.Tensor):
            sig.append((x.data_ptr(),tuple(x.shape),tuple(x.stride()),
                        x.dtype,x.device.index))
        else:
            sig.append(x)
    return tuple(sig)


def _raw_replay_launch(workspace,key,kernel,grid,args,kwargs):
    """无 hook/debug 时直接 run 已缓存 compiled kernel，跳过包装与元数据重算。"""
    if workspace.get('_nfa_eager_record') is not None:
        return False
    context=workspace.get('_nfa_launch_context')
    if context is None:
        from triton import knobs
        stream=None
        enter_hook=knobs.runtime.launch_enter_hook
        exit_hook=knobs.runtime.launch_exit_hook
        runtime_debug=knobs.runtime.debug
    else:
        stream,enter_hook,exit_hook,runtime_debug=context
    if enter_hook is not None or exit_hook is not None or runtime_debug:
        return False
    arg_key=tuple((x.dtype,x.device,x.data_ptr()%16)
                  if isinstance(x,torch.Tensor) else x for x in args)
    cache_key=('triton_launch',key,arg_key,tuple(sorted(kwargs.items())))
    entry=workspace.get(cache_key)
    if entry is None:
        return False
    compiled,names=entry
    grid_1=grid[1] if len(grid)>1 else 1
    grid_2=grid[2] if len(grid)>2 else 1
    tail=tuple(kwargs[name] for name in names[len(args):])
    values=args+tail
    if stream is None:
        stream=torch.cuda.current_stream().cuda_stream
    compiled.run(grid[0],grid_1,grid_2,stream,compiled.function,
                 compiled.packed_metadata,None,None,None,*values)
    return True


def _cached_launch_metadata(workspace,plan_key,compiled,grid,stream,values):
    """缓存 launch_metadata；指针/shape/stride 任一变化即自动失效重算。"""
    sig=_launch_value_signature(values)
    cache=workspace.get('_nfa_launch_metadata_cache')
    if cache is None:
        cache=workspace['_nfa_launch_metadata_cache']={}
    bucket=cache.get(plan_key)
    if bucket is not None and bucket[0]==sig:
        return bucket[1]
    metadata=compiled.launch_metadata(grid,stream,*values)
    if len(cache)>=8:
        cache.pop(next(iter(cache)))
    cache[plan_key]=(sig,metadata)
    return metadata


def _fast_launch(workspace,key,kernel,grid,*args,**kwargs):
    if workspace is None:
        return kernel[grid](*args,**kwargs)
    raw_replay=kwargs.pop('raw_replay',False)
    if raw_replay and _raw_replay_launch(workspace,key,kernel,grid,args,kwargs):
        return
    record=workspace.get('_nfa_eager_record')
    if record is not None:
        record.append((key,kernel,grid,args,kwargs.copy()))
    context=workspace.get('_nfa_launch_context')
    if context is None:
        from triton import knobs
        stream=None
        enter_hook=knobs.runtime.launch_enter_hook
        exit_hook=knobs.runtime.launch_exit_hook
        runtime_debug=knobs.runtime.debug
    else:
        stream,enter_hook,exit_hook,runtime_debug=context
    kwargs['debug']=kwargs.get('debug',kernel.debug) or runtime_debug
    geometry=workspace.get('_nfa_launch_geometry')
    plan_key=('triton_plan',key)
    plan=workspace.get(plan_key) if geometry is not None else None
    if plan is not None and plan[0] is geometry and plan[1]==kwargs['debug']:
        _,_,run,function,packed_metadata,compiled,tail,grid_1,grid_2=plan
        values=args+tail
    else:
        arg_key=tuple((x.dtype,x.device,x.data_ptr()%16)
                      if isinstance(x,torch.Tensor) else x for x in args)
        cache_key=('triton_launch',key,arg_key,tuple(sorted(kwargs.items())))
        entry=workspace.get(cache_key)
        if entry is None:
            compiled=kernel[grid](*args,**kwargs)
            names=kernel.arg_names
            workspace[cache_key]=(compiled,names)
            if geometry is not None:
                grid_1=grid[1] if len(grid)>1 else 1
                grid_2=grid[2] if len(grid)>2 else 1
                tail=tuple(kwargs[name] for name in names[len(args):])
                workspace[plan_key]=(geometry,kwargs['debug'],compiled.run,
                    compiled.function,compiled.packed_metadata,compiled,tail,
                    grid_1,grid_2)
            return
        compiled,names=entry
        tail=tuple(kwargs[name] for name in names[len(args):])
        values=args+tail
        grid_1=grid[1] if len(grid)>1 else 1
        grid_2=grid[2] if len(grid)>2 else 1
        if geometry is not None:
            workspace[plan_key]=(geometry,kwargs['debug'],compiled.run,
                compiled.function,compiled.packed_metadata,compiled,tail,
                grid_1,grid_2)
        run=compiled.run;function=compiled.function
        packed_metadata=compiled.packed_metadata
    if stream is None:
        stream=torch.cuda.current_stream().cuda_stream
    launch_metadata=(None if enter_hook is None else
                     _cached_launch_metadata(
                         workspace,plan_key,compiled,grid,stream,values))
    run(grid[0],grid_1,grid_2,stream,function,packed_metadata,
        launch_metadata,enter_hook,exit_hook,*values)


@triton.jit
def _tile_keys(IDS,P,VALID,LOW,HIGH,QUOTA,OUT,R:tl.constexpr,S:tl.constexpr,
               GROUPS:tl.constexpr,K:tl.constexpr,META_GROUP:tl.constexpr,
               TOKEN_BITS:tl.constexpr):
    h=tl.program_id(0);tile=tl.program_id(1);i=tl.arange(0,2*K)
    r=tile*(2*K)+i
    token=tl.load(IDS+h*R+r,r<R,other=0)
    probability=tl.load(P+h*R+r,r<R,other=0)
    keep=tl.load(VALID+h*R+r,r<R,other=0)
    b=0 if META_GROUP==0 else h//META_GROUP
    good=keep&(token>=tl.load(LOW+b))&(token<tl.load(HIGH+b))&(probability>0)&(tl.load(QUOTA+b)>0)
    bits=probability.to(tl.uint16,bitcast=True).to(tl.int32)
    delta=(S-token).to(tl.int32)
    if TOKEN_BITS==32:
        bits=bits.to(tl.int64)
        delta=delta.to(tl.int64)
    key=tl.where(good,(bits<<TOKEN_BITS)+delta,0)
    ordered=tl.topk(key,K)
    i=tl.arange(0,K)
    tl.store(OUT+(h*GROUPS+tile)*K+i,ordered)


@triton.jit
def _tile_keys_from_scores_body(h,tile,IDS,SCORES,MSTAT,LSTAT,VALID,LOW,HIGH,QUOTA,OUT,
               R:tl.constexpr,S:tl.constexpr,GROUPS:tl.constexpr,K:tl.constexpr,
               META_GROUP:tl.constexpr,TOKEN_BITS:tl.constexpr,G:tl.constexpr,
               PROB_BF16:tl.constexpr):
    i=tl.arange(0,2*K)
    r=tile*(2*K)+i
    token=tl.load(IDS+h*R+r,r<R,other=0)
    keep=tl.load(VALID+h*R+r,r<R,other=0)
    prob_sum=tl.zeros((2*K,),tl.float32)
    for j in tl.static_range(G):
        score=tl.load(SCORES+(h*G+j)*R+r,r<R,other=float('-inf')).to(tl.float32)
        m=tl.load(MSTAT+h*G+j);l=tl.load(LSTAT+h*G+j)
        safe_l=tl.where(l>0.0,l,1.0)
        head_p=tl.exp(score-m)/safe_l
        head_p=tl.where(l>0.0,head_p,0.0)
        if PROB_BF16:
            prob_sum+=head_p.to(tl.bfloat16).to(tl.float32)
        else:
            prob_sum+=head_p.to(tl.float16).to(tl.float32)
    probability=prob_sum*(1.0/G)
    if PROB_BF16:
        probability=probability.to(tl.bfloat16)
    else:
        probability=probability.to(tl.float16)
    b=0 if META_GROUP==0 else h//META_GROUP
    good=keep&(token>=tl.load(LOW+b))&(token<tl.load(HIGH+b))&(probability>0)&(tl.load(QUOTA+b)>0)
    bits=probability.to(tl.uint16,bitcast=True).to(tl.int32)
    delta=(S-token).to(tl.int32)
    if TOKEN_BITS==32:
        bits=bits.to(tl.int64)
        delta=delta.to(tl.int64)
    key=tl.where(good,(bits<<TOKEN_BITS)+delta,0)
    ordered=tl.topk(key,K)
    i=tl.arange(0,K)
    tl.store(OUT+(h*GROUPS+tile)*K+i,ordered)


@triton.jit
def _tile_keys_from_scores(IDS,SCORES,MSTAT,LSTAT,VALID,LOW,HIGH,QUOTA,OUT,
               R:tl.constexpr,S:tl.constexpr,GROUPS:tl.constexpr,K:tl.constexpr,
               META_GROUP:tl.constexpr,TOKEN_BITS:tl.constexpr,G:tl.constexpr,
               PROB_BF16:tl.constexpr):
    _tile_keys_from_scores_body(tl.program_id(0),tl.program_id(1),IDS,SCORES,
        MSTAT,LSTAT,VALID,LOW,HIGH,QUOTA,OUT,R=R,S=S,GROUPS=GROUPS,K=K,
        META_GROUP=META_GROUP,TOKEN_BITS=TOKEN_BITS,G=G,PROB_BF16=PROB_BF16)


@triton.jit
def _merge_topk(IN,OUT,GROUPS:tl.constexpr,NEXT:tl.constexpr,K:tl.constexpr):
    h=tl.program_id(0);tile=tl.program_id(1);i=tl.arange(0,2*K)
    group=2*tile+(i>=K).to(tl.int32)
    rank=tl.where(i<K,i,2*K-1-i)
    value=tl.load(IN+(h*GROUPS+group)*K+rank,group<GROUPS,other=0)
    # 两条降序 run 反转拼接成 bitonic 序列。首个跨 K 比较后，较大的
    # K 项已完整包含 Top-K；无需再排序会被丢弃的另外 K 项。
    peer=tl.gather(value,i^K,axis=0)
    value=tl.where(i<K,tl.maximum(value,peer),tl.minimum(value,peer))
    i=tl.arange(0,K)
    value=tl.gather(value,i,axis=0)
    for bit in tl.static_range(9,-1,-1):
        stride=1<<bit
        peer=tl.gather(value,i^stride,axis=0)
        value=tl.where((i&stride)==0,tl.maximum(value,peer),tl.minimum(value,peer))
    tl.store(OUT+(h*NEXT+tile)*K+i,value)


def tree_topk(ids,p,valid,lower,upper,quota,capacity,width,*,metadata_group=None):
    hk,r=ids.shape
    metadata_group=0 if metadata_group is None else metadata_group
    groups=triton.cdiv(r,2*width)
    token_bits=32 if capacity>=65536 else 16
    packed_dtype=torch.int64 if token_bits==32 else torch.int32
    current=torch.empty((hk,groups,width),device=ids.device,dtype=packed_dtype)
    _tile_keys[(hk,groups)](ids,p,valid,lower,upper,quota,current,r,capacity,groups,width,metadata_group,token_bits,num_warps=8)
    while groups>1:
        following=triton.cdiv(groups,2)
        output=torch.empty((hk,following,width),device=ids.device,dtype=current.dtype)
        _merge_topk[(hk,following)](current,output,groups,following,width,num_warps=8)
        current=output;groups=following
    return current.view(hk,width)


@triton.jit
def _finish_sorted_key(h,key,LOW,HIGH,QUOTA,OUT,COUNT,S:tl.constexpr,K:tl.constexpr,
                       SINK:tl.constexpr,META_GROUP:tl.constexpr,TOKEN_BITS:tl.constexpr):
    i=tl.arange(0,K)
    b=0 if META_GROUP==0 else h//META_GROUP
    q=tl.load(QUOTA+b).to(tl.int32)
    if q==0:
        tl.store(OUT+h*K+i,0)
        tl.store(COUNT+h,0)
    else:
        positive=(key>>TOKEN_BITS)>0
        token_mask=(1<<TOKEN_BITS)-1
        chosen=tl.where((i<q)&positive,S-(key&token_mask),S).to(tl.int32)
        npositive=tl.sum(((i<q)&positive).to(tl.int32),0)
        if npositive==q:
            fast_packed=tl.sort(tl.where(chosen>=SINK,chosen,S),descending=False)
            tl.store(OUT+h*K+i,tl.where(fast_packed<S,fast_packed,0))
            tl.store(COUNT+h,tl.sum((fast_packed<S).to(tl.int32),0))
        else:
            chosen=tl.sort(chosen,descending=False)
            low=tl.load(LOW+b).to(tl.int32);high=tl.load(HIGH+b).to(tl.int32)
            boundaries=tl.where(chosen<S,chosen-low-i,S)
            lo=tl.zeros((K,),tl.int32);hi=tl.full((K,),K,tl.int32)
            for _ in range(11):
                mid=lo+(hi-lo)//2
                value=tl.gather(boundaries,tl.minimum(mid,K-1),axis=0)
                right=value<=i
                active=lo<hi
                lo=tl.where(active&right,mid+1,lo)
                hi=tl.where(active&~right,mid,hi)
            zero=low+i+lo
            zero=tl.where((i<q-npositive)&(zero<high),zero,S)
            combined=tl.cat(chosen,zero,can_reorder=True)
            combined=tl.where(combined>=SINK,combined,S)
            packed=tl.sort(combined,descending=False)
            j=tl.arange(0,2*K)
            tl.store(OUT+h*K+j,tl.where(packed<S,packed,0),j<K)
            tl.store(COUNT+h,tl.sum((packed<S).to(tl.int32),0))


@triton.jit
def _merge_tree_finish(IN,LOW,HIGH,QUOTA,OUT,COUNT,GROUPS:tl.constexpr,
                       S:tl.constexpr,K:tl.constexpr,SINK:tl.constexpr,
                       META_GROUP:tl.constexpr,TOKEN_BITS:tl.constexpr):
    """单 CTA 顺序合并所有降序 run；Top-K 集合与原归并树一致。"""
    h=tl.program_id(0)
    i=tl.arange(0,K)
    current=tl.load(IN+h*GROUPS*K+i)
    pair=tl.arange(0,2*K)
    rank=tl.where(pair<K,pair,2*K-1-pair)
    for group in tl.static_range(1,GROUPS):
        incoming=tl.load(IN+(h*GROUPS+group)*K+rank)
        retained=tl.gather(current,rank,axis=0)
        value=tl.where(pair<K,retained,incoming)
        peer=tl.gather(value,pair^K,axis=0)
        value=tl.where(pair<K,tl.maximum(value,peer),tl.minimum(value,peer))
        current=tl.gather(value,i,axis=0)
        for bit in tl.static_range(9,-1,-1):
            stride=1<<bit
            peer=tl.gather(current,i^stride,axis=0)
            current=tl.where((i&stride)==0,tl.maximum(current,peer),tl.minimum(current,peer))
    _finish_sorted_key(h,current,LOW,HIGH,QUOTA,OUT,COUNT,S=S,K=K,SINK=SINK,
                       META_GROUP=META_GROUP,TOKEN_BITS=TOKEN_BITS)


@triton.jit
def _merge_finish(IN,LOW,HIGH,QUOTA,OUT,COUNT,S:tl.constexpr,K:tl.constexpr,SINK:tl.constexpr,META_GROUP:tl.constexpr,TOKEN_BITS:tl.constexpr):
    h=tl.program_id(0);i=tl.arange(0,2*K)
    group=(i>=K).to(tl.int32)
    rank=tl.where(i<K,i,2*K-1-i)
    key=tl.load(IN+(h*2+group)*K+rank)
    # 最后一级归并直接消费寄存器中的 key；跨 K 比较后只保留并排序
    # 较大的 K 项，省掉一次中间写读、一个 launch 和无效半边排序。
    peer=tl.gather(key,i^K,axis=0)
    key=tl.where(i<K,tl.maximum(key,peer),tl.minimum(key,peer))
    i=tl.arange(0,K)
    key=tl.gather(key,i,axis=0)
    for bit in tl.static_range(9,-1,-1):
        stride=1<<bit
        peer=tl.gather(key,i^stride,axis=0)
        key=tl.where((i&stride)==0,tl.maximum(key,peer),tl.minimum(key,peer))
    _finish_sorted_key(h,key,LOW,HIGH,QUOTA,OUT,COUNT,S=S,K=K,SINK=SINK,
                       META_GROUP=META_GROUP,TOKEN_BITS=TOKEN_BITS)


@triton.jit
def _finish(TOP,LOW,HIGH,QUOTA,OUT,COUNT,S:tl.constexpr,K:tl.constexpr,SINK:tl.constexpr,META_GROUP:tl.constexpr,TOKEN_BITS:tl.constexpr):
    h=tl.program_id(0);i=tl.arange(0,K)
    b=0 if META_GROUP==0 else h//META_GROUP
    q=tl.load(QUOTA+b).to(tl.int32)
    if q==0:
        tl.store(OUT+h*K+i,0)
        tl.store(COUNT+h,0)
    else:
        key=tl.load(TOP+h*K+i)
        positive=(key>>TOKEN_BITS)>0
        token_mask=(1<<TOKEN_BITS)-1
        chosen=tl.where((i<q)&positive,S-(key&token_mask),S).to(tl.int32)
        npositive=tl.sum(((i<q)&positive).to(tl.int32),0)
        if npositive==q:
            fast_packed=tl.sort(tl.where(chosen>=SINK,chosen,S),descending=False)
            tl.store(OUT+h*K+i,tl.where(fast_packed<S,fast_packed,0))
            tl.store(COUNT+h,tl.sum((fast_packed<S).to(tl.int32),0))
        else:
            chosen=tl.sort(chosen,descending=False)
            low=tl.load(LOW+b).to(tl.int32);high=tl.load(HIGH+b).to(tl.int32)
            boundaries=tl.where(chosen<S,chosen-low-i,S)
            lo=tl.zeros((K,),tl.int32);hi=tl.full((K,),K,tl.int32)
            for _ in range(11):
                mid=lo+(hi-lo)//2
                value=tl.gather(boundaries,tl.minimum(mid,K-1),axis=0)
                right=value<=i
                active=lo<hi
                lo=tl.where(active&right,mid+1,lo)
                hi=tl.where(active&~right,mid,hi)
            zero=low+i+lo
            zero=tl.where((i<q-npositive)&(zero<high),zero,S)
            combined=tl.cat(chosen,zero,can_reorder=True)
            combined=tl.where(combined>=SINK,combined,S)
            packed=tl.sort(combined,descending=False)
            j=tl.arange(0,2*K)
            tl.store(OUT+h*K+j,tl.where(packed<S,packed,0),j<K)
            tl.store(COUNT+h,tl.sum((packed<S).to(tl.int32),0))


def rank_unique(ids,p,valid,lower,upper,quota,*,capacity,width,sink):
    hk,r=ids.shape
    top=tree_topk(ids,p,valid,lower,upper,quota,capacity,width)
    out=torch.empty((1,hk,width),device=ids.device,dtype=torch.int32)
    counts=torch.empty((1,hk),device=ids.device,dtype=torch.int32)
    _finish[(hk,)](top,lower,upper,quota,out,counts,capacity,width,sink,0,32 if capacity>=65536 else 16,num_warps=8)
    return TokenSelection(out,counts)


def rank_unique_batched(ids,p,valid,lower,upper,quota,*,capacity,width=1024,sink=0,
                        validate=True,workspace=None,score_state=None,
                        prevalidated=False,fuse_tree=False,fuse_tiles=False,
                        raw_replay=False):
    """Rank deduplicated [B,Hkv,R] candidates with live per-request metadata.

    Uses the pinned unique-ranking contract; not a replacement for selecting
    and deduplicating candidates. No request loop or capacity-wide storage.
    """
    fused=score_state is not None
    if not prevalidated and (ids.ndim!=3 or valid.shape!=ids.shape
            or (not fused and (p is None or p.shape!=ids.shape))
            or type(capacity) is not int or not 0<capacity<2**31 or width!=1024
            or type(sink) is not int or not 0<=sink<=capacity
            or ids.shape[2]<width or min(ids.shape[:2])<1 or ids.numel()>=2**31
            or ids.dtype!=torch.int64 or valid.dtype!=torch.bool
            or (not fused and p.dtype not in (torch.float16,torch.bfloat16))):
        raise ValueError('unsupported batched unique ranking geometry')
    b,h,r=ids.shape
    if fused:
        scores,mstat,lstat,gqa_group,prob_dtype=score_state
        if not prevalidated and (type(gqa_group) is not int or gqa_group<1
                or scores.shape!=(b,h,gqa_group,r)
                or mstat.shape!=(b,h,gqa_group) or lstat.shape!=mstat.shape
                or scores.dtype!=torch.bfloat16
                or mstat.dtype!=torch.float32 or lstat.dtype!=torch.float32
                or prob_dtype not in (torch.float16,torch.bfloat16)):
            raise ValueError('unsupported fused score-state ranking geometry')
        tensors=(ids,valid,lower,upper,quota,scores,mstat,lstat)
    else:
        tensors=(ids,p,valid,lower,upper,quota)
    if not prevalidated and (any(x.shape!=(b,) or x.dtype!=torch.int64 for x in (lower,upper,quota))
            or not all(x.is_cuda and x.device==ids.device and x.is_contiguous()
                       for x in tensors)):
        raise ValueError('same-device contiguous CUDA candidates and int64 bounds[B] required')
    if validate:
        torch._assert_async(((lower>=0)&(upper>=lower)&(upper<=capacity)
                             &(quota>=0)&(quota<=width)).all(),'invalid batch ranking bounds/quota')
    hk=b*h
    groups=triton.cdiv(r,2*width)
    token_bits=32 if capacity>=65536 else 16
    packed_dtype=torch.int64 if token_bits==32 else torch.int32
    current=_workspace_empty(workspace,'rank_tile',(hk,groups,width),packed_dtype,ids.device)
    if fused:
        if not fuse_tiles:
            _fast_launch(workspace,'rank_tile_scores_launch',_tile_keys_from_scores,(hk,groups),ids.view(hk,r),scores,mstat,lstat,valid.view(hk,r),lower,upper,quota,current,r,capacity,groups,width,h,token_bits,gqa_group,prob_dtype==torch.bfloat16,num_warps=8,raw_replay=raw_replay)
    else:
        _fast_launch(workspace,'rank_tile_launch',_tile_keys,(hk,groups),ids.view(hk,r),p.view(hk,r),valid.view(hk,r),lower,upper,quota,current,r,capacity,groups,width,h,token_bits,num_warps=8,raw_replay=raw_replay)
    if fuse_tree and groups>2:
        out=_workspace_empty(workspace,'rank_out',(b,h,width),torch.int32,ids.device)
        counts=_workspace_empty(workspace,'rank_counts',(b,h),torch.int32,ids.device)
        _fast_launch(workspace,'rank_tree_finish_launch',_merge_tree_finish,(hk,),
                     current,lower,upper,quota,out,counts,groups,capacity,width,
                     sink,h,token_bits,num_warps=8,raw_replay=raw_replay)
        return TokenSelection(out,counts)
    while groups>2:
        following=triton.cdiv(groups,2)
        output=_workspace_empty(workspace,'rank_merge',(hk,following,width),current.dtype,ids.device)
        _fast_launch(workspace,('rank_merge_launch',following),_merge_topk,(hk,following),current,output,groups,following,width,num_warps=8,raw_replay=raw_replay)
        current=output;groups=following
    out=_workspace_empty(workspace,'rank_out',(b,h,width),torch.int32,ids.device)
    counts=_workspace_empty(workspace,'rank_counts',(b,h),torch.int32,ids.device)
    if groups==2:
        _fast_launch(workspace,'rank_merge_finish_launch',_merge_finish,(hk,),current,lower,upper,quota,out,counts,capacity,width,sink,h,token_bits,num_warps=8,raw_replay=raw_replay)
    else:
        _fast_launch(workspace,'rank_finish_launch',_finish,(hk,),current,lower,upper,quota,out,counts,capacity,width,sink,h,token_bits,num_warps=8,raw_replay=raw_replay)
    return TokenSelection(out,counts)
