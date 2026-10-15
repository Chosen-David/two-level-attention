"""Opt-in host adapter building blocks, not an installed GPU runner.

The caller supplies the actual synchronization function and frozen identities.
This timer only measures the wrapped boundary; it cannot infer request scope.
"""
from contextlib import contextmanager
import time

@contextmanager
def synchronized_wall(sync):
    if not callable(sync): raise ValueError('explicit completion synchronization required')
    sync();start=time.perf_counter();receipt={}
    try:
        yield receipt
        sync();receipt.update(status='ok',wall_ms=(time.perf_counter()-start)*1000)
    except BaseException:
        receipt['status']='failed'
        raise


def runtime_counters():
    import sys
    result={}
    for name,field in [('sparse_attn.ops.eager_prefill','RUNTIME_COUNTS'),
                       ('sparse_attn.ops.batch_select','_GRAPH_STATS'),
                       ('sparse_attn.ops.fused_attn_triton','FUSED_LAUNCH_COUNTS')]:
        module=sys.modules.get(name)
        if module is not None: result['candidate_prefill' if name.endswith('.eager_prefill') else name.rsplit('.',1)[-1]]=dict(getattr(module,field))
    return result


def counter_delta(before,after):
    return {module:{key:value-before.get(module,{}).get(key,0) for key,value in counts.items()}
            for module,counts in after.items()}


def collect_timing_call(protocol, protocol_sha, seed_row, call, synchronize, observe):
    """Measure a host-owned boundary then read actual support/routes outside it.

    No GPU launch policy, inferred lifecycle, warmup or model adapter is supplied.
    observe(result) must supply lifecycle, routes, support and actual token counts.
    Its support must exactly cover the frozen units. Returns only validated rows;
    exceptions propagate and must be recorded as failed by the host runner.
    """
    from .contract import protocol_errors, need, validate_timing, TIMING
    need(not protocol_errors(protocol),'unfrozen collection protocol')
    need(protocol['timing_level']!='T0','host wall collector cannot measure kernel-only T0')
    if not callable(call) or not callable(observe): raise ValueError('actual call/observer required')
    before=runtime_counters()
    with synchronized_wall(synchronize) as timer:
        result=call()
    after=runtime_counters()  # end of measured call; observer is outside this window
    if any(module not in after or not set(counts)<=set(after[module]) for module,counts in before.items()):
        raise ValueError("runtime counters disappeared during measurement")
    observed=observe(result)
    if not isinstance(observed,dict): raise ValueError('observer must return actual receipts')
    delta=counter_delta(before,after)
    fallback=sum(value for counts in delta.values() for key,value in counts.items() if 'fallback' in key)
    if any(value<0 for counts in delta.values() for value in counts.values()):
        raise ValueError('counters reset during measurement')
    row=dict(seed_row,**observed)
    row.update(protocol_sha256=protocol_sha,status='ok',measured_this_run=True,
               aggregation='direct-wall',fallback_count=fallback)
    row[TIMING[protocol['timing_level']]]=timer['wall_ms']
    validate_timing(row,protocol,protocol_sha)
    return row

from contextvars import ContextVar
_QWEN_REQUEST = ContextVar('paper_qwen_request', default=None)


def record_qwen_route(layer, route):
    capture = _QWEN_REQUEST.get()
    if capture is not None: capture.route(layer, route)


class QwenRequestCapture:
    """Opt-in real manual-path telemetry; no support/scientific acceptance.

    Hashing/hook installation excluded; guard enter/exit and completion included.
    Internal compile/plan/kernel lifecycle stays unknown. No support inspection.
    """
    def __init__(self, model, material_ids, question_ids, *, sync=None, clock=None):
        import hashlib
        import torch
        from pathlib import Path
        if any(t.ndim!=2 or t.shape[0]!=1 for t in (material_ids,question_ids)) or not material_ids.shape[1]:
            raise ValueError('capture requires B1 and nonempty material')
        ids=torch.cat((material_ids,question_ids),dim=-1).detach().cpu().numpy().astype('<i8')
        root=Path(__file__).resolve().parents[2]
        producer={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                  ('benchmark/LongBench/pred.py','sparse_attn/patches/qwen3_attn_patch.py')}
        self.binding=dict(producer_files_sha256=producer,input_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
            material_tokens=material_ids.shape[1],question_tokens=question_ids.shape[1],
            actual_input_tokens=ids.shape[1],input_ids_encoding='concatenated int64 little-endian',
            collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.sync=sync if sync is not None else lambda: torch.cuda.synchronize(material_ids.device)
        self.clock=clock or time.perf_counter
        self.model=model;self.handles=[];self.calls=[];self.current=None;self.first=None;self.last=None
        self.generated=[];self.receipt=None;self.layers=list(range(len(model.model.layers)))
        if not self.layers:raise ValueError('no Qwen attention layers to observe')
        self.native=[]
        self.generation_policy=None
        for i,layer in enumerate(model.model.layers):
            fn=layer.self_attn.forward;fn=getattr(fn,'__func__',fn)
            if fn.__module__=='sparse_attn.patches.qwen3_attn_patch' and fn.__name__=='qwen3_attn_forward':continue
            if fn.__module__=='transformers.models.qwen3.modeling_qwen3' and fn.__qualname__=='Qwen3Attention.forward':self.native.append(i)
            else:raise ValueError('unsupported attention forward; route hooks unavailable')

    def __enter__(self):
        if _QWEN_REQUEST.get() is not None:raise ValueError('nested request capture unsupported')
        def hook(layer):
            def observe(module,args,output):self.route(layer,'native-qwen3-forward')
            return observe
        try:
            for i in self.native:self.handles.append(self.model.model.layers[i].self_attn.register_forward_hook(hook(i)))
            self.before=runtime_counters();self.sync();self.start=self.clock()
            self.token=_QWEN_REQUEST.set(self)
            return self
        except BaseException:
            for handle in self.handles:handle.remove()
            raise

    def route(self,layer,route):
        if self.current is None:raise ValueError('attention outside measured model call')
        self.current['layers'].append(dict(layer=layer,route=route))

    def call(self,phase,model,**kwargs):
        if model is not self.model:raise ValueError('different model supplied to capture')
        if phase not in ('material','question','decode') or self.current is not None:raise ValueError('invalid model phase')
        event=dict(phase=phase,q_tokens=int(kwargs['input_ids'].shape[-1]),layers=[])
        if phase!='material' and event['q_tokens']!=1:raise ValueError('question/decode must be token-wise')
        self.current=event
        try:
            result=model(**kwargs)
            if [r['layer'] for r in event['layers']]!=self.layers:raise ValueError('actual route coverage incomplete/duplicated')
            self.calls.append(event);return result
        finally:self.current=None

    def token_ready(self,token):
        self.generated.append(int(token))
        if self.first is None:self.sync();self.first=self.clock()

    def generation_complete(self):
        self.sync();self.last=self.clock()

    def __exit__(self,kind,value,tb):
        try:
            if kind is not None:return False
            self.sync();end=self.clock()  # outer context: guard has already exited
            after=runtime_counters()
            if any(m not in after or not set(c)<=set(after[m]) for m,c in self.before.items()):raise ValueError('request counters disappeared')
            delta=counter_delta(self.before,after)
            if any(v<0 for c in delta.values() for v in c.values()):raise ValueError('request counters reset')
            import math
            n=len(self.generated)
            if any(type(t) not in (int,float) or not math.isfinite(t) for t in (self.start,self.first,self.last,end)):
                raise ValueError('nonfinite request timestamp')
            if [c['phase'] for c in self.calls]!=['material']+['question']*self.binding['question_tokens']+['decode']*max(0,n-1):raise ValueError('request phase sequence mismatch')
            if n<1 or self.first is None or self.last is None or not (self.start<=self.first<=self.last<=end and end>self.start):raise ValueError('invalid request timing')
            durations=((self.first-self.start)*1000,(self.last-self.first)*1000,(end-self.start)*1000)
            if not all(math.isfinite(t) for t in durations):raise ValueError('nonfinite duration')
            self.receipt=dict(self.binding,schema='longbench-qwen-request-v1',
                status='candidate-telemetry-not-paper-acceptance',measured_this_run=True,
                generation_policy=self.generation_policy,
                fallback_count=sum(v for c in delta.values() for key,v in c.items() if 'fallback' in key),
                boundary='before-guard-enter-to-after-guard-exit-complete',
                ttft_ms=(self.first-self.start)*1000,decode_after_first_wall_ms=(self.last-self.first)*1000,
                tpot_ms=(self.last-self.first)*1000/(n-1) if n>1 else None,
                request_wall_ms=(end-self.start)*1000,actual_generated_tokens=n,
                phase_calls=self.calls,runtime_counter_delta=delta,
                lifecycle=dict(guard='included',tokenization='excluded',input_hashing='excluded',support_observation='disabled',
                    compile=None,plan=None,capture=None,prepare=None,metadata_update=None,injected_copy=None,replay=None,
                    sync='initial excluded; completion syncs included'),
                support=None,support_coverage='not-observed',scientific_acceptance=False,
                limitations=['native internal kernel choice not observed','lightweight route-hook overhead included',
                    'no complete support audit','internal lifecycle durations unavailable; never substitute zeros'])
        finally:
            _QWEN_REQUEST.reset(self.token)
            for handle in self.handles:handle.remove()
        return False
