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
