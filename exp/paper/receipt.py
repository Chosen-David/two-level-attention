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
    try: yield receipt
    except BaseException:
        receipt['status']='failed'
        raise
    else:
        sync();receipt.update(status='ok',wall_ms=(time.perf_counter()-start)*1000)


def runtime_counters():
    import sys
    result={}
    for name,field in [('sparse_attn.ops.candidate_prefill','RUNTIME_COUNTS'),
                       ('sparse_attn.ops.batch_select','_GRAPH_STATS'),
                       ('sparse_attn.ops.fused_attn_triton','FUSED_LAUNCH_COUNTS')]:
        module=sys.modules.get(name)
        if module is not None: result[name.rsplit('.',1)[-1]]=dict(getattr(module,field))
    return result


def counter_delta(before,after):
    return {module:{key:value-before.get(module,{}).get(key,0) for key,value in counts.items()}
            for module,counts in after.items()}
