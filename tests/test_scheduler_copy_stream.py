"""Source-bound stream dependency regression; optional native CUDA check."""
import ast
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

SOURCE = Path(os.environ.get('BASELINE_SOURCE', str(
    Path(__file__).resolve().parents[1] / 'sparse_attn/scheduler/scheduler.py')))

class Stream:
    def __init__(self): self.queue=[]; self.done=0
    def run(self, limit):
        while self.done < limit:
            op=self.queue[self.done]; self.done+=1; op()
    def wait_stream(self, other):
        limit=len(other.queue)
        self.queue.append(lambda: other.run(limit))
    def wait_event(self, event):
        stream,limit=event.snapshot
        self.queue.append(lambda: stream.run(limit))

class Event:
    def __init__(self, stream): self.record(stream)
    def record(self, stream): self.snapshot=(stream,len(stream.queue))
    def synchronize(self): self.snapshot[0].run(self.snapshot[1])

class Tensor:
    def __init__(self, runtime, values, lo=0, hi=None):
        self.runtime=runtime; self.values=values; self.lo=lo
        self.hi=len(values) if hi is None else hi
    def __getitem__(self, s):
        a,b,_=s.indices(self.hi-self.lo)
        return Tensor(self.runtime,self.values,self.lo+a,self.lo+b)
    def copy_(self, src, **kwargs):
        self.runtime.current.queue.append(lambda: self.values.__setitem__(
            slice(self.lo,self.hi),src.values[src.lo:src.hi]))

class Runtime:
    def __init__(self): self.current=Stream()
    def current_stream(self): return self.current
    @contextmanager
    def stream(self, stream):
        old=self.current; self.current=stream
        try: yield
        finally: self.current=old

def transfer_suffix(runtime):
    # Execute the production transfer block from idx selection to publication.
    tree=ast.parse(SOURCE.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)
                and n.name=='record_next_tokens')
    start=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='idx' for t in n.targets))
    method.body=method.body[start:]
    method.args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=a) for a in
        ['self','next_tokens','batch']],kwonlyargs=[],kw_defaults=[],defaults=[])
    ns={'torch':NS(cuda=runtime)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns['record_next_tokens']

class CopyStreamHostTest(unittest.TestCase):
    def check(self, batches, change_producer=False):
        rt=Runtime(); copy=Stream(); enqueue=transfer_suffix(rt)
        state=NS(_step_counter=0,_copy_stream=copy,
            _copy_events=[Event(copy),Event(copy)],
            _token_dev=[Tensor(rt,[-1]*8) for _ in range(2)],
            _token_cpu=[Tensor(rt,[-99]*8) for _ in range(2)])
        for step,batch in enumerate(batches):
            if change_producer: rt.current=Stream()
            producer=rt.current
            values=list(range(100+step*8,100+step*8+batch))
            enqueue(state,Tensor(rt,values),batch)
            idx=state._pending_copy_idx
            # Event completion must include D2H, without draining later work.
            later=[];producer.queue.append(lambda target=later:target.append(True))
            state._copy_events[idx].synchronize()
            self.assertEqual(state._token_cpu[idx].values[:batch],values)
            self.assertEqual(later,[])
            self.assertIs(rt.current,producer)
            self.assertEqual(state._pending_batch,batch)
    def test_completion_covers_cpu_data(self): self.check([4])
    def test_alternating_buffers_and_batch_changes(self): self.check([8,1,3,2]*8)
    def test_nondefault_producers(self): self.check([1,8,2,3]*8,True)

try:
    import torch
except ImportError:
    torch=None

@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA unavailable')
class CopyStreamCudaTest(unittest.TestCase):
    def test_native_tokens_eos_and_reuse(self):
        spec=importlib.util.spec_from_file_location('scheduler_under_test',SOURCE)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        removed=[]
        cache=NS(layers=[NS(kv_pool_k=torch.empty(1,device='cuda'))],
                 remove_request=removed.append)
        scheduler=mod.ContinuousBatchScheduler(cache=cache,max_batch=4,eos_ids=[2],pad_token_id=0)
        producer=torch.cuda.Stream()
        for step in range(64):
            with torch.cuda.stream(producer):
                scheduler.active_req_ids=[0,2]
                scheduler._cpu_req_staging.copy_(torch.tensor([0,2,-1,-1]))
                scheduler._device_req_ids_buf.copy_(scheduler._cpu_req_staging)
                tokens=[3,2 if step%2 else 4,0,0]
                logits=torch.full((4,1,8),-100.,device='cuda')
                for b,t in enumerate(tokens): logits[b,0,t]=100.
                scheduler.record_next_tokens([0,2,-1,-1],logits)
                scheduler._drain_pending_next_tokens()
                self.assertEqual(scheduler.next_tokens[0],3)
                if step%2:
                    self.assertNotIn(2,scheduler.active_req_ids)
                    self.assertEqual(removed[-1],2)
                else: self.assertEqual(scheduler.next_tokens[2],4)
        torch.cuda.synchronize()

if __name__=='__main__': unittest.main()
