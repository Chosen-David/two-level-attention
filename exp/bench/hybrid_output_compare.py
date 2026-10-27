"""CUDA output-assembly microbenchmark; excludes attention and model TPOT."""
import argparse
import json
import time
from pathlib import Path
import torch


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--steps', type=int, default=100)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required; no CPU performance substitution')
    records = []
    for batch in (1,2,8,32):
        outputs = [torch.randn(1,1,32,128,device='cuda',dtype=torch.bfloat16) for _ in range(batch)]
        arms = {'cat': lambda: torch.cat(outputs,dim=0),
                'single_direct': lambda: outputs[0] if len(outputs)==1 else torch.cat(outputs,dim=0)}
        assert torch.equal(arms['cat'](), arms['single_direct']())
        for fn in arms.values():
            for _ in range(20): fn()
        for repeat in range(a.repeats):
            for name in (list(arms) if repeat%2==0 else list(reversed(arms))):
                torch.cuda.synchronize()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                t=time.perf_counter();start.record()
                for _ in range(a.steps): result=arms[name]()
                end.record();end.synchronize()
                records.append(dict(batch=batch,arm=name,event_ms=start.elapsed_time(end)/a.steps,
                                    wall_ms=(time.perf_counter()-t)*1000/a.steps))
                del result
    a.out.write_text(json.dumps(dict(torch=torch.__version__,device=torch.cuda.get_device_name(),
        scope='output assembly only; direct arm launches no tensor work at B1; not TPOT',
        shape_per_row=[1,1,32,128],dtype='bf16',records=records),indent=2)+'\n')


if __name__ == '__main__': main()
