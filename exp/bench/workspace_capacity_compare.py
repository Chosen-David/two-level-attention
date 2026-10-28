"""Pair workspace backing storage; CUDA allocator deltas only when available."""
import gc
import hashlib
import json
from pathlib import Path
import torch
from sparse_attn.ops.far_skip_workspace import FarSkipWorkspace

def main():
    source=Path(__file__).resolve().parents[2]/'sparse_attn/ops/far_skip_workspace.py'
    result=dict(torch=torch.__version__,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='workspace allocation only; not KV eviction or latency',cases=[])
    for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
        for fixed in (False,True):
            gc.collect()
            if device=='cuda':torch.cuda.synchronize();torch.cuda.empty_cache()
            like=torch.empty(0,device=device,dtype=torch.bfloat16)
            before=torch.cuda.memory_allocated() if device=='cuda' else None
            ws=FarSkipWorkspace(fixed_shapes=fixed)
            pair=[ws.scratch(n,(8,1156,128),like) for n in ('cuda_gather_x','cuda_gather_y')]
            pointers=[x.data_ptr() for x in pair]
            for _ in range(100):
                for i,n in enumerate(('cuda_gather_x','cuda_gather_y')):
                    assert ws.scratch(n,(8,1156,128),like).data_ptr()==pointers[i]
            result['cases'].append(dict(device=device,fixed_shapes=fixed,payload_bytes=sum(x.numel()*x.element_size() for x in pair),backing_storage_bytes=sum(x.untyped_storage().nbytes() for x in pair),cuda_allocated_delta=None if before is None else torch.cuda.memory_allocated()-before))
            del pair,ws,like
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
