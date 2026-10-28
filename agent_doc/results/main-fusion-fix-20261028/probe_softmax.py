import json
from pathlib import Path
import torch
from main_kernels import _iter3_fused_softmax_kernel

torch.set_num_threads(1)
cases=[]
for r in (1024,2048,4224):
    g=4
    x=torch.zeros(g,r)
    # Different tile offsets disappear under tile-local normalization.
    x[:,1024:]=10
    x[:,0]=1
    keep=torch.ones(1,r,dtype=torch.bool)
    mid=torch.ones_like(keep)
    out=torch.empty(1,r)
    _iter3_fused_softmax_kernel[(1,(r+1023)//1024)](x,keep,mid,out,R_CAP=r,GROUP_SIZE=g,BLOCK=1024,num_warps=8)
    expected=x.softmax(-1).nan_to_num(nan=0).view(1,g,r).mean(1)
    cases.append(dict(r=r,sum_actual=float(out.sum()),sum_expected=float(expected.sum()),max_error=float((out-expected).abs().max()),actual_top=int(out.argmax()),expected_top=int(expected.argmax())))
assert cases[1]['actual_top']==0 and cases[1]['expected_top']>=1024
assert cases[2]['sum_actual']>4.9
result=dict(source='main@fb335e5ceb742b31d83206c670a10286c7fba182',source_blob='950c53ed27a884f0cc220a82226c64440674e202',scope='Exact extracted kernel, Triton3.4 CPU interpreter; not CUDA performance; main regression',cases=cases)
Path(__file__).with_name('receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
