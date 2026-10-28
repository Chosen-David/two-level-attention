"""Main-versus-fixed compact kernel; padded MID backing avoids unsafe probe reads."""
import json
from pathlib import Path
import torch
from triton.runtime import interpreter
from main_kernels import _iter3_compact_kernel as baseline
from fixed_kernels import _iter3_compact_kernel as kernel

# Triton 3.4 uses int(ndarray) for loop bounds; NumPy 2.5 rejects size-one
# nonzero-dimensional arrays. Keep the kernel unchanged, use scalar .item().
original_patch = interpreter._patch_lang_tensor
def compatible_patch(tensor):
 original_patch(tensor)
 tensor.__index__ = lambda self: int(self.handle.data.item())
interpreter._patch_lang_tensor = compatible_patch

torch.set_num_threads(1)
records=[]
for s in (65,129):
 for h in (1,8):
  for r in (17,2049,4224):
   for mode in ('empty','mixed','full','tail'):
    torch.manual_seed(20261028)
    ids=torch.randint(s,(h,r))
    keep=torch.rand(h,r)>.4
    if mode=='empty':keep.zero_()
    if mode=='full':keep.fill_(True)
    if mode=='tail':keep[:,:r//2]=False;keep[:,r//2:]=True
    ids=torch.where(keep,ids,s+7)
    # Logical domain is [0,s); padded backing is deliberately valid memory.
    # It exposes the read without asking the interpreter to access OOB memory.
    mid=torch.zeros(s+16,dtype=torch.bool);mid[:s:3]=True;mid[s-1]=False;mid[s:]=True
    safe=torch.where(keep,ids,s-1)
    def run(x, fn):
     out=torch.empty_like(x); ko=torch.empty_like(keep);mo=torch.empty_like(keep);cl=torch.empty(h,dtype=torch.long)
     fn[(h,)](x,keep,mid,out,ko,mo,cl,r,S_CAP_M1=s-1,BLOCK=2048,num_warps=8)
     return out,ko,mo,cl
    old=run(ids,kernel);unchecked=run(ids,baseline)
    order=torch.argsort(~keep,dim=-1,stable=True)
    expected=(safe.gather(-1,order),keep.gather(-1,order),mid[safe.gather(-1,order)],keep.sum(-1))
    for a,b in zip(old,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    valid_prefix=torch.arange(r)[None]<old[3][:,None]
    assert torch.equal(old[0][valid_prefix],unchecked[0][valid_prefix])
    assert torch.equal(old[3],unchecked[3])
    # Both versions sanitize stored IDs; only the fixed version also sanitizes MID reads.
    scatter_rejects=False
    try:torch.zeros(h,s).scatter_add_(-1,unchecked[0],torch.zeros(h,r))
    except RuntimeError:scatter_rejects=True
    assert not scatter_rejects
    assert torch.equal(old[0],unchecked[0])
    assert bool((old[2]!=unchecked[2]).any()) == bool((~keep).any())
    records.append(dict(s=s,h=h,r=r,mode=mode,reference_exact=True,
      invalid=int((~keep).sum()),changed_ids=int((old[0]!=unchecked[0]).sum()),
      changed_mid=int((old[2]!=unchecked[2]).sum()),scatter_rejects=scatter_rejects))
Path(__file__).with_name('pairs.json').write_text(json.dumps(dict(scope='CPU interpreter; logical OOB guarded with padded backing; no GPU timing',cases=records),indent=2)+'\n')
print(json.dumps(dict(cases=len(records),reference_exact=True,baseline_mid_failures=sum(x['changed_mid']>0 for x in records))))
