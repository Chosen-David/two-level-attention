import json, torch, math
from sparse_attn.ops.frozen_region_cache import RegionCacheLayer, RegionGeometry
from sparse_attn.ops.frozen_region_gate import GateConfig, decide_regions, profile_regions
torch.set_num_threads(1)
with torch.inference_mode():
 g=RegionGeometry(sink=2,near=6,window=3,page=4,max_new_tokens=40)
 layer=RegionCacheLayer(g);x=torch.arange(17.).reshape(1,1,17,1);layer.update(x,x*2)
 before=sum(n.resident_bytes() for n in layer.nodes.values());layer.freeze(skip_near=True,skip_far=False)
 assert before==sum(n.resident_bytes() for n in layer.nodes.values())
 assert 8 not in layer.gather()[2]
 for t in range(17,49):
  x=torch.tensor([[[[float(t)]]]]);layer.update(x,x*2)
  k,v,positions=layer.gather();far_end=max(2,((t+1-3-6)//4)*4)
  expected=list(range(far_end))+list(range(t+1-3,t+1))
  assert positions==expected,(t,positions,expected)
  assert k.flatten().tolist()==[float(i) for i in expected]
  assert v.flatten().tolist()==[2.*i for i in expected]
  assert layer.permanently_dropped==0 and layer.report()['stored_tokens']==t+1
 assert 8 in layer.gather()[2]
 for n,f,want in [(.03,.04,(True,True)),(.03,.06,(True,False)),(.06,.04,(False,True)),(.06,.07,(False,False)),(.05,.05,(False,False))]:
  d=decide_regions({'near':{'mean':n},'far':{'mean':f}},GateConfig(epsilon=.05))
  assert (d['skip_near'],d['skip_far'])==want
 torch.manual_seed(7);q=torch.randn(1,4,20,8);k=torch.randn(1,2,20,8)
 actual=profile_regions(q,k,g,rows=8,chunk=3);expected={'near':[],'far':[]}
 for pos in range(12,20):
  prob=(q[:,:,pos:pos+1]@k[:,:,:pos+1].repeat_interleave(2,1).transpose(-1,-2)/math.sqrt(8)).softmax(-1)
  end=pos+1-3;far_end=max(2,((pos+1-3-6)//4)*4)
  expected['near'].append(float(prob[...,far_end:end].sum(-1).mean()))
  expected['far'].append(float(prob[...,2:far_end].sum(-1).mean()))
 for key,values in expected.items():assert abs(actual[key]['mean']-sum(values)/8)<1e-6
print(json.dumps(dict(status='passed',device='cpu',migration_steps=32,threshold_cases=5,profile_queries=8,scope='small CPU coordinate/reference checks only; not independent result admission or Qwen3-8B quality validation')))
