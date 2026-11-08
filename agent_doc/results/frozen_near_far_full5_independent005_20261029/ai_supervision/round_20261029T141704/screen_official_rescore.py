import json,pathlib,hashlib,math
from benchmark.LongBench.eval import sample_score
from benchmark.LongBench.metrics import SCORER_BACKEND_ID
root=pathlib.Path('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/results/frozen_near_far_independent005_pool_v2_20261029')
errors=[];checked=0;files=[]
for p in sorted((root/'requests').glob('*/*.json')):
 data=p.read_bytes();r=json.loads(data);files.append(dict(path=str(p),sha256=hashlib.sha256(data).hexdigest()))
 for arm,v in r['arms'].items():
  s=sample_score('repobench' if r['dataset']=='repobench-p' else r['dataset'],v['prediction'],r['answers'],[]);checked+=1
  if not math.isfinite(s) or abs(s-v['score'])>1e-12:errors.append(dict(dataset=r['dataset'],row=r['row_index'],arm=arm,recomputed=s,recorded=v['score']))
print(json.dumps(dict(checked=checked,errors=errors,scorer_backend=SCORER_BACKEND_ID,files=files),indent=2))
