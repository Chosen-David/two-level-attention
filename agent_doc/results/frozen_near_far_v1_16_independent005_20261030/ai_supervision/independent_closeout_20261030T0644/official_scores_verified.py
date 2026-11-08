from pathlib import Path
import sys,json,hashlib,math
B=Path(__file__).resolve().parent;S=Path('/home/wangyuanshuo02/two-level-attention-frozen-main');sys.path.insert(0,str(S))
from benchmark.LongBench.eval import sample_score
errors=[];n=0;hashes={};counts={}
rows={ds.stem:[json.loads(l) for l in ds.read_text().splitlines()] for ds in Path('/home/wangyuanshuo02/datasets/LongBench/data').glob('*.jsonl') if ds.stem in [p.name for p in (B/'fixed/v1/requests').iterdir()]}
for f in sorted((B/'fixed/v1/requests').glob('*/*.json')):
 r=json.loads(f.read_bytes());ds=r['dataset'];counts[ds]=counts.get(ds,0)+1
 for arm,a in r['arms'].items():
  s=sample_score('repobench' if ds=='repobench-p' else ds,a['prediction'],r['answers'],rows[ds][r['row_index']].get('all_classes',[]))
  if not math.isfinite(s) or abs(s-a['score'])>1e-12:errors.append([str(f),arm,s,a['score']])
  n+=1
for name in ['benchmark/LongBench/eval.py','benchmark/LongBench/metrics.py']:
 hashes[str(S/name)]=hashlib.sha256((S/name).read_bytes()).hexdigest()
result=dict(outputs_checked=n,counts=counts,errors=errors,source_sha256=hashes,scope='Official frozen scorer cross-check; independent recheck score path separately required')
(B/'OFFICIAL_SCORES_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));raise SystemExit(bool(errors))
