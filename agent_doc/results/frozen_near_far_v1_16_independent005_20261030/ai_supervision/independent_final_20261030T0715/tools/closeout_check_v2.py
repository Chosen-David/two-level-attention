"""Read-only independent coverage + paired-summary check; never generate or modify production.
Requires completed production outputs and a separate full recheck.py result.
Writes only --output below this review directory. Pending runs publish no metrics.
"""
import argparse,collections,hashlib,json,pathlib,datetime,sys,math
ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=pathlib.Path(a.run);out=pathlib.Path(a.output).resolve();root=pathlib.Path(__file__).resolve().parent
if root not in out.parents:raise SystemExit('output must be a child of this review directory')
out.mkdir(parents=True,exist_ok=True);errors=[];bindings={}
def read(f):
 b=f.read_bytes();bindings[str(f.resolve())]=hashlib.sha256(b).hexdigest();return json.loads(b)
def ck(ok,msg):
 if not ok:errors.append(msg)
def finish(status,extra=None):
 changed=[f for f,h in bindings.items() if not pathlib.Path(f).exists() or hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()!=h]
 if changed:errors.append('files changed during check')
 r=dict(status=status if not errors else 'pending_or_failed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),errors=errors,changed=changed,artifact_hashes=bindings,full_acceptance='pending',limitations=['No GPU rerun','Full input/scorer/gate/cache recheck still required','Only actual tested branches are evidence']);r.update(extra or {});(out/'closeout_checks.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['status','errors','full_acceptance']}));return 0 if status=='scoped_closeout_pass' and not errors else 2
q=read(p/'queue.json');ms=[read(f) for f in sorted(p.glob('manifest_rank*.json'))];cfg=ms[0]['protocol'];ident=ms[0]['identity'];ds=cfg['datasets'];arms=['full','far_0.05','dual_0.05'];jobs=q['jobs']
ck(len(jobs)==128 and {j['rank'] for j in jobs}==set(range(128)) and all(j['status']=='done' for j in jobs),'all128 queue done')
ck(len(ms)==128 and {m['rank'] for m in ms}==set(range(128)),'all128 unique manifests')
ck(all(m['identity']==ident and m['protocol']==cfg for m in ms),'manifest identity')
ck(hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()==ident,'protocol digest')
expected_counts={d:150 if d=='multifieldqa_en' else 500 if d in ['lcc','repobench-p'] else 200 for d in ds};expected={(d,i) for d in ds for i in range(expected_counts[d])};records={}
for f in sorted(p.glob('requests/*/*.json')):
 r=read(f);key=(r['dataset'],r['row_index']);ck(key not in records and key in expected,'unique expected key');ck(f==p/'requests'/key[0]/('%04d.json'%key[1]),'canonical path');ck(r['status']=='complete' and r['identity']==ident and set(r['arms'])==set(arms),'complete paired identity');records[key]=r
ck(set(records)==expected and len(records)==3750,'all3750 exact rows')
for rank in range(128):
 f=p/('worker_done_rank%d.json'%rank)
 if not f.exists():ck(False,'missing done '+str(rank));continue
 d=read(f);m=next((m for m in ms if m['rank']==rank),{});j=next((j for j in jobs if j['rank']==rank),{})
 ck(d['identity']==ident and d['pid']==m.get('pid')==j.get('pid'),'done identity/pid '+str(rank))
if errors:sys.exit(finish('pending',dict(observed_requests=len(records),observed_manifests=len(ms))))
# Only a genuinely complete input reaches summary/CI/3pp checks.
import numpy as np
summary=read(p/'summary.json');validation=read(p/'validation.json');index=read(p/'raw_index.json');ck(summary['status']=='complete' and summary['identity']==ident and validation['status']=='complete-validated' and validation['identity']==ident,'summary/validation identity')
ordered=sorted(records.values(),key=lambda r:(ds.index(r['dataset']),r['row_index']));rng=np.random.default_rng(42)
def near(x,y):return math.isfinite(float(x)) and math.isfinite(float(y)) and abs(float(x)-float(y))<1e-10
for arm in arms:
 means={d:float(np.mean([r['arms'][arm]['score'] for r in ordered if r['dataset']==d])*100) for d in ds}
 ck(all(near(summary['arms'][arm]['per_task'][d],v) for d,v in means.items()),'task means '+arm);ck(near(summary['arms'][arm]['avg16'],np.mean(list(means.values()))),'AVG16 '+arm)
comparisons={}
for arm,base in [('far_0.05','full'),('dual_0.05','full'),('dual_0.05','far_0.05')]:
 name=arm+'-vs-'+base;c=summary['comparisons'][name];groups=[];result={}
 for d in ds:
  v=np.array([100*(r['arms'][arm]['score']-r['arms'][base]['score']) for r in ordered if r['dataset']==d]);groups.append(v);ci=np.quantile(v[rng.integers(0,len(v),(5000,len(v)))].mean(1),[.025,.975]);entry=c['per_task'][d]
  ck(entry['n']==len(v) and near(entry['delta_pp'],v.mean()) and all(near(x,y) for x,y in zip(entry['ci95_pp'],ci)),'paired task '+name+'/'+d)
  ck(entry['degraded']==int(sum(v<0)) and entry['improved']==int(sum(v>0)),'sign counts '+name+'/'+d);result[d]=dict(n=len(v),delta_pp=float(v.mean()),ci95_pp=ci.tolist(),within_3pp_point_tolerance=bool(v.mean()>=-3))
 boot=np.stack([g[rng.integers(0,len(g),(5000,len(g)))].mean(1) for g in groups]).mean(0);delta=float(np.mean([g.mean() for g in groups]));ci=np.quantile(boot,[.025,.975]);ck(near(c['avg16']['delta_pp'],delta) and all(near(x,y) for x,y in zip(c['avg16']['ci95_pp'],ci)),'paired AVG16 '+name);result['AVG16']=dict(delta_pp=delta,ci95_pp=ci.tolist(),within_3pp_point_tolerance=delta>=-3);comparisons[name]=result
ck(len(index['records'])==3750 and len({(x['dataset'],x['row_index']) for x in index['records']})==3750,'index unique')
for item in index['records']:ck(bindings.get(str((p/item['path']).resolve()))==item['sha256'],'index raw hash')
f=p/'predictions.jsonl';b=f.read_bytes();bindings[str(f.resolve())]=hashlib.sha256(b).hexdigest();preds=[json.loads(x) for x in b.splitlines()];ck(len(preds)==11250 and len({(r['dataset'],r['row_index'],r['arm']) for r in preds})==11250,'predictions unique11250')
for r in preds:
 key=(r['dataset'],r['row_index']);raw=records[key];arm=raw['arms'][r['arm']];ck(all(r[k]==arm[k] for k in ['score','prediction','generated_ids']),'prediction content');ck(p/r['raw_path']==p/'requests'/key[0]/('%04d.json'%key[1]),'prediction path')
from extra_checks import verify
verify(p,records,summary,index,preds,cfg,ident,ck,bindings)
sys.exit(finish('scoped_closeout_pass',dict(comparisons=comparisons,numpy=np.__version__)))
