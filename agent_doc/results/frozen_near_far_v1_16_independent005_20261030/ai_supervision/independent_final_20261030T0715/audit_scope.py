from pathlib import Path
import json,hashlib,subprocess,collections,datetime,sys,importlib.metadata
import numpy as np
B=Path(__file__).resolve().parent; V=B/'fixed/v1'; M=B/'fixed/mk2'; errors=[]
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def ck(ok,msg):
 if not ok:errors.append(msg)
def sealed_file(root,rel):
 seal=read(root/'EVIDENCE_SHA256.json'); files=seal['files']
 expected=files[rel]['sha256'] if isinstance(files,dict) else next(x['sha256'] for x in files if x['path']==rel)
 b=(root/rel).read_bytes();ck(sha(b)==expected,'prior seal '+rel);return b
oldv=B.parent/'independent_review_20261030T0514'
oldm=oldv
prior=json.loads(sealed_file(oldv,'v1_checked/cpu_checks.json'))
ck(prior['errors']==[],'old v1 errors'); shared=0
for r in prior['records']:
 rel=f"requests/{r['dataset']}/{r['row']:04d}.json"
 b=sealed_file(oldv,'fixed/v1/'+rel)
 ck(sha(b)==r['sha256'] and b==(V/rel).read_bytes(),'v1 old changed '+rel);shared+=1
for rel in ['REPORT_DETAIL.md','SCOPED_VALIDATION.json','FINAL_BINDINGS_CHECK.json','SCOPE_AUDIT.json','OFFICIAL_SCORES_VERIFIED.json','tools/results/v1_cache.json']:
 sealed_file(oldv,rel)
manifest=read(V/'manifest_rank0.json');cfg=manifest['protocol'];ident=manifest['identity']
ordered=[(ds,i) for ds in cfg['datasets'] for i in cfg['selected_rows'][ds]]
for ds in cfg['datasets']:ck(np.random.default_rng(cfg['seed']).permutation(len(cfg['selected_rows'][ds])).tolist()==cfg['selected_rows'][ds],'permutation '+ds)
rankmap={key:n%128 for n,key in enumerate(ordered)}; manifests={read(f)['rank']:read(f) for f in V.glob('manifest_rank*.json')};counts=collections.Counter();rankcounts=collections.Counter();activation=collections.defaultdict(collections.Counter)
for f in V.glob('requests/*/*.json'):
 r=read(f);key=(r['dataset'],r['row_index']);rank=rankmap[key]
 ck(f.name==f"{key[1]:04d}.json",'filename '+str(f));ck(rank in manifests,'missing rank manifest '+str(rank));counts[key[0]]+=1;rankcounts[rank]+=1
 for arm,a in r['arms'].items():activation[arm].update(a['gate_counts'])
for rank,ma in manifests.items():ck(ma['identity']==ident and ma['protocol']==cfg,'manifest '+str(rank))
done=[]
for f in V.glob('worker_done_rank*.json'):
 d=read(f);rank=int(f.stem.replace('worker_done_rank',''));ck(d['identity']==ident,'done identity');ck(rankcounts[rank]==sum(n%128==rank for n in range(3750)),'done coverage '+str(rank));done.append(rank)
source_states={};source_bind={}
for label,src,ma in [('v1',Path('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/frozen005_v1_16_source_20261030'),manifest)]:
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip();dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=src,text=True)
 source_states[label]=dict(source=str(src),head=head,tracked_dirty=dirty);ck(not dirty,label+' dirty')
 for rel,h in ma['protocol']['source_sha256'].items():
  b=(src/rel).read_bytes();ck(sha(b)==h,label+' source '+rel);ck(sha(subprocess.check_output(['git','show',head+':'+rel],cwd=src))==h,label+' git object '+rel)
  dest=B/'source'/label/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b);source_bind[str(src/rel)]=h
pkgs={p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','tokenizers','rouge']}
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),errors=errors,v1_counts=dict(counts),v1_total=sum(counts.values()),v1_previous_covered=shared,v1_new=sum(counts.values())-shared,manifest_count=len(manifests),done_count=len(done),rank_counts=dict(rankcounts),activation={a:dict(v) for a,v in activation.items()},source_states=source_states,source_bindings=source_bind,environment=dict(python=sys.version,packages=pkgs),prior_scope='0514 V1 2296 full CPU records. MK2 not rechecked this round. 0313/0343/0355/0415 tool or inventory reports do not grant newer record admission.')

(B/'SCOPE_AUDIT.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['source_bindings','rank_counts']},indent=2));raise SystemExit(bool(errors))
