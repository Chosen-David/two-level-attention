"""Supplement recheck.py: linked/attention order, skip flags, stored sum, staging bytes.
Use only on a fixed raw snapshot. Does not recompute Q/K or certify GPU execution.
"""
import json,hashlib,argparse
from pathlib import Path
def check_record(r):
 errors=[]
 for arm,a in r['arms'].items():
  for i,d in a['decisions'].items():
   sn,sf=d['skip_near'],d['skip_far'];names=['sink']+([] if sf else ['far'])+([] if sn and sf else ['near_staging' if sn else 'near'])+['swa']
   for phase in ['frozen_kv','final_kv']:
    st=a[phase]['layers'][int(i)]
    if not(st['linked_order']==names and st['attention_order']==[n for n in names if n!='near_staging'] and st['skip_near']==sn and st['skip_far']==sf and st['stored_tokens']==sum(n['tokens'] for n in st['nodes'].values())):errors.append([arm,i,phase])
   if sn and not sf and sum(v['kv_bytes'] for v in a['prefill_full_kv']['layers'][int(i)]['nodes'].values())!=sum(v['kv_bytes'] for v in a['frozen_kv']['layers'][int(i)]['nodes'].values()):errors.append(['near bytes',arm,i])
   if len(a['generated_ids'])>1:
    rc=a['tli_receipts'][i]
    if rc['skip_near']!=sn or rc['skip_far']!=sf:errors.append(['receipt',arm,i])
 return errors
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--snapshot',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output).resolve();root=Path(__file__).resolve().parent
 if root not in out.parents:raise SystemExit('output must remain below this review')
 files=sorted(Path(a.snapshot).glob('requests/*/*.json'));errors=[];bindings={}
 for f in files:
  b=f.read_bytes();r=json.loads(b);bindings[str(f.resolve())]=hashlib.sha256(b).hexdigest()
  if r['status']!='complete':errors.append([str(f),'not complete']);continue
  errors.extend([[str(f),x] for x in check_record(r)])
 changed=[f for f,h in bindings.items() if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h]
 result=dict(status='scoped_cache_receipts_pass' if files and not errors and not changed else 'failed',full_acceptance='pending',requests=len(files),errors=errors,changed=changed,bindings=bindings,limitations=['No Q/K mass recomputation','Must accompany full recheck.py; this is only supplemental cache metadata'])
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='bindings'}));raise SystemExit(0 if result['status']=='scoped_cache_receipts_pass' else 2)
