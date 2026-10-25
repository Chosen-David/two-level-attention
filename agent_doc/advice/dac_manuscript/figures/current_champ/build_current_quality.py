"""Render current pooled Qwen3-8B RULER quality from six hashed JSON snapshots.
Requires matplotlib. Run in any cwd: python3 path/to/build_current_quality.py.
No models, experiments, prediction replay, or synthetic task scores.
"""
from pathlib import Path
from decimal import Decimal
import os,json,csv,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/tla-current-quality-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'tla-current-pooled-v1'})
prov=json.loads((R/'provenance.json').read_text());shas={f['file']:f['sha256'] for f in prov['files']}
def load(n):
 raw=(R/'inputs'/n).read_bytes();assert hashlib.sha256(raw).hexdigest()==shas[n]
 j=json.loads(raw,parse_float=Decimal);assert len(j['scores'])==1
 k=next(iter(j['scores']));assert len(j['scores'][k])==11;assert all(n==100 for n in j['n'][k].values());assert not j['incomplete_cells']
 return k,j['scores'][k]
rows=[];means=[]
for label,length,dn in [('32K',32768,'e116e_ruler32_formal_fullkv.json'),('64K',65536,'e119_ruler64k_formal_fullkv.json'),('128K',131072,'e119_ruler128k_formal_fullkv.json')]:
 pn=f't6_ruler_q8b_L{length}_champ.json';pk,p=load(pn);dk,d=load(dn)
 assert set(p)==set(d);assert pk.startswith(f'L{length}/') and dk.startswith(f'L{length}/');assert '_ntoff_nl4096_' in pk and '_np32_' in pk
 for task in p:
  dv,pv=Decimal(str(d[task])),Decimal(str(p[task]));rows.append({'length':label,'task':task,'dense':str(dv),'pooled':str(pv),'delta':str(pv-dv),'n_per_arm':100,'dense_file':dn,'dense_key':dk,'pooled_file':pn,'pooled_key':pk,'dense_sha256':shas[dn],'pooled_sha256':shas[pn]})
 dm=sum(Decimal(str(v)) for v in d.values())/11;pm=sum(Decimal(str(v)) for v in p.values())/11
 means.append({'length':label,'dense':str(dm),'pooled':str(pm),'delta':str(pm-dm)})
assert len(rows)==33 and f"{Decimal(means[-1]['delta']):+.2f}"=='+0.38'
with (R/'current_pooled_task_scores.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(R/'current_pooled_summary.json').write_text(json.dumps(means,indent=2)+'\n')
fig=plt.figure(figsize=(7,3.15));fig.patch.set_facecolor('white')
fig.text(.025,.94,'Current pooled configuration',fontsize=13,fontweight='bold')
fig.text(.025,.89,'Qwen3-8B · RULER · 11 reported tasks per length · K = 1024',fontsize=9)
fig.text(.025,.80,'(a) Macro scores',fontweight='bold');fig.text(.50,.80,'(b) All 33 task-score differences',fontweight='bold')
a=fig.add_axes([.025,.32,.43,.42]);a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
for x,t in [(.02,'Length'),(.46,'FullKV'),(.75,'Pooled'),(.98,'Δ')]:a.text(x,.92,t,ha='left' if x==.02 else 'right',va='center',fontsize=9)
a.plot([.01,.99],[.80,.80],lw=.7,color='#374550')
for i,m in enumerate(means):
 y=.64-i*.25
 for x,k in [(.02,'length'),(.46,'dense'),(.75,'pooled'),(.98,'delta')]:
  val=m[k] if k=='length' else (f'{Decimal(m[k]):+.2f}' if k=='delta' else f'{Decimal(m[k]):.2f}')
  a.text(x,y,val,ha='left' if k=='length' else 'right',va='center',fontweight='bold' if k=='delta' else 'normal',fontsize=10)
a.plot([.01,.99],[.02,.02],lw=.7,color='#374550')
b=fig.add_axes([.56,.33,.405,.40]);b.set_xlim(-25,30);b.set_ylim(-.45,2.55)
b.axvline(0,color='#707c85',lw=.8);b.set_xticks([-20,-10,0,10,20,30]);b.grid(axis='x',color='#e2e6e9',lw=.6);b.set_axisbelow(True)
for i,m in enumerate(means):
 y=2-i;counts={}
 for row in [r for r in rows if r['length']==m['length']]:
  d=float(row['delta']);key=round(d/1.7);idx=counts.get(key,0);counts[key]=idx+1
  offset=[0,.13,-.13,.26,-.26,.39,-.39][idx]
  b.plot(d,y+offset,'o',ms=3.5,color='#536471',alpha=.95)
 b.plot(float(m['delta']),y-.26,'D',ms=4.7,color='#087e91')
b.set_yticks([2,1,0],['32K','64K','128K']);b.tick_params(axis='both',length=0,labelsize=8)
for spine in b.spines.values():spine.set_visible(False)
b.set_xlabel('Pooled − FullKV (score points)',fontsize=8,labelpad=3)
fig.text(.50,.19,'● Task     ◆ Equal-task macro',fontsize=8,color='#3d5a68')
fig.text(.025,.14,'100 examples/task/arm. QA tasks (SQuAD, HotpotQA) unreported; excluded from means.',fontsize=7.5)
fig.text(.025,.095,'Differences computed before rounding. Dot offsets separate nearby values only.',fontsize=7.5)
fig.text(.025,.05,'Descriptive comparisons: execution identities not fully matched; no paired confidence intervals.',fontsize=7.5)
for ext in ['pdf','svg','png']:
 kw={'dpi':200} if ext=='png' else ({'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {})
 fig.savefig(R/f'current_pooled_quality.{ext}',**kw)
print(json.dumps(means,indent=2))
