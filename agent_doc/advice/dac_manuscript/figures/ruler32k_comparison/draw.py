#!/usr/bin/env python3
"""Reproduce descriptive RULER charts from fixed-revision author summaries. No raw replay."""
from pathlib import Path
import json,csv,hashlib,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
D=Path(__file__).resolve().parent
TASKS=['niah_single_1','niah_single_2','niah_single_3','niah_multikey_1','niah_multikey_2','niah_multikey_3','niah_multiquery','niah_multivalue','cwe','fwe','vt']
LABELS=['Single-1','Single-2','Single-3','MultiKey-1','MultiKey-2','MultiKey-3','MultiQuery','MultiValue','CWE','FWE','VT','AVG']
PAIRS=[('8B','e147c_r32k_formal.json','e116e_ruler32_formal_fullkv.json'),('32B','e143a_r32k_tli_formal.json','e143a_r32k_fullkv_formal.json')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
ZH=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size=8)
def get(n):
 x=json.loads((D/'inputs'/n).read_text());assert len(x['scores'])==1
 s=next(iter(x['scores'].values()));assert set(s)==set(TASKS)
 assert not x['incomplete_cells'];assert set(next(iter(x['n'].values())).values())=={100}
 return [s[t] for t in TASKS]
data=[]
for model,m,b in PAIRS:
 a,c=get(m),get(b);data.append((model,a+[statistics.mean(a)],c+[statistics.mean(c)]))
with (D/'scores.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['model','task','method_reported_score','FullKV_reported_score','difference_score_points'])
 for model,a,b in data:
  for task,x,y in zip(TASKS+['AVG'],a,b):w.writerow([model,task,x,y,x-y])
for lang in ['en','zh']:
 plt.rcParams['pdf.fonttype']=3 if lang=='zh' else 42  # CFF TTC must not be embedded as Type42.
 fp=ZH if lang=='zh' else None
 fig,axs=plt.subplots(1,2,figsize=(7.1,4.5))
 fig.subplots_adjust(left=.13,right=.90,bottom=.25,top=.76,wspace=.73)
 for ax,(model,a,b) in zip(axs,data):
  yy=list(range(12));ax.axhspan(10.5,11.5,color='#edf0f2',zorder=0)
  ax.scatter(b,[y-.10 for y in yy],s=22,facecolors='white',edgecolors='#272727',linewidth=.9,marker='o',zorder=4,label='FullKV')
  ax.scatter(a,[y+.10 for y in yy],s=21,color='#176e9e',marker='D',zorder=5,label='Method configuration' if lang=='en' else '本方法配置')
  ax.set_xlim(-3,103);ax.set_ylim(11.6,-.6);ax.set_yticks(yy,LABELS);ax.set_xticks([0,25,50,75,100]);ax.tick_params(axis='both',length=2,pad=3)
  ax.grid(axis='x',color='#d5d5d5',linewidth=.5);ax.set_axisbelow(True)
  for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
  ax.set_xlabel('Reported task score (0–100)' if lang=='en' else '报告的任务得分（0–100）',fontproperties=fp,labelpad=5)
  ax.set_title('Qwen3-'+model,fontsize=10,fontweight='bold',pad=32)
  sub=('Historical comparison;\ndifferent scorers' if model=='8B' else 'Matched manifests;\npredictions not replayed') if lang=='en' else ('历史对照，评分器不一致' if model=='8B' else '清单匹配；预测未独立重放')
  ax.text(.5,1.055,sub,transform=ax.transAxes,ha='center',fontproperties=fp,fontsize=7.5,color='#555555')
  ax.text(1.10,1.02,'Δ pts' if lang=='en' else '差值',transform=ax.transAxes,ha='center',fontproperties=fp,fontsize=7.5)
  for y,x,z in zip(yy,a,b):ax.text(1.10,y,f'{x-z:+.2f}',transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=7.5,fontweight='bold' if y==11 else 'normal')
  ax.text(.5,-.21,f'AVG: {a[-1]:.2f} vs {b[-1]:.2f}',transform=ax.transAxes,ha='center',fontsize=8.2,fontweight='bold')
 handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.99),ncol=2,frameon=False,prop=fp)
 foot=('RULER 32K · 11 tasks × 100 examples/arm · author-reported aggregates\nΔ = method − FullKV; no paired confidence intervals or non-inferiority claim.') if lang=='en' else ('RULER 32K · 11 个任务，每任务每组 100 例 · 作者报告的汇总结果\n差值＝本方法－FullKV；尚无配对置信区间，不支持精度无损结论。')
 fig.text(.5,.055,foot,ha='center',va='center',fontproperties=fp,fontsize=7.5,linespacing=1.5,color='#444444')
 for ext in ['pdf','svg','png']:fig.savefig(D/f'ruler32k_comparison_{lang}.{ext}',dpi=240)
 plt.close(fig)
 Image.open(D/f'ruler32k_comparison_{lang}.png').convert('L').save(D/f'ruler32k_comparison_{lang}_gray.png')
# Exact source-file hashes and arithmetic assertions, independent of rounding used in plots.
summary={'source_revision':'b5a97b314119006dc7e1624bf60268ff2501e44d','measurements':'author-reported aggregates; local arithmetic only','models':{},'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (D/'inputs').glob('*.json')}}
for model,a,b in data:
 ds=[x-y for x,y in zip(a,b)];summary['models'][model]={'method_avg':a[-1],'fullkv_avg':b[-1],'delta_avg':ds[-1],'negative_tasks':{t:d for t,d in zip(TASKS,ds) if d<0},'positive_tasks':{t:d for t,d in zip(TASKS,ds) if d>0},'ties':[t for t,d in zip(TASKS,ds) if d==0]}
assert abs(summary['models']['8B']['delta_avg']+1/11)<1e-10
assert abs(summary['models']['32B']['delta_avg']+17.33/11)<1e-10
(D/'arithmetic_check.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary['models'],indent=2))
# Compact AVG-only alternative for main-text placement; full task plot remains available.
for lang in ['en','zh']:
 plt.rcParams['pdf.fonttype']=3 if lang=='zh' else 42  # CFF TTC must not be embedded as Type42.
 fp=ZH if lang=='zh' else None
 fig,axs=plt.subplots(1,2,figsize=(7.1,2.1))
 fig.subplots_adjust(left=.08,right=.98,bottom=.36,top=.73,wspace=.30)
 for ax,(model,a,b) in zip(axs,data):
  ax.barh([1,0],[b[-1],a[-1]],height=.48,color=['white','#176e9e'],edgecolor=['#272727','#176e9e'],linewidth=.8)
  ax.set_yticks([1,0],['FullKV','Method' if lang=='en' else '本方法'],fontproperties=fp)
  ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x',color='#ddd',lw=.5);ax.set_axisbelow(True)
  ax.tick_params(length=2);ax.set_title('Qwen3-'+model,fontsize=9,fontweight='bold',pad=18)
  for y,z in [(1,b[-1]),(0,a[-1])]:ax.text(z+2,y,f'{z:.2f}',va='center',fontsize=8)
  for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
  txt=('Different scorers; historical comparison' if model=='8B' else 'Matched manifests; predictions not replayed') if lang=='en' else ('评分器不同；历史对照' if model=='8B' else '清单匹配；预测未独立重放')
  ax.text(.5,1.045,txt,transform=ax.transAxes,ha='center',fontsize=7.5,fontproperties=fp)
  ax.set_xlabel('11-task AVG (0–100)' if lang=='en' else '11 任务 AVG（0–100）',fontsize=8,fontproperties=fp)
 footer='RULER 32K · author-reported aggregates · no non-inferiority claim' if lang=='en' else 'RULER 32K · 作者报告汇总 · 尚不支持精度无损结论'
 fig.text(.5,.055,footer,ha='center',fontsize=7.5,fontproperties=fp)
 for ext in ['pdf','svg','png']:fig.savefig(D/f'ruler32k_avg_compact_{lang}.{ext}',dpi=240)
 plt.close(fig)
 Image.open(D/f'ruler32k_avg_compact_{lang}.png').convert('L').save(D/f'ruler32k_avg_compact_{lang}_gray.png')
