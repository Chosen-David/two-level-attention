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
TRIPLE64=('e147h_r64k_formal.json','e146c_r64k_formal.json','e119_ruler64k_formal_fullkv.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
ZH=FontProperties(fname=str(D.parent.parent/'fonts/noto/NotoSansCJKsc-Regular.otf'),size=8)
def get(n):
 x=json.loads((D/'inputs'/n).read_text());assert len(x['scores'])==1
 s=next(iter(x['scores'].values()));assert set(s)==set(TASKS)
 assert not x['incomplete_cells'];assert set(next(iter(x['n'].values())).values())=={100}
 return [s[t] for t in TASKS]
data=[]
for model,m,b in PAIRS:
 a,c=get(m),get(b);data.append((model,a+[statistics.mean(a)],c+[statistics.mean(c)]))
# 64K triple: beta=.0625 (e147h) vs beta=.125 (e146c) vs reused FullKV anchor (e119); sparse arms share one machine.
a64,b64,c64=get(TRIPLE64[0]),get(TRIPLE64[1]),get(TRIPLE64[2])
t64=(a64+[statistics.mean(a64)],b64+[statistics.mean(b64)],c64+[statistics.mean(c64)])
with (D/'scores.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['model','task','method_reported_score','FullKV_reported_score','difference_score_points'])
 for model,a,b in data:
  for task,x,y in zip(TASKS+['AVG'],a,b):w.writerow([model,task,x,y,x-y])
 for arm,a,b in [('8B-64K-beta.0625',t64[0],t64[2]),('8B-64K-beta.125',t64[1],t64[2])]:
  for task,x,y in zip(TASKS+['AVG'],a,b):w.writerow([arm,task,x,y,x-y])
for lang in ['en','zh']:
 # mpl 3.11 PDF glyph-subset tracker reads ps.fonttype, not pdf.fonttype; set both.
 plt.rcParams['pdf.fonttype']=plt.rcParams['ps.fonttype']=3 if lang=='zh' else 42  # CFF OTF must not be embedded as Type42.
 fp=ZH if lang=='zh' else None
 fig,axs=plt.subplots(1,3,figsize=(7.1,4.5))
 fig.subplots_adjust(left=.12,right=.955,bottom=.25,top=.76,wspace=.58)
 for ax,(model,a,b) in zip(axs[:2],data):
  yy=list(range(12));ax.axhspan(10.5,11.5,color='#edf0f2',zorder=0)
  ax.scatter(b,[y-.10 for y in yy],s=22,facecolors='white',edgecolors='#272727',linewidth=.9,marker='o',zorder=4,label='FullKV')
  ax.scatter(a,[y+.10 for y in yy],s=21,color='#176e9e',marker='D',zorder=5,label='Method configuration' if lang=='en' else '本方法配置')
  ax.set_xlim(-3,103);ax.set_ylim(11.6,-.6);ax.set_yticks(yy,LABELS);ax.set_xticks([0,25,50,75,100]);ax.tick_params(axis='both',length=2,pad=3)
  if model=='32B':ax.tick_params(axis='y',labelleft=False)
  ax.grid(axis='x',color='#d5d5d5',linewidth=.5);ax.set_axisbelow(True)
  for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
  ax.set_xlabel('Reported task score (0–100)' if lang=='en' else '报告的任务得分（0–100）',fontproperties=fp,labelpad=5)
  ax.set_title(('Qwen3-'+model+' · 32K') if lang=='en' else ('Qwen3-'+model+' · 32K'),fontsize=10,fontweight='bold',pad=32)
  sub=('Historical comparison;\ndifferent scorers' if model=='8B' else 'Matched manifests;\npredictions not replayed') if lang=='en' else ('历史对照，评分器不一致' if model=='8B' else '清单匹配；预测未独立重放')
  ax.text(.5,1.055,sub,transform=ax.transAxes,ha='center',fontproperties=fp,fontsize=7.5,color='#555555')
  ax.text(1.10,1.02,'Δ pts' if lang=='en' else '差值',transform=ax.transAxes,ha='center',fontproperties=fp,fontsize=7.5)
  for y,x,z in zip(yy,a,b):ax.text(1.10,y,f'{x-z:+.2f}',transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=7.5,fontweight='bold' if y==11 else 'normal')
  ax.text(.5,-.21,f'AVG: {a[-1]:.2f} vs {b[-1]:.2f}',transform=ax.transAxes,ha='center',fontsize=8.2,fontweight='bold')
 ax=axs[2];a,b,c=t64
 yy=list(range(12));ax.axhspan(10.5,11.5,color='#edf0f2',zorder=0)
 ax.scatter(c,[y-.18 for y in yy],s=22,facecolors='white',edgecolors='#272727',linewidth=.9,marker='o',zorder=4,label='FullKV (anchor reused)' if lang=='en' else 'FullKV（锚复用）')
 ax.scatter(b,[y for y in yy],s=21,facecolors='white',edgecolors='#8fb8d4',linewidth=1.0,marker='D',zorder=5,label='β=.125' if lang=='en' else 'β=.125')
 ax.scatter(a,[y+.18 for y in yy],s=21,color='#176e9e',marker='D',zorder=6,label='β=.0625' if lang=='en' else 'β=.0625')
 ax.set_xlim(-3,103);ax.set_ylim(11.6,-.6);ax.set_yticks(yy,LABELS);ax.set_xticks([0,25,50,75,100]);ax.tick_params(axis='both',length=2,pad=3)
 ax.tick_params(axis='y',labelleft=False)
 ax.grid(axis='x',color='#d5d5d5',linewidth=.5);ax.set_axisbelow(True)
 for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
 ax.set_xlabel('Reported task score (0–100)' if lang=='en' else '报告的任务得分（0–100）',fontproperties=fp,labelpad=5)
 ax.set_title('Qwen3-8B · 64K',fontsize=10,fontweight='bold',pad=32)
 sub='Reported same-machine sparse arms;\nanchor from a separate run' if lang=='en' else '作者报告稀疏两臂同机；锚独立运行'
 ax.text(.5,1.055,sub,transform=ax.transAxes,ha='center',fontproperties=fp,fontsize=7.5,color='#555555')
 ax.text(1.10,1.02,'Δ pts' if lang=='en' else '差值',transform=ax.transAxes,ha='center',fontproperties=fp,fontsize=7.5)
 for y,x,z in zip(yy,a,c):ax.text(1.10,y,f'{x-z:+.2f}',transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=7.5,fontweight='bold' if y==11 else 'normal')
 ax.text(.5,-.21,f'AVG: {a[-1]:.2f} / {b[-1]:.2f} / {c[-1]:.2f}',transform=ax.transAxes,ha='center',fontsize=8.2,fontweight='bold')
 handles,labels=axs[0].get_legend_handles_labels();h2,l2=axs[2].get_legend_handles_labels();fig.legend(handles+h2,labels+l2,loc='upper center',bbox_to_anchor=(.53,.99),ncol=5,frameon=False,prop=fp)
 foot=('RULER 32K/64K · 11 tasks × 100 examples/arm · author-reported aggregates\nΔ = sparse − FullKV (64K: β=.0625); no paired CI or non-inferiority claim.') if lang=='en' else ('RULER 32K/64K · 11 个任务，每任务每组 100 例 · 作者报告的汇总结果\n差值＝稀疏－FullKV（64K取β=.0625）；无配对区间或非劣结论。')
 fig.text(.5,.055,foot,ha='center',va='center',fontproperties=fp,fontsize=7.5,linespacing=1.5,color='#444444')
 for ext in ['pdf','svg','png']:fig.savefig(D/f'ruler32k_comparison_{lang}.{ext}',dpi=240)
 plt.close(fig)
 Image.open(D/f'ruler32k_comparison_{lang}.png').convert('L').save(D/f'ruler32k_comparison_{lang}_gray.png')
# Exact source-file hashes and arithmetic assertions, independent of rounding used in plots.
summary={'source_revision':'b5a97b314119006dc7e1624bf60268ff2501e44d + 51b4679c52feebc5327775c660db41550b1c8782 (64K)','measurements':'author-reported aggregates; local arithmetic only','models':{},'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (D/'inputs').glob('*.json') if not p.name.startswith('provenance')}}
for model,a,b in data:
 ds=[x-y for x,y in zip(a,b)];summary['models'][model]={'method_avg':a[-1],'fullkv_avg':b[-1],'delta_avg':ds[-1],'negative_tasks':{t:d for t,d in zip(TASKS,ds) if d<0},'positive_tasks':{t:d for t,d in zip(TASKS,ds) if d>0},'ties':[t for t,d in zip(TASKS,ds) if d==0]}
a,b,c=t64
summary['models']['8B-64K']={'beta0625_avg':a[-1],'beta125_avg':b[-1],'fullkv_anchor_avg':c[-1],'delta_beta0625_vs_beta125':a[-1]-b[-1],'delta_beta0625_vs_fullkv':a[-1]-c[-1],'delta_beta125_vs_fullkv':b[-1]-c[-1]}
assert abs(summary['models']['8B']['delta_avg']+1/11)<1e-10
assert abs(summary['models']['32B']['delta_avg']+17.33/11)<1e-10
assert abs(summary['models']['8B-64K']['delta_beta0625_vs_beta125']-(48.85-48.04545454545455))<1e-6
assert abs(summary['models']['8B-64K']['delta_beta0625_vs_fullkv']-(48.85-48.53636363636364))<1e-6
(D/'arithmetic_check.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary['models'],indent=2))
# Compact AVG-only alternative for main-text placement; full task plot remains available.
for lang in ['en','zh']:
 plt.rcParams['pdf.fonttype']=plt.rcParams['ps.fonttype']=3 if lang=='zh' else 42  # keep in sync with the loop above.
 fp=ZH if lang=='zh' else None
 fig,axs=plt.subplots(1,3,figsize=(7.1,2.1))
 fig.subplots_adjust(left=.08,right=.98,bottom=.33,top=.68,wspace=.30)
 for ax,(model,a,b) in zip(axs[:2],data):
  ax.barh([1,0],[b[-1],a[-1]],height=.48,color=['white','#176e9e'],edgecolor=['#272727','#176e9e'],linewidth=.8)
  ax.set_yticks([1,0],['FullKV','Method' if lang=='en' else '本方法'],fontproperties=fp)
  ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x',color='#ddd',lw=.5);ax.set_axisbelow(True)
  ax.tick_params(length=2);ax.set_title(('Qwen3-'+model+' · 32K') if lang=='en' else ('Qwen3-'+model+' · 32K'),fontsize=9,fontweight='bold',pad=30)
  for y,z in [(1,b[-1]),(0,a[-1])]:ax.text(z+2,y,f'{z:.2f}',va='center',fontsize=8)
  for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
  txt=('Different scorers;\nhistorical comparison' if model=='8B' else 'Matched manifests;\npredictions not replayed') if lang=='en' else ('评分器不同；历史对照' if model=='8B' else '清单匹配；预测未独立重放')
  ax.text(.5,1.045,txt,transform=ax.transAxes,ha='center',fontsize=7.5,fontproperties=fp)
  ax.set_xlabel('11-task AVG (0–100)' if lang=='en' else '11 任务 AVG（0–100）',fontsize=8,fontproperties=fp)
 ax=axs[2];a,b,c=t64
 ax.barh([2,1,0],[c[-1],b[-1],a[-1]],height=.52,color=['white','#a9cadd','#176e9e'],edgecolor=['#272727','#5b8db4','#176e9e'],linewidth=.8)
 ax.set_yticks([2,1,0],['FullKV','β=.125','β=.0625'] if lang=='en' else ['FullKV','β=.125','β=.0625'],fontproperties=fp)
 ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x',color='#ddd',lw=.5);ax.set_axisbelow(True)
 ax.tick_params(length=2);ax.set_title('Qwen3-8B · 64K',fontsize=9,fontweight='bold',pad=30)
 for y,z in [(2,c[-1]),(1,b[-1]),(0,a[-1])]:ax.text(z+2,y,f'{z:.2f}',va='center',fontsize=8)
 for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
 txt='Reported same-host sparse arms;\nanchor reused' if lang=='en' else '作者报告同机；锚复用'
 ax.text(.5,1.045,txt,transform=ax.transAxes,ha='center',fontsize=7.5,fontproperties=fp)
 ax.set_xlabel('11-task AVG (0–100)' if lang=='en' else '11 任务 AVG（0–100）',fontsize=8,fontproperties=fp)
 footer='RULER 32K/64K · author-reported aggregates · no non-inferiority claim' if lang=='en' else 'RULER 32K/64K · 作者报告汇总 · 尚不支持精度无损结论'
 fig.text(.5,.055,footer,ha='center',fontsize=7.5,fontproperties=fp)
 for ext in ['pdf','svg','png']:fig.savefig(D/f'ruler32k_avg_compact_{lang}.{ext}',dpi=240)
 plt.close(fig)
 Image.open(D/f'ruler32k_avg_compact_{lang}.png').convert('L').save(D/f'ruler32k_avg_compact_{lang}_gray.png')
