#!/usr/bin/env python3
"""Reproduce descriptive RULER charts from fixed-revision author summaries. No raw replay.

2026-10-23: T6 absolute-quota champion (np32_ntoff_nl4096, i.e. nb32 / single-pool free
competition / nl4096 / 4-page & 128-token guards / K1024 / B64 / K1=128) added at 32K/64K/128K,
plus a third 128K panel. The 23.98 ratio-interface (k_f=0) number is an author-reported
aggregate (S6 disclosure); no per-task JSON exists for it, so it is plotted as an AVG-level
bar/marker only and never as per-task points.

Layout notes (glyph-gate driven, 2026-10-23): both languages embed Type42; pymupdf reports
DejaVu Type42 char boxes at 1.70em and Noto Type42 CJK char boxes at 1.45em of the font size.
Vertical pitches/anchors below were sized so adjacent text boxes stay separated under those
metrics: single-line subtitles/footers, 7.5pt ticks, figsize 7.1x4.7 (full) / 7.1x2.6
(compact), Delta column at axes fraction 1.12, AVG line at -0.33, legend rows via
labelspacing. The earlier zh Type3 path (2.86em inflated boxes, 17/21 hard pt^2-gate pairs)
is retired.
"""
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
PAIRS=[('8B','e147c_r32k_formal.json','e116e_ruler32_formal_fullkv.json')]
TRIPLE64=('e147h_r64k_formal.json','e146c_r64k_formal.json','e119_ruler64k_formal_fullkv.json')
CHAMP='np32_ntoff_nl4096'
RATIO128=23.98  # author-reported aggregate (S6): ratio interface, k_f=0, nominal 128K, YaRN 4, alpha.125/beta.375/gamma.625; no per-task JSON in repo.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
ZH=FontProperties(fname=str(D.parent.parent/'fonts/noto/NotoSansCJKsc-Regular.otf'),size=7.5)
def get(n,need=None):
 x=json.loads((D/'inputs'/n).read_text());arms=[k for k in x['scores'] if need is None or need in k];assert len(arms)==1
 s=x['scores'][arms[0]];assert set(s)==set(TASKS)
 assert not x['incomplete_cells'];assert set(x['n'][arms[0]].values())=={100}
 return [s[t] for t in TASKS],arms[0]
data=[]
for model,m,b in PAIRS:
 a,_=get(m);c,_=get(b);a,c=a+[statistics.mean(a)],c+[statistics.mean(c)];data.append((model,a,c))
# 64K quadruple: beta=.0625 (e147h) vs beta=.125 (e146c) vs reused FullKV anchor (e119) vs T6 champion; e147h/e146c share one machine, T6 is a separate run.
a64,_=get(TRIPLE64[0]);b64,_=get(TRIPLE64[1]);c64f,_=get(TRIPLE64[2]);c64,_=get('t6_ruler_q8b_L65536_champ.json',CHAMP)
t64=tuple(v+[statistics.mean(v)] for v in (a64,b64,c64f,c64))
# 32K T6 champion vs e116e anchor (different scorers, historical comparison).
q32,arm32q=get('e147c_r32k_formal.json');f32,_=get('e116e_ruler32_formal_fullkv.json');c32,arm32c=get('t6_ruler_q8b_L32768_champ.json',CHAMP)
d32=tuple(v+[statistics.mean(v)] for v in (f32,q32,c32))
# 128K: same-machine beta reruns (e147h/e146c), dense anchor from a separate run (e119), T6 champion, plus author-reported ratio interface AVG.
b1280,_=get('e147h_r128k_formal.json');b1281,_=get('e146c_r128k_formal.json');f128,_=get('e119_ruler128k_formal_fullkv.json');c128,_=get('t6_ruler_q8b_L131072_champ.json',CHAMP)
d128=tuple(v+[statistics.mean(v)] for v in (f128,b1280,b1281,c128))
with (D/'scores.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['model','task','method_reported_score','FullKV_reported_score','difference_score_points'])
 for model,a,b in data:
  for task,x,y in zip(TASKS+['AVG'],a,b):w.writerow([model,task,x,y,x-y])
 for arm,a,b in [('8B-64K-beta.0625',t64[0],t64[2]),('8B-64K-beta.125',t64[1],t64[2]),
                 ('8B-32K-T6-champ',d32[2],d32[0]),('8B-64K-T6-champ',t64[3],t64[2]),
                 ('8B-128K-T6-champ',d128[3],d128[0]),('8B-128K-beta.0625',d128[1],d128[0]),('8B-128K-beta.125',d128[2],d128[0])]:
  for task,x,y in zip(TASKS+['AVG'],a,b):w.writerow([arm,task,x,y,x-y])
 w.writerow(['8B-128K-ratio-kf0-author-reported','AVG',RATIO128,d128[0][-1],RATIO128-d128[0][-1]])
for lang in ['en','zh']:
 # mpl 3.11 PDF glyph-subset tracker reads ps.fonttype, not pdf.fonttype; set both.
 # mpl 3.11 PDF glyph-subset tracker reads ps.fonttype, not pdf.fonttype; CJK needs both synced at 42.
 # Type3 CJK char boxes render at 2.86em in pymupdf and fail the pt^2 glyph gate on every tight row;
 # Type42 (as in fig_o_observation.py) embeds OTF subsets with normal bboxes and passes.
 plt.rcParams['pdf.fonttype']=plt.rcParams['ps.fonttype']=42
 fp=ZH if lang=='zh' else None;en=lang=='en'
 fig,axs=plt.subplots(1,3,figsize=(7.1,4.7))
 fig.subplots_adjust(left=.09,right=.925,bottom=.28,top=.76,wspace=.82)
 # ---- panel 1: 32K, three arms ----
 ax=axs[0];F,Q,C=d32
 yy=list(range(12));ax.axhspan(10.5,11.5,color='#edf0f2',zorder=0)
 ax.scatter(F,[y-.14 for y in yy],s=22,facecolors='white',edgecolors='#272727',linewidth=.9,marker='o',zorder=4,label='FullKV (dense anchor)' if en else 'FullKV（稠密锚）')
 ax.scatter(Q,[y for y in yy],s=21,color='#176e9e',marker='D',zorder=5,label='quota (8,320)' if en else '配额 (8,320)')
 ax.scatter(C,[y+.14 for y in yy],s=21,color='#0d5580',marker='s',zorder=6,label='absolute champ (this run)' if en else '绝对配额冠军（本次运行）')
 ax.set_xlim(-3,103);ax.set_ylim(11.6,-.6);ax.set_yticks(yy,LABELS);ax.set_xticks([0,25,50,75,100]);ax.tick_params(axis='both',length=2,pad=3,labelsize=7.5)
 ax.grid(axis='x',color='#d5d5d5',linewidth=.5);ax.set_axisbelow(True)
 for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
 ax.set_xlabel('Reported task score (0–100)' if en else '报告的任务得分（0–100）',fontproperties=fp,labelpad=7,fontsize=7.5)
 ax.set_title('Qwen3-8B · 32K',fontsize=10,fontweight='bold',pad=30)
 ax.text(.5,1.06,'Different scorers' if en else '历史对照，评分器不一致',transform=ax.transAxes,ha='center',va='bottom',fontproperties=fp,fontsize=7.5,color='#555555')
 ax.text(1.12,1.02,'Δ pts' if en else '差值',transform=ax.transAxes,ha='center',fontproperties=fp,fontsize=7.5)
 for y,x,z in zip(yy,Q,F):ax.text(1.12,y,f'{x-z:+.2f}',transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=7.5,fontweight='bold' if y==11 else 'normal')
 ax.text(.5,-.30,f'AVG: {F[-1]:.2f} / {Q[-1]:.2f} / {C[-1]:.2f}',transform=ax.transAxes,ha='center',fontsize=7.5,fontweight='bold')
 # ---- panel 2: 64K, four arms ----
 ax=axs[1];F,B1,B0,C=t64
 yy=list(range(12));ax.axhspan(10.5,11.5,color='#edf0f2',zorder=0)
 ax.scatter(F,[y-.21 for y in yy],s=22,facecolors='white',edgecolors='#272727',linewidth=.9,marker='o',zorder=4,label='FullKV (dense anchor)' if en else 'FullKV（稠密锚）')
 ax.scatter(B1,[y-.07 for y in yy],s=21,facecolors='white',edgecolors='#8fb8d4',linewidth=1.0,marker='D',zorder=5,label='β=.125')
 ax.scatter(B0,[y+.07 for y in yy],s=21,color='#176e9e',marker='D',zorder=6,label='β=.0625')
 ax.scatter(C,[y+.21 for y in yy],s=21,color='#0d5580',marker='s',zorder=7,label='absolute champ (this run)' if en else '绝对配额冠军（本次运行）')
 ax.set_xlim(-3,103);ax.set_ylim(11.6,-.6);ax.set_yticks(yy,LABELS);ax.set_xticks([0,25,50,75,100]);ax.tick_params(axis='both',length=2,pad=3,labelsize=7.5)
 ax.tick_params(axis='y',labelleft=False)
 ax.grid(axis='x',color='#d5d5d5',linewidth=.5);ax.set_axisbelow(True)
 for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
 ax.set_xlabel('Reported task score (0–100)' if en else '报告的任务得分（0–100）',fontproperties=fp,labelpad=7,fontsize=7.5)
 ax.set_title('Qwen3-8B · 64K',fontsize=10,fontweight='bold',pad=30)
 ax.text(.5,1.06,'Sparse arms across runs' if en else '稀疏臂跨多次运行',transform=ax.transAxes,ha='center',va='bottom',fontproperties=fp,fontsize=7.5,color='#555555')
 ax.text(1.12,1.02,'Δ pts' if en else '差值',transform=ax.transAxes,ha='center',fontproperties=fp,fontsize=7.5)
 for y,x,z in zip(yy,B0,F):ax.text(1.12,y,f'{x-z:+.2f}',transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=7.5,fontweight='bold' if y==11 else 'normal')
 ax.text(.5,-.30,f'AVG: {F[-1]:.2f} / {B1[-1]:.2f} / {B0[-1]:.2f} / {C[-1]:.2f}',transform=ax.transAxes,ha='center',fontsize=7.5,fontweight='bold')
 # ---- panel 3: 128K, FullKV anchor vs champion, plus author-reported ratio-interface AVG marker ----
 ax=axs[2];F,_,_,C=d128
 yy=list(range(12));ax.axhspan(10.5,11.5,color='#edf0f2',zorder=0)
 ax.scatter(F,[y-.10 for y in yy],s=22,facecolors='white',edgecolors='#272727',linewidth=.9,marker='o',zorder=4,label='FullKV (dense anchor)' if en else 'FullKV（稠密锚）')
 ax.scatter(C,[y+.10 for y in yy],s=21,color='#0d5580',marker='s',zorder=6,label='absolute champ (this run)' if en else '绝对配额冠军（本次运行）')
 ax.plot([21.5,26.5],[11,11],color='#8a8a8a',lw=2.4,solid_capstyle='butt',zorder=5,label='ratio k_f=0 (author-reported)' if en else '比例接口 k_f=0（作者报告）')
 ax.text(24,10.42,'23.98 (k_f=0)',ha='center',va='center',fontsize=7.5,color='#666666')
 ax.set_xlim(-3,103);ax.set_ylim(11.6,-.6);ax.set_yticks(yy,LABELS);ax.set_xticks([0,25,50,75,100]);ax.tick_params(axis='both',length=2,pad=3,labelsize=7.5)
 ax.tick_params(axis='y',labelleft=False)
 ax.grid(axis='x',color='#d5d5d5',linewidth=.5);ax.set_axisbelow(True)
 for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
 ax.set_xlabel('Reported task score (0–100)' if en else '报告的任务得分（0–100）',fontproperties=fp,labelpad=7,fontsize=7.5)
 ax.set_title('Qwen3-8B · 128K',fontsize=10,fontweight='bold',pad=30)
 ax.text(.5,1.06,'Anchors separate runs' if en else '冠军单次运行；锚独立运行',transform=ax.transAxes,ha='center',va='bottom',fontproperties=fp,fontsize=7.5,color='#555555')
 ax.text(1.12,1.02,'Δ pts' if en else '差值',transform=ax.transAxes,ha='center',fontproperties=fp,fontsize=7.5)
 for y,x,z in zip(yy,C,F):ax.text(1.12,y,f'{x-z:+.2f}',transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=7.5,fontweight='bold' if y==11 else 'normal')
 ax.text(.5,-.30,f'AVG: {F[-1]:.2f} / {C[-1]:.2f}',transform=ax.transAxes,ha='center',fontsize=7.5,fontweight='bold')
 pairs=[]
 for ax_ in axs:
  pairs+=list(zip(*ax_.get_legend_handles_labels()))
 seen=set();hs=[];ls=[]
 for h,l in pairs:
  if l not in seen:seen.add(l);hs.append(h);ls.append(l)
 fig.legend(hs,ls,loc='upper center',bbox_to_anchor=(.507,1.0),ncol=3,frameon=False,prop=fp,labelspacing=.8)
 if en:
  foot='RULER 32K/64K/128K · 11 tasks × 100/arm · author-reported aggregates + this-run champion; 23.98 author-reported (S6)\nΔ per panel vs FullKV: 32K quota (8,320), 64K β=.0625, 128K champion; no paired CI or non-inferiority claim'
  fig.text(.5,.065,foot,ha='center',va='baseline',fontsize=7.5,linespacing=1.5,color='#444444')
 else:
  foot='RULER 32K/64K/128K · 11 任务 × 每任务 100 例 · 作者报告汇总 + 本次冠军 · 23.98 作者报告（S6）· 无配对区间，不支持非劣结论'
  fig.text(.5,.059,foot,ha='center',va='baseline',fontproperties=fp,fontsize=7.5,color='#444444')
 for ext in ['pdf','svg','png']:fig.savefig(D/f'ruler32k_comparison_{lang}.{ext}',dpi=240)
 plt.close(fig)
 Image.open(D/f'ruler32k_comparison_{lang}.png').convert('L').save(D/f'ruler32k_comparison_{lang}_gray.png')
# Exact source-file hashes and arithmetic assertions, independent of rounding used in plots.
summary={'source_revision':'b5a97b314119006dc7e1624bf60268ff2501e44d + 51b4679c52feebc5327775c660db41550b1c8782 (64K) + T6 local runs (uncommitted exp/) 2026-10-23','measurements':'author-reported aggregates + this-run T6 champion runs; local arithmetic only','models':{},'t6_receipts':{'L32768':{'run_id':'20261023003349-2837950-5569','bs':8,'yarn':'off'},'L65536':{'run_id':'20261023005422-2845695-2513','bs':2,'yarn_factor':2.0},'L131072':{'run_id':'20261023005422-2845702-1841','bs':2,'yarn_factor_effective':4.0}},'ratio_interface_128k':{'avg':RATIO128,'provenance':'author-reported aggregate (S6): nominal 128K, YaRN factor 4, alpha=.125/beta=.375/gamma=.625, k_f=0; no per-task JSON in repo'},'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (D/'inputs').glob('*.json') if not p.name.startswith('provenance')}}
for model,a,b in data:
 ds=[x-y for x,y in zip(a,b)];summary['models'][model]={'method_avg':a[-1],'fullkv_avg':b[-1],'delta_avg':ds[-1],'negative_tasks':{t:d for t,d in zip(TASKS,ds) if d<0},'positive_tasks':{t:d for t,d in zip(TASKS,ds) if d>0},'ties':[t for t,d in zip(TASKS,ds) if d==0]}
a,b,c,f=t64
summary['models']['8B-64K']={'beta0625_avg':a[-1],'beta125_avg':b[-1],'fullkv_anchor_avg':c[-1],'t6_champ_avg':f[-1],'delta_beta0625_vs_beta125':a[-1]-b[-1],'delta_beta0625_vs_fullkv':a[-1]-c[-1],'delta_beta125_vs_fullkv':b[-1]-c[-1],'delta_t6_champ_vs_fullkv':f[-1]-c[-1]}
f32v,q32v,c32v=d32
summary['models']['8B-32K-T6-champ']={'fullkv_anchor_avg':f32v[-1],'quota_8_320_avg':q32v[-1],'t6_champ_avg':c32v[-1],'delta_t6_champ_vs_fullkv':c32v[-1]-f32v[-1],'delta_quota_vs_fullkv':q32v[-1]-f32v[-1]}
f128v,b0625,b125,c128v=d128
summary['models']['8B-128K']={'fullkv_anchor_avg':f128v[-1],'beta0625_avg':b0625[-1],'beta125_avg':b125[-1],'t6_champ_avg':c128v[-1],'ratio_kf0_author_reported_avg':RATIO128,'delta_t6_champ_vs_fullkv':c128v[-1]-f128v[-1],'delta_beta0625_vs_fullkv':b0625[-1]-f128v[-1],'delta_beta125_vs_fullkv':b125[-1]-f128v[-1],'delta_ratio_vs_fullkv':RATIO128-f128v[-1]}
# Historical assertions (unchanged semantics).
assert abs(summary['models']['8B']['delta_avg']+1/11)<1e-10
assert abs(summary['models']['8B-64K']['delta_beta0625_vs_beta125']-(48.85-48.04545454545455))<1e-6
assert abs(summary['models']['8B-64K']['delta_beta0625_vs_fullkv']-(48.85-48.53636363636364))<1e-6
# T6 fail-closed assertions: exact per-task sums, champion arm identity, rounded AVG anchors (9).
assert abs(sum(c32)-649.23)<1e-9 and abs(sum(c64)-525.8)<1e-9 and abs(sum(c128)-514.98)<1e-9
assert CHAMP in arm32c and CHAMP in get('t6_ruler_q8b_L65536_champ.json',CHAMP)[1] and CHAMP in get('t6_ruler_q8b_L131072_champ.json',CHAMP)[1]
assert abs(c32v[-1]-59.02)<5e-3 and abs(t64[3][-1]-47.8)<5e-3 and abs(c128v[-1]-46.82)<5e-3  # rounded display anchors; exact sums asserted above
# 128K historical references: same-machine beta reruns and separate-run dense anchor (3).
assert abs(sum(b1280)-502.98)<1e-9 and abs(sum(b1281)-506.22)<1e-9 and abs(sum(f128)-510.75)<1e-9
assert abs(b0625[-1]-45.73)<5e-3 and abs(b125[-1]-46.02)<5e-3 and abs(f128v[-1]-46.43)<5e-3
(D/'arithmetic_check.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary['models'],indent=2))
# Compact AVG-only alternative for main-text placement; full task plot remains available.
for lang in ['en','zh']:
 plt.rcParams['pdf.fonttype']=plt.rcParams['ps.fonttype']=42  # keep in sync with the loop above.
 fp=ZH if lang=='zh' else None;en=lang=='en'
 fig,axs=plt.subplots(1,3,figsize=(7.1,2.6))
 fig.subplots_adjust(left=.13,right=.985,bottom=.33,top=.66,wspace=.64)
 # ---- compact panel 1: 32K, three bars ----
 ax=axs[0];F,Q,C=d32
 ax.barh([2,1,0],[F[-1],Q[-1],C[-1]],height=.5,color=['white','#a9cadd','#0d5580'],edgecolor=['#272727','#5b8db4','#0d5580'],linewidth=.8)
 ax.set_yticks([2,1,0],['FullKV','quota (8,320)','absolute champ'] if en else ['FullKV','配额 (8,320)','绝对冠军'],fontproperties=fp)
 ax.set_xlim(0,100);ax.set_ylim(-.6,2.6);ax.set_xticks([0,25,50,75,100]);ax.tick_params(length=2,labelsize=7.5);ax.grid(axis='x',color='#ddd',lw=.5);ax.set_axisbelow(True)
 ax.set_title('Qwen3-8B · 32K',fontsize=9,fontweight='bold',pad=30)
 for y,z in [(2,F[-1]),(1,Q[-1]),(0,C[-1])]:ax.text(z+2,y,f'{z:.2f}',va='center',fontsize=7.5)
 for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
 ax.text(.5,1.06,'Different scorers; champion: this run' if en else '评分器不同；冠军为本次运行',transform=ax.transAxes,ha='center',va='bottom',fontproperties=fp,fontsize=7.5)
 ax.set_xlabel('11-task AVG (0–100)' if en else '11 任务 AVG（0–100）',fontproperties=fp,fontsize=7.5,labelpad=7)
 # ---- compact panel 2: 64K, four bars ----
 ax=axs[1];F,B1,B0,C=t64
 ax.barh([3,2,1,0],[F[-1],B1[-1],B0[-1],C[-1]],height=.58,color=['white','#a9cadd','#176e9e','#0d5580'],edgecolor=['#272727','#5b8db4','#176e9e','#0d5580'],linewidth=.8)
 ax.set_yticks([3,2,1,0],['FullKV','β=.125','β=.0625','absolute champ'] if en else ['FullKV','β=.125','β=.0625','绝对冠军'],fontproperties=fp)
 ax.set_xlim(0,100);ax.set_ylim(-.6,3.6);ax.set_xticks([0,25,50,75,100]);ax.tick_params(length=2,labelsize=7.5);ax.grid(axis='x',color='#ddd',lw=.5);ax.set_axisbelow(True)
 ax.set_title('Qwen3-8B · 64K',fontsize=9,fontweight='bold',pad=30)
 for y,z in [(3,F[-1]),(2,B1[-1]),(1,B0[-1]),(0,C[-1])]:ax.text(z+2,y,f'{z:.2f}',va='center',fontsize=7.5)
 for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
 ax.text(.5,1.06,'Sparse arms across runs; anchor separate' if en else '稀疏臂跨多次运行；锚独立运行',transform=ax.transAxes,ha='center',va='bottom',fontproperties=fp,fontsize=7.5)
 ax.set_xlabel('11-task AVG (0–100)' if en else '11 任务 AVG（0–100）',fontproperties=fp,fontsize=7.5,labelpad=7)
 # ---- compact panel 3: 128K, three bars incl. hatched author-reported ratio interface ----
 ax=axs[2];F,_,_,C=d128
 ax.barh([2,1,0],[RATIO128,F[-1],C[-1]],height=.5,color=['#e8e8e8','white','#0d5580'],edgecolor=['#8a8a8a','#272727','#0d5580'],linewidth=.8,hatch=['///','',''])
 ax.set_yticks([2,1,0],['FullKV (anchor)','ratio k_f=0','absolute champ'] if en else ['FullKV（锚）','比例接口 k_f=0','绝对冠军'],fontproperties=fp)
 ax.set_xlim(0,100);ax.set_ylim(-.6,2.6);ax.set_xticks([0,25,50,75,100]);ax.tick_params(length=2,labelsize=7.5);ax.grid(axis='x',color='#ddd',lw=.5);ax.set_axisbelow(True)
 ax.set_title('Qwen3-8B · 128K',fontsize=9,fontweight='bold',pad=30)
 for y,z in [(2,RATIO128),(1,F[-1]),(0,C[-1])]:ax.text(z+2,y,f'{z:.2f}',va='center',fontsize=7.5)
 for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
 ax.text(.5,1.06,'Champion single run; no CI' if en else '冠军单次运行；无区间',transform=ax.transAxes,ha='center',va='bottom',fontproperties=fp,fontsize=7.5)
 ax.set_xlabel('11-task AVG (0–100)' if en else '11 任务 AVG（0–100）',fontproperties=fp,fontsize=7.5,labelpad=7)
 if en:
  footer='RULER 32K/64K/128K · author-reported aggregates + this-run champion; 23.98 author-reported (S6)\nno paired CI or non-inferiority claim'
  fig.text(.5,.060,footer,ha='center',va='baseline',fontsize=7.5,linespacing=1.5,color='#444444')
 else:
  footer='RULER 32K/64K/128K · 作者报告汇总 + 本次运行冠军 · 23.98 为作者报告（S6）· 无配对区间，不支持非劣结论'
  fig.text(.5,.058,footer,ha='center',va='baseline',fontproperties=fp,fontsize=7.5,color='#444444')
 for ext in ['pdf','svg','png']:fig.savefig(D/f'ruler32k_avg_compact_{lang}.{ext}',dpi=240)
 plt.close(fig)
 Image.open(D/f'ruler32k_avg_compact_{lang}.png').convert('L').save(D/f'ruler32k_avg_compact_{lang}_gray.png')
