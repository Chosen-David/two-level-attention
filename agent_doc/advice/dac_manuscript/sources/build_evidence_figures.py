"""Replot archived descriptive evidence; no model execution. See evidence_bindings.json."""
from pathlib import Path
import json, statistics, hashlib, os, subprocess
from matplotlib.text import Text
from matplotlib.font_manager import FontProperties
LANG=os.environ.get("DAC_FIG_LANG","en")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from observation_bins import diagnose
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'sources/data'; OUT=ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
d=json.loads((DATA/'observation_plot_input.json').read_text())
rows=[(c,o,diagnose(o)) for c in d['cells'] for o in c['o1']]
median=statistics.median(r['far_share'] for c,o,r in rows)
c,o,r=min((x for x in rows if x[1]['q_pos']>9000),key=lambda x:abs(x[2]['far_share']-median))
far=statistics.mean(x[2]['far_capture'] for x in rows);near=statistics.mean(x[2]['near_capture'] for x in rows)
TRANSLATE={'near':'近区','far':'远区','Far':'远区','Near':'近区','Query–key distance (tokens)':'查询—键距离（词元）','Mass / valid token':'注意力概率质量\n／有效词元','(a) Dense-reference distance density':'(a) 稠密参考距离密度','Selected mass / regional mass':'选中概率质量／区域概率质量','(b) Feasible whole-bin diagnostic':'(b) 可行整箱选择诊断','Dense':'稠密','Archived sparse configuration':'历史稀疏配置','Input length (tokens)':'输入长度（词元）','Diagnostic TPOT (ms)':'诊断TPOT（毫秒）','Macro-average task score (128K)':'宏平均任务分（128K）','Tail32':'低频32','Random32 (unpaired)':'随机32（未配对）','High32':'高频32','Selected / reference-mid-TopK mass':'选中／参考mid-TopK质量'}
def save(fig,name):
 if LANG=='zh':
  name += '_zh'
  for ax in fig.axes:
   xlim,ylim=ax.get_xlim(),ax.get_ylim()
   ax.set_xticks(ax.get_xticks(),[TRANSLATE.get(t.get_text(),t.get_text()) for t in ax.get_xticklabels()])
   ax.set_yticks(ax.get_yticks(),[TRANSLATE.get(t.get_text(),t.get_text()) for t in ax.get_yticklabels()])
   ax.set_xlim(xlim);ax.set_ylim(ylim)
  for t in fig.findobj(Text):
   t.set_text(TRANSLATE.get(t.get_text(),t.get_text()));t.set_fontproperties(FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size=t.get_fontsize()))
 for ext in ['svg','png']:fig.savefig(OUT/(name+'.'+ext),dpi=220)
 subprocess.run(['inkscape',str(OUT/(name+'.svg')),'--export-type=pdf','--export-filename='+str(OUT/(name+'.pdf'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(3.35,2.75),layout='constrained')
a=axs[0]
for region,color in [('near','#007C91'),('far','#C25E24')]:
 bs=[b for b in r['bins'] if b['region']==region]
 for i,b in enumerate(bs):
  lo=o['q_pos']-b['end']+1;hi=o['q_pos']-b['start']+1
  a.plot([lo,hi],[b['mass']/b['n_tok']]*2,color=color,lw=.8,label=region if i==0 else None)
a.set_yscale('log');a.set_xlabel('Query–key distance (tokens)');a.set_ylabel('Mass / valid token');a.set_title('(a) Dense-reference distance density',loc='left');a.legend(frameon=False,ncol=2,loc='upper right')
a=axs[1];a.barh(['Far','Near'],[far,near],color=['#C25E24','#007C91'],height=.5)
a.set_xlim(0,1.15);a.set_xticks([0,.5,1]);a.set_xlabel('Selected mass / regional mass');a.set_title('(b) Feasible whole-bin diagnostic',loc='left')
for i,v in enumerate([far,near]):a.text(v+.02,i,f'{v:.4f}',va='center',fontsize=8)
save(fig,'observation_current')
x=json.loads((DATA/'decode_tpot_plot_input.json').read_text())['results_median']
fig,a=plt.subplots(figsize=(3.35,2.15),layout='constrained')
for arm,label,color,marker in [('dense_dynamic','Dense','#65737D','o'),('tli_dynamic','Archived sparse configuration','#007C91','^')]:
 vals=[x[f'L{L}'][arm]['tpot_ms'] for L in [32768,65536,131072]]
 a.plot([32,64,128],vals,label=label,color=color,marker=marker,lw=1.4)
 for xx,v in zip([32,64,128],vals):a.annotate(f'{v:.1f}',(xx,v),xytext=(0,6),textcoords='offset points',ha='center',fontsize=7)
a.set_xticks([32,64,128],['32K','64K','128K']);a.set_ylim(0,140);a.set_xlabel('Input length (tokens)');a.set_ylabel('Diagnostic TPOT (ms)');a.legend(loc='upper left',frameon=False);save(fig,'decode_tpot_current')
x=json.loads((DATA/'e146_r128k_formal.json').read_text())['scores']; vals=[statistics.mean(x['L131072/none'].values()),statistics.mean(x['L131072/tli_64_128_1024_c4_Aa0.125_b0.375_g0.625_hb4f1e46605'].values())]
fig,a=plt.subplots(figsize=(3.35,1.8),layout='constrained');a.barh(['Dense','Archived sparse configuration'],vals,color=['#65737D','#007C91'],height=.5);a.set_xlim(0,58);a.set_xlabel('Macro-average task score (128K)');a.invert_yaxis()
for i,v in enumerate(vals):a.text(v+1,i,f'{v:.2f}',va='center')
save(fig,'ruler128k_current')
vals=[statistics.mean(c['o3'][arm]['selected_mass_over_reference_topk_mass']['mean'] for c in d['cells']) for arm in ['tail32','random32','high32']]
fig,a=plt.subplots(figsize=(3.35,1.9),layout='constrained');a.barh(['Tail32','Random32 (unpaired)','High32'],vals,color=['#007C91','#83919A','#C25E24'],height=.55);a.invert_yaxis();a.set_xlim(0,1.1);a.set_xlabel('Selected / reference-mid-TopK mass')
for i,v in enumerate(vals):a.text(v+.02,i,f'{v:.3f}',va='center')
save(fig,'coordinate_diagnostic_current')
summary={'representative':{k:c[k] for k in ['task','sample','layer']},'q_pos':o['q_pos'],'n_source_sequences':len({(cell['task'],cell['sample']) for cell in d['cells']}),'n_query_layer_observations':len(rows),'far_capture':far,'near_capture':near,'selected_middle_tokens_range':[min(r['selected_tokens'] for c,o,r in rows),max(r['selected_tokens'] for c,o,r in rows)],'density_protected_mass':o['mass_protected_union'],'density_protected_tokens':o['region_bounds']['sink_hi']+o['region_bounds']['length']-max(o['region_bounds']['sink_hi'],o['region_bounds']['swa_lo'])}
(ROOT/'sources/data/figure_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
