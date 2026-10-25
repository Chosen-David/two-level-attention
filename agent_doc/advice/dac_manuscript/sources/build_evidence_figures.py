from paper_font_paths import cjk_regular, register_matplotlib
"""Replot archived descriptive evidence; no model execution. See evidence_bindings.json."""
from pathlib import Path
import json, statistics, hashlib, os, subprocess
from matplotlib.text import Text
from matplotlib.font_manager import FontProperties
LANG=os.environ.get("DAC_FIG_LANG","en")
import argparse
parser = argparse.ArgumentParser(description='Replot archived descriptive evidence without model execution.')
parser.add_argument('--only', choices=['all', 'decode-ruler'], default='all',
                    help='Limit output to the bilingual decode/RULER layout assets.')
ONLY = parser.parse_args().only
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
register_matplotlib()
from observation_bins import diagnose
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'sources/data'; OUT=ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':8,'axes.labelsize':8.5,'legend.fontsize':7.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
d=json.loads((DATA/'observation_plot_input.json').read_text())
rows=[(c,o,diagnose(o)) for c in d['cells'] for o in c['o1']]
median=statistics.median(r['far_share'] for c,o,r in rows)
c,o,r=min((x for x in rows if x[1]['q_pos']>9000),key=lambda x:abs(x[2]['far_share']-median))
far=statistics.mean(x[2]['far_capture'] for x in rows);near=statistics.mean(x[2]['near_capture'] for x in rows)
TRANSLATE={'near':'近区','far':'远区','Far':'远区','Near':'近区','Query–key distance (tokens)':'查询—键距离（词元）','Mass / valid token':'平均注意力权重\n（每有效词元）','(a) Dense-reference distance density':'(a) 稠密参考距离密度','Regional attention retention':'区域注意力保留率','(b) Feasible whole-bin diagnostic':'(b) 可行整箱选择诊断','Dense':'稠密','Dense FullKV':'稠密FullKV','Archived sparse configuration':'历史稀疏配置','Archived sparse\nconfiguration':'历史稀疏配置','LoDA-SA (CUDA Graph)':'LoDA-SA（整步CUDA Graph）','no graph':'无graph','128K mean task score':'宏平均任务分（128K）','Input length (tokens)':'输入长度（词元）','Diagnostic TPOT (ms)':'诊断TPOT（毫秒）','Decode TPOT (ms)':'解码TPOT（毫秒）','Macro-average task score (128K)':'宏平均任务分（128K）','Tail32':'低频32','Random32 (unpaired)':'随机32（未配对）','High32':'高频32','Selected / reference-mid-TopK mass':'选中／参考mid-TopK权重和'}
def save(fig,name):
 if LANG=='zh':
  name += '_zh'
  for ax in fig.axes:
   xlim,ylim=ax.get_xlim(),ax.get_ylim()
   ax.set_xticks(ax.get_xticks(),[TRANSLATE.get(t.get_text(),t.get_text()) for t in ax.get_xticklabels()])
   ax.set_yticks(ax.get_yticks(),[TRANSLATE.get(t.get_text(),t.get_text()) for t in ax.get_yticklabels()])
   ax.set_xlim(xlim);ax.set_ylim(ylim)
  for t in fig.findobj(Text):
   t.set_text(TRANSLATE.get(t.get_text(),t.get_text()));t.set_fontproperties(FontProperties(fname=cjk_regular(),size=t.get_fontsize()))
 for ext in ['svg','png']:fig.savefig(OUT/(name+'.'+ext),dpi=220)
 import shutil
 if shutil.which('inkscape'):
  subprocess.run(['inkscape',str(OUT/(name+'.svg')),'--export-type=pdf','--export-filename='+str(OUT/(name+'.pdf'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 else:
  import cairosvg;cairosvg.svg2pdf(url=str(OUT/(name+'.svg')),write_to=str(OUT/(name+'.pdf')))
 plt.close(fig)
# Figure1 uses the remote four-panel design, maintained in a single renderer.
if ONLY == 'all':
 subprocess.run([__import__('sys').executable,str(ROOT/'sources/build_remote_observation.py')],check=True)
tp=json.loads((DATA/'decode_tpot_plot_input.json').read_text());x=tp['results_mean_tpot_ms'];sp=tp['speedup_graph_vs_dense']
fig,a=plt.subplots(figsize=(3.35,2.15),layout='constrained')
# Measured E152v4 (pc40): dense native-eager vs LoDA-SA whole-step CUDA Graph replay;
# the dashed no-graph sparse arm is disclosed inline (system-lever honesty, no legend slot).
# Per-point placement keeps wide value+speedup labels inside the axes:
# 32K/64K below-right of the graph point, 128K above-left (right edge of the axes).
GRAPHLBL={32768:((5,-4),'left','top'),65536:((5,-4),'left','top'),131072:((-5,5),'right','bottom')}
for arm,label,color,marker,ls in [('dense','Dense FullKV','#65737D','o','-'),('tli_graph','LoDA-SA (CUDA Graph)','#007C91','^','-'),('tli_eager',None,'#A6B4BE','D','--')]:
 vals=[x[f'L{L}'][arm] for L in [32768,65536,131072]]
 a.plot([32,64,128],vals,label=label,color=color,marker=marker,lw=1.4,ls=ls)
 for L,xx,v in zip([32768,65536,131072],[32,64,128],vals):
  if arm=='tli_graph':
   off,ha,va=GRAPHLBL[L];a.annotate(f'{v:.1f} ({sp[f"L{L}"]:.2f}x)',(xx,v),xytext=off,textcoords='offset points',ha=ha,va=va,fontsize=7)
  else:a.annotate(f'{v:.1f}',(xx,v),xytext=(0,5),textcoords='offset points',ha='center',va='bottom',fontsize=7)
a.annotate('no graph',(96,127),ha='center',va='bottom',fontsize=7,style='italic',color='#A6B4BE')
a.set_xticks([32,64,128],['32K','64K','128K']);a.set_ylim(0,150);a.set_yticks(range(0, 141, 20));a.set_xlabel('Input length (tokens)');a.set_ylabel('Decode TPOT (ms)');a.legend(loc='upper left', bbox_to_anchor=(0, 1.32), frameon=False, borderaxespad=0);save(fig,'decode_tpot_current')
x=json.loads((DATA/'e146_r128k_formal.json').read_text())['scores']; vals=[statistics.mean(x['L131072/none'].values()),statistics.mean(x['L131072/tli_64_128_1024_c4_Aa0.125_b0.375_g0.625_hb4f1e46605'].values())]
fig,a=plt.subplots(figsize=(3.35,2.05),layout='constrained');a.barh(['Dense','Archived sparse\nconfiguration'],vals,color=['#65737D','#007C91'],height=.5);a.set_xlim(0,58);a.set_xlabel('128K mean task score');a.invert_yaxis()
for i,v in enumerate(vals):a.text(v+1,i,f'{v:.2f}',va='center')
a.set_title('稀疏臂：名义远区词元配额 = 0\n（代码／配置推论）' if LANG=='zh' else 'Sparse arm: nominal far-token quota = 0\n(code/config inference)',fontsize=7.5)
save(fig,'ruler128k_current')
if ONLY == 'all':
 vals=[statistics.mean(c['o3'][arm]['selected_mass_over_reference_topk_mass']['mean'] for c in d['cells']) for arm in ['tail32','random32','high32']]
 fig,a=plt.subplots(figsize=(3.35,1.9),layout='constrained');a.barh(['Tail32','Random32 (unpaired)','High32'],vals,color=['#007C91','#83919A','#C25E24'],height=.55);a.invert_yaxis();a.set_xlim(0,1.1);a.set_xlabel('Selected / reference-mid-TopK mass')
 for i,v in enumerate(vals):a.text(v+.02,i,f'{v:.3f}',va='center')
 save(fig,'coordinate_diagnostic_current')
 summary={'representative':{k:c[k] for k in ['task','sample','layer']},'q_pos':o['q_pos'],'n_source_sequences':len({(cell['task'],cell['sample']) for cell in d['cells']}),'n_query_layer_observations':len(rows),'far_capture':far,'near_capture':near,'selected_middle_tokens_range':[min(r['selected_tokens'] for c,o,r in rows),max(r['selected_tokens'] for c,o,r in rows)],'density_protected_mass':o['mass_protected_union'],'density_protected_tokens':o['region_bounds']['sink_hi']+o['region_bounds']['length']-max(o['region_bounds']['sink_hi'],o['region_bounds']['swa_lo'])}
 (ROOT/'sources/data/figure_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
