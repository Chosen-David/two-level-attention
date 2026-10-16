from pathlib import Path
import os,subprocess
os.environ.setdefault("MPLCONFIGDIR","/tmp/dac_detail_mpl")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont(os.environ.get("PAPER_CJK_FONT", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
cjk=font_manager.FontProperties(fname=os.environ.get("PAPER_CJK_FONT", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")).get_name()
p=Path(__file__).resolve().parents[1]/'figures'
for lang in ['en','zh']:
 plt.rcParams.update({'font.family':'DejaVu Sans' if lang=='en' else cjk,'font.size':10,'pdf.fonttype':42,'svg.fonttype':'path'})
 fig,ax=plt.subplots(figsize=(5.7,2.6))
 labels=['Nominal full-dimensional\npage scoring','Shared-subspace\nscoring','Nominal full-dimensional\ntoken scoring'] if lang=='en' else ['名义全维页评分','共享子空间评分','名义全维词元评分']
 ds=[0,-.322,-.244];means=[47.910,47.588,47.666];cs=['#687988','#128982','#305c86']
 for y,(d,m,c) in enumerate(zip(ds,means,cs)):
  ax.plot([0,d],[y,y],color=c,lw=2);ax.scatter([d],[y],color=c,s=28,zorder=3);ax.annotate(f'{d:+.3f}  ({m:.3f})',(d,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
 ax.axvline(0,color='#687988',lw=.7);ax.set_yticks(range(3),labels);ax.set_ylim(2.5,-.6);ax.set_xlim(-.42,.11)
 ax.set_xlabel('Mean score difference from nominal page-scoring variant\n(points; parenthesized values are mean scores)' if lang=='en' else '相对名义全维页评分配置的平均分差\n（分；括号内为平均得分）')
 ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.15);fig.tight_layout(pad=.5)
 stem='e134_quality_current'+('_zh' if lang=='zh' else '')
 for ext in ['svg','png']:fig.savefig(p/f'{stem}.{ext}',dpi=250,bbox_inches='tight')
 subprocess.run(['inkscape',str(p/f'{stem}.svg'),f'--export-filename={p/stem}.pdf'],check=True,capture_output=True)
 plt.close(fig)
