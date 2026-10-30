"""Render only independently admitted E155 paired uncertainty, no synthetic data."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/wangyuanshuo02/two-level-attention/agent_doc/results/nfa_ablation_20261030')
review=R/'reuse_E_review.md'
assert 'usable-with-scope' in review.read_text()
p=R/'near_length_paired.json';data=json.loads(p.read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(7,2.65),sharex=True,sharey=True)
plotdata=[]
for ax,task,title in zip(axes,['hotpotqa','gov_report'],['HotpotQA','GovReport']):
    x=data['tasks'][task]
    for y,nl,color,marker in [(1,2048,'#0078a8','o'),(0,8192,'#c87624','s')]:
        c=x['contrasts'][f'{nl}_minus_4096'];delta=c['delta_points'];lo,hi=c['ci95']
        ax.errorbar(delta,y,xerr=[[delta-lo],[hi-delta]],color=color,fmt=marker,markersize=6,
            capsize=4,elinewidth=1.7,capthick=1.3,zorder=3)
        plotdata.append(dict(task=task,near_len=nl,delta=delta,ci95=[lo,hi]))
    baseline=sum(x['arms']['4096']['per_request_score'].values())/200
    ax.set_title(title,pad=20,fontweight='bold')
    ax.text(.5,1.04,f'4096-token reference: {baseline:.2f}',ha='center',transform=ax.transAxes,color='#60666b',fontsize=9)
    ax.axvline(0,color='#7c8388',ls=(0,(3,3)),lw=1,zorder=1)
    ax.set_yticks([0,1],['8192','2048']);ax.set_ylim(-.55,1.55)
    ax.set_xlim(-2,3);ax.set_xticks([-2,-1,0,1,2,3])
    ax.grid(axis='x',color='#e5e8eb',lw=.6);ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color('#b5bdc4')
    ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',color='#b5bdc4')
    ax.set_xlabel('Score difference vs. 4096 (points)')
axes[0].set_ylabel('Near-region length (tokens)')
fig.subplots_adjust(left=.115,right=.98,bottom=.27,top=.72,wspace=.2)
fig.text(.115,.035,'sink = 4 · SWA = 128 · 200 requests/task · paired bootstrap 95% CI',fontsize=9,color='#505960')
for ext in ('png','pdf','svg'):fig.savefig(R/'figures'/f'E_near_length_delta.{ext}',dpi=250,facecolor='white')
(R/'figures/E_plot_data.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),review_sha256=hashlib.sha256(review.read_bytes()).hexdigest(),points=plotdata),indent=2)+'\n')
