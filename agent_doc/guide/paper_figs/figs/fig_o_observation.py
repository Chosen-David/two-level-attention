# -*- coding: utf-8 -*-
"""CPU-only diagnostic rendering from existing O1/O3 aggregates.

(a,b) feasible whole-bin mass-rank heuristic, NOT token TopK or an upper bound.
All tokens in a selected bin are counted; bins that cannot fit are skipped.
Exact recorded region bounds are used, with no within-bin interpolation.
(c) source O3 ratios, not full-attention mass or a paired-random comparison.
(d) blocked: historical E6 cannot certify the current gate.
Default refuses publication. Explicit --diagnostic writes only diagnostic assets.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from exp.paper.observation_bins import diagnose, finite_summary, paired_comparison, finite_nonnegative
from exp.paper.e140_asset_provenance import add_render_warning


def build_data(source, budget=2048):
    cells=json.loads(source.read_text())['cells']
    rows=[]
    for cell in cells:
        for o in cell['o1']:
            result=diagnose(o,budget)
            result.update(task=cell['task'],sample=cell['sample'],layer=cell['layer'],q_pos=o['q_pos'])
            result['region_bounds']=o['region_bounds']
            rows.append(result)
    ids=[(r['task'],r['sample'],r['layer'],r['q_pos']) for r in rows]
    if not rows or len(ids)!=len(set(ids)):
        raise ValueError('empty or duplicate observation identities')
    far=[r['far_capture'] for r in rows];near=[r['near_capture'] for r in rows]
    summary=dict(far=finite_summary(far),near=finite_summary(near),paired=paired_comparison(near,far),
                 selected_tokens_min=min(r['selected_tokens'] for r in rows),
                 selected_tokens_max=max(r['selected_tokens'] for r in rows),
                 n_rows=len(rows),n_source_sequences=len({(r['task'],r['sample']) for r in rows}))
    arms={a:sum(finite_nonnegative(c['o3'][a]['selected_mass_over_reference_topk_mass']['mean']) for c in cells)/len(cells)
          for a in ('tail32','random32','high32')}
    return dict(protocol='feasible-whole-bin-mass-rank-v1',status='diagnostic-only-not-scientific-acceptance',
                budget=budget,rows=rows,summary=summary,o3=arms,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                representative_rule='display-only: q>9000, closest far share to median of finite rows; fallback all finite rows; ties source order',
                limitations=['whole-bin heuristic; not token TopK, optimum or upper bound',
                             'query/layer repeats are not independent documents',
                             'E6 gate panel blocked; no gate safety or skip thresholds',
                             'O3 random32 is unpaired; ratios are relative reference-mid-TopK mass'])


def render(data, out):
    import statistics
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, ListedColormap
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    rows=data['rows'];summary=data['summary']
    finite=[r for r in rows if r['far_share'] is not None]
    if not finite:
        raise ValueError('no nonzero mid mass for representative strip')
    median=statistics.median(r['far_share'] for r in finite)
    eligible=[r for r in finite if r['q_pos']>9000] or finite
    selected=min(eligible,key=lambda r:abs(r['far_share']-median))
    data['representative_identity']={k:selected[k] for k in ('task','sample','layer','q_pos')}
    fig,axes=plt.subplots(1,4,figsize=(21,5.6),gridspec_kw={'width_ratios':[1.35,1.15,.9,1.0]})
    a,b,c,d=axes
    bins=selected['bins'];edges=[v['start'] for v in bins]+[bins[-1]['end']]
    mass=np.array([v['mass'] for v in bins]);chosen=np.array([i in selected['selected'] for i in range(len(bins))])
    positive=mass[mass>0]
    if not len(positive):raise ValueError('no positive bin mass')
    vmin=float(positive.min());vmax=max(float(positive.max()),vmin*1.01)
    mesh=a.pcolormesh(edges,[1,2],np.ma.masked_less_equal(mass[None,:],0),cmap='Blues',norm=LogNorm(vmin=vmin,vmax=vmax),shading='flat')
    a.pcolormesh(edges,[0,1],chosen[None,:].astype(float),cmap=ListedColormap(['#eeeeee','#df783c']),vmin=0,vmax=1,shading='flat')
    rb=selected['region_bounds'];a.axvline(rb['near_lo'],color='#b14921',ls='--')
    a.axvspan(0,rb['mid_lo'],color='.7',alpha=.4)
    a.axvspan(rb['mid_hi'],selected['q_pos']+1,color='.7',alpha=.4)
    a.set(xlim=(0,selected['q_pos']+1),ylim=(0,2.5),yticks=[.5,1.5],yticklabels=['whole-bin\nselection','bin mass'],xlabel='absolute key position (tokens)')
    for region,left,right in [('far',rb['mid_lo'],rb['near_lo']),('near',rb['near_lo'],rb['mid_hi'])]:
        a.text((left+right)/2,2.2,region,ha='center')
    a.set_title('(a) Existing mass bins',loc='left',weight='bold')
    a.text(0,-.40,f"{selected['task']} s{selected['sample']} L{selected['layer']} q{selected['q_pos']}\nSelected {selected['selected_tokens']} / cap {data['budget']} mid tokens\nAll tokens in each selected bin; gray = protected",transform=a.transAxes,va='top',fontsize=9)
    fig.colorbar(mesh,cax=fig.add_axes([.06,.18,.20,.022]),orientation='horizontal',label='mass per bin (log); exact bin widths')
    for task,color,marker in [('hotpotqa','#176c91','o'),('musique','#bd5425','^')]:
        subset=[r for r in rows if r['task']==task and r['far_share'] is not None and r['far_capture'] is not None]
        b.scatter([r['far_share'] for r in subset],[r['far_capture'] for r in subset],s=18,color=color,marker=marker,alpha=.65,label=f'{task} (n={len(subset)})')
    b.set(xlim=(-.03,1.03),ylim=(-.03,1.05),xlabel='far share of mid mass',ylabel='far mass captured by whole-bin heuristic')
    b.set_title('(b) Budget-feasible diagnostic',loc='left',weight='bold')
    b.legend(loc='lower right',fontsize=8)
    def fmt(v):return 'undefined' if v is None else f'{v:.3f}'
    pair=summary['paired']
    stats=(f"Far mean {fmt(summary['far']['mean'])} (n={summary['far']['n_valid']})\n"
           f"Near mean {fmt(summary['near']['mean'])} (n={summary['near']['n_valid']})\n"
           f"Near > far: {pair['near_gt_far']}/{pair['n_pairs']} finite pairs\n"
           f"Selected {summary['selected_tokens_min']}–{summary['selected_tokens_max']} tokens; cap {data['budget']}")
    b.text(0,-.25,stats,transform=b.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='.8'))
    arms=['tail32','random32','high32'];values=[data['o3'][arm] for arm in arms]
    c.bar(range(3),values,color=['#176c91','#999999','#df783c'])
    for i,v in enumerate(values):c.text(i,v+.015,f'{v:.3f}',ha='center')
    c.set(xticks=range(3),xticklabels=['tail32','random32\n(unpaired)','high32'],ylim=(0,1.05),ylabel='selected mass / reference-mid-TopK mass')
    c.set_title('(c) Existing O3 ratios',loc='left',weight='bold')
    c.text(0,-.25,'Equal cell means; 8 task/sample/layer cells.\nNot full-attention mass; no paired-random control.',transform=c.transAxes,va='top',fontsize=9)
    d.axis('off');d.set_title('(d) Gate evidence blocked',loc='left',weight='bold')
    d.text(.03,.9,'Historical E6 is not comparable:\n\n• summed logits before softmax\n• last sampled query only\n• far = [64, t − 2048)\n\nCurrent gate safety and skip\nthresholds cannot be inferred.\n\nNo E6 heatmap or skip count\nis used in this diagnostic.',transform=d.transAxes,va='top',fontsize=11,linespacing=1.6,color='#763919')
    fig.subplots_adjust(left=.055,right=.985,bottom=.29,top=.80,wspace=.48)
    add_render_warning(fig)
    fig.text(.5,.015,f"{summary['n_rows']} query/layer observations from {summary['n_source_sequences']} source sequences; document overlap unresolved. Repeated observations are descriptive, not independent trials.",ha='center',fontsize=10)
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):fig.savefig(out/f'O_observation.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    (out/'O_observation_diagnostic.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    with (out/'O_observation_figure_data.tsv').open('w',newline='') as f:
        fields=['task','sample','layer','q_pos','budget','selected_tokens','available_tokens','far_share','far_capture','near_capture']
        writer=csv.DictWriter(f,fields,delimiter='\t',lineterminator='\n',extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,allow_nan=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic',action='store_true')
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    if not args.diagnostic:
        raise SystemExit('Diagnostic opt-in required; unsupported publication claims blocked. No files written.')
    render(build_data(ROOT/'exp/trace/results/e140_o_trace_v2.json'),args.out)


if __name__=='__main__':main()
