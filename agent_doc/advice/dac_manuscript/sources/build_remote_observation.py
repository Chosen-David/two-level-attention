from paper_font_paths import cjk_regular, register_matplotlib
"""Adapt the remote abcd renderer, blob d98ee75..., without recollection.
Same2x2design: saved example / region-complementarity miss / coordinate bars / historical matrix.
2026-10-21 fourth revision: panel (b) quota-floor paired arms retired (nt demoted;
nb/nl absolute-quota caliber) and replaced by per-region missed reference mass
(block-avg misses far, minmax misses near) from region_miss_stats.json; panel (a)
gains near/far background tints. Hybrid near-budget scan stays a data-file
disclosure, not a figure arm (budget division belongs to the E155d grid axis).
"""
from pathlib import Path
import json,os,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
register_matplotlib()
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';DATA=ROOT/'sources/data'
d=json.loads((DATA/'observation_plot_input.json').read_text());extra=json.loads((DATA/'remote_observation_input.json').read_text());cells=d['cells'];rs=json.loads((DATA/'region_miss_stats.json').read_text())
LANG=os.environ.get('DAC_FIG_LANG','en')
plt.rcParams.update({'font.family':'DejaVu Sans','axes.linewidth':1.5,'svg.fonttype':'path','pdf.fonttype':42})
C_MAIN='#0077b6';C_SEC='#00b4d8';C_HL='#ff6b35';C_NEU='#6c757d'
T=15;L=13;TK=11.5;AN=10.5;LG=11.5;LW=2.4
c_a=next(c for c in cells if (c['task'],c['sample'],c['layer'])==('musique',1,24));o_a=next(o for o in c_a['o1'] if o['q_pos']==14212)
q_a=o_a['q_pos'];rb=o_a['region_bounds'];pg=extra['representative']['pages'];qa_q=extra['representative']
bin_lo=pg['pos_lo'];ntok_a=np.array(pg['n_tok']);mass_a=np.array(pg['mass']);pos_c=np.array(bin_lo)+(ntok_a-1)/2
is_far=np.array([lo+n<=rb['near_lo'] for lo,n in zip(bin_lo,ntok_a)])
sel_o=set(pg['sel_oracle']);sel_avg=set(pg['sel_avg']);sel_mm=set(pg['sel_minmax']);sel_mask=np.array([i in sel_o for i in range(len(bin_lo))])
missed=sorted(sel_o-sel_avg);recov=sorted(set(missed)&sel_mm);spike=qa_q['detail']['spike'];spike_top1=max(spike['token_mass'])/sum(spike['token_mass']);far_tok_frac=ntok_a[is_far].sum()/ntok_a.sum()
actual_budget={a:sum(pg['n_tok'][i] for i in pg['sel_'+a]) for a in ['oracle','avg','minmax']}
if actual_budget!={'oracle':1989,'avg':1989,'minmax':2048}:raise ValueError('Saved example budget changed')
# Panel (b) numbers come from region_miss_stats.json (fail-closed rebuild:
# selection reproduction and page/region alignment asserted there).
agg=rs['aggregates']
if rs['n_queries_region']!=128 or rs['n_queries_total']!=128:raise ValueError('Region-miss record incomplete')
arms=['tail32','random32','high32'];arm_lab={'tail32':'tail32\n(low-freq)','random32':'random32\n(unpaired)','high32':'high32\n(high-freq)'}
arm_rm={a:float(np.mean([c['o3'][a]['selected_mass_over_reference_topk_mass']['mean'] for c in cells])) for a in arms}
ROW_ORDER=list(extra['profiles']);ROW_LAB=dict(zip(ROW_ORDER,['gov_0','gov_1','hot_0','hot_1','mfld_0','mfld_1','mus_0','mus_1','nar_0','nar_1','pr_0','pr_1','qas_0','qas_1','natural','needle']))
M=np.array([extra['profiles'][k] for k in ROW_ORDER]);n_light=int((np.median(M,axis=0)<.01).sum());skip_rng=(M<.01).sum(axis=1).tolist()
if M.shape!=(16,36) or n_light!=24 or (min(skip_rng),max(skip_rng))!=(12,32):raise ValueError('Historical profile identity changed')

fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(
    2, 2, figsize=(9.5, 6.5),
    gridspec_kw={"wspace": 0.48, "hspace": 0.44,
                 "width_ratios": [1.2, 1.0]})


# ---- (a) raw page mass and symmetric saved-set comparison ----
cmap_mass = mcolors.LinearSegmentedColormap.from_list("mass", ["#ffffff", C_MAIN])
norm_mass = mcolors.LogNorm(vmin=2e-4, vmax=2e-2)
edges=np.r_[bin_lo,bin_lo[-1]+int(ntok_a[-1])]
ax_a.pcolormesh(edges,[3,3.8],mass_a[None,:],cmap=cmap_mass,norm=norm_mass,shading='flat')
C_MISS = "#d62728"
# Identical classification for each scorer: intersection, reference-only, scorer-only.
# Background means neither selected. Reference is a saved whole-page mass ranking,
# not exact token TopK, a combinatorial optimum, or DSA.
for y,chosen in [(2,sel_o),(1,sel_avg),(0,sel_mm)]:
    for j in range(len(bin_lo)):
        color="#e9e9e9";hatch=None;edge='none';width=0
        if j in chosen and j in sel_o:color=C_MAIN
        elif j in sel_o:color=C_MISS;hatch='////';edge='black';width=.45
        elif j in chosen:color=C_HL;hatch='..';edge='black';width=.45
        ax_a.add_patch(plt.Rectangle((edges[j],y),edges[j+1]-edges[j],.8,
            facecolor=color,edgecolor=edge,linewidth=width,hatch=hatch))
ax_a.axvspan(0, rb["sink_hi"], color=C_NEU, alpha=0.35, lw=0)
ax_a.axvspan(rb["swa_lo"], q_a + 1, color=C_NEU, alpha=0.35, lw=0)
# Region tints across the mid span: far (highlight) and near (primary), so the
# region of each missed page reads at a glance; protected spans stay grey above.
ax_a.axvspan(rb["sink_hi"], rb["near_lo"], color=C_HL, alpha=0.12, lw=0)
ax_a.axvspan(rb["near_lo"], rb["swa_lo"], color=C_MAIN, alpha=0.12, lw=0)
_tagbox = dict(boxstyle="round,pad=0.2", facecolor="white",edgecolor="none",alpha=0.85)
ax_a.text((rb["sink_hi"] + rb["near_lo"]) / 2, 3.4,
          f"far ({far_tok_frac:.0%} of mid tokens)",ha="center",fontsize=AN,
          color=C_HL,fontweight="bold",bbox=_tagbox)
ax_a.text((rb["near_lo"] + rb["swa_lo"]) / 2,3.4,"near",ha="center",
          fontsize=AN,color=C_MAIN,fontweight="bold",bbox=_tagbox)
ax_a.axvline(rb["near_lo"],color=C_HL,lw=1.6,linestyle="--")
i_sp=bin_lo.index(spike["pos_lo"])
ax_a.plot([pos_c[i_sp]],[3.95],marker="v",color=C_MISS,markersize=8,clip_on=False,zorder=6)
leg_a=[Patch(fc=C_MAIN,label="selected in reference"),
       Patch(fc=C_MISS,ec='black',linewidth=.45,hatch='////',label="reference page missed"),
       Patch(fc=C_HL,ec='black',linewidth=.45,hatch='..',label="selected outside reference")]
ax_a.legend(handles=leg_a,loc="upper left",bbox_to_anchor=(.005,1.005),
            fontsize=9.5,framealpha=.92,handlelength=1.4,borderpad=.45,labelspacing=.4)
ax_a.set_xlim(0,q_a+1);ax_a.set_xticks(range(0,15000,3000));ax_a.set_ylim(0,6.1)
ax_a.set_yticks([.4,1.4,2.4,3.4])
ax_a.set_yticklabels(["minmax","avg pool","mass rank","page mass"],fontsize=TK-1.5)
ax_a.set_xlabel("key position (tokens); query at right end", fontsize=L)
ax_a.set_title("(a) Symmetric page selections",
               loc="left", pad=10, fontsize=T, fontweight="bold")
ax_a.tick_params(axis="x", labelsize=TK)
sm = plt.cm.ScalarMappable(cmap=cmap_mass, norm=norm_mass)
cb = fig.colorbar(sm, ax=ax_a, fraction=0.04, pad=0.015)
cb.set_label("mass (log)", fontsize=AN - 1.5)
cb.ax.tick_params(labelsize=10)

# ---- (b) 区域互补漏失：avg 漏远区、minmax 漏近区（128 条相关观测均值）----
# Stacked bars: each scorer's missed reference mass split by page region
# (near bottom / far top). Numbers live in the caption, not the panel.
xb_m = np.arange(2)
near_vals = [agg["avg_miss_near"], agg["mm_miss_near"]]
far_vals = [agg["avg_miss_far"], agg["mm_miss_far"]]
ax_b.bar(xb_m, near_vals, width=0.52, color=C_MAIN,
         edgecolor="black", linewidth=.7, label="missed near")
ax_b.bar(xb_m, far_vals, width=0.52, bottom=near_vals, color=C_HL,
         edgecolor="black", linewidth=.7, label="missed far")
ax_b.set_xticks(xb_m)
ax_b.set_xticklabels(["avg pool", "minmax"], fontsize=TK)
ax_b.set_ylabel("missed reference mass", fontsize=L)
ax_b.set_ylim(0, 0.30)
ax_b.set_yticks([0.0, 0.1, 0.2, 0.3])
ax_b.set_title("(b) Missed mass by region", loc="left", pad=10,
               fontsize=T, fontweight="bold")
ax_b.tick_params(labelsize=TK)
ax_b.legend(loc="upper right", fontsize=LG - 2, framealpha=0.92)

# ---- (c) O3 三臂 ----
colors3 = [C_MAIN, C_NEU, C_SEC]
for i, (a, col) in enumerate(zip(arms, colors3)):
    ax_c.bar([i], [arm_rm[a]], width=0.58, color=col, edgecolor="black",
             linewidth=1.2)
    ax_c.text(i, arm_rm[a] + 0.025, f"{arm_rm[a]:.2f}", ha="center",
              fontsize=AN, fontweight="bold")
ax_c.set_xticks(np.arange(3))
ax_c.set_xticklabels([arm_lab[a] for a in arms], fontsize=TK - 1)
ax_c.set_ylabel("selected mass /\nreference mid-TopK mass", fontsize=L)
ax_c.set_ylim(0, 1.0)
ax_c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_c.set_title("(c) Which 32 of 128 dims", loc="left", pad=10, fontsize=T,
               fontweight="bold")
ax_c.tick_params(labelsize=TK)

# ---- (d) E6 逐层 far mass 热图 ----
norm6 = mcolors.LogNorm(vmin=1e-7, vmax=1.0)
im = ax_d.imshow(M, aspect="auto", cmap=cmap_mass, norm=norm6,
                 interpolation="nearest")
ax_d.set_xticks([0, 8, 16, 24, 32])
ax_d.set_yticks(range(len(ROW_ORDER)))
ax_d.set_yticklabels([ROW_LAB[k] for k in ROW_ORDER], fontsize=9.5)
ax_d.set_xlabel("layer", fontsize=L)
ax_d.set_title("(d) Historical far proxy (16 seqs)", loc="left", pad=10,
               fontsize=T, fontweight="bold")
ax_d.tick_params(axis="x", labelsize=TK)
cb6 = fig.colorbar(im, ax=ax_d, fraction=0.04, pad=0.015)
cb6.set_label("far proxy (log)", fontsize=AN - 0.5)
cb6.ax.tick_params(labelsize=10)

fig.subplots_adjust(left=0.10, right=0.93, top=0.94, bottom=0.09)


# Bilingual labels only; preserve the remote abcd layout and encodings.
if LANG=='zh':
 translations={'selected in reference':'选中参考页','reference page missed':'参考页漏选','selected outside reference':'选中参考外页','page mass':'页权重和','minmax':'极值评分','(a) Symmetric page selections':'(a) 对称页选集比较','(b) Missed mass by region':'(b) 按区域的漏失权重和','(c) Which 32 of 128 dims':'(c) 128维中选择32维','(d) Historical far proxy (16 seqs)':'(d) 历史远区代理（16序列）','page not selected':'未选中页','selected by avg pool':'均值评分选中','mass-ranked page missed by avg\n(✓ = also selected by minmax)':'质量排序选中、均值漏选\n（✓：极值评分也选中）','avg pool':'均值评分','mass rank':'质量排序','attn mass':'注意力权重和','key position (tokens); query at right end':'键位置（词元）；查询在右端','mass (log)':'权重和（对数）','missed reference mass':'漏失的参考权重和','missed near':'漏失近区','missed far':'漏失远区','selected mass /\nreference mid-TopK mass':'选中／参考mid-TopK\n注意力权重和','layer':'层','far proxy (log)':'远区代理（对数）','near':'近区','tail32\n(low-freq)':'尾部32\n（低频）','random32\n(unpaired)':'随机32\n（未配对）','high32\n(high-freq)':'高频32\n（高频）',f'far ({far_tok_frac:.0%} of mid tokens)':f'远区（中间词元的{far_tok_frac:.0%}）'}
 for ax in fig.axes:
  xlim,ylim=ax.get_xlim(),ax.get_ylim()
  ax.set_xticks(ax.get_xticks(), [translations.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
  ax.set_yticks(ax.get_yticks(), [translations.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()])
  ax.set_xlim(xlim);ax.set_ylim(ylim)
 for t in fig.findobj(Text):
  t.set_text(translations.get(t.get_text(),t.get_text()));t.set_fontproperties(FontProperties(fname=cjk_regular(),size=t.get_fontsize()))
stem='o_observation'+('_zh' if LANG=='zh' else '')
fig.savefig(OUT/(stem+'.svg'),bbox_inches='tight');fig.savefig(OUT/(stem+'.png'),dpi=220,bbox_inches='tight')
svg=OUT/(stem+'.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
subprocess.run(['inkscape',str(OUT/(stem+'.svg')),'--export-type=pdf','--export-filename='+str(OUT/(stem+'.pdf'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE) if __import__('shutil').which('inkscape') else __import__('cairosvg').svg2pdf(url=str(OUT/(stem+'.svg')),write_to=str(OUT/(stem+'.pdf')))
plt.close(fig)
summary={'representative':'musique:s1:L24:q14212','a_actual_tokens':actual_budget,'a_selected_page_counts':{a:len(pg['sel_'+a]) for a in ['oracle','avg','minmax']},'a_mass_rank_missed_by_avg':len(missed),'a_missed_also_selected_by_minmax':len(recov),'a_spike_top1_share':spike_top1,'a_symmetric_sets':{'page_count':len(bin_lo),'reference_missed_by_avg':len(sel_o-sel_avg),'reference_missed_by_minmax':len(sel_o-sel_mm),'reference_missed_by_both':len(sel_o-sel_avg-sel_mm),'avg_missed_minmax_selected':len((sel_o-sel_avg)&sel_mm),'minmax_missed_avg_selected':len((sel_o-sel_mm)&sel_avg),'avg_selected_outside_reference':len(sel_avg-sel_o),'minmax_selected_outside_reference':len(sel_mm-sel_o)},'b_region_miss':{'avg_near':agg['avg_miss_near'],'avg_far':agg['avg_miss_far'],'mm_near':agg['mm_miss_near'],'mm_far':agg['mm_miss_far'],'avg_total':agg['avg_miss_total'],'mm_total':agg['mm_miss_total'],'avg_pages_near':agg['avg_pages_near'],'avg_pages_far':agg['avg_pages_far'],'mm_pages_near':agg['mm_pages_near'],'mm_pages_far':agg['mm_pages_far']},'b_hybrid_scan_disclosure':{'best_near_budget':rs['hybrid_scan']['best_near_budget'],'best_miss':rs['hybrid_scan']['best_miss'],'global_minmax_miss':rs['hybrid_scan']['global_minmax_miss'],'note':rs['hybrid_scan']['note']},'b_retired_quota_floor_panel':'2026-10-21: far-quota paired-arm panel withdrawn (nt demoted; nb/nl absolute-quota caliber); old per-query rows remain in git history','b_independent_sequences':4,'b_correlated_observations':rs['n_queries_region'],'c_reference_ratios':arm_rm,'d_low_proxy_layer_medians':n_light,'d_per_sequence_low_counts':dict(zip(ROW_ORDER,skip_rng))}
(DATA/'remote_observation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
