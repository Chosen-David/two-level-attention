"""Adapt the remote abcd renderer, blob d98ee75..., without recollection.
Same2x2design: saved example / regional capture / coordinate bars / historical matrix.
Only statistical labels, exact boundaries, budget admission and readable fonts change.
"""
from pathlib import Path
import json,os,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';DATA=ROOT/'sources/data'
d=json.loads((DATA/'observation_plot_input.json').read_text());extra=json.loads((DATA/'remote_observation_input.json').read_text());cells=d['cells']
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
if actual_budget!={'oracle':2053,'avg':2053,'minmax':2048}:raise ValueError('Saved example budget changed')
# Whole-bin admission with deterministic position ties; no fractional-bin mass assumptions.
def select(bs,budget):
 selected=[];used=0
 for b in sorted(bs,key=lambda b:(-b['mass'],b['start'])):
  if b['n_tok']<=budget-used:selected.append(b);used+=b['n_tok']
 return selected,used
rows=[]
for c in cells:
 for o in c['o1']:
  bs=[];bounds=o['region_bounds'];q=o['q_pos']
  for b in o['bins']:
   lo=q-b['d_hi'];hi=q-b['d_lo']+1
   if hi-lo!=b['n_tok'] or not(bounds['mid_lo']<=lo<hi<=bounds['mid_hi']):raise ValueError('Invalid bin')
   if hi<=bounds['near_lo']:region='far'
   elif lo>=bounds['near_lo']:region='near'
   else:raise ValueError('Region-straddling bin')
   bs.append(dict(b,start=lo,end=hi,region=region))
  bs.sort(key=lambda b:b['start'])
  if bs[0]['start']!=bounds['mid_lo'] or bs[-1]['end']!=bounds['mid_hi'] or any(a['end']!=b['start'] for a,b in zip(bs,bs[1:])):raise ValueError('Incomplete bins')
  g,ng=select(bs,2048);fs=[b for b in bs if b['region']=='far'];ns=[b for b in bs if b['region']=='near'];f,nf=select(fs,1024);n,nn=select(ns,1024)
  fm=sum(b['mass'] for b in fs);nm=sum(b['mass'] for b in ns)
  rows.append(dict(id=f"{c['task']}:s{c['sample']}:L{c['layer']}:q{q}",far_share=fm/(fm+nm),global_far=sum(b['mass'] for b in g if b['region']=='far')/fm,global_near=sum(b['mass'] for b in g if b['region']=='near')/nm,quota_far=sum(b['mass'] for b in f)/fm,global_tokens=ng,quota_far_tokens=nf,quota_near_tokens=nn,global_starts=[b['start'] for b in g],quota_far_starts=[b['start'] for b in f],quota_near_starts=[b['start'] for b in n]))
sh=np.array([r['far_share'] for r in rows]);fc=np.array([r['global_far'] for r in rows]);nc=np.array([r['global_near'] for r in rows]);qc=np.array([r['quota_far'] for r in rows]);fc_mean=float(fc.mean());nc_mean=float(nc.mean());qc_mean=float(qc.mean())
FAR_QUOTA=1024;BUDGET_TOK=2048
arms=['tail32','random32','high32'];arm_lab={'tail32':'tail32\n(low-freq)','random32':'random32\n(unpaired)','high32':'high32\n(high-freq)'}
arm_rm={a:float(np.mean([c['o3'][a]['selected_mass_over_reference_topk_mass']['mean'] for c in cells])) for a in arms}
ROW_ORDER=list(extra['profiles']);ROW_LAB=dict(zip(ROW_ORDER,['gov_0','gov_1','hot_0','hot_1','mfld_0','mfld_1','mus_0','mus_1','nar_0','nar_1','pr_0','pr_1','qas_0','qas_1','natural','needle']))
M=np.array([extra['profiles'][k] for k in ROW_ORDER]);n_light=int((np.median(M,axis=0)<.01).sum());skip_rng=(M<.01).sum(axis=1).tolist()
if M.shape!=(16,36) or n_light!=24 or (min(skip_rng),max(skip_rng))!=(12,32):raise ValueError('Historical profile identity changed')

fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(
    2, 2, figsize=(9.5, 6.9),
    gridspec_kw={"wspace": 0.48, "hspace": 0.52,
                 "width_ratios": [1.2, 1.0]})


# ---- (a) token strip：质量 / oracle 单池 / avg 单池 三条带 ----
cmap_mass = mcolors.LinearSegmentedColormap.from_list("mass", ["#ffffff", C_MAIN])
norm_mass = mcolors.LogNorm(vmin=2e-4, vmax=2e-2)
X0, X1 = float(min(pos_c)), float(max(pos_c))
# Saved mass-ranked and mean-score selections; unequal actual budgets are disclosed.
edges=np.r_[bin_lo,bin_lo[-1]+int(ntok_a[-1])]
ax_a.pcolormesh(edges,[2,3],mass_a[None,:],cmap=cmap_mass,norm=norm_mass,shading='flat')
cmap_sel = mcolors.ListedColormap(["#e9e9e9", C_HL])
ax_a.pcolormesh(edges,[1,2],sel_mask[None,:].astype(float),cmap=cmap_sel,vmin=0,vmax=1,shading='flat')
C_MISS = "#d62728"
rgb = np.tile(np.array(mcolors.to_rgb("#e9e9e9")), (len(mass_a), 1))
for i in range(len(mass_a)):
    if i in sel_avg:
        rgb[i] = mcolors.to_rgb(C_MAIN)
for i in missed:
    rgb[i] = mcolors.to_rgb(C_MISS)
for j in range(len(bin_lo)):
    ax_a.add_patch(plt.Rectangle((edges[j],0),edges[j+1]-edges[j],1,facecolor=rgb[j],edgecolor='none'))
for i in recov:
    ax_a.text(pos_c[i], 0.5, "✓", ha="center", va="center",
              fontsize=TK - 2, color="white", fontweight="bold")
# sink / swa 保护区（不在 mid bins 内；区域太窄无法内嵌文字，见说明框）
ax_a.axvspan(0, rb["sink_hi"], color=C_NEU, alpha=0.35, lw=0)
ax_a.axvspan(rb["swa_lo"], q_a + 1, color=C_NEU, alpha=0.35, lw=0)
# near/far 区域标注（置于顶带中部）
_tagbox = dict(boxstyle="round,pad=0.2", facecolor="white",
               edgecolor="none", alpha=0.85)
ax_a.text((rb["sink_hi"] + rb["near_lo"]) / 2, 2.5,
          f"far ({far_tok_frac:.0%} of mid tokens)", ha="center",
          fontsize=AN, color=C_HL, fontweight="bold", bbox=_tagbox)
ax_a.text((rb["near_lo"] + rb["swa_lo"]) / 2, 2.5, "near", ha="center",
          fontsize=AN, color=C_MAIN, fontweight="bold", bbox=_tagbox)
ax_a.axvline(rb["near_lo"], color=C_HL, lw=1.6, linestyle="--")
# Fixed-example spike marker; top token share is computed, not assumed to be100%.
i_sp = bin_lo.index(spike["pos_lo"])
ax_a.plot([pos_c[i_sp]], [3.10], marker="v", color=C_MISS, markersize=8,
          clip_on=False, zorder=6)
# 图例替代原说明框（用户 2026-10-16：图内不堆文字，数字解释并入 caption）
leg_a = [Patch(fc="#e9e9e9", ec="0.6", label="page not selected"),
         Patch(fc=C_MAIN, label="selected by avg pool"),
         Patch(fc=C_MISS,
               label="mass-ranked page missed by avg\n(✓ = also selected by minmax)")]
ax_a.legend(handles=leg_a, loc="upper left", bbox_to_anchor=(0.005, 1.005),
            fontsize=9.5, framealpha=0.92, handlelength=1.4,
            borderpad=0.45, labelspacing=0.4)
ax_a.set_xlim(0, q_a + 1)
ax_a.set_xticks(range(0, 15000, 3000))
ax_a.set_ylim(0, 5.2)
ax_a.set_yticks([0.5, 1.5, 2.5])
ax_a.set_yticklabels(["avg pool", "mass rank", "attn mass"],
                     fontsize=TK - 1.5)
ax_a.set_xlabel("key position (tokens); query at right end", fontsize=L)
ax_a.set_title("(a) Example page selections",
               loc="left", pad=10, fontsize=T, fontweight="bold")
ax_a.tick_params(axis="x", labelsize=TK)
sm = plt.cm.ScalarMappable(cmap=cmap_mass, norm=norm_mass)
cb = fig.colorbar(sm, ax=ax_a, fraction=0.04, pad=0.015)
cb.set_label("mass (log)", fontsize=AN - 1.5)
cb.ax.tick_params(labelsize=10)

# ---- (b) 排序捕获率曲线：全局单池 vs far 保底配额 ----
# Same query/layer order for both policies; 43 paired changes are negative.
order_b = np.array(sorted(range(len(rows)), key=lambda i:(fc[i],rows[i]["id"])))
rk = np.arange(1, len(fc) + 1)
fc_s, qc_s = fc[order_b], qc[order_b]
ax_b.plot(rk, fc_s, color=C_MAIN, lw=LW,
          label=f"single pool (mean {fc_mean:.2f})")
ax_b.plot(rk, qc_s, color=C_HL, lw=LW, linestyle="--",
          label=f"far/near 1024/1024 ({qc_mean:.2f})")
ax_b.fill_between(rk, fc_s, qc_s, where=qc_s > fc_s,
                  color=C_HL, alpha=0.15, lw=0)
ax_b.axhline(nc_mean, color=C_NEU, lw=1.8, linestyle=":",
             label=f"global near: {nc_mean:.2f}")
ax_b.set_xlabel("128 observations, sorted by global far capture", fontsize=L)
ax_b.set_ylabel("far mass capture\n(whole-bin selection)", fontsize=L - 1)
ax_b.set_xlim(0, len(fc) + 1)
ax_b.set_ylim(0, 1.05)
ax_b.set_title("(b) Regional budget changes capture", loc="left", pad=10,
               fontsize=T, fontweight="bold")
ax_b.tick_params(labelsize=TK)
ax_b.legend(loc="upper left", fontsize=LG - 2, framealpha=0.92)

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
 translations={'(a) Example page selections':'(a) 页选集示例','(b) Regional budget changes capture':'(b) 区域预算改变保留率','(c) Which 32 of 128 dims':'(c) 128维中选择32维','(d) Historical far proxy (16 seqs)':'(d) 历史远区代理（16序列）','page not selected':'未选中页','selected by avg pool':'均值评分选中','mass-ranked page missed by avg\n(✓ = also selected by minmax)':'质量排序选中、均值漏选\n（✓：极值评分也选中）','avg pool':'均值评分','mass rank':'质量排序','attn mass':'注意力权重和','key position (tokens); query at right end':'键位置（词元）；查询在右端','mass (log)':'权重和（对数）','128 observations, sorted by global far capture':'128条观测，按单池远区保留率排序','far mass capture\n(whole-bin selection)':'远区注意力保留率','selected mass /\nreference mid-TopK mass':'选中／参考mid-TopK\n注意力权重和','layer':'层','far proxy (log)':'远区代理（对数）','near':'近区','tail32\n(low-freq)':'尾部32\n（低频）','random32\n(unpaired)':'随机32\n（未配对）','high32\n(high-freq)':'高频32\n（高频）',f'far ({far_tok_frac:.0%} of mid tokens)':f'远区（中间词元的{far_tok_frac:.0%}）',f'single pool (mean {fc_mean:.2f})':f'单池（均值{fc_mean:.2f}）',f'far/near 1024/1024 ({qc_mean:.2f})':f'远/近1024/1024（{qc_mean:.2f}）',f'global near: {nc_mean:.2f}':f'单池近区：{nc_mean:.2f}'}
 for ax in fig.axes:
  xlim,ylim=ax.get_xlim(),ax.get_ylim()
  ax.set_xticks(ax.get_xticks(), [translations.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
  ax.set_yticks(ax.get_yticks(), [translations.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()])
  ax.set_xlim(xlim);ax.set_ylim(ylim)
 for t in fig.findobj(Text):
  t.set_text(translations.get(t.get_text(),t.get_text()));t.set_fontproperties(FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size=t.get_fontsize()))
stem='o_observation'+('_zh' if LANG=='zh' else '')
fig.savefig(OUT/(stem+'.svg'),bbox_inches='tight');fig.savefig(OUT/(stem+'.png'),dpi=220,bbox_inches='tight')
subprocess.run(['inkscape',str(OUT/(stem+'.svg')),'--export-type=pdf','--export-filename='+str(OUT/(stem+'.pdf'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
plt.close(fig)
summary={'representative':'musique:s1:L24:q14212','a_actual_tokens':actual_budget,'a_selected_page_counts':{a:len(pg['sel_'+a]) for a in ['oracle','avg','minmax']},'a_mass_rank_missed_by_avg':len(missed),'a_missed_also_selected_by_minmax':len(recov),'a_spike_top1_share':spike_top1,'b_global_far_mean':fc_mean,'b_global_near_mean':nc_mean,'b_partition_far_mean':qc_mean,'b_partition_worse_count':int((qc<fc).sum()),'b_far_share_median':float(np.median(sh)),'b_global_far_min':float(fc.min()),'b_partition_far_min':float(qc.min()),'b_global_far_p10':float(np.percentile(fc,10)),'b_partition_far_p10':float(np.percentile(qc,10)),'b_independent_sequences':4,'b_correlated_observations':len(rows),'c_reference_ratios':arm_rm,'d_low_proxy_layer_medians':n_light,'d_per_sequence_low_counts':dict(zip(ROW_ORDER,skip_rng)),'rows':rows}
(DATA/'remote_observation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
