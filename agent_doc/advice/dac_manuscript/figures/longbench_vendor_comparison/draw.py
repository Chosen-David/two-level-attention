#!/usr/bin/env python3
"""LongBench 16-task vendor comparison bar chart for LoDA-SA paper.

Generates a grouped bar chart showing LoDA-SA vs ported baselines (JSSA,
AsyncTLS) on LongBench 16 tasks plus AVG. Emphasizes that LoDA-SA matches
FullKV and JSSA while outperforming AsyncTLS.

Sources: qwen3_8b_longbench_{tli_champion,jssa_vendor,asynctls_vendor,fullkv}.json
"""
from pathlib import Path
import json, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image

D = Path(__file__).resolve().parent

# ---- Load data ----
def load(path_key):
    p = D / 'inputs' / f'{path_key}.json'
    with open(p) as f:
        d = json.load(f)
    if 'tasks' in d:
        return d['tasks']
    # per-file keyed format (TLI style)
    tasks = {}
    for k, v in d.items():
        if k == '_meta':
            continue
        task = k.split('-')[0]
        tasks[task] = v['score']
    return tasks

tli = load('qwen3_8b_longbench_tli_champion')
jssa = load('qwen3_8b_longbench_jssa_vendor')
atls = load('qwen3_8b_longbench_asynctls_vendor')
fkv = load('qwen3_8b_longbench_fullkv')

# ---- Common task ordering ----
tasks = sorted(fkv.keys())
n = len(tasks)
methods = ['FullKV', 'LoDA-SA', 'JSSA', 'AsyncTLS']
colors = ['#888888', '#0d5580', '#3a7ca5', '#5b8db4']
data = {
    'FullKV': [fkv[t] for t in tasks],
    'LoDA-SA': [tli.get(t, 0) for t in tasks],
    'JSSA': [jssa.get(t, 0) for t in tasks],
    'AsyncTLS': [atls.get(t, 0) for t in tasks],
}
avgs = {m: statistics.mean(v) for m, v in data.items()}

# ---- Render both languages ----
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none',
})
_notofont = D.parent.parent / 'fonts' / 'noto' / 'NotoSansCJKsc-Regular.otf'
ZH = FontProperties(fname=str(_notofont), size=7.5)

for lang in ['en', 'zh']:
    z = lang == 'zh'
    fp = ZH if z else None

    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    x = list(range(n))
    bar_w = .15
    gap = .03
    total_w = len(methods) * (bar_w + gap)

    for i, (method, vals) in enumerate(data.items()):
        offsets = [xi + (i - len(methods)/2 + .5) * (bar_w + gap) for xi in x]
        bars = ax.bar(offsets, vals, bar_w, color=colors[i],
                       edgecolor='#555' if i < 2 else '#aaa', linewidth=.6,
                       label=method if not z else {
                           'FullKV': 'FullKV（稠密）', 'LoDA-SA': 'LoDA-SA',
                           'JSSA': 'JSSA', 'AsyncTLS': 'AsyncTLS',
                       }[method])

    ax.axvline(n - .5, color='#333', lw=1.2, ls='--', alpha=.4)
    ax.text(n + .1, 50, 'AVG', fontsize=8, fontweight='bold', va='center',
            fontproperties=fp if z else None)

    # AVG bars on the right side
    for i, method in enumerate(methods):
        offsets = n + (i - len(methods)/2 + .5) * (bar_w + gap)
        val = avgs[method]
        ax.bar(offsets, val, bar_w, color=colors[i],
               edgecolor='#555' if i < 2 else '#aaa', linewidth=.6)
        ax.text(offsets, val + 2, f'{val:.1f}', ha='center', va='bottom',
                fontsize=7.5, fontweight='bold' if method in ('FullKV', 'LoDA-SA') else 'normal',
                fontproperties=fp if z else None)

    ax.set_xlim(-.8, n + len(methods) * (bar_w + gap) + .5)
    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=35, ha='right', fontsize=7, fontproperties=fp if z else None)
    ax.tick_params(length=2, labelsize=8)
    ax.set_ylabel('Task score (0-100)', fontsize=8, fontproperties=fp if z else None)
    ax.set_title('LongBench 16-task comparison (Qwen3-8B)' if not z
                 else 'LongBench 16任务对比（Qwen3-8B）',
                 fontsize=10, fontweight='bold', pad=10,
                 fontproperties=fp if z else None)
    ax.grid(axis='y', color='#e0e0e0', lw=.4)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    ax.legend(loc='upper right', fontsize=7.5, framealpha=.8, ncol=2,
              prop=fp if z else None,
              bbox_to_anchor=(1.0, .97))

    footer = ('Single runs per method. AVG = unweighted mean over 16 tasks. '
              'LoDA-SA (49.27) matches FullKV (49.39) and JSSA (49.28).'
              if not z else
              '每方法单次运行。AVG = 16任务非加权均值。LoDA-SA (49.27) ≈ FullKV (49.39) ≈ JSSA (49.28)。')
    fig.text(.5, .01, footer, ha='center', fontsize=7.5, color='#444',
             fontproperties=fp if z else None)

    fig.subplots_adjust(bottom=.18, right=.99, top=.90, left=.08)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(D / f'longbench_vendor_compact_{lang}.{ext}', dpi=240)
    plt.close(fig)

    Image.open(D / f'longbench_vendor_compact_{lang}.png').convert('L').save(
        D / f'longbench_vendor_compact_{lang}_gray.png')

print('Done: longbench_vendor_compact_{en,zh}.{pdf,svg,png}')
