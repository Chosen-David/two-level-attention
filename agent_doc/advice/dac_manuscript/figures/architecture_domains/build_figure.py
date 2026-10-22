"""Bilingual LoDA-SA architecture figure (fig2_domains).

cairosvg misplaces baseline-shift tspan pen positions (subscript runs land on
neighbour labels), so subscripts are emitted as separately measured <text>
runs: a probe SVG rendered through the same cairosvg+font stack yields exact
per-run widths, then each run is placed at a cumulative anchor-start x.
Glyph-level acceptance: exp/trace/check_glyph_overlap.py (45% ratio) plus the
pt^2-area recheck must both report zero collisions for en and zh.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import cairosvg
import pymupdf

R = Path(__file__).resolve().parent; O = R; O.mkdir(exist_ok=True)
FAMILY = {'en': 'Liberation Sans', 'zh': 'Noto Sans CJK SC'}
SUB = 11  # subscript font size (SVG px), matches the retired tspan size

# Subscript-bearing labels. Runs: (text, is_sub, gap_before_px).
SUB_LABELS = [
    # (key, x, y, size, weight) + per-lang runs
    (462, 121, 15, 'bold', 'near',
     [('Near · n', 0, 0), ('b', 1, 1), ('pages', 0, 5)],
     [('近区 · n', 0, 0), ('b', 1, 1), ('页', 0, 5)]),
    (662, 121, 15, 'bold', 'far',
     [('Far · K', 0, 0), ('1', 1, 1), ('−n', 0, 1), ('b', 1, 1), ('pages', 0, 5)],
     [('远区 · K', 0, 0), ('1', 1, 1), ('−n', 0, 1), ('b', 1, 1), ('页', 0, 5)]),
    (470, 218, 14, 'normal', 'topk_n',
     [('group mean → TopK(P', 0, 0), ('n', 1, 1), (')', 0, 1)],
     [('组均值 → TopK(P', 0, 0), ('n', 1, 1), (')', 0, 1)]),
    (668, 218, 14, 'normal', 'topk_f',
     [('group mean → TopK(P', 0, 0), ('f', 1, 1), (')', 0, 1)],
     [('组均值 → TopK(P', 0, 0), ('f', 1, 1), (')', 0, 1)]),
    (559, 358, 16, 'normal', 'pool',
     [('Single-pool TopK · K', 0, 0), ('mid', 1, 1), ('slots', 0, 5)],
     [('单池 TopK · K', 0, 0), ('mid', 1, 1), ('槽位', 0, 5)]),
]


def probe_widths(lang):
    """Render every atomic run through cairosvg and measure its ink width (SVG px)."""
    specs = []
    for _, _, size, weight, _, en_runs, zh_runs in SUB_LABELS:
        for t, sub, _ in (zh_runs if lang == 'zh' else en_runs):
            spec = (t, SUB if sub else size, weight)
            if spec not in specs:
                specs.append(spec)
    height = 40 + 26 * len(specs)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="700" height="{height}">'
             f'<g font-family="{FAMILY[lang]}">']
    for i, (t, sz, w) in enumerate(specs):
        parts.append(f'<text x="350" y="{50 + 26 * i}" font-size="{sz}" '
                     f'font-weight="{w}" text-anchor="middle">{escape(t)}</text>')
    parts.append('</g></svg>')
    pdf = cairosvg.svg2pdf(bytestring=''.join(parts).encode())
    doc = pymupdf.open(stream=pdf, filetype='pdf')
    rows = []
    for b in doc[0].get_text('rawdict')['blocks']:
        for l in b.get('lines', []):
            for sp in l.get('spans', []):
                txt = ''.join(c['c'] for c in sp.get('chars', []))
                if txt.strip():
                    rows.append((sp['bbox'][1], (sp['bbox'][2] - sp['bbox'][0]) / 0.75))
    doc.close()
    rows.sort()
    if len(rows) != len(specs):
        raise ValueError(f'probe span count {len(rows)} != spec count {len(specs)}')
    return {spec: w for spec, (_, w) in zip(specs, rows)}


for lang in ('en', 'zh'):
    z = lang == 'zh'
    W = probe_widths(lang)
    s = []

    def text(x, y, a, b='', size=16, anchor='middle', color='#202529', weight='normal'):
        s.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
                 f'fill="{color}" font-weight="{weight}">{escape(b if z and b else a)}</text>')

    def mtext(x, y, size, weight, runs):
        """Emit a centered multi-run label with measured cumulative pen positions."""
        widths = [W[(t, SUB if sub else size, weight)] for t, sub, _ in runs]
        total = sum(widths) + sum(g for _, _, g in runs)
        cx = x - total / 2
        for (t, sub, gap), w in zip(runs, widths):
            cx += gap
            sz = SUB if sub else size
            yy = y + (3 if sub else 0)
            s.append(f'<text x="{cx:.1f}" y="{yy}" font-size="{sz}" text-anchor="start" '
                     f'fill="#202529" font-weight="{weight}">{escape(t)}</text>')
            cx += w

    def rect(x, y, w, h, fill='white', stroke='#424a50', sw=1):
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"/>')

    def line(ps, arrow=True, dash=False, width=1.3, color='#424a50'):
        s.append('<polyline points="' + ' '.join(f'{x},{y}' for x, y in ps) +
                 f'" fill="none" stroke="{color}" stroke-width="{width}"' +
                 (' stroke-dasharray="4 3"' if dash else '') +
                 (' marker-end="url(#a)"' if arrow else '') + '/>')

    def grid(x, y, rows, cols, cw=12, ch=12, kept=(), fill='#edf0f2'):
        for r in range(rows):
            for c in range(cols):
                rect(x + c * cw, y + r * ch, cw, ch,
                     '#668398' if r in kept else fill, '#58656e', .65)

    def page(x, y, keep=False):
        rect(x, y, 28, 42, '#668398' if keep else 'white')
        for j in (10, 20, 30):
            line([(x + 4, y + j), (x + 24, y + j)], False, width=.55,
                 color='#d9e0e5' if keep else '#c2c8cc')

    # Three claim-bearing domains, one per design decision.
    text(91, 23, 'Unmodified KV cache', '原维KV缓存', 17, weight='normal')
    text(495, 23, 'One 32-coordinate view serves both stages', '同一32坐标视图服务两级', 17, weight='normal')
    text(911, 23, 'Single pool, original width', '单池 · 原维注意力', 17, weight='normal')
    rect(218, 68, 562, 330, '#fafbfc', '#aeb9c0', 1)
    text(23, 54, 'Q', 'Q', 18)
    line([(39, 49), (998, 49), (998, 322), (975, 322), (975, 338)], width=2)
    text(907, 45, 'original 128D', '原始128维', 14)
    # Raw storage, pages grouped in rows.
    text(8, 153, 'K', 'K', 17); text(124, 94, 'V', 'V', 17)
    grid(20, 101, 8, 5, 12, 13); grid(94, 101, 8, 5, 12, 13)
    text(87, 224, 'tokens × 128D', 'token × 128维', 14)
    text(87, 247, 'Unmodified cache', '缓存保持原维', 14)
    # The shared coordinate view is shown as a tied two-half selection.
    line([(50, 101), (50, 73), (173, 73), (173, 120), (239, 120)], width=1.4)
    line([(197, 49), (197, 98), (239, 98)])
    text(310, 92, 'Shared 32-coordinate view', '共享32坐标表示', 16)
    for y in (105, 126):
        for i in range(8):
            rect(252 + i * 13, y, 11, 10, '#668398' if i >= 6 else 'white', '#58656e', .8)
    for i in (6, 7):
        line([(257 + i * 13, 115), (257 + i * 13, 126)], False, width=.8)
    text(310, 154, '16 complete RoPE pairs', '16对完整RoPE坐标', 13)
    text(510, 94, 'L1 · page budgets', 'L1 · 页预算', 16)
    # Two regional lanes represented with memory pages and selected supports.
    mtext(462, 121, 15, 'bold', SUB_LABELS[0][6 if z else 5])
    mtext(662, 121, 15, 'bold', SUB_LABELS[1][6 if z else 5])
    line([(364, 115), (397, 115), (397, 151), (414, 151)])
    line([(397, 115), (397, 102), (596, 102), (596, 151), (611, 151)])
    for i in range(4):
        page(414 + i * 29, 134, i in (0, 2))
    for i in range(4):
        page(611 + i * 29, 134, i in (1, 3))
    text(470, 197, 'mean-key scores', '均值键打分', 14)
    text(668, 197, 'sign-aware min/max', '符号感知极值', 14)
    mtext(470, 218, 14, 'normal', SUB_LABELS[2][6 if z else 5])
    mtext(668, 218, 14, 'normal', SUB_LABELS[3][6 if z else 5])
    # Shared mask gathers the two regional supports (no separate probability normalization).
    line([(470, 226), (470, 245), (532, 245)])
    line([(668, 226), (668, 245), (595, 245)])
    for i in range(8):
        rect(532 + i * 8, 237, 7, 15, '#668398' if i in (0, 2, 5, 7) else 'white', '#58656e', .7)
    text(563, 273, 'shared page mask · boundary pages bypass quotas', '共享页掩码 · 边界保送页在配额外', 13)
    text(563, 292, 'filled = kept', '实色 = 保留', 12)
    # L2 shares representation but full-history scoring precedes application of the mask.
    line([(310, 161), (310, 292)])
    text(310, 309, 'L2 · full-history logits', 'L2 · 全历史logits', 15)
    text(310, 329, 'float QDQ · 1/√128', '浮点QDQ · 1/√128', 13)
    line([(395, 309), (425, 309)])
    text(559, 313, 'mask → softmax/Q → KV-group mean', '掩码 → 逐Q softmax → KV组均值', 14)
    line([(563, 253), (563, 258), (430, 258), (430, 299)])
    line([(559, 321), (559, 341)])
    mtext(559, 358, 16, 'normal', SUB_LABELS[4][6 if z else 5])
    text(310, 374, 'prefix ∪ recent IDs', '前缀 ∪ 滑窗ID', 14)
    line([(397, 370), (455, 370), (455, 384), (698, 384), (698, 367)])
    line([(678, 351), (693, 351)])
    text(730, 356, '∪ / dedup', '合并去重', 14)
    text(730, 377, '|IDs| ≤ K', '|IDs| ≤ K', 13)
    # Thin ID route and thick original KV route meet only at gather.
    line([(779, 351), (807, 351), (807, 207), (838, 207)])
    text(815, 193, 'IDs', 'ID', 13)
    rect(838, 190, 149, 105, 'white', '#88949d', 1)
    text(908, 212, 'Retained K/V', '所选K/V支持集', 15)
    grid(850, 229, 3, 5, 11, 10); grid(913, 229, 3, 5, 11, 10)
    text(877, 279, 'K', 'K', 16); text(940, 279, 'V', 'V', 16)
    line([(87, 255), (87, 422), (821, 422), (821, 275), (838, 275)], width=2)
    text(451, 441, 'Original K/V · 128D', '原始K/V · 128维', 14)
    line([(877, 295), (877, 320), (898, 320), (898, 338)], width=2)
    line([(940, 295), (940, 320), (898, 320)], False, width=2)
    rect(838, 338, 149, 56, '#edf2f5', '#424a50', 1.2)
    text(912, 360, 'QKᵀ → softmax → V', 'QKᵀ → softmax → V', 14)
    text(912, 381, 'original 128D', '原始128维', 14)
    line([(987, 365), (1005, 365)], width=2); text(996, 408, 'o', 'o', 18)
    # Optional far gate strip: disabled by default, mirrors observation panel (d).
    rect(0, 459, 1008, 27, '#faf0e5', '#ae6b2d', 1)
    text(504, 477, 'Optional far gate: prefill-calibrated, frozen at decode · disabled in this reference',
         '可选远区门控：预填充标定 · 解码冻结 · 本参考配置不启用', 13, color='#ae6b2d')

    defs = ('<defs><marker id="a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" '
            'markerHeight="5" orient="auto"><path d="M0,0 L8,4 L0,8z" fill="#424a50"/>'
            '</marker></defs>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="7in" height="3.47in" '
           f'viewBox="0 0 1008 490">{defs}<g font-family="{FAMILY[lang]}">' +
           ''.join(s) + '</g></svg>')
    p = O / f'fig2_domains_{lang}.svg'
    p.write_text(svg)
    cairosvg.svg2pdf(url=str(p), write_to=str(O / f'fig2_domains_{lang}.pdf'))
    cairosvg.svg2png(url=str(p), write_to=str(O / f'fig2_domains_{lang}.png'), output_width=1600)

# The editable optional-control inset is retained as a separately scoped source.
for lang in ('en', 'zh'):
    p = O / f'fig2_gate_{lang}.svg'
    cairosvg.svg2pdf(url=str(p), write_to=str(O / f'fig2_gate_{lang}.pdf'))
    cairosvg.svg2png(url=str(p), write_to=str(O / f'fig2_gate_{lang}.png'), output_width=1600)
