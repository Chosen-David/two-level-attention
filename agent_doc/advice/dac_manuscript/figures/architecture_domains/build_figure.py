"""Canonical selected-configuration diagram. Old PPT/boundary scenes are archives.
Run from the pinned manuscript font environment; requires cairosvg.
SVG millimetre coordinates fix physical text sizes; outputs share one scene.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import argparse, hashlib, json, subprocess
import cairosvg
P=argparse.ArgumentParser();P.add_argument('--language',choices=('en','zh','both'),default='both');A=P.parse_args()
R=Path(__file__).resolve().parent
C={'white':'#FFFFFF','ink':'#263442','muted':'#63717C','border':'#CAD3DA','view':'#6557A8','near':'#247D8A','far':'#B86C35','protect':'#A77A12'}
for lang in (('en','zh') if A.language=='both' else (A.language,)):
 z=lang=='zh';s=[]
 def t(x,y,en,zh='',pt=8.5,c='ink',bold=False,anchor='middle'):
  s.append(dict(kind='text',x=x,y=y,text=zh if z and zh else en,pt=pt,c=c,bold=bold,anchor=anchor))
 def box(x,y,w,h,fill='white',c='border',radius=.7):s.append(dict(kind='rect',x=x,y=y,w=w,h=h,fill=fill,c=c,r=radius))
 def line(pts,c='ink',dash=False,arrow=True):s.append(dict(kind='line',points=pts,c=c,dash=dash,arrow=arrow))
 def page(x,y,c):
  box(x,y,3.8,5,'white',c,.3)
  for j in range(3):line([(x+.7,y+1.1+j*1.25),(x+3.1,y+1.1+j*1.25)],c,arrow=False)
 for x,w,a,b,za,zb in [(0,30,'Shared view','','共享坐标',''),(34,47,'L1 · Page admission','','L1 · 区域选页',''),(85,57,'L2 · Token competition','','L2 · 词元竞争',''),(146,34,'Original-width','attention','原始维度','注意力')]:
  t(x+w/2,4.6,a,za,pt=9.5,bold=True)
  if b:t(x+w/2,9,b,zb,pt=9.5,bold=True)
  line([(x,11),(x+w,11)],'border',arrow=False)
 # One paired post-RoPE scoring view, V excluded.
 t(15,22,'Q, K · 128D','Q、K · 128维',bold=True)
 t(15,28,'post-RoPE','旋转后',pt=8,c='muted')
 for y in (34,40):
  for i in range(16):box(1+i*1.75,y,1.25,3,'#6557A8','white',.1)
 for i in range(16):line([(1.625+i*1.75,37),(1.625+i*1.75,40)],'view',arrow=False)
 t(15,50,'32D shared view','32维共享视图',pt=9,bold=True,c='view')
 t(15,56,'16 complete','16个完整',pt=8)
 t(15,60,'low-frequency','低频',pt=8)
 t(15,64,'RoPE pairs','RoPE旋转对',pt=8)
 # Shared view feeds L1 and L2 directly, not through L1 shortlisting.
 line([(29,47),(32,47),(32,15),(113.5,15),(113.5,19)],'view')
 line([(32,47),(32,18),(45,18),(45,21)],'view')
 line([(45,18),(69,18),(69,21)],'view')
 for x,c,fill,name,zh,num,score,score2,zs,zs2 in [(35,'near','#F0F7F8','Near','近区','32','Mean-key','scores','均值键','评分'),(59,'far','#FBF4EF','Far','远区','96','Sign-aware','extrema','符号感知','极值')]:
  box(x,21,22,41,fill,c)
  t(x+11,25,name,zh,pt=8.5,bold=True,c=c)
  for j in range(3):page(x+3+j*5.5,28,c)
  t(x+11,37,score,zs,pt=8)
  t(x+11,41,score2,zs2,pt=8)
  t(x+11,46,'Mean scores','均值评分',pt=7.5)
  t(x+11,50,'in KV group','KV组内',pt=7.5)
  t(x+11,55,'Top pages','页TopK',pt=8,bold=True)
  t(x+11,60,num+' pages',num+' 页',pt=9,bold=True,c=c)
 line([(46,62),(46,66),(83,66),(83,32),(88,32)],'muted',dash=True)
 line([(70,62),(70,66)],'muted',dash=True,arrow=False)
 # Separate page eligibility bypass.
 page(36,72,'muted')
 t(61,74,'Boundary pages','边界页',pt=8)
 t(60,78,'outside page quotas','页配额之外',pt=7.5,c='muted')
 line([(80,75),(83,75),(83,66)],'muted',dash=True,arrow=False)
 # Full-history logits -> mask -> per-query probabilities -> group mean -> pooled slots.
 box(88,19,51,56,'#F8F6FB')
 t(113.5,24,'Full-history logits · 32D','全历史logits · 32维',pt=8.5,bold=True)
 line([(113.5,25.5),(113.5,28.5)],'view')
 t(113.5,33,'Page mask','页掩码',pt=8.5)
 line([(113.5,35),(113.5,38)],'view')
 t(113.5,42,'Softmax per query','逐查询softmax',pt=8.5)
 line([(113.5,44),(113.5,47)],'view')
 t(113.5,51,'Mean probabilities','概率均值',pt=8.5)
 t(113.5,55,'within KV group','KV组内',pt=8)
 line([(113.5,57),(113.5,60)],'view')
 t(113.5,65,'One pooled TopK','单池TopK',pt=9,bold=True,c='view')
 t(113.5,69,'892 nominal slots','892个名义槽位',pt=8)
 for i in range(16):box(93+i*2.6,71,2,1.7,C['near' if i%3 else 'far'],'white',.1)
 # Token protection bypasses L2 and merges only with pooled winners.
 box(0,72,30,10,'#FCF8EB','protect')
 t(15,76,'Protected tokens','保护词元',pt=8,c='protect',bold=True)
 t(15,80,'sink 4 · recent 128','前缀4 · 最近128',pt=7.5,c='protect')
 line([(30,79),(32,79),(32,83),(144,83),(144,29),(147,29)],'protect',dash=True)
 line([(139,64),(142,64),(142,23),(147,23)],'muted',dash=True)
 box(147,19,32,20,'#F3F6F8')
 t(163,24,'Union + deduplicate','合并 + 去重',pt=8,bold=True)
 t(163,34,'≤1024 tokens','≤1024词元',pt=9,bold=True)
 line([(163,39),(163,49)],'muted',dash=True)
 t(168,45,'IDs','ID',pt=7.5,c='muted',anchor='start')
 box(147,49,32,20,'#F1F5F8','ink')
 t(163,54,'Attention on support','支持集上注意力',pt=8,bold=True)
 t(163,60,'Original Q, K, V','原始 Q、K、V',pt=8)
 t(163,65,'128D','128维',pt=8.5,bold=True)
 line([(163,69),(163,73)],'ink');t(163,78,'o',pt=9)
 # Low-emphasis raw-width lane enters attention, never the scoring probability path.
 t(1,87,'Original 128D Q / K / V','原始128维 Q / K / V',pt=7.5,anchor='start')
 line([(43,87),(179.8,87),(179.8,63),(179,63)],'ink')
 t(62,91,'Solid: tensors','实线：数值张量',pt=7.5,c='muted')
 t(121,91,'Dashed: masks / IDs','虚线：掩码／ID',pt=7.5,c='muted')
 # Physical millimetre scene preserves all declared point sizes.
 H=92
 family='Noto Sans CJK SC' if z else 'Liberation Sans'
 defs='<defs>'+''.join(f'<marker id="{k}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="3" markerHeight="3" orient="auto"><path d="M0,0L8,4L0,8Z" fill="{v}"/></marker>' for k,v in C.items())+'</defs>'
 out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{H}mm" viewBox="0 0 180 {H}"><rect width="180" height="{H}" fill="white"/>{defs}<g font-family="{family}">']
 for o in s:
  if o['kind']=='text':out.append(f'<text x="{o["x"]}" y="{o["y"]}" font-size="{o["pt"]*25.4/72}" text-anchor="{o["anchor"]}" fill="{C[o["c"]]}" font-weight="{700 if o["bold"] else 400}">{escape(o["text"])}</text>')
  elif o['kind']=='rect':out.append(f'<rect x="{o["x"]}" y="{o["y"]}" width="{o["w"]}" height="{o["h"]}" rx="{o["r"]}" fill="{o["fill"]}" stroke="{C[o["c"]]}" stroke-width=".22"/>')
  else:out.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in o['points'])+f'" fill="none" stroke="{C[o["c"]]}" stroke-width="{.28 if not o["dash"] else .24}"'+(' stroke-dasharray="1.2 .9"' if o['dash'] else '')+(f' marker-end="url(#{o["c"]})"' if o['arrow'] else '')+'/>')
 out.append('</g></svg>');p=R/f'fig2_domains_{lang}.svg';p.write_text(''.join(out))
 (R/f'architecture_scene_{lang}.json').write_text(json.dumps({'width_mm':180,'height_mm':H,'font':family,'objects':s},ensure_ascii=False,indent=2)+'\n')
 cairosvg.svg2pdf(url=str(p),write_to=str(p.with_suffix('.pdf')))
 cairosvg.svg2png(url=str(p),write_to=str(p.with_suffix('.png')),output_width=1800)

files=[p for p in R.glob('fig2_domains_*.*') if p.suffix in ('.svg','.pdf','.png')] + list(R.glob('architecture_scene_*.json')) + [Path(__file__).resolve()]
manifest={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'generator':'build_figure.py','dimensions_mm':[180,92],'fonts':{'en':'Liberation Sans','zh':'Noto Sans CJK SC'},'declared_text_pt':{'titles':9.5,'body':[8,9],'secondary':7.5},'archive_note':'Old boundary_scene and PPT objects are archived; not synchronized to this selected-configuration scene.','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
(R/'architecture_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
