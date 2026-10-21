"""Editable bilingual schematic; source semantics pinned in source_identity.json.
All numbers in the top strip are an integer illustration, not runtime defaults.
One scene creates SVG/PDF and native PowerPoint objects.
"""
from pathlib import Path
import json, subprocess
from xml.sax.saxutils import escape
R=Path(__file__).resolve().parent
W,H=1100,600
FAR,NEAR,PROT='#F4DDD0','#DCECF5','#F1D478'
for lang in ('en','zh'):
 z=lang=='zh'; scene=[]
 def text(x,y,en,zh,size=19,color='#243541',anchor='middle'):
  scene.append(dict(type='text',x=x,y=y,text=zh if z else en,size=size,color=color,anchor=anchor))
 def rect(x,y,w,h,fill='#FFFFFF',stroke='#4E626E',sw=1):
  scene.append(dict(type='rect',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,sw=sw))
 def line(x1,y1,x2,y2,color='#4E626E',arrow=True):
  scene.append(dict(type='line',x1=x1,y1=y1,x2=x2,y2=y2,color=color,arrow=arrow))
 text(550,25,'One chronological sequence · page IDs remain unchanged','同一时间顺序 · 页面 ID 始终不变',23)
 text(211,57,'far','远区',21);text(600,57,'near (unprotected middle)','近区（未保护中间区）',21)
 text(944,57,'SWA','保护滑窗',21)
 # 6 physical pages, last has only two real tokens. B=8, S=42, sink=4, w=8.
 for i,name in enumerate('abcdef'):
  x=55+i*166
  for j in range(8):
   t=8*i+j
   fill=PROT if t<4 or 34<=t<42 else FAR if t<16 else NEAR
   if t>=42:fill='#FFFFFF'
   rect(x+j*18,76,18,29,fill,'#76858D',.7)
   if t>=42:line(x+j*18,76,x+(j+1)*18,105,'#BAC1C5',False)
  rect(x,76,144,29,'none','#1F3440',2 if i in (0,4) else 1)
  text(x+72,132,f'{name}: [{8*i},{min(8*i+8,42)})',f'{name}: [{8*i},{min(8*i+8,42)})',18)
 text(73,157,'sink','前缀',17)
 text(1010,157,'no padding','不读填充',17)
 line(375,66,375,111,'#1F3440',False)
 text(550,186,'Illustration only: B=8, S=42, sink=4, SWA=8; near/far cut=16','仅示例：页宽8，长度42，前缀4，滑窗8；近远分界16',18)
 # Three explicitly distinct paths; protected tokens never depend on page TopK.
 rect(35,216,300,95,'#FAF0E9');text(185,242,'L1 · page competition','L1 · 区域页竞争',21)
 text(185,270,'far: b   |   near: c, d','远区：b   |   近区：c、d',19)
 text(185,296,'32D: far extrema / near mean','32维：远区极值／近区均值',17)
 rect(370,216,350,95,'#EDF5F9');text(545,242,'Boundary pages a, e → eligibility','边界页 a、e → 候选资格',21)
 text(545,270,'outside K1 / regional page quotas','在 K1／区域页配额之外追加',19)
 text(545,296,'sink end / SWA start inside a page','前缀结束／滑窗起点位于页内',17)
 rect(755,216,310,95,'#FCF7DF');text(910,242,'Protected → final support','保护词元 → 最终支持集',21)
 text(910,270,'[0,4) ∪ [34,42)','[0,4) ∪ [34,42)',19)
 text(910,296,'count within final token cap K','计入最终词元上限 K',18)
 line(185,311,185,339);line(545,311,545,339)
 rect(35,339,685,93,'#F5F8FA')
 text(377,365,'Shared 32-coordinate Q/K · L2 logits over full history','共享32坐标 Q/K · L2 对全历史评分',21)
 text(377,394,'page mask → per-Q softmax → KV-group probability mean','页掩码 → 逐Q softmax → KV组概率均值',18)
 text(377,419,'unprotected tokens compete under regional token quotas','未保护词元仍按区域词元配额竞争',18)
 line(377,432,377,462);line(1048,311,1048,485);line(1048,485,730,485)
 rect(35,462,695,48,'#EDF2F5');text(382,493,'Selected ∪ protected IDs · deduplicate · |support| ≤ K','入选 ∪ 保护 ID · 去重 · 支持集大小 ≤ K',21)
 line(382,510,382,543)
 rect(35,543,1030,40,'#F4F6F7')
 text(550,570,'Original-width Q, K, V → one attention softmax on selected support','原维 Q、K、V → 在合并支持集上进行一次注意力 softmax',21)
 text(910,371,'Partitioned · bypass on','分区路径 · 边界保送开启',21)
 text(910,400,'gate / candidate gather off','门控／候选收集关闭',18)
 text(910,429,'Eligibility ≠ final retention','候选资格 ≠ 最终保留',18)
 # Font sizes and positions shared with native editable slide export.
 (R/f'boundary_scene_{lang}.json').write_text(json.dumps(dict(width=W,height=H,objects=scene),ensure_ascii=False,indent=2)+'\n')
 out=['<svg xmlns="http://www.w3.org/2000/svg" width="7in" height="3.82in" viewBox="0 0 1100 600"><defs><marker id="a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L8,4 L0,8z" fill="#4E626E"/></marker></defs><g font-family="Noto Sans CJK SC, DejaVu Sans">']
 for o in scene:
  if o['type']=='text':out.append(f'<text x="{o["x"]}" y="{o["y"]}" font-size="{o["size"]}" fill="{o["color"]}" text-anchor="{o["anchor"]}">{escape(o["text"])}</text>')
  elif o['type']=='rect':out.append(f'<rect x="{o["x"]}" y="{o["y"]}" width="{o["w"]}" height="{o["h"]}" fill="{o["fill"]}" stroke="{o["stroke"]}" stroke-width="{o["sw"]}"/>')
  else:out.append(f'<line x1="{o["x1"]}" y1="{o["y1"]}" x2="{o["x2"]}" y2="{o["y2"]}" stroke="{o["color"]}" stroke-width="1.5"'+(' marker-end="url(#a)"' if o['arrow'] else '')+'/>')
 out.append('</g></svg>');p=R/f'fig2_domains_{lang}.svg';p.write_text(''.join(out))
 for ext in ('pdf','png'):
  subprocess.run(['inkscape',str(p),f'--export-type={ext}','--export-width=1800',f'--export-filename={R/f"fig2_domains_{lang}.{ext}"}'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 # Gate inset remains a separately scoped, editable SVG authority.
 gate=R/f'fig2_gate_{lang}.svg'
 subprocess.run(['inkscape',str(gate),'--export-type=pdf',f'--export-filename={gate.with_suffix(".pdf")}'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
