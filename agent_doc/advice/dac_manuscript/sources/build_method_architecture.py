from pathlib import Path
import html, subprocess, argparse, json
P=argparse.ArgumentParser(); P.add_argument('--out',required=True); A=P.parse_args(); OUT=Path(A.out); OUT.mkdir(parents=True,exist_ok=True)
C={'ink':'#243746','muted':'#617481','line':'#CBD5DC','pale':'#EDF1F4','purple':'#76608D','pl':'#EEE9F3','near':'#4B7FB3','nl':'#E7EFF7','far':'#12695F','fl':'#E5F3F0','gate':'#AE6B2D','gl':'#FAF0E5','white':'#FFFFFF'}
class S:
 def __init__(self,w,h,lang):
  self.w=w; self.h=h; self.lang=lang; self.a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}pt" height="{h}pt" viewBox="-4 -4 {w+8} {h+8}"><defs>']
  for c,v in C.items(): self.a.append(f'<marker id="{c}" markerWidth="5" markerHeight="5" refX="4.5" refY="2.5" orient="auto"><path d="M0 0L5 2.5L0 5Z" fill="{v}"/></marker>')
  self.a+=['</defs>',f'<rect width="{w}" height="{h}" fill="white"/>']
 def t(self,x,y,en,zh=None,size=8,c='ink',bold=False,anchor='start'):
  txt=en if self.lang=='en' or zh is None else zh
  self.a.append(f'<text x="{x}" y="{y}" font-family="{ "Liberation Sans" if self.lang=="en" else "Noto Sans CJK SC"}" font-size="{size}" fill="{C[c]}" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{html.escape(txt)}</text>')
 def r(self,x,y,w,h,c='pale',stroke=None,rx=0):self.a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{C[c]}"'+(f' stroke="{C[stroke]}" stroke-width=".65"' if stroke else '')+'/>')
 def line(self,pts,c='line',arrow=False,dash=False):self.a.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in pts)+f'" fill="none" stroke="{C[c]}" stroke-width=".8"'+(f' marker-end="url(#{c})"' if arrow else '')+(' stroke-dasharray="3 2"' if dash else '')+'/>')
 def grid(self,x,y,rows,cols,c='purple',cw=4,ch=4,gap=1,sel=None):
  for r in range(rows):
   for col in range(cols):self.r(x+col*(cw+gap),y+r*(ch+gap),cw,ch,c if sel is None or col in sel else 'pale')
 def title(self,n,en,zh):self.t(0,10,n,size=10,c='purple',bold=True);self.t(17,10,en,zh,size=9.2,bold=True);self.line([(0,17),(self.w,17)])
 def save(self,n):
  f=OUT/f'{n}_{self.lang}.svg';f.write_text('\n'.join(self.a+['</svg>']))
  import cairosvg
  cairosvg.svg2pdf(url=str(f),write_to=str(f.with_suffix('.pdf')))
  cairosvg.svg2png(url=str(f),write_to=str(f.with_suffix('.png')),output_width=1300)

def page(s,x,y,c,selected=True):
 s.r(x,y,16,22,'white',c if selected else 'line',2)
 s.r(x+3,y+4,10,2,c if selected else 'line')
 for r in range(2):
  for k in range(3):s.r(x+3+k*3.6,y+9+r*5,2.6,3,c if selected else 'pale')
def pill(s,x,y,w,en,zh,c):
 s.r(x,y,w,16,{'purple':'pl','near':'nl','far':'fl','gate':'gl'}[c],None,3)
 s.t(x+w/2,y+11,en,zh,size=8,c=c,anchor='middle')
for lang in ['en','zh']:
 s=S(252,166,lang);s.title('a','One coordinate set, two scoring stages','同一坐标集，服务两级评分')
 s.t(0,31,'Post-RoPE Q / K',size=8.5,bold=True)
 s.t(252,31,'128D → 32D','128 维 → 32 维',size=9,c='purple',bold=True,anchor='end')
 # Split halves are drawn as paired strips, with the selected tail in both.
 for y,left,right in [(42,'0 … 47','48 … 63'),(65,'64 … 111','112 … 127')]:
  s.r(0,y,98,10,'pale',None,1)
  for j in range(16):s.r(103+j*3.1,y,2.2,10,'purple',None,.6)
  s.t(4,y+8,left,size=7.5,c='muted')
  s.t(159,y+8,right,size=8,c='purple')
 for j in range(16):s.line([(104.1+j*3.1,52),(104.1+j*3.1,65)],'purple')
 s.t(0,61,'j ↔ j+64',size=8,c='muted')
 s.t(0,94,'16 complete pairs','16 个完整旋转对',size=8,c='purple')
 s.line([(126,76),(126,99)],'purple',True)
 # Connected shared view feeds both stages; Q rows are explicitly separate.
 s.r(0,103,252,59,'pl',None,4)
 s.t(8,116,'Same J for every Q and K','所有 Q / K 使用同一个 J',size=8.5,bold=True,c='purple')
 s.grid(8,123,4,8,'purple',cw=3,ch=3,gap=1)
 s.t(8,153,'Shared K','共享 K',size=8,c='purple',bold=True)
 s.line([(42,134),(54,134),(54,126),(67,126)],'purple',True)
 s.line([(54,134),(54,147),(67,147)],'purple',True)
 s.t(72,129,'L1 summaries','L1 页摘要',size=8.5)
 s.t(72,150,'L2 Q–DQ keys','L2 量化反量化键',size=8.5)
 s.line([(178,122),(178,154)],'line')
 for r in range(4):s.grid(189,123+r*6,1,8,'purple',cw=4,ch=4,gap=1)
 s.t(184,154,'Separate Q heads','独立 Q 头',size=8,c='purple')
 s.save('subspace')
 s=S(252,228,lang);s.title('b','Regional pages, one token pool','区域选页，单一词元池')
 # Two equal, aligned region cards. Page glyphs instead of anonymous grids.
 for x,c,en,zh,quota in [(0,'far','Far · extrema','远区 · 极值',96),(130,'near','Near · mean key','近区 · 均值键',32)]:
  s.r(x,25,122,62,'fl' if c=='far' else 'nl',None,4)
  s.t(x+7,37,en,zh,size=8.5,c=c,bold=True)
  for i in range(4):page(s,x+7+i*21,44,c,i!=1)
  pill(s,x+6,69,110,f'L1 · {quota} pages',f'L1 · {quota} 页',c)
 s.line([(61,88),(61,96),(126,96)],'far');s.line([(191,88),(191,96),(126,96)],'near')
 s.line([(126,96),(126,103)],'ink',True)
 s.t(126,113,'Full-history scores → L1 mask','全历史评分 → L1 掩码',size=8.5,bold=True,anchor='middle')
 # Matrix makes shared support across query heads immediately visible.
 for r in range(4):
  s.t(0,123+r*7.5,f'q{r+1}',size=7.5,c='muted')
  for j in range(24):s.r(20+j*6.8,117+r*7.5,5.8,4.5,'pale' if j in [4,5,12,13,19] else ('far' if j<16 else 'near'),None,.5)
 s.t(190,127,'Shared mask','共享掩码',size=7.5,c='muted');s.t(190,137,'grey = masked','灰色 = 屏蔽',size=7.5,c='muted')
 s.line([(126,146),(126,149)],'ink',True)
 s.r(0,151,252,19,'pale',None,3)
 s.t(126,164,'Per-Q softmax → GQA probability mean','逐 Q softmax → GQA 组内概率均值',size=8.5,bold=True,anchor='middle')
 s.line([(126,171),(126,177),(61,177),(61,183)],'far',True)
 s.line([(126,177),(191,177),(191,183)],'near',True)
 s.t(126,194,'Single pool · free-competition TopK','单池 · 自由竞争 TopK',size=8.5,c='ink',bold=True,anchor='middle')
 s.grid(0,200,1,28,'purple',cw=8,ch=4,gap=1)
 s.line([(116,206),(116,211),(126,211)],'purple');s.line([(126,211),(126,215)],'ink',True)
 s.t(126,224,'Indices ∪ protection → original-space attention','词元索引 ∪ 保护集 → 原维注意力',size=8.5,bold=True,anchor='middle')
 s.save('near_far')
 s=S(252,218,lang);s.title('c','Layerwise gates: prefill → decode','逐层门控：prefill → decode')
 s.t(31,31,'Prefill',size=9,bold=True);s.t(103,31,'Freeze','冻结',size=9,c='gate',bold=True);s.t(200,31,'Decode','Decode',size=9,bold=True,anchor='middle')
 s.t(31,45,'Full-D Q/K','全维 Q/K',size=8)
 s.t(103,45,'at entry','入口处',size=8,c='gate')
 for x,step in [(151,'t'),(190,'t+1'),(229,'t+2')]:s.t(x+9,45,step,size=8,anchor='middle')
 s.line([(117,49),(117,177)],'gate',False,True)
 # Three layer states are illustrative, not measured outcomes. Prefill always observes full attention.
 for row,(label,on) in enumerate([('ℓ₁',True),('ℓ₂',False),('ℓ₃',True)]):
  y=61+row*39
  s.r(0,y-6,252,35,'pale' if row%2==0 else 'white',None,3)
  s.t(3,y+12,label,size=10,bold=True)
  s.grid(31,y,4,8,'purple',cw=3,ch=3,gap=1)
  s.t(31,y+26,'far weight sum ≥ ε' if on else 'far weight sum < ε','远区注意力权重总和 ≥ ε' if on else '远区注意力权重总和 < ε',size=6.2,c='muted')
  s.line([(67,y+8),(100,y+8)],'gate',True)
  s.r(103,y-1,28,19,'gl','gate',3);s.t(117,y+12,'ON' if on else 'OFF',size=8,c='gate',bold=True,anchor='middle')
  s.line([(132,y+8),(146,y+8)],'gate',True)
  for x in [151,190,229]:
   # upper = near L1/L2; lower = far L1/L2. Both far stages are crossed when OFF.
   for dx in [0,11]:
    s.r(x+dx,y,9,6,'near',None,1)
    s.r(x+dx,y+10,9,6,'far' if on else 'white',None if on else 'line',1)
   if not on:s.line([(x-1,y+18),(x+21,y+8)],'gate')
   if x<229:s.line([(x+22,y+8),(x+34,y+8)],'muted',True)
 s.r(0,178,9,6,'near',None,1);s.t(14,185,'Near (top)','近区（上）',size=8,c='near',bold=True)
 s.r(98,178,9,6,'far',None,1);s.t(112,185,'Far (bottom)','远区（下）',size=8,c='far',bold=True)
 s.t(252,185,'╱ = skip','╱ = 跳过',size=8,c='gate',anchor='end')
 s.t(0,199,'OFF: L2 refill → near; near L1 cap fixed','OFF：L2 余量回填近区；近区 L1 上限不变',size=8)
 s.t(0,212,'Illustrative layers · gate disabled by default','层级状态仅作示意 · 门控默认不启用',size=8,c='gate',bold=True)
 s.save('gate')
 s=S(516,190,lang)
 s.t(0,12,'Shared-subspace near–far selection','共享子空间近远区域选择',size=11,bold=True)
 for x,n,en,zh in [(0,'01','Shared view','共享评分视图'),(123,'02','Regional L1','区域 L1'),(261,'03','Token L2','词元 L2'),(404,'04','Original attention','原维注意力')]:
  s.t(x,33,n,size=9,c='purple',bold=True);s.t(x+19,33,en,zh,size=9,bold=True)
 s.line([(0,41),(516,41)])
 s.t(0,55,'Post-RoPE Q / K','RoPE 后 Q / K',size=8);s.grid(0,64,4,16,cw=4,ch=4,sel=range(12,16));s.t(0,94,'128D → 32D','128 维 → 32 维',c='purple',bold=True);s.t(0,107,'16 complete pairs','16 个完整旋转对',size=7.8);s.t(0,136,'Original Q/K/V','原始 Q/K/V',size=8,bold=True)
 s.line([(85,77),(111,77)],'purple',True)
 s.t(123,55,'Far: extrema','远区：极值',c='far',bold=True);s.t(123,103,'Near: mean key','近区：均值键',c='near',bold=True)
 for y,c in [(64,'far'),(112,'near')]:
  for i in range(5):s.grid(123+i*22,y,2,3,c if i!=2 else 'pale',cw=4,ch=4)
 s.t(123,88,'96 pages','96 页',size=7.7,c='far');s.t(123,136,'16 pages','16 页',size=7.7,c='near')
 s.line([(234,77),(250,77)],'far',True);s.line([(234,120),(250,120)],'near',True)
 
 for r in range(4):
  for j in range(22):s.r(261+j*5,52+r*5,4,4,'pale' if j in [0,1,6,7,16,17,21] else ('far' if j<12 else 'near'))
 s.t(282,79,'far',size=6.5,c='far');s.t(331,79,'near',size=6.5,c='near');s.t(261,90,'Full-history scoring','全历史评分',size=8,bold=True)
 s.t(261,102,'Mask → per-Q softmax','掩码 → 逐 Q softmax',size=7.8);s.t(261,114,'→ GQA probability mean','→ GQA 概率均值',size=7.8)
 s.t(261,126,'Single-pool TopK','单池 TopK',size=8,bold=True);s.grid(261,134,1,17,'purple',cw=5,ch=5)
 s.line([(382,77),(397,77)],'ink',True)
 s.t(404,55,'Indices ∪ protection','词元索引 ∪ 保护集',size=8,bold=True);s.grid(404,65,1,3,'far',cw=5,ch=5);s.grid(424,65,1,8,'near',cw=5,ch=5);s.grid(474,65,1,4,'ink',cw=5,ch=5);s.t(404,79,'far','远区',size=6.5,c='far',bold=True);s.t(438,79,'near','近区',size=6.5,c='near',bold=True);s.t(477,79,'protected','保护',size=6.5,c='ink',bold=True);s.t(404,93,'Gather original K/V','聚集原始 K/V',size=8);s.t(404,109,'Original Q · 128D','原始 Q · 128 维',size=8);s.t(404,124,'Per-Q attention','每个 Q 独立注意力',size=8,bold=True)
 s.line([(0,140),(0,154),(459,154),(459,135)],'muted',True);s.t(145,151,'Original Q/K/V bypass · protected tokens retained','原始 Q/K/V 旁路 · 始终保留保护词元',size=7.7,c='muted')
 s.r(0,167,516,23,'gl');s.t(7,182,'Optional far gate: prefill → freeze → reuse; OFF skips far L1 + L2','可选远区门控：prefill → 冻结 → 复用；OFF 跳过远区 L1 + L2',size=8,c='gate');s.t(509,182,'Disabled by default','默认不启用',size=8,c='gate',bold=True,anchor='end')
 s.save('overview')
