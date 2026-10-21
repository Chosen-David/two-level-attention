from pathlib import Path
import json
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from lxml import etree
R=Path(__file__).resolve().parents[1]
P=R/'presentation/TwoLevel_UserEdited_20261014.pptx'
p=Presentation(P)
assert len(p.slides)==17
def replace_text(shape,text):
 tf=shape.text_frame
 first=tf.paragraphs[0]
 # Preserve existing font/paragraph properties while replacing visible content.
 if first.runs:
  first.runs[0].text=text.replace("\v", "\n").replace("_x000B_", "\n")
  for r in list(first.runs)[1:]:r._r.getparent().remove(r._r)
 else:first.text=text
 for par in list(tf.paragraphs)[1:]:par._p.getparent().remove(par._p)
def replace_picture(slide,old,path):
 new=slide.shapes.add_picture(str(path),old.left,old.top,old.width,old.height)
 old._element.getparent().remove(old._element)
 return new
replace_text(p.slides[1].shapes[7],'极值在两区的漏失均较低；混合配额诊断未胜全局极值（.135 / .121）。历史稀疏解码更慢，当前设计尚缺匹配质量—成本结果。')
s=p.slides[2]
replace_picture(s,next(sh for sh in s.shapes if sh.shape_type==13),R/'figures/o_observation_zh.png')
replace_text(s.shapes[4] if s.shapes[4].has_text_frame else s.shapes[3], '(a) 均值／极值同32维，参考为全维；实际词元数不同；(b) 各区漏失 ÷ 每查询参考集总质量，再取均值；\n极值两区均较低；23种拆分最佳 .135，仍差于全局极值 .121；(c) 随机未配对；(d) 历史代理，非解码保证。')
replace_text(p.slides[4].shapes[9],'nl 指定近区范围；nb 预留近区页\n合并候选统一 L2；最终原维 Q / K / V')
s=p.slides[5]
replace_text(s.shapes[2],'分区参考路径：边界保送开启，门控／候选收集关闭；示例数字不是默认配置；归档评测不验证新路径')
for sh in list(s.shapes)[4:]:sh._element.getparent().remove(sh._element)
scene=json.loads((R/'figures/architecture_domains/boundary_scene_zh.json').read_text())
scale=6.6/600
ox,oy=(16.6666666667-1100*scale)/2,2.04
def X(x):return Inches(ox+x*scale)
def Y(y):return Inches(oy+y*scale)
def color(v):return RGBColor.from_string(v.lstrip('#'))
for o in scene['objects']:
 if o['type']=='rect':
  sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,X(o['x']),Y(o['y']),Inches(o['w']*scale),Inches(o['h']*scale))
  if o['fill']=='none':sh.fill.background()
  else:sh.fill.solid();sh.fill.fore_color.rgb=color(o['fill'])
  sh.line.color.rgb=color(o['stroke']);sh.line.width=Pt(o['sw']*scale*72)
 elif o['type']=='line':
  sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,X(o['x1']),Y(o['y1']),X(o['x2']),Y(o['y2']))
  sh.line.color.rgb=color(o['color']);sh.line.width=Pt(1)
  if o['arrow']:
   ln=sh._element.spPr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ln')
   etree.SubElement(ln,'{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd',type='triangle',w='sm',len='sm')
 else:
  width=1100 # Centered text box retains exact center; no word-wrap.
  sh=s.shapes.add_textbox(X(o['x']-width/2),Y(o['y']-o['size']*1.05),Inches(width*scale),Inches(o['size']*1.5*scale))
  tf=sh.text_frame;tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0;tf.word_wrap=False
  par=tf.paragraphs[0];par.alignment=PP_ALIGN.CENTER
  run=par.add_run();run.text=o['text'];run.font.name='Noto Sans CJK SC';run.font.size=Pt(o['size']*scale*72);run.font.color.rgb=color(o['color'])
replace_text(p.slides[6].shapes[2],'16完整旋转对，两级共用；图示QDQ为历史量化变体，主文bf16配置不以该量化为必要条件；最终为原维注意力')
# Current method replaces the obsolete numeric-token-quota schematic with editable objects.
s=p.slides[7]
replace_text(s.shapes[1],'nl / nb：区域页准入，统一词元竞争')
replace_text(s.shapes[2],'保护不重叠的长历史；4/128保护，K=1024；无近远词元配额')
for sh in list(s.shapes)[4:]: sh._element.getparent().remove(sh._element)
def card(x,y,w,h,title,body):
 sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
 sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string('EEF3F8');sh.line.color.rgb=RGBColor.from_string('63849B')
 tf=sh.text_frame;tf.clear();tf.word_wrap=True
 for i,txt in enumerate([title,body]):
  par=tf.paragraphs[0] if i==0 else tf.add_paragraph();par.text=txt;par.font.name='Noto Sans CJK SC';par.font.size=Pt(24 if i==0 else 20);par.font.color.rgb=RGBColor.from_string('18354A')
card(.7,2.2,4.8,2.0,'1  近区范围 nl','同一时间序列按页对齐切分近远；nl 为请求的近区词元长度，至少一页。')
card(5.9,2.2,4.8,2.0,'2  页预算 nb','近区 nb 页，远区 K1 − nb 页；非对齐保护边界页另行加入。')
card(11.1,2.2,4.8,2.0,'3  统一 L2','两区与边界页的候选合并；逐查询 softmax → KV组概率平均 → 一次 TopK。')
card(.7,4.7,7.4,2.3,'候选资格 ≠ 最终保护','边界页内未保护词元仍须竞争；仅 sink4 与近期128词元强制进入最终支持集。')
card(8.5,4.7,7.4,2.3,'名义892个共享中间槽位','无预定近远词元份额；部分sink页可重复选中保护位置，去重后未保护数量可能更少。')

replace_text(p.slides[8].shapes[1],'历史数值词元配额门控：非当前统一 L2 算法')
replace_text(p.slides[8].shapes[2],'当前 nl/nb 设计关闭门控；下图回填属于历史固定词元配额分支，不移用于当前方法')
for idx,fig in [(13,'ruler32k_avg_compact_zh.png'),(15,'ruler32k_comparison_zh.png')]:
 slide=p.slides[idx]
 replace_picture(slide,next(sh for sh in slide.shapes if sh.shape_type==13),R/'figures/ruler32k_comparison'/fig)
 if idx==13:
  replace_text(slide.shapes[2],'Qwen3-8B：RULER 32K／64K 与稠密 FullKV 的 AVG')
  replace_text(slide.shapes[3],'11任务 × 100例／任务／臂；32K比较跨机器、评分器不同。\n64K稀疏两臂报告同机、评分器相同，生成脚本／有效配置不同；稠密锚另次运行、不同评分器、legacy-partial。\n仅稀疏臂 K=1024、4/128保护；门控与评分维数未核，不证非劣或匹配质量—成本收益。')
for sh in p.slides[8].shapes:
 if sh.has_text_frame and 'OFF：跳过远区点积' in sh.text:
  replace_text(sh,sh.text.replace('OFF：跳过远区点积','OFF：移除远区竞争'))
 if sh.has_text_frame and 'L1 k=0；L2跳过' in sh.text:
  replace_text(sh,'L1 k=0；远区无L2配额')
replace_text(p.slides[1].shapes[5],'共享32个低频RoPE坐标；nl定义近区范围、nb预留近区页；合并候选统一L2竞争，最终原维注意力。')
replace_text(p.slides[4].shapes[8],'显式近区范围与页预算')
replace_text(p.slides[4].shapes[10],'页准入不保证最终区域份额')
replace_text(p.slides[4].shapes[14],'历史固定配额门控另行研究\n当前统一 L2 设计关闭门控')
replace_text(p.slides[9].shapes[29],'统一 TopK → 组共享词元索引')
replace_text(p.slides[16].shapes[12],'nb 控制候选页准入，不保证最终区域份额\nnl/nb统一L2仍缺匹配质量与完整请求成本')
for idx in [10,11,12,13,14,15]:
 sh=p.slides[idx].shapes[2 if idx in [13,15] else 1]
 if sh.has_text_frame and not sh.text.startswith('历史'):replace_text(sh,'历史记录：'+sh.text)
# LoDA-SA naming and distance-first explanation; execution still prepares the shared view first.
replace_text(p.slides[0].shapes[1],'LoDA-SA：低频距离感知稀疏注意力')
replace_text(p.slides[0].shapes[3],'Low-Frequency Distance-Aware Sparse Attention')
replace_text(p.slides[0].shapes[12],'距离区域候选准入 → 共享低频评分 → 统一 L2')
replace_text(p.slides[1].shapes[1],'LoDA-SA：有限预算下保留不同距离的候选入口')
replace_text(p.slides[1].shapes[3],'全局排名不预留位置区域入口；先明确候选覆盖，再控制筛选成本。\n不假设近处总重要，也不假设注意力随距离单调衰减。')
replace_text(p.slides[1].shapes[5],'nl定义近区、nb预留L1近区页；两级共享32个低频RoPE坐标；合并候选统一L2，最终原维Q/K/V。')
replace_text(p.slides[4].shapes[1],'LoDA-SA：先说明距离策略，再说明共享评分')
replace_text(p.slides[4].shapes[3],'距离区域与页预算')
replace_text(p.slides[4].shapes[4],'nl 指定近区范围；nb 预留近区页\n远区获得剩余 L1 页预算')
replace_text(p.slides[4].shapes[5],'页准入不保证最终区域份额')
replace_text(p.slides[4].shapes[8],'共享低频 RoPE 评分')
replace_text(p.slides[4].shapes[9],'16完整旋转对，两级共用\n合并候选统一L2，最终原维注意力')
replace_text(p.slides[4].shapes[10],'执行仍先准备表示，再进行区域选择')
replace_text(p.slides[5].shapes[1],'LoDA-SA：同一时间序列上的两级筛选')
replace_text(p.slides[6].shapes[1],'LoDA-SA：两级共享完整低频 RoPE 对')
replace_text(p.slides[7].shapes[1],'LoDA-SA：nl / nb 页准入与统一词元竞争')
replace_text(p.slides[9].shapes[1],'LoDA-SA：统一 L2 与原维注意力')
replace_text(p.slides[16].shapes[1],'LoDA-SA：距离准入、共享评分与完整请求成本')
assert len(p.slides)==17
p.save(P)
