from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pathlib import Path
R=Path(__file__).resolve().parents[1];p=R/'presentation/TwoLevel_UserEdited_20261014.pptx';r=Presentation(p)
def text(sl,x,y,w,h,value,size=22):
 s=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=s.text_frame;tf.word_wrap=True
 for i,line in enumerate(value.split('\n')):
  a=tf.paragraphs[0] if i==0 else tf.add_paragraph();a.text=line;a.font.name='Noto Sans CJK SC';a.font.size=Pt(size);a.font.color.rgb=RGBColor.from_string('243746');a.space_after=Pt(14)
 return s
for num,keep,title in [(3,[0,1,90],'四项观察：区域选集、紧凑坐标与历史层差异'),(12,[0,1,17],'历史动态缓存：同步模型级解码计时'),(14,[0,1,2],'保留历史128K质量负结果')]:
 sl=r.slides[num-1];orig=list(sl.shapes)
 for shape in orig:
  is_footer=shape.has_text_frame and shape.text.strip()==str(num) and shape.top>Inches(8.5)
  if (shape.top>=Inches(1.9) and not is_footer) or (num==3 and shape.shape_type==MSO_SHAPE_TYPE.PICTURE):shape._element.getparent().remove(shape._element)
 for shape in sl.shapes:
  if shape.has_text_frame and Inches(.7)<shape.top<Inches(1.8):shape.text=title
 if num==3:
  sl.shapes.add_picture(str(R/'figures/o_observation_zh.png'),Inches(2.9),Inches(1.85),height=Inches(6.55))
  text(sl,.9,8.47,14.9,.70,'(a) 页权重排序≠词元TopK；实际1989/1989/2048词元，集合差异不代表质量优劣；\v(b) 43/128条下降；(c) 随机未配对；(d) 历史代理，不是当前门控或解码保证。',16)
 elif num==12:
  sl.shapes.add_picture(str(R/'figures/decode_tpot_current_zh.png'),Inches(.8),Inches(2.2),width=Inches(8))
  text(sl,9.2,2.2,6.6,6.2,'作者报告中位数 · 非完整请求E2E\nQwen3-8B，固定128输出词元\n127个首输出后间隔，逐步同步\n历史稀疏配置 α=.125 / β=.375 / γ=.625\n不同于当前4/128候选；运行路径未独立核验\n32K/64K/128K报告中位数之比（稀疏／稠密）：3.38 / 2.46 / 1.78\n三档比值减小，但稀疏TPOT仍上升且高于稠密；汇总不能定位因果瓶颈。',23)
 else:
  sl.shapes.add_picture(str(R/'figures/ruler128k_current_zh.png'),Inches(.8),Inches(2.3),width=Inches(8))
  text(sl,9.2,2.2,6.6,6.2,'历史128K得分：46.43 / 23.98\n11任务；每任务每臂100例\n报告 β=.375，K1=128，保护128/128\n中间预算768；页长64，γ=.625\n按已检公式：名义远区词元配额=0\n代码／配置推论；实际执行未核验\n不能据此认定得分差的唯一原因；不验证当前4/128候选。',23)
# Terminology and attribution only; retain the existing17-slide structure.
for slide in r.slides:
 for shape in slide.shapes:
  if not shape.has_text_frame:continue
  for para in shape.text_frame.paragraphs:
   for run in para.runs:
    run.text=run.text.replace('远区概率质量','远区注意力权重总和').replace('远区 概率求和 → 均值','远区注意力权重求和 → 均值').replace('组共享的两级候选选择','页到词元的级联稀疏选择').replace('全维页评分、校准精筛通道与异步执行','相关工作的级联选择先例')
# Keep the statistic label within its existing gate node.
label=r.slides[8].shapes[84]
label.left=Inches(3.88); label.top=Inches(5.06)
label.width=Inches(1.04); label.height=Inches(.35)
label.text_frame.margin_left=0; label.text_frame.margin_right=0
label.text_frame.margin_top=0; label.text_frame.margin_bottom=0
label.text_frame.paragraphs[0].text='远区注意力\v权重总和'
for para in label.text_frame.paragraphs:
 for run in para.runs:
  run.font.size=Pt(10); run.font.name='Noto Sans CJK SC'; run.font.color.rgb=RGBColor.from_string('7B6598')

# Full-manuscript narrative revision: preserve17slides and existing diagrams.
updates = {
 (2,1): '固定词元预算下：候选访问与选择成本',
 (2,3): '有用词元所在页一旦被排除，精筛无法恢复；检索本身也有成本。',
 (2,5): '共享32坐标映射与区域页配额协同选择，\n最终在选集上使用原128维 Q / K / V。',
 (2,6): '证据与范围',
 (2,7): '历史128/128配置五任务均值47.588；不是当前紧凑4/128候选的匹配质量—成本验证。',
 (5,1): '两项核心机制，一个可选扩展',
 (5,5): '共享坐标映射，不要求共享物理缓冲区',
 (11,1): 'Q1：现有得分能否验证紧凑评分？',
 (13,1): '区域访问与门控：证据支持到哪里？',
 (17,1): '候选访问与选择成本仍需匹配验证',
 (17,12): '表示与分配机制已具体化\n当前4/128候选仍缺匹配质量—成本验证',
 (17,13): '门控默认关闭；历史任务存在损失\n预填充统计不保证解码需求',
}
for (num, idx), value in updates.items():
 tf=r.slides[num-1].shapes[idx].text_frame
 runs=[run for para in tf.paragraphs for run in para.runs]
 if runs:
  runs[0].text=value
  for run in runs[1:]: run.text=''
 else: tf.paragraphs[0].text=value

r.save(p)
