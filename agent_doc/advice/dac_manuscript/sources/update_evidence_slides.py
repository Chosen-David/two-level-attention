from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pathlib import Path
R=Path(__file__).resolve().parents[1];p=R/'presentation/TwoLevel_UserEdited_20261014.pptx';r=Presentation(p)
def text(sl,x,y,w,h,value,size=22):
 s=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=s.text_frame;tf.word_wrap=True
 for i,line in enumerate(value.split('\n')):
  a=tf.paragraphs[0] if i==0 else tf.add_paragraph();a.text=line;a.font.name='Noto Sans CJK SC';a.font.size=Pt(size);a.font.color.rgb=RGBColor.from_string('243746');a.space_after=Pt(14)
 return s
for num,keep,title in [(3,[0,1,90],'稠密参考：距离密度与整箱捕获诊断'),(12,[0,1,17],'历史动态缓存：同步模型级解码计时'),(14,[0,1,2],'保留历史128K质量负结果')]:
 sl=r.slides[num-1];orig=list(sl.shapes)
 for shape in orig:
  is_footer=shape.has_text_frame and shape.text.strip()==str(num) and shape.top>Inches(8.5)
  if shape.top>=Inches(1.9) and not is_footer:shape._element.getparent().remove(shape._element)
 for shape in sl.shapes:
  if shape.has_text_frame and Inches(.7)<shape.top<Inches(1.8):shape.text=title
 if num==3:
  sl.shapes.add_picture(str(R/'figures/observation_current_zh.png'),Inches(.8),Inches(2.1),width=Inches(7.2))
  text(sl,8.6,2.2,7.1,6.2,'Qwen3-8B · 归档128/128保护\n密度：MuSiQue样本1 / 层24 / q9996\n每有效词元平均注意力权重\n四条源序列，128条相关查询／层观测\n注意力保留率：远0.6759，近0.9315\n各区选中权重和／该区总权重\n每观测同一全局整箱选集≤2048中间词元；不是K1024方法选集、最优上界或任务质量。',23)
 elif num==12:
  sl.shapes.add_picture(str(R/'figures/decode_tpot_current_zh.png'),Inches(.8),Inches(2.2),width=Inches(8))
  text(sl,9.2,2.2,6.6,6.2,'作者报告中位数 · 非完整请求E2E\nQwen3-8B，固定128输出词元\n127个首输出后间隔，逐步同步\n历史稀疏配置 α=.125 / β=.375 / γ=.625\n不同于当前4/128候选；运行路径未独立核验\n三档稀疏计时均高于稠密；汇总不能定位因果瓶颈。',23)
 else:
  sl.shapes.add_picture(str(R/'figures/ruler128k_current_zh.png'),Inches(.8),Inches(2.3),width=Inches(8))
  text(sl,9.2,2.2,6.6,6.2,'历史稀疏配置 · 作者报告得分\n11任务 × 每任务每臂100例\n差值−22.45分，宏平均分比51.6%\n历史稀疏配置 α=.125 / β=.375 / γ=.625\n原始预测与评分身份未独立验证\n不是样本配对推断；当前4/128工作点质量仍未验证。',23)
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
r.save(p)
