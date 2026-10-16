import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const base=import.meta.dirname;
await fs.mkdir(path.join(base,'output'),{recursive:true});
await fs.mkdir(path.join(base,'build'),{recursive:true});
for(const lang of ['en','zh']){
 const p=Presentation.create({slideSize:{width:1008,height:475}});const slide=p.slides.add();slide.background.fill='#ffffff';
 const objs=JSON.parse(await fs.readFile(path.join(import.meta.dirname,lang+'.json'),'utf8'));
 let i=0;
 for(const o of objs){
  i++;const num=k=>Number(o[k]);const name=`R3-${lang}-${i}-${o.type}`;
  if(o.type==='rect')slide.shapes.add({name,geometry:'rect',position:{left:num('x'),top:num('y'),width:num('width'),height:num('height')},fill:o.fill||'none',line:{fill:o.stroke||'none',width:num('stroke-width')||1}});
  if(o.type==='polyline'){
   const pts=o.points.split(' ').map(t=>t.split(',').map(Number));
   const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);const x=Math.min(...xs),y=Math.min(...ys),w=Math.max(1,Math.max(...xs)-x),h=Math.max(1,Math.max(...ys)-y);
   slide.shapes.add({name,geometry:'custom',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:o.stroke,width:num('stroke-width'),style:o['stroke-dasharray']?'dashed':'solid'},customPaths:[{width:w,height:h,commands:pts.map((v,j)=>({[j?'lineTo':'moveTo']:{x:v[0]-x,y:v[1]-y}}))}]});
   if(o['marker-end']){
    const a=pts.at(-2),b=pts.at(-1),dx=b[0]-a[0],dy=b[1]-a[1],len=Math.hypot(dx,dy),ux=dx/len,uy=dy/len;
    const v=[[b[0],b[1]],[b[0]-5*ux+2.4*uy,b[1]-5*uy-2.4*ux],[b[0]-5*ux-2.4*uy,b[1]-5*uy+2.4*ux]];
    const lx=Math.min(...v.map(q=>q[0])),ly=Math.min(...v.map(q=>q[1])),ww=Math.max(1,Math.max(...v.map(q=>q[0]))-lx),hh=Math.max(1,Math.max(...v.map(q=>q[1]))-ly);
    slide.shapes.add({name:name+'-arrow',geometry:'custom',position:{left:lx,top:ly,width:ww,height:hh},fill:o.stroke,line:{fill:'none',width:0},customPaths:[{width:ww,height:hh,commands:[...v.map((q,j)=>({[j?'lineTo':'moveTo']:{x:q[0]-lx,y:q[1]-ly}})),{close:{}}]}]});
   }
  }
  if(o.type==='text'){
   const size=num('font-size'),center=num('x'),wid=Math.min(600,Math.max(22,[...o.text].reduce((s,c)=>s+(c.charCodeAt(0)>255?1.02:.57)*size,0)+8));
   let left=o['text-anchor']==='start'?center:o['text-anchor']==='end'?center-wid:center-wid/2;
   left=Math.max(0,Math.min(1008-wid,left));
   const t=slide.shapes.add({name,geometry:'textbox',position:{left,top:num('y')-size*1.02,width:wid,height:size*1.4},fill:'none',line:{fill:'none',width:0}});t.text=o.text;t.text.style={fontSize:size,typeface:lang==='zh'?'Noto Sans CJK SC':'Liberation Sans',bold:o['font-weight']==='bold',color:o.fill||'#202529',alignment:o['text-anchor']==='start'?'left':o['text-anchor']==='end'?'right':'center',verticalAlignment:'top',insets:0,wrap:'none',autoFit:'none'};
  }
 }
 await fs.writeFile(path.join(base,'output',lang+'.png'),new Uint8Array(await (await p.export({slide,format:'png',scale:1.5})).arrayBuffer()));
 const candidate=path.join(base,'build',lang+'.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
 await fs.copyFile(candidate,path.join(base,'output',`fig2_native_${lang}.pptx`));

}
