#!/usr/bin/env python3
"""Compile an independent 7 x 3.4 inch font proof, never manuscript figures/PDFs."""
from pathlib import Path
import argparse,json,subprocess,os,re
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 subprocess.run(['python3',str(ROOT/'check_fonts.py'),'--check'],check=True)
 manifest=json.loads((ROOT/'MANIFEST.json').read_text());faces=[(n,x) for n,x in manifest['files'].items() if 'family'in x and not n.startswith('lm-math') and not n.endswith('.ttc')]
 # Include TTC SC faces as well as SC OTF; explicit collection index is checked.
 faces += [(n,x) for n,x in manifest['files'].items() if n.endswith('.ttc')]
 lines=[r'\documentclass{article}',r'\usepackage[paperwidth=7in,paperheight=3.4in,margin=0.12in]{geometry}',r'\usepackage{fontspec,unicode-math}',r'\pagestyle{empty}',r'\setmainfont{NimbusSans-Regular.otf}',r'\setmathfont{latinmodern-math.otf}',r'\begin{document}',r'\fontsize{8}{10}\selectfont\noindent']
 expected={'LatinModernMath-Regular'}
 for n,x in faces:
  options='Path={'+str((ROOT/n).parent)+'/'+'}'
  if n.endswith('.ttc'):options+=',FontIndex=2'
  label=x['family']+' '+x['style'];sample='检索块、驻留缓存与增量传输' if 'CJK'in label else 'Query → block selection → token selection; 32 / 64 / 128K'
  lines.append(r'{\fontspec['+options+']{'+Path(n).name+'}'+label+': '+sample+r'}\\')
  expected.add(x['postscript'])
 lines += [r'Math: $\sum_{i=1}^{n}\alpha_i=1,\quad QK^{\mathsf T}/\sqrt{d}$',r'\end{document}']
 (out/'font-proof.tex').write_text('\n'.join(lines)+'\n')
 # Host TeX distribution remains an explicit prerequisite. Preserve wrapper font paths.
 env=os.environ.copy()
 if Path('/usr/share/texlive/texmf-dist').is_dir():
  env['TEXINPUTS']='.:/usr/share/texlive/texmf-dist/tex//:'
  env['TFMFONTS']='/usr/share/texlive/texmf-dist/fonts/tfm//:/usr/share/texmf/fonts/tfm//:'
  env['TEXFORMATS']=str(out)+':'
  subprocess.run(['xetex','-ini','-etex','-interaction=nonstopmode','-jobname=xelatex','xelatex.ini'],cwd=out,env=env,stdout=subprocess.DEVNULL,check=True)
 with (out/'compile.log').open('w') as log:subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','font-proof.tex'],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 log=(out/'font-proof.log').read_text()
 if re.search(r'Missing character|LaTeX Font Warning|Overfull',log):raise SystemExit('Font proof has missing/substituted glyphs or overflow; inspect log')
 text=subprocess.check_output(['pdffonts',str(out/'font-proof.pdf')],text=True);(out/'pdffonts.txt').write_text(text)
 actual=set()
 for row in text.splitlines()[2:]:
  fields=row.split();actual.add(re.sub(r'^[A-Z]{6}\+','',fields[0]).removesuffix('-Identity-H'))
  if fields[-5]!='yes':raise SystemExit('Unembedded PDF font: '+row)
 if not expected.issubset(actual):raise SystemExit('Missing proof font embedding: '+str(expected-actual))
 info=subprocess.check_output(['pdfinfo',str(out/'font-proof.pdf')],text=True);(out/'pdfinfo.txt').write_text(info)
 if '504 x 244.8 pts' not in info or not re.search(r'Pages:\s+1\b',info):raise SystemExit('Incorrect proof dimensions/page count')
 subprocess.run(['pdftoppm','-scale-to','1800','-png','-singlefile',str(out/'font-proof.pdf'),str(out/'font-proof')],check=True)
 print('Font proof compiled, all requested faces embedded; 1 page, 7 x 3.4 inches.')

if __name__=='__main__':main()
