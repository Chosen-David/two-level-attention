from pathlib import Path
import subprocess,sys,shutil
root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(root/'sources/build_method_architecture.py'),'--out',str(out)],check=True)
for lang in ['en','zh']:
 for ext in ['svg','pdf']:
  shutil.copyfile(out/f'overview_{lang}.{ext}',root/f'figures/ppt_overview_{lang}.{ext}')
