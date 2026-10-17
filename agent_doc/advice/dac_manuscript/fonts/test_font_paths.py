"""Isolated font path regressions; never modifies manuscript figures or bundle."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
class FontPathTests(unittest.TestCase):
    def test_relative_cache_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            code='export PAPER_FONT_CACHE=relative-cache; source "$1/env.sh" && test "$FONTCONFIG_FILE" = "$PWD/relative-cache/fonts.conf" && python3 "$1/check_fonts.py" --check'
            subprocess.run(['bash','-c',code,'bash',str(ROOT)],cwd=tmp,check=True,capture_output=True)
    def test_checked_path_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'fonts';root.mkdir()
            manifest=json.loads((ROOT/'MANIFEST.json').read_text())
            for rel in manifest['files']:
                target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
            for name in ('MANIFEST.json','check_fonts.py'):shutil.copyfile(ROOT/name,root/name)
            cache=Path(tmp)/'cache';env=dict(os.environ,FONTCONFIG_FILE=str(cache/'fonts.conf'))
            command=[sys.executable]+(['-O'] if sys.flags.optimize else [])+[str(root/'check_fonts.py')]
            subprocess.run(command+['--prepare',str(cache)],check=True,env=env)
            subprocess.run(command+['--check'],check=True,env=env,capture_output=True)
            original=root/'urw/NimbusSans-Regular.otf';raw=original.read_bytes()
            for mode in ('missing','tampered','duplicate'):
                if mode=='missing':original.unlink()
                elif mode=='tampered':original.write_bytes(raw+b'changed')
                else:
                    # Same names, distinct unchecked bytes, explicitly preferred by Fontconfig.
                    duplicate=root/'urw/000-Unchecked-Nimbus.otf';duplicate.write_bytes(raw+b'unchecked')
                    conf=cache/'fonts.conf';conf.write_text(conf.read_text().replace('</fontconfig>',f'<match target="pattern"><test name="family"><string>Nimbus Sans</string></test><edit name="file" mode="assign"><string>{duplicate}</string></edit></match></fontconfig>'))
                result=subprocess.run(command+['--check'],env=env,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0,mode)
                original.write_bytes(raw)
    def test_duplicate_family_matplotlib_explicit_file(self):
        spec=importlib.util.spec_from_file_location('paper_font_paths',ROOT.parent/'sources/paper_font_paths.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        import matplotlib;matplotlib.use('Agg')
        from matplotlib import pyplot as plt,font_manager
        with tempfile.TemporaryDirectory() as tmp:
            external=Path(tmp)/'external.otf';shutil.copyfile(module.cjk_regular(),external)
            font_manager.fontManager.addfont(str(external))
            # Put an external duplicate before the bundled face; explicit fname must win.
            entry=font_manager.fontManager.ttflist.pop();font_manager.fontManager.ttflist.insert(0,entry)
            module.register_matplotlib()
            fig,ax=plt.subplots();text=ax.text(.2,.3,'中文',fontsize=13,weight='bold',style='italic')
            module.pin_cjk_text(fig)
            self.assertEqual(Path(font_manager.findfont(text.get_fontproperties())).resolve(),(ROOT/'noto/NotoSansCJKsc-Bold.otf').resolve())
            self.assertEqual(text.get_fontsize(),13);self.assertEqual(text.get_fontstyle(),'italic');self.assertEqual(text.get_fontweight(),'bold')
            fig.canvas.draw();plt.close(fig)

if __name__=='__main__':unittest.main()
