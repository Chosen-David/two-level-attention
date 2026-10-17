"""Explicit project font files for plotting; independent of venv/system paths."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]/'fonts'

def checked_font(relative):
    item=json.loads((ROOT/'MANIFEST.json').read_text())['files'][relative]
    path=ROOT/relative
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        raise RuntimeError('Missing/modified project font: '+str(path))
    return str(path)

def cjk_regular():
    # SC-specific OTF avoids Matplotlib choosing TTC's first (JP) face.
    return checked_font('noto/NotoSansCJKsc-Regular.otf')

def register_matplotlib():
    from matplotlib import font_manager
    for face in ('Regular','Bold','Italic','BoldItalic'):
        font_manager.fontManager.addfont(checked_font('urw/NimbusSans-'+face+'.otf'))
    for face in ('Regular','Bold'):
        font_manager.fontManager.addfont(checked_font('noto/NotoSansCJKsc-'+face+'.otf'))
    for family in ('Nimbus Sans','Noto Sans CJK SC'):
        font_manager.findfont(family,fallback_to_default=False)
