"""Annotate historical E140 exports without recomputing or certifying data.

CPU-only, explicit apply command. Original bytes remain recoverable at the pinned
Git commit; the legacy manifest identities are retained separately from current
annotated assets. This is not a renderer or a collector.
"""
import argparse
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

WARNING = 'DIAGNOSTIC ONLY - NOT VALIDATED PAPER EVIDENCE'
DETAIL = 'Whole-bin heuristic, not token TopK or an upper bound. E6 gate evidence blocked.'
REFERENCE = 'TL-OBS-PLOT-CORRECTNESS-20261016 | Existing aggregates; no new collection'
FIGDIR = Path('agent_doc/guide/paper_figs/figs')
MANIFEST = Path('agent_doc/guide/paper_figs/observation/e140_validity.json')
EXTENSIONS = ('svg', 'png', 'pdf')
STATUS = 'whole-bin-diagnostic-gate-blocked'
SCHEMA = 'e140-historical-validity-v4'


def add_render_warning(fig):
    """Keep future explicitly opted-in Matplotlib exports visibly diagnostic."""
    fig.text(0.5, .90, WARNING + '\n' + DETAIL + '\n' + REFERENCE,
             ha='center', va='bottom', fontsize=13, color='#7a2100',
             bbox=dict(facecolor='#fff0d2', edgecolor='#7a2100', pad=9))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def annotate(data, extension):
    if extension == 'svg':
        ns = 'http://www.w3.org/2000/svg'
        ET.register_namespace('', ns)
        ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
        root = ET.fromstring(data)
        x, y, width, height = map(float, root.attrib['viewBox'].split())
        if x or y:
            raise ValueError('expected zero-origin SVG')
        band = width * .05
        root.set('viewBox', f'0 0 {width} {height + band}')
        # Matplotlib exports use pt; preserve their physical scale.
        if not root.attrib['height'].endswith('pt'):
            raise ValueError('expected SVG height in pt')
        root.set('height', f'{float(root.attrib["height"][:-2]) * (height + band) / height}pt')
        drawing = ET.Element(f'{{{ns}}}g', {'transform': f'translate(0 {band})'})
        drawing.extend(list(root))
        root[:] = []
        root.append(drawing)
        ET.SubElement(root, f'{{{ns}}}rect', {'width': str(width), 'height': str(band), 'fill': '#fff0d2'})
        for line, fraction, size in ((WARNING,.31,.013), (DETAIL,.59,.009), (REFERENCE,.83,.008)):
            t = ET.SubElement(root, f'{{{ns}}}text', {'x': str(width/2), 'y': str(band*fraction), 'text-anchor': 'middle', 'font-family': 'DejaVu Sans, sans-serif', 'font-size': str(width*size), 'fill': '#7a2100'})
            t.text = line
        return ET.tostring(root, encoding='utf-8', xml_declaration=True)
    if extension == 'png':
        from PIL import Image, ImageDraw, ImageFont
        source = Image.open(BytesIO(data)).convert('RGBA')
        width, height = source.size
        band = round(width * .05)
        result = Image.new('RGBA', (width, height + band), '#fff0d2')
        result.paste(source, (0, band))
        draw = ImageDraw.Draw(result)
        for line, fraction, size in ((WARNING,.25,.013), (DETAIL,.55,.009), (REFERENCE,.80,.008)):
            font = ImageFont.truetype('DejaVuSans.ttf', max(10, round(width*size)))
            draw.text((width/2, band*fraction), line, font=font, fill='#7a2100', anchor='mm')
        output = BytesIO()
        result.save(output, format='PNG')
        return output.getvalue()
    if extension == 'pdf':
        import fitz
        source = fitz.open(stream=data, filetype='pdf')
        if len(source) != 1:
            raise ValueError('expected one-page E140 PDF')
        width, height = source[0].rect.width, source[0].rect.height
        band = width * .05
        result = fitz.open()
        page = result.new_page(width=width, height=height+band)
        page.show_pdf_page(fitz.Rect(0, band, width, height+band), source, 0)
        page.draw_rect(fitz.Rect(0, 0, width, band), color=None, fill=(1,.94,.82))
        for line, fraction, size in ((WARNING,.08,.013), (DETAIL,.43,.009), (REFERENCE,.70,.008)):
            remaining = page.insert_textbox(fitz.Rect(0, band*fraction, width, band), line, fontsize=width*size, align=1, color=(.48,.13,0))
            if remaining < 0:
                raise ValueError('warning text did not fit PDF')
        output = result.tobytes(garbage=4, deflate=True, no_new_id=True)
        source.close(); result.close()
        return output
    raise ValueError('unsupported asset extension')


def verify(root):
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest.get('status') != STATUS or manifest.get('schema') != SCHEMA:
        raise ValueError('historical validity status/schema changed')
    assets = manifest['current_assets']
    expected = {str(FIGDIR / f'O_observation.{ext}') for ext in EXTENSIONS}
    if {a['path'] for a in assets} != expected or len(assets) != 3:
        raise ValueError('current asset set incomplete')
    for entry in assets + manifest.get('derived_outputs', []) + manifest.get('renderer_sources', []):
        if sha((root / entry['path']).read_bytes()) != entry['sha256']:
            raise ValueError('asset/input identity mismatch: ' + entry['path'])
    reviewed = {e['path'] for e in manifest.get('legacy_inputs_review', {}).values()
                if isinstance(e, dict)} if manifest.get('legacy_inputs_review') else set()
    for entry in manifest['unchanged_inputs']:
        if entry.get('status') == 'missing':
            if (root / entry['path']).exists() and entry['path'] not in reviewed:
                raise ValueError('previously missing input requires identity review: '+entry['path'])
            continue
        if sha((root / entry['path']).read_bytes()) != entry['sha256']:
            raise ValueError('asset/input identity mismatch: ' + entry['path'])
    if not manifest.get('prior_asset_sets'):
        raise ValueError('prior asset generations missing from manifest')
    return manifest


def apply(root, source_commit):
    old = json.loads((root / MANIFEST).read_text())
    if old.get('schema') != 'e140-historical-validity-v1' or old.get('status') != 'historical-diagnostic-not-corrected':
        raise ValueError('requires original v1 manifest; refusing repeat annotation')
    commit = subprocess.check_output(['git','rev-parse', source_commit+'^{commit}'], cwd=root, text=True).strip()
    outputs, previous = {}, []
    # Preflight every input before writing any asset.
    for ext in EXTENSIONS:
        path = str(FIGDIR / f'O_observation.{ext}')
        data = (root / path).read_bytes()
        pinned = subprocess.check_output(['git','show',commit+':'+path], cwd=root)
        if data != pinned:
            raise ValueError('uncommitted asset differs from pinned source: '+path)
        previous.append({'path': path, 'sha256': sha(data), 'git_commit': commit,
                         'git_blob': subprocess.check_output(['git','rev-parse',commit+':'+path],cwd=root,text=True).strip()})
        outputs[path] = annotate(data, ext)
    new = deepcopy(old)
    new['schema'] = 'e140-historical-validity-v2'
    new['artifacts_scope'] = 'Legacy v1 snapshot, NOT current paths; bytes recoverable at legacy_snapshot_commit.'
    new['legacy_snapshot_commit'] = '096414368d2f7c102cb526a62c222e181b97620c'
    new['unannotated_current_snapshot'] = previous
    new['current_assets'] = [{'path': p, 'sha256': sha(b), 'status': STATUS} for p,b in outputs.items()]
    new['annotation'] = {'warning': WARNING, 'detail': DETAIL, 'reference': REFERENCE,
                         'operation': 'Add visible warning outside original plot bounds; no data correction or scientific validation.'}
    inputs = ['exp/trace/results/e140_o_trace.json', 'exp/trace/results/e140_o4_select_trace.json', str(FIGDIR/'O_observation_figure_data.tsv')]
    new['unchanged_inputs'] = [
        {'path': p, 'sha256': sha((root/p).read_bytes()), 'status': 'present-unchanged'}
        if (root/p).is_file() else {'path': p, 'sha256': None, 'status': 'missing'}
        for p in inputs]
    new['source_provenance'] = 'Historical figure lineage only; missing inputs and collector validity defects remain unresolved.'
    for p,b in outputs.items():
        (root/p).write_bytes(b)
    (root/MANIFEST).write_text(json.dumps(new, indent=2)+'\n')
    return verify(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    if args.apply and not args.source_commit:
        parser.error('--apply requires --source-commit')
    manifest = apply(args.root, args.source_commit) if args.apply else verify(args.root)
    print(json.dumps({'status':manifest['status'], 'assets':manifest['current_assets']}))
