"""Artifact identity and visible diagnostic boundary, CPU only."""
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import fitz
from PIL import Image
from exp.paper import e140_asset_provenance as asset

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / asset.FIGDIR / 'fig_o_observation.py'


class AssetProvenance(unittest.TestCase):
    def test_manifest_preserves_legacy_hashes_and_exact_git_sources(self):
        manifest = asset.verify(ROOT)
        for entry in manifest['artifacts']:
            data = subprocess.check_output(['git','show',manifest['legacy_snapshot_commit']+':'+entry['path']],cwd=ROOT)
            self.assertEqual(asset.sha(data),entry['sha256'])
        for snapshot in manifest['historical_snapshots']:
            for entry in snapshot['files']:
                data = subprocess.check_output(['git','show',snapshot['git_commit']+':'+entry['path']],cwd=ROOT)
                self.assertEqual(asset.sha(data),entry['sha256'])
        self.assertTrue(manifest['prior_asset_sets'])
        self.assertTrue(any(x['status']=='missing' for x in manifest['unchanged_inputs']))

    def test_current_assets_carry_diagnostic_warning(self):
        # Feasible whole-bin diagnostic, blocked E6 panel; no scientific promotion.
        manifest = asset.verify(ROOT)
        for entry in manifest['current_assets']:
            self.assertEqual(entry['status'], asset.STATUS)
            data = (ROOT/entry['path']).read_bytes()
            ext = Path(entry['path']).suffix
            if ext == '.svg':
                self.assertIn(asset.WARNING, ''.join(ET.fromstring(data).itertext()))
            elif ext == '.pdf':
                doc = fitz.open(stream=data,filetype='pdf')
                self.assertIn(asset.WARNING, doc[0].get_text())
                doc.close()
            else:
                img = Image.open(BytesIO(data)).convert('RGBA')
                self.assertGreater(img.height, 0)

    def test_v2_source_bytes_pinned(self):
        # Panels (a)(b)(c) read e140_o_trace_v2.json; its bytes stay pinned so
        # the quarantined combined figure remains traceable to its sources.
        manifest = asset.verify(ROOT)
        by_path = {e['path']: e for e in manifest['unchanged_inputs'] if e.get('status') != 'missing'}
        for p in ('exp/trace/results/e140_o_trace_v2.json',
                  'exp/trace/results/e140_o4_select_trace_v2.json'):
            self.assertIn(p, by_path)
            self.assertEqual(asset.sha((ROOT/p).read_bytes()), by_path[p]['sha256'])

    def test_tampered_asset_or_promoted_status_fails(self):
        manifest = asset.verify(ROOT)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            checked = [i for i in manifest['current_assets']+manifest['unchanged_inputs']+manifest['derived_outputs']+manifest['renderer_sources'] if i.get('status')!='missing']
            for item in checked:
                target=root/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/item['path'],target)
            target=root/asset.MANIFEST;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(manifest));asset.verify(root)
            for item in manifest['current_assets']+manifest['derived_outputs']+manifest['renderer_sources']:
                path=root/item['path'];original=path.read_bytes();path.write_bytes(original+b'changed')
                with self.assertRaisesRegex(ValueError,'identity mismatch'):asset.verify(root)
                path.write_bytes(original)
            manifest['status']='paper-ready';target.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'status/schema'):asset.verify(root)

    def test_default_renderer_fails_closed(self):
        # Explicit diagnostic opt-in is required; default never publishes.
        before={str(p):asset.sha(p.read_bytes()) for p in (ROOT/asset.FIGDIR).glob('O_observation*')}
        env=dict(os.environ);env.pop('E140_ALLOW_HISTORICAL_DIAGNOSTIC',None)
        flag=['-O'] if sys.flags.optimize else []
        result=subprocess.run([sys.executable,*flag,str(RENDERER)],env=env,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('No files written',result.stderr)
        self.assertEqual(before,{str(p):asset.sha(p.read_bytes()) for p in (ROOT/asset.FIGDIR).glob('O_observation*')})

    def test_opt_in_warning_helper_exports_visible_text(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(12,3));ax.plot([0,1],[0,1])
        asset.add_render_warning(fig)
        self.assertIn('add_render_warning(fig)',RENDERER.read_text())
        output=BytesIO();fig.savefig(output,format='pdf',bbox_inches='tight')
        doc=fitz.open(stream=output.getvalue(),filetype='pdf')
        self.assertIn(asset.WARNING,doc[0].get_text())
        doc.close();plt.close(fig)

    def test_repeat_annotation_is_rejected_without_writes(self):
        manifest=asset.verify(ROOT)
        before={a['path']:a['sha256'] for a in manifest['current_assets']}
        with self.assertRaisesRegex(ValueError,'refusing repeat annotation'):
            asset.apply(ROOT,'HEAD')
        self.assertEqual(before,{a['path']:a['sha256'] for a in manifest['current_assets']})


if __name__=='__main__':unittest.main()
