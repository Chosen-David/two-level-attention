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
ANNOTATED_COMMIT = '21b595d'


class AssetProvenance(unittest.TestCase):
    def test_manifest_preserves_legacy_hashes_and_exact_git_sources(self):
        manifest = asset.verify(ROOT)
        # v1 manifest did not exist in the original figure commit. Its frozen
        # bytes are instead retained at the pre-annotation snapshot below.
        before = manifest['unannotated_current_snapshot'][0]['git_commit']
        old = json.loads(subprocess.check_output(['git','show',before+':'+str(asset.MANIFEST)],cwd=ROOT))
        self.assertEqual(old['artifacts'], manifest['artifacts'])
        for entry in manifest['artifacts']:
            data = subprocess.check_output(['git','show',manifest['legacy_snapshot_commit']+':'+entry['path']],cwd=ROOT)
            self.assertEqual(asset.sha(data),entry['sha256'])
        for entry in manifest['unannotated_current_snapshot']:
            data = subprocess.check_output(['git','show',entry['git_commit']+':'+entry['path']],cwd=ROOT)
            self.assertEqual(asset.sha(data),entry['sha256'])
        for entry in manifest['historical_annotated_assets']:
            data = subprocess.check_output(['git','show',entry['git_commit']+':'+entry['path']],cwd=ROOT)
            self.assertEqual(asset.sha(data),entry['sha256'])
        self.assertTrue(any(x['status']=='missing' for x in manifest['unchanged_inputs']))

    def test_annotated_legacy_assets_preserve_plot_and_expose_warning(self):
        manifest = asset.verify(ROOT)
        # The annotated historical figures no longer sit at the live paths
        # (replaced by the E140v2 recollection render); verify their pinned
        # git blobs instead of the working tree. before = unannotated
        # snapshot, after = annotated snapshot.
        annotated = {e['path']: e for e in manifest['historical_annotated_assets']}
        for entry in manifest['unannotated_current_snapshot']:
            before = subprocess.check_output(['git','show',entry['git_commit']+':'+entry['path']],cwd=ROOT)
            after = subprocess.check_output(['git','show',annotated[entry['path']]['git_commit']+':'+entry['path']],cwd=ROOT)
            ext = Path(entry['path']).suffix
            if ext == '.png':
                old = Image.open(BytesIO(before)).convert('RGBA')
                new = Image.open(BytesIO(after)).convert('RGBA')
                band = new.height-old.height
                self.assertGreater(band,0)
                self.assertEqual(new.width,old.width)
                self.assertEqual(old.tobytes(),new.crop((0,band,new.width,new.height)).tobytes())
                self.assertGreater(len(new.crop((0,0,new.width,band)).getcolors(100000)),2)
            elif ext == '.svg':
                new = ET.fromstring(after)
                self.assertIn(asset.WARNING, ''.join(new.itertext()))
            else:
                new = fitz.open(stream=after,filetype='pdf')
                self.assertIn(asset.WARNING,new[0].get_text())
                new.close()

    def test_current_assets_are_v2_renders_without_historical_banner(self):
        # Current assets render the E140v2 recollection (fixed collectors);
        # the historical-diagnostic banner must not mislabel valid v2 evidence.
        manifest = asset.verify(ROOT)
        for entry in manifest['current_assets']:
            self.assertEqual(entry['status'], asset.V3_ASSET_STATUS)
            data = (ROOT/entry['path']).read_bytes()
            ext = Path(entry['path']).suffix
            if ext == '.svg':
                self.assertNotIn(asset.WARNING, ''.join(ET.fromstring(data).itertext()))
            elif ext == '.pdf':
                doc = fitz.open(stream=data,filetype='pdf')
                self.assertNotIn(asset.WARNING, doc[0].get_text())
                doc.close()
        for entry in manifest['v2_recollection']['sources']:
            self.assertTrue((ROOT/entry['path']).is_file())

    def test_tampered_asset_or_promoted_status_fails(self):
        manifest = asset.verify(ROOT)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            checked = manifest['current_assets']+manifest['v2_recollection']['sources']
            for item in checked:
                target=root/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/item['path'],target)
            target=root/asset.MANIFEST;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(manifest));asset.verify(root)
            for item in manifest['current_assets']:
                path=root/item['path'];original=path.read_bytes();path.write_bytes(original+b'changed')
                with self.assertRaisesRegex(ValueError,'identity mismatch'):asset.verify(root)
                path.write_bytes(original)
            manifest['status']='paper-ready';target.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'status/schema'):asset.verify(root)

    def test_renderer_uses_v2_sources_without_banner_call(self):
        # The renderer reads the E140v2 recollection fail-closed and must not
        # stamp the historical-diagnostic banner on valid v2 evidence.
        text = RENDERER.read_text()
        self.assertIn('e140_o_trace_v2.json', text)
        self.assertIn('e140_o4_select_trace_v2.json', text)
        self.assertIn('missing E140v2 recollection', text)
        self.assertNotIn('add_render_warning', text)

    def test_opt_in_warning_helper_exports_visible_text(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(12,3));ax.plot([0,1],[0,1])
        asset.add_render_warning(fig)
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
