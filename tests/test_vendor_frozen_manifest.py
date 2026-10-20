"""Host-only metadata tests: execute info.py with all tensor access forbidden."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_info():
    class NoTorch:
        def __getattr__(self, name):
            raise AssertionError('unexpected tensor work: ' + name)
    path = ROOT / 'sparse_attn/info.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.Import) and any(a.name == 'torch' for a in n.names))]
    scope = dict(__name__=__name__, __file__=str(path), torch=NoTorch())
    exec(compile(tree, str(path), 'exec'), scope)
    return SimpleNamespace(**scope)


class VendorFrozenManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info = load_info()

    def check_frozen(self, method, mutation):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b'{"0": {"0": ["vertical_and_slash", 64, 64, 1]}}'
            args = SimpleNamespace(method=method, tli_enable_layer_skip=False)
            if method == 'minference_vendor':
                source = root / 'pattern.json'
                source.write_bytes(payload)
                args.minference_config_path = str(source)
                self.info.freeze_minference_pattern_sha8(args)
                key = 'minference_config_path'
            else:
                source = root / 'model/qasper_agreement.pkl'
                source.parent.mkdir()
                source.write_bytes(payload)
                args.model = 'model'
                args.fasa_agreement_dir = str(root)
                self.info.stage_fasa_agreement(args, str(root / 'repo'))
                key = 'fasa_agreement'
            before = self.info.get_treatment_manifest_json(args)
            name_before = self.info.get_method_name_with_info(args)
            if mutation == 'replace':
                source.write_bytes(b'changed after runtime freeze')
            elif mutation == 'delete':
                source.unlink()
            elif mutation == 'symlink':
                target = source.with_name('new-target')
                target.write_bytes(b'new bytes')
                source.unlink()
                source.symlink_to(target)
            manifest = self.info.resolve_treatment_snapshot(args).manifest
            self.assertEqual(manifest[key]['sha256'], hashlib.sha256(payload).hexdigest())
            self.assertEqual(self.info.get_treatment_manifest_json(args), before)
            self.assertEqual(self.info.get_method_name_with_info(args), name_before)

    def test_minference_replacement(self):
        self.check_frozen('minference_vendor', 'replace')

    def test_minference_deletion(self):
        self.check_frozen('minference_vendor', 'delete')

    def test_fasa_replacement(self):
        self.check_frozen('fasa_vendor', 'replace')

    def test_fasa_deletion(self):
        self.check_frozen('fasa_vendor', 'delete')

    def test_unchanged_sources(self):
        for method in ('minference_vendor', 'fasa_vendor'):
            with self.subTest(method=method):
                self.check_frozen(method, 'none')

    def test_source_replaced_by_symlink(self):
        for method in ('minference_vendor', 'fasa_vendor'):
            with self.subTest(method=method):
                self.check_frozen(method, 'symlink')

    def test_unfrozen_file_identity_tracks_live_content(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'pattern.json'
            args = SimpleNamespace(method='minference_vendor',
                minference_config_path=str(source), tli_enable_layer_skip=False)
            for data in (b'first', b'second'):
                source.write_bytes(data)
                identity = self.info.resolve_treatment_snapshot(args).manifest['minference_config_path']
                self.assertEqual(identity['sha256'], hashlib.sha256(data).hexdigest())
            source.unlink()
            with self.assertRaises(SystemExit):
                self.info.resolve_treatment_snapshot(args)


if __name__ == '__main__':
    unittest.main()
