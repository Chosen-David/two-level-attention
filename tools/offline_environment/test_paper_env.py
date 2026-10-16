"""Safety/portability regressions for offline installation (no network)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('paper_env',Path(__file__).with_name('paper_env.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@unittest.skipUnless(m.sys.version_info[:2] == (3, 12) and m.platform.system() == 'Linux' and m.platform.machine() == 'x86_64', 'bundle-specific tests require CPython 3.12 Linux x86_64')
class OfflineSafety(unittest.TestCase):
    def test_real_bundle_verifies(self):
        self.assertEqual(len(m.verify()['files']),15)
    def test_missing_wheel_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'manifest.json').write_text(json.dumps({'files':[{'filename':'missing.whl'}]}))
            with self.assertRaisesRegex(ValueError,'missing'):m.verify(p)
    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'manifest.json').write_text(json.dumps({'files':[{'filename':'../secret.whl'}]}))
            with self.assertRaisesRegex(ValueError,'unsafe'):m.verify(p)
    def test_tampered_wheel_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'wheelhouse').mkdir();(p/'wheelhouse/x.whl').write_bytes(b'bad')
            (p/'manifest.json').write_text(json.dumps({'files':[{'filename':'x.whl','bytes':3,'sha256':'0'*64}]}))
            with self.assertRaisesRegex(ValueError,'mismatch'):m.verify(p)
    def test_existing_environment_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); (root/'.venv-existing').mkdir()
            with patch.object(m,'ROOT',root):
                with self.assertRaisesRegex(SystemExit,'NEW'):m.install(root/'.venv-existing')
    def test_other_platform_rejected_before_creation(self):
        with patch.object(m.platform,'machine',return_value='aarch64'):
            with self.assertRaisesRegex(SystemExit,'targets'):m.install(m.ROOT/'.venv-never-created')
    def test_musl_rejected(self):
        with patch.object(m.platform,'libc_ver',return_value=('musl','1.2')):
            with self.assertRaisesRegex(SystemExit,'glibc'):m.install(m.ROOT/'.venv-never-created')
    def test_outside_root_rejected(self):
        with self.assertRaisesRegex(SystemExit,'NEW'):m.install('/tmp/.venv-forbidden')

if __name__=='__main__':unittest.main()
