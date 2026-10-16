"""Actual local-git interruption/retry test; no network or GPU."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('fetch_baselines',Path(__file__).with_name('fetch_baselines.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SourceResume(unittest.TestCase):
    def test_interrupted_fetch_resumes_without_deleting_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);up=root/'upstream';up.mkdir()
            subprocess.run(['git','init',str(up)],check=True,capture_output=True)
            (up/'LICENSE').write_text('fixture only')
            subprocess.run(['git','-C',str(up),'add','LICENSE'],check=True,capture_output=True)
            subprocess.run(['git','-C',str(up),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','fixture'],check=True,capture_output=True)
            sha=m.git(up,'rev-parse','HEAD')
            row={'name':'fixture','local_cache':'third_party/offline-sources/fixture','repository':'https://github.com/fixture/upstream','requested_object':sha,'commit':sha,'license':'fixture'}
            actual=m.git
            def interrupted(path,*args):
                if args[0]=='fetch':raise subprocess.CalledProcessError(128,['git','fetch'],output='simulated network loss')
                return actual(path,*args)
            with patch.object(m,'ROOT',root),patch.object(m,'git',side_effect=interrupted):
                with self.assertRaises(subprocess.CalledProcessError):m.fetch(row)
            cache=root/row['local_cache'];self.assertTrue((cache/'.git/tla-offline-source.json').is_file())
            def local_fetch(path,*args):
                if args[0]=='fetch':return actual(path,'fetch','--depth=1',str(up),sha)
                return actual(path,*args)
            with patch.object(m,'ROOT',root),patch.object(m,'git',side_effect=local_fetch):
                result=m.fetch(row)
            self.assertTrue(result['source_present']);self.assertEqual((cache/'LICENSE').read_text(),'fixture only')
            # Dirty real work is still preserved and rejected on subsequent fetch.
            (cache/'LICENSE').write_text('user change')
            with patch.object(m,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'changes'):m.fetch(row)
            self.assertEqual((cache/'LICENSE').read_text(),'user change')

if __name__=='__main__':unittest.main()
