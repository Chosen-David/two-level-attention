import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]/'exp/vendor_baselines/jssa_vendor'
spec=importlib.util.spec_from_file_location('jssa_manage',ROOT/'environment/manage.py')
manage=importlib.util.module_from_spec(spec);spec.loader.exec_module(manage)
class PackageTests(unittest.TestCase):
    def test_integrity_and_rejection(self):
        manifest=manage.check()
        self.assertEqual(manifest['commit'],'34ce5abd82453b16ca6d5f16a1a8ae50fac5b920')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'UPSTREAM.json').write_text(json.dumps({'files':{'upstream/a.py':'0'*64}}))
            with self.assertRaises(ValueError):manage.check(root)
            (root/'upstream').mkdir();(root/'upstream/a.py').write_text('changed')
            with self.assertRaises(ValueError):manage.check(root)
    def test_default_cli_offline_and_install_rejected(self):
        cmd=[sys.executable]+(['-O'] if sys.flags.optimize else [])
        result=subprocess.run(cmd+[str(ROOT/'environment/manage.py')],capture_output=True,text=True,check=True)
        data=json.loads(result.stdout)
        self.assertFalse(data['gpu_initialized']);self.assertFalse(data['installation_started'])
        self.assertFalse(data['execution_ready']);self.assertFalse(data['dependency_lock_complete'])
        blockers=' '.join(data['blockers'])
        for requirement in ('flash-attn','HF kernels','model_type','model path identity','effective RoPE'):
            self.assertIn(requirement,blockers)
        bad=subprocess.run(['bash',str(ROOT/'environment/provision.sh'),'--install'],capture_output=True)
        self.assertEqual(bad.returncode,2)
    def test_original_basis_cpu_spectral_optimum(self):
        # Extract the exact official pure function: avoid importing model loaders.
        source=ast.parse((ROOT/'upstream/calibration/jssa_calibrate.py').read_text())
        func=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='compute_basis')
        scope={'torch':torch};exec(compile(ast.Module(body=[func],type_ignores=[]),'official-compute-basis','exec'),scope)
        a=torch.diag(torch.tensor([8.,4.,2.,1.]))[None]
        b=torch.diag(torch.tensor([1.,2.,4.,8.]))[None]
        with patch.object(torch.cuda,'is_available',return_value=False):m=scope['compute_basis']({0:a},{0:b},2)[0]
        self.assertEqual(m.device.type,'cpu')
        torch.testing.assert_close(m@m.transpose(-1,-2),torch.eye(2)[None])
        torch.testing.assert_close(m.transpose(-1,-2)@m,torch.diag(torch.tensor([1.,0.,0.,1.]))[None])
if __name__=='__main__':unittest.main()
