"""Exercise the native verifier's real failure exit without GPU or downloads."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('E141_TEST_SOURCE','/tmp/tla-hisa-official/TransMLA_NeurIPS_2025/models/deepseek_v2_lite/modeling_deepseek.py'))
SOURCE_SHA='040246d626f2bb94075177c9cc17999fc1641a96bdb4eaa45a9efd3cce4df0aa'
DRIVER=r'''
import sys, torch
sys.path.insert(0,'exp/trace')
import verify_e141_native_forward as v
side,value,source,sha=sys.argv[1:]
if side=='row_error_nan':
    from unittest.mock import patch
    with patch.object(torch.Tensor,'__sub__',return_value=torch.tensor([float('nan')])):
        v.finite_probability_error(torch.tensor([0.5]),torch.tensor([0.5]))
    raise SystemExit(0)
if side=='row_error':
    # Both operands are finite, but fp32 subtraction overflows.
    v.finite_probability_error(torch.tensor([3e38]),torch.tensor([-3e38]))
    raise SystemExit(0)
original=v.definitions
def definitions(path,names,namespace):
    result=original(path,names,namespace)
    if side=='got' and 'head_mean_probs' in names:
        function=namespace['head_mean_probs']
        def poisoned(*args,**kwargs):
            output=function(*args,**kwargs).clone()
            output[0]=float(value)
            return output
        namespace['head_mean_probs']=poisoned
    if side=='ref' and 'DeepseekV2Attention' in names:
        cls=namespace['DeepseekV2Attention'];forward=cls.forward
        def poisoned(self,*args,**kwargs):
            output,probabilities,cache=forward(self,*args,**kwargs)
            probabilities=probabilities.clone()
            probabilities[0,0,0,0]=float(value)
            return output,probabilities,cache
        cls.forward=poisoned
    return result
v.definitions=definitions
sys.argv=['verifier','--upstream-source',source,'--require-sha256',sha]
raise SystemExit(v.main())
'''

class FiniteGate(unittest.TestCase):
    def run_case(self,side,value='nan'):
        flags=['-O'] if sys.flags.optimize else []
        result=subprocess.run([sys.executable,*flags,'-c',DRIVER,side,value,str(SOURCE),SOURCE_SHA],
                              cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        evidence=os.environ.get('E141_FINITE_LOG_DIR')
        if evidence:
            folder=Path(evidence);folder.mkdir(parents=True,exist_ok=True)
            (folder/f'{side}-{value}.log').write_text(result.stdout+f'\nexit_code={result.returncode}\n')
        return result
    def test_finite_native_surrogate_still_passes(self):
        result=self.run_case('finite')
        self.assertEqual(result.returncode,0,result.stdout)
        self.assertIn('"parity": true',result.stdout)
    def test_nonfinite_got_and_ref_exit_one(self):
        for side in ('got','ref'):
            for value in ('nan','inf','-inf'):
                with self.subTest(side=side,value=value):
                    result=self.run_case(side,value)
                    self.assertEqual(result.returncode,1,result.stdout)
                    self.assertIn('ValueError: nonfinite got/ref probabilities',result.stdout)
                    self.assertNotIn('"parity": true',result.stdout)
    def test_nonfinite_row_error_exits_one(self):
        for side in ('row_error','row_error_nan'):
            with self.subTest(side=side):
                result=self.run_case(side)
                self.assertEqual(result.returncode,1,result.stdout)
                self.assertIn('ValueError: nonfinite probability row error',result.stdout)

if __name__=='__main__':unittest.main()
