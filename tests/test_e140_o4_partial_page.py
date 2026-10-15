"""CPU contracts for O4 valid partial pages and fixed near/far quotas."""
import importlib.util
from pathlib import Path
import unittest
import os
import subprocess
import sys
import tempfile
import torch

PATH = Path(__file__).resolve().parents[1] / 'agent_doc/guide/paper_figs/trace/e140_o4_select_trace.py'
spec = importlib.util.spec_from_file_location('o4', PATH)
o4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o4)

class PartialPages(unittest.TestCase):
    def select(self, n, k1=64, near=None):
        q = torch.ones(32,128)
        k = torch.zeros(n+256,8,128)
        k[128+n-1] = 100 if n else 0
        geo = dict(mid_lo=128, mid_hi=128+n, near_lo=128 if near is None else near)
        return o4.tli_select(q,k,geo,k1,torch.arange(32))

    def test_valid_reductions(self):
        for n in (1,63,64,65,257):
            for sign in (-1,1):
                x = sign * torch.arange(1,n*2+1,dtype=torch.float32).view(n,2)
                stats = o4.valid_page_stats(x)
                for i, start in enumerate(range(0,n,64)):
                    page=x[start:start+64]
                    for actual, expected in zip(stats,(page.mean(0),page.amax(0),page.amin(0))):
                        torch.testing.assert_close(actual[i],expected)

    def test_partial_high_score_and_aligned(self):
        for n in (1,63,64,65,257):
            selected = self.select(n)
            self.assertTrue(selected[:,-1].all())
            self.assertTrue(selected.all())
            self.assertEqual(selected.shape,(8,n))

    def test_high_scoring_tail_wins_single_page(self):
        for n in (65,127):
            selected=self.select(n,1)
            self.assertTrue(selected[:,-1].all())
            self.assertFalse(selected[:,:64].any())
            self.assertEqual(selected.sum(1).tolist(),[min(n-64,40)]*8)

    def test_aligned_selective_independent_oracle(self):
        generator=torch.Generator().manual_seed(140)
        values=torch.randn(256,generator=generator)
        q=torch.ones(32,128)
        k=torch.zeros(512,8,128)
        k[128:384]=values[:,None,None]
        geo=dict(mid_lo=128,mid_hi=384,near_lo=128)
        actual=o4.tli_select(q,k,geo,8,torch.arange(32))
        page=int(values.view(4,64).mean(1).argmax())
        chosen=values[page*64:(page+1)*64].topk(40).indices+page*64
        expected=torch.zeros(8,256,dtype=torch.bool)
        expected[:,chosen]=True
        self.assertTrue(torch.equal(actual,expected))

    def test_dry_run_and_execute_identity_guard(self):
        flags=['-O'] if sys.flags.optimize else []
        env=dict(os.environ)
        for key in ('E140_MODEL_REVISION','E140_TOKENIZER_REVISION','E140_DATASET_REVISION'):
            env.pop(key,None)
        with tempfile.TemporaryDirectory() as folder:
            env['E140_OUT']=str(Path(folder)/'fresh.json')
            dry=subprocess.run([sys.executable,*flags,str(PATH)],env=env,capture_output=True,text=True)
            self.assertEqual(dry.returncode,0)
            self.assertIn('"gpu_started": false',dry.stdout)
            bad=subprocess.run([sys.executable,*flags,str(PATH),'--execute'],env=env,capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)
            self.assertIn('identity declarations required',bad.stderr)
            self.assertNotIn('set_device',bad.stderr)
            self.assertFalse(Path(env['E140_OUT']).exists())

    def test_q512_no_refill(self):
        geo=o4.zone_geometry(512)
        self.assertEqual((geo['mid_lo'],geo['near_lo'],geo['mid_hi']),(128,320,385))
        for k1, count in ((64,257),(128,193)):
            selected=self.select(257,k1,320)
            self.assertEqual(selected.sum(1).tolist(),[count]*8)
            self.assertTrue(selected[:,-1].all())
            self.assertEqual(selected[:,192:].sum(1).tolist(),[65]*8)

    def test_empty_zones_budgets_and_far_tail(self):
        self.assertEqual(self.select(0).numel(),0)
        self.assertFalse(self.select(65,0).any())
        self.assertTrue(self.select(65,64,193).all()) # near empty, far partial
        old=o4.K2_MID
        try:
            o4.K2_MID=0
            self.assertFalse(self.select(65).any())
        finally:
            o4.K2_MID=old
        for near in (127,194,129):
            with self.assertRaises(ValueError): self.select(65,64,near)
        with self.assertRaises(ValueError): self.select(65,-1)

if __name__ == '__main__': unittest.main()
