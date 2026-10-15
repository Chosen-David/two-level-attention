"""CPU-only bookkeeping tests; actual forward comparison is in the verifier."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'agent_doc/guide/paper_figs/trace/e141_mla_nope_probe.py'
spec=importlib.util.spec_from_file_location('e141_probe',SCRIPT)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class Collector(unittest.TestCase):
    def inputs(self,t):
        return dict(nope=torch.zeros(2,128),rope=torch.zeros(2,64)),torch.zeros(t,2,128),torch.zeros(t,1,64)
    def test_regions_mass_and_saturation(self):
        for t in (1,127,128,129,255,256,257,259,1024,1025):
            q,k,r=self.inputs(t)
            c=p.probe_cell(q,k,r,192**-.5,{'full192':('full',None)})
            o=c['o1'];b=o['region_bounds']
            self.assertEqual(o['schema'],'e141-o1-head-mean-v2')
            self.assertNotIn('far_max_kv_group_mass',o)
            self.assertEqual(o['aggregate_far_mass'],o['mass_far'])
            self.assertAlmostEqual(o['mass_protected_union']+o['mass_near']+o['mass_far'],1,places=5)
            self.assertEqual(b['swa_lo'],max(0,t-128))
            self.assertEqual(c['mid_count'],max(0,t-256))
            self.assertEqual(c['budget_saturated'],t-256<=768)
            if t<=256:self.assertEqual(c['skip_reason'],'empty_mid')
            else:
                self.assertEqual(c['status'],'used')
                self.assertEqual(c['arms']['full192']['selected_count'],min(768,t-256))
                self.assertGreaterEqual(b['near_lo'],b['mid_lo'])
                if t==1024:self.assertEqual(b['near_lo'],704)
    def test_partial_page_budgeted_selection_and_denominators(self):
        q,k,r=self.inputs(1025)
        scores=torch.arange(1025,dtype=torch.float32)
        with patch.object(p,'K1',1),patch.object(p,'K2MID',4):
            sel,cand=p.two_level_select(scores,128,897)
            self.assertEqual(cand.tolist(),[896])
            self.assertEqual(sel.tolist(),[896])
            with patch.object(p,'arm_score',return_value=scores):
                c=p.probe_cell(q,k,r,192**-.5,{'full192':('full',None)})
            arm=c['arms']['full192']
            self.assertEqual(arm['reference_token_overlap_recall'],float(896 in c['reference_ids'])/4)
            self.assertAlmostEqual(arm['selected_mass_over_reference_topk_mass'],.25)
            self.assertEqual(arm['selected_count'],1)
            self.assertEqual(c['reference_count'],4)
            self.assertFalse(c['budget_saturated'])
    def test_skipped_queries_keep_actual_masses_positions_and_o1(self):
        q,k,r=self.inputs(300)
        for midmass in (0.,5e-10):
            probs=torch.zeros(300);probs[0]=1-midmass;probs[150]=midmass
            with patch.object(p,'head_mean_probs',return_value=probs):
                c=p.probe_cell(q,k,r,192**-.5,{'full192':('full',None)})
            self.assertEqual(c['query_position'],299)
            self.assertEqual(c['skip_reason'],'reference_mass_below_threshold')
            self.assertAlmostEqual(c['mid_mass'],midmass,delta=1e-16)
            self.assertAlmostEqual(c['reference_topk_mass'],midmass,delta=1e-16)
            self.assertIn('o1',c);self.assertEqual(c['arms'],{})
        summary=p.summarize_cells([c],['full192'])
        self.assertEqual(summary['n_query_used'],0);self.assertEqual(summary['n_query_skipped'],1)
        self.assertIsNone(summary['arms']['full192']);self.assertEqual(len(summary['raw_queries']),1)
        used=[]
        for length in (300,1025):
            q,k,r=self.inputs(length)
            used.append(p.probe_cell(q,k,r,192**-.5,{'full192':('full',None)}))
        summary=p.summarize_cells([c,*used],['full192'])
        self.assertEqual(summary['n_query_planned'],3)
        self.assertEqual(summary['n_query_used'],2)
        self.assertEqual(summary['n_query_skipped'],1)
        for label in ('saturated','budget_constrained'):
            self.assertEqual(summary['strata'][label]['n_query_used'],1)
    def test_invalid_values_and_geometry_fail(self):
        q,k,r=self.inputs(300)
        for scale in (False,0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):p.head_mean_probs(q,k,r,scale)
        with self.assertRaises(ValueError):p.head_mean_probs(q,k+float('nan'),r,.1)
        with self.assertRaises(ValueError):p.head_mean_probs(q,k[:0],r[:0],.1)
        with self.assertRaises(ValueError):p.head_mean_probs(q,k.double(),r,.1)
        with self.assertRaises(ValueError):p.two_level_select(torch.ones(3),-1,3)
        with patch.object(p,'K1',0):
            with self.assertRaises(ValueError):p.two_level_select(torch.ones(3),0,3)
    def test_default_and_preflight_never_load_model_or_cuda(self):
        run=subprocess.run([sys.executable,str(SCRIPT)],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr);self.assertIn('"gpu_started": false',run.stdout)
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'trace.json'
            with patch.object(p,'OUT',str(out)),patch.object(p,'MODEL',folder),patch.object(sys,'argv',['probe','--execute']),patch.dict(os.environ,{},clear=True),patch.object(p,'load_model',side_effect=AssertionError('model accessed')),patch.object(torch.cuda,'set_device',side_effect=AssertionError('cuda accessed')):
                with self.assertRaisesRegex(ValueError,'identity'):p.main()
                with patch.dict(os.environ,{k:'declared' for k in ('E141_MODEL_REVISION','E141_TOKENIZER_REVISION','E141_DATASET_REVISION')}):
                    with self.assertRaisesRegex(ValueError,'pinned'):p.main()
                out.write_text('historical')
                with self.assertRaisesRegex(ValueError,'output exists'):p.main()
                self.assertEqual(out.read_text(),'historical')

if __name__=='__main__':unittest.main()
