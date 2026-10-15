"""CPU contracts; toy hooks are not real-model/GPU parity evidence."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'agent_doc/guide/paper_figs/trace/e140_o_trace.py'
spec=importlib.util.spec_from_file_location('e140_trace',SCRIPT)
trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)

class Observation(unittest.TestCase):
    def test_near_is_right_and_swa_exact(self):
        p=torch.zeros(2,1024);p[:,750]=.7;p[:,150]=.3
        o=trace.o1_trace(p,1023)
        self.assertAlmostEqual(o['mass_near'],.7,places=6)
        self.assertAlmostEqual(o['mass_far'],.3,places=6)
        self.assertEqual(o['region_bounds']['near_lo'],704)
        self.assertEqual(1024-o['region_bounds']['swa_lo'],128)
        self.assertNotIn('far_worst_head',o)
        self.assertAlmostEqual(o['far_max_kv_group_mass'],.3,places=6)
    def test_short_overlap_and_nonaligned_conservation(self):
        for length in (1,127,128,129,255,256,257,1025):
            p=torch.ones(2,length)/length;o=trace.o1_trace(p,length-1)
            self.assertAlmostEqual(o['mass_protected_union']+o['mass_near']+o['mass_far'],1,places=5)
            self.assertEqual(sum(b['n_tok'] for b in o['bins']),max(0,length-256))
            r=o['region_bounds'];self.assertLessEqual(r['mid_lo'],r['near_lo']);self.assertLessEqual(r['near_lo'],r['mid_hi'])
        self.assertEqual(trace.regions(1024)['near_lo'],704)
    def test_negative_probabilities_geometry(self):
        for p,q in [(torch.ones(1,3),2),(torch.tensor([[float('nan')]]),0),(torch.tensor([[-1.,2.]]),1),(torch.ones(1,1),True),(torch.ones(1,2)/2,2)]:
            with self.assertRaises(ValueError):trace.o1_trace(p,q)
        for kw in ({'alpha':False},{'alpha':float('nan')},{'bs':0},{'swa':-1}):
            with self.assertRaises(ValueError):trace.regions(300,**kw)
    def test_o3_partial_page_and_explicit_denominators(self):
        q=torch.zeros(4,128);k=torch.zeros(259,1,128)
        row=trace.o3_trace(q,k,128**-.5,trace.SUBSPACES['tail32'])[0]
        self.assertEqual(row['mid_count'],3);self.assertEqual(row['selected_count'],3)
        self.assertEqual(row['selected_ids'],row['reference_ids'])
        self.assertAlmostEqual(row['selected_full_attention_mass'],3/259,places=6)
        self.assertAlmostEqual(row['selected_mass_over_reference_topk_mass'],1)
        self.assertEqual(row['reference_token_page_coverage'],1)
        with self.assertRaises(ValueError):trace.o3_trace(q,k[:200],128**-.5,trace.SUBSPACES['tail32'])
    def test_actual_hook_receives_layer_normalized_projections_and_rope(self):
        class Attention(torch.nn.Module):
            def __init__(self):super().__init__();self.q_norm=torch.nn.Identity();self.k_norm=torch.nn.Identity()
            def forward(self,hidden_states,position_embeddings):
                self.q_norm(hidden_states.unsqueeze(2));self.k_norm(hidden_states.unsqueeze(2));return hidden_states
        class Layer(torch.nn.Module):
            def __init__(self):super().__init__();self.self_attn=Attention()
            def forward(self,x):
                normalized=x*3
                return self.self_attn(hidden_states=normalized,position_embeddings=(torch.zeros_like(x),torch.ones_like(x)))
        class Inner(torch.nn.Module):
            def __init__(self):super().__init__();self.layers=torch.nn.ModuleList([Layer()])
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.model=Inner()
            def forward(self,input_ids,use_cache):return self.model.layers[0](input_ids.unsqueeze(-1).expand(-1,-1,128).float())
        model=Model();ids=torch.tensor([[1,2]])
        with patch.object(trace,'LAYERS',[0]),patch.object(trace,'DEV','cpu'):
            q,k=trace.get_qk(model,10000,ids)
        expected=trace.rotate_half(ids.unsqueeze(-1).expand(-1,-1,128).float()*3)[0].unsqueeze(1)
        torch.testing.assert_close(q[0],expected);torch.testing.assert_close(k[0],expected)
        self.assertEqual(len(model.model.layers[0].self_attn.q_norm._forward_hooks),0)
    def test_query_schedule_no_duplicate_weighting(self):
        with self.assertRaises(ValueError):trace.query_schedule(577)
        with self.assertRaises(ValueError):trace.query_schedule(True)
        positions=trace.query_schedule(1024)
        self.assertEqual(len(set(positions)),16)
    def test_cli_dry_run_does_not_load_model(self):
        result=subprocess.run([sys.executable,str(SCRIPT)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('"gpu_started": false',result.stdout)

if __name__=='__main__':unittest.main()
