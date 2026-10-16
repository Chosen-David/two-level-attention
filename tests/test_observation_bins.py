"""CPU synthetic arithmetic plus separate existing-data consistency checks."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import unittest
import tempfile
from exp.paper.observation_bins import diagnose, validated_bins, select_whole_bins, finite_summary, paired_comparison

ROOT=Path(__file__).resolve().parents[1]

def example(widths=(64,64,1),near=192,masses=(.2,.3,.1)):
    lo=128;hi=lo+sum(widths);q=hi+127;start=lo;bins=[]
    for width,mass in zip(widths,masses):
        end=start+width
        bins.append(dict(d_lo=q-end+1,d_hi=q-start,n_tok=width,mass=mass));start=end
    far=sum(b['mass'] for b in bins if q-b['d_lo']+1<=near)
    return dict(q_pos=q,near_len=hi-near,region_bounds=dict(length=q+1,sink_hi=lo,swa_lo=hi,mid_lo=lo,mid_hi=hi,near_lo=near),bins=bins,mass_far=far,mass_near=sum(masses)-far)

class BinArithmetic(unittest.TestCase):
    def test_budget_never_overshoots_and_skips_nonfitting(self):
        bins=[dict(n_tok=64,mass=.8),dict(n_tok=64,mass=.7),dict(n_tok=1,mass=.2)]
        for budget in range(131):
            selected,used=select_whole_bins(bins,budget)
            self.assertLessEqual(used,budget)
            self.assertEqual(used,sum(bins[i]['n_tok'] for i in selected))
        self.assertEqual(select_whole_bins(bins,65),({0,2},65))
        self.assertEqual(select_whole_bins(bins,0),(set(),0))
        self.assertEqual(select_whole_bins([],2048),(set(),0))

    def test_swa_offset_and_partial_bin(self):
        x=example();result=diagnose(x,65)
        self.assertEqual([b['region'] for b in result['bins']],['far','near','near'])
        self.assertEqual(result['selected_tokens'],65)
        self.assertAlmostEqual(result['far_capture'],0)
        self.assertAlmostEqual(result['near_capture'],1)
        self.assertEqual(result['bins'][-1]['end'],x['region_bounds']['swa_lo'])
        self.assertEqual(result['bins'][-1]['n_tok'],1)

    def test_empty_regions_and_denominators(self):
        for near,empty in ((128,'far'),(257,'near')):
            x=example(near=near);self.assertIsNone(diagnose(x)[empty+'_capture'])
        x=example(masses=(0,0,0));result=diagnose(x)
        self.assertIsNone(result['far_share']);self.assertIsNone(result['near_capture'])
        x=example(widths=(),near=128,masses=());self.assertEqual(diagnose(x)['selected_tokens'],0)
        self.assertIsNone(finite_summary([])['mean'])
        self.assertEqual(paired_comparison([.9,float('nan'),None,.2],[.5,.1,.2,.3]),dict(near_gt_far=1,n_pairs=2,n_total=4,fraction=.5))
        self.assertEqual(finite_summary([.9,float('nan'),None,.1]),dict(mean=.5,n_valid=2,n_total=4))
        with self.assertRaises(ValueError):paired_comparison([1],[])

    def test_bad_bins_rejected(self):
        mutations=[lambda x:x['bins'][0].update(n_tok=63),
                   lambda x:x['bins'][1].update(d_hi=x['bins'][1]['d_hi']+1),
                   lambda x:x['bins'].pop(),
                   lambda x:x['region_bounds'].update(near_lo=193),
                   lambda x:x['region_bounds'].update(swa_lo=258),
                   lambda x:x.update(mass_far=.99),
                   lambda x:x['bins'][0].update(mass=float('nan')),
                   lambda x:x['bins'][0].update(mass=-1)]
        for mutate in mutations:
            x=example();mutate(x)
            with self.assertRaises(ValueError):validated_bins(x)
        for budget in (-1,1.5,True):
            with self.assertRaises(ValueError):select_whole_bins([],budget)

    def test_exact_token_topk_is_not_identifiable(self):
        # Identical two-bin sums permit different exact token Top-1 masses.
        a=[.6,0,.2,.2];b=[.3,.3,.4,0]
        self.assertEqual([sum(a[:2]),sum(a[2:])],[sum(b[:2]),sum(b[2:])])
        self.assertNotEqual(max(a),max(b))
        self.assertEqual(select_whole_bins([dict(n_tok=2,mass=.6),dict(n_tok=2,mass=.4)],1),(set(),0))

    def test_renderer_rejects_bad_o3_and_duplicate_ids(self):
        path=ROOT/'agent_doc/guide/paper_figs/figs/fig_o_observation.py'
        spec=importlib.util.spec_from_file_location('diagnostic_renderer',path)
        renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
        original=json.loads((ROOT/'exp/trace/results/e140_o_trace_v2.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'input.json'
            for bad in (float('nan'),float('inf'),-.1):
                value=deepcopy(original)
                value['cells'][0]['o3']['tail32']['selected_mass_over_reference_topk_mass']['mean']=bad
                source.write_text(json.dumps(value))
                with self.assertRaises(ValueError):renderer.build_data(source)
            value=deepcopy(original);value['cells'].append(value['cells'][0])
            source.write_text(json.dumps(value))
            with self.assertRaises(ValueError):renderer.build_data(source)

    def test_existing_data_consistency_only(self):
        cells=json.loads((ROOT/'exp/trace/results/e140_o_trace_v2.json').read_text())['cells']
        rows=[diagnose(o) for c in cells for o in c['o1']]
        self.assertEqual(len(rows),128)
        self.assertTrue(all(r['selected_tokens']<=2048 for r in rows))
        self.assertTrue(all(len(r['selected'])==len(set(r['selected'])) for r in rows))
        self.assertEqual(paired_comparison([r['near_capture'] for r in rows],[r['far_capture'] for r in rows])['n_pairs'],128)

if __name__=='__main__':unittest.main()
