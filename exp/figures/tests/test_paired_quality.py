import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from exp.figures.paired_quality import paired_stats, validate_plan, analyze, IDENTITY, ROOT
from benchmark.LongBench.eval import sample_score, scorer, SCORER_BACKEND_ID
from benchmark.LongBench.run_cap_manifest import build_run_cap_core, finalize_run_cap_manifest


def fixture(base):
    identity={k:'fixture-value' for k in IDENTITY}
    identity['scorer_backend']=SCORER_BACKEND_ID
    identity['scorer_sha256']=hashlib.sha256(b''.join((ROOT/f).read_bytes() for f in ('benchmark/LongBench/eval.py','benchmark/LongBench/metrics.py','benchmark/LongBench/lbv2_choice.py'))).hexdigest()
    p=dict(schema='paired-longbench-v1',status='frozen',run_id='synthetic-software-only',run_mode='natural',noninferiority_delta_pp=None,bootstrap_draws=100,seed=17,
           arms={a:dict(selection=s,identity=copy.deepcopy(identity)) for a,s in [('dense','dense_fullkv'),('sub_sub','sparse_sub_sub')]},tasks={})
    task='hotpotqa'; spec={'samples':{}}
    for arm in ('dense','sub_sub'):
        rows=[dict(_id=str(i),pred='yes' if arm=='dense' or i else 'no',answers=['yes'],all_classes=[],length=12) for i in range(3)]
        raw=('\n'.join(json.dumps(r) for r in rows)+'\n').encode();path=base/(arm+'.jsonl');path.write_bytes(raw)
        sha=hashlib.sha256(raw).hexdigest()
        core=build_run_cap_core('fixture','/fixture/model','none' if arm=='dense' else 'tli',task,'fixture-fingerprint','fixture-run',100,None,10,{'fixture':True})
        manifest=finalize_run_cap_manifest(core,'complete',3,[12]*3,[False]*3,path.name,sha,3)
        mpath=Path(str(path)+'.runcap.json');mpath.write_text(json.dumps(manifest))
        spec[arm]=dict(path=path.name,sha256=sha,runcap_sha256=hashlib.sha256(mpath.read_bytes()).hexdigest())
        for i,row in enumerate(rows):
            item=spec['samples'].setdefault(str(i),dict(document_id='doc'+str(i),input_ids_sha256=hashlib.sha256(str(i).encode()).hexdigest()))
            item['record_sha256_'+arm]=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    p['tasks'][task]=spec
    return p


class PairedTests(unittest.TestCase):
    def test_recorded_failures_and_class_contract(self):
        for field, value, expected in (
                ('error', 'OOM', 'error-present'),
                ('failed', True, 'failed-flag'),
                ('all_classes', ['different'], None),
                ('answers', [], None)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                base = Path(d); p = fixture(base)
                spec = p['tasks']['hotpotqa']; source = spec['sub_sub']
                path = base / source['path']
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0][field] = value
                raw = b''.join((json.dumps(row) + '\n').encode() for row in rows)
                path.write_bytes(raw); source['sha256'] = hashlib.sha256(raw).hexdigest()
                sidecar = Path(str(path) + '.runcap.json')
                manifest = json.loads(sidecar.read_text())
                manifest['result_sha256'] = source['sha256']
                sidecar_raw = json.dumps(manifest).encode()
                sidecar.write_bytes(sidecar_raw)
                source['runcap_sha256'] = hashlib.sha256(sidecar_raw).hexdigest()
                for row in rows:
                    spec['samples'][row['_id']]['record_sha256_sub_sub'] = hashlib.sha256(
                        json.dumps(row, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=False).encode()).hexdigest()
                if expected is None:
                    with self.assertRaises(ValueError):
                        analyze(p, base)
                else:
                    result = analyze(p, base)
                    self.assertIsNone(result['overall'])
                    self.assertEqual(result['audit']['hotpotqa']['excluded'][0]['sub_sub_status'], expected)

    def test_unknown_task_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = fixture(Path(d))
            p['tasks']['unknown_task'] = p['tasks'].pop('hotpotqa')
            with self.assertRaises(ValueError):
                analyze(p, Path(d))

    def test_seed_requires_nonnegative_integer(self):
        for seed in (None, True, [1, 2], -1, 1.5, '17'):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    paired_stats({'a': {'x': [0], 'y': [1]}}, 100, seed)
                with tempfile.TemporaryDirectory() as d:
                    p = fixture(Path(d)); p['seed'] = seed
                    with self.assertRaises(ValueError):
                        validate_plan(p)

    def test_source_hash_required(self):
        for field in ('sha256', 'runcap_sha256'):
            for value in (None, '', '0' * 63, 'z' * 64):
                with self.subTest(field=field, value=value):
                    with tempfile.TemporaryDirectory() as d:
                        p = fixture(Path(d))
                        p['tasks']['hotpotqa']['dense'][field] = value
                        with self.assertRaises(ValueError):
                            analyze(p, Path(d))

    def test_official_accumulation_order(self):
        with patch('benchmark.LongBench.eval.sample_score', return_value=0.10005):
            expected = 0.0
            for _ in range(6):
                expected += 0.10005
            self.assertEqual(scorer('hotpotqa', ['x'] * 6, [['y']] * 6, []),
                             round(100 * expected / 6, 2))

    def test_cluster_estimand(self):
        r,_=paired_stats({'a':{'x':[0]*9,'y':[100]}},1000,5)
        self.assertEqual(r['a']['estimate'],10)
        self.assertEqual(r['a']['n_independent_seq'],2)
    def test_task_macro(self):
        _,r=paired_stats({'a':{'x':[0],'y':[0]},'b':{'x':[100]*50,'y':[100]*50}},100,5)
        self.assertEqual(r['estimate'],50)
    def test_single_doc_rejected(self):
        with self.assertRaises(ValueError): paired_stats({'a':{'x':[0,1]}},100)
    def test_nan_rejected(self):
        with self.assertRaises(ValueError): paired_stats({'a':{'x':[float('nan')],'y':[0]}},100)
    def test_official_unchanged(self):
        self.assertEqual(scorer('triviaqa',['\nyes\nno','no'],[['yes'],['yes']],[]),50)
        self.assertEqual(sample_score('triviaqa','\nyes\nno',['yes'],[]),1)
    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d));r=analyze(p,Path(d))
            self.assertAlmostEqual(r['tasks']['hotpotqa']['estimate'],-100/3)
            self.assertIsNone(r['winner'])
    def test_failures_retained(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d));p['tasks']['hotpotqa']['samples']['absent']=copy.deepcopy(p['tasks']['hotpotqa']['samples']['0'])
            r=analyze(p,Path(d));self.assertEqual(r['audit']['hotpotqa']['n_excluded'],1);self.assertIsNone(r['overall'])
    def test_wrong_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d));p['tasks']['hotpotqa']['dense']['sha256']='0'*64
            with self.assertRaises(ValueError):analyze(p,Path(d))
    def test_full_full(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d));p['arms']['dense']['selection']='full/full'
            with self.assertRaises(ValueError):validate_plan(p)
    def test_phase(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d));p['arms']['dense']['identity']['phase']='other'
            with self.assertRaises(ValueError):validate_plan(p)
    def test_duplicate_document_input(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d));s=p['tasks']['hotpotqa']['samples'];s['1']['input_ids_sha256']=s['0']['input_ids_sha256']
            with self.assertRaises(ValueError):analyze(p,Path(d))

if __name__=='__main__':unittest.main()
