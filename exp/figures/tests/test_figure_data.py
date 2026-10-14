import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
f = load('figure_data_contract')
o = load('o3_retrieval')
import numpy as np


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'result.json'
        self.source.write_text(json.dumps({'a': {'mean': .6, 'lo': .4, 'hi': .8, 'n': 4}}))
        self.spec = {'schema': 'figure-bindings-v1', 'status': 'frozen', 'sources': {
            'raw': {'path': 'result.json', 'sha256': o.digest(self.source), 'run_id': 'synthetic'}},
            'cells': [dict(figure_id='O3', panel_id='test', arm_id='tail', reference_id='full',
                          run_mode='fixed-QK', metric_id='mass', unit='probability', aggregation='equal_document_mean',
                          transform='none', source_id='raw', fields=dict(estimate='/a/mean', ci_low='/a/lo',
                          ci_high='/a/hi', n_independent_seq='/a/n'))]}
        self.binding = self.root / 'bindings.json'
    def build(self):
        self.binding.write_text(json.dumps(self.spec))
        return f.build_rows(self.binding)[0]
    def test_export_and_check(self):
        rows = self.build()
        path = self.root / 'figure_data.tsv'
        f.write_new(path, f.tsv_bytes(rows))
        self.assertEqual(f.validate_tsv(path, rows), 1)
        with self.assertRaises(FileExistsError):
            f.write_new(path, b'overwrite')
        self.assertEqual(path.read_bytes(), f.tsv_bytes(rows))
    def test_hash_mismatch(self):
        self.spec['sources']['raw']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): self.build()
    def test_null_hash_cannot_disable_check(self):
        self.spec['sources']['raw']['sha256'] = None
        with self.assertRaisesRegex(ValueError, 'source hash'): self.build()
    def test_missing_ci(self):
        del self.spec['cells'][0]['fields']['ci_low']
        with self.assertRaisesRegex(ValueError, 'source pointers'): self.build()
    def test_unfrozen(self):
        self.spec['status'] = 'UNFROZEN'
        with self.assertRaisesRegex(ValueError, 'UNFROZEN'): self.build()
    def test_duplicate_cells(self):
        self.spec['cells'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate'): self.build()
    def test_mixed_panels(self):
        for field, value in [('run_mode', 'natural'), ('unit', 'ms')]:
            other = copy.deepcopy(self.spec['cells'][0]); other['arm_id'] = 'high'; other[field] = value
            self.spec['cells'] = [self.spec['cells'][0], other]
            with self.assertRaisesRegex(ValueError, 'mixed'): self.build()
    def test_missing_arm_and_tampered_value(self):
        rows = self.build(); path = self.root / 'test.tsv'
        path.write_bytes(f.tsv_bytes([]))
        with self.assertRaisesRegex(ValueError, 'missing/extra'): f.validate_tsv(path, rows)
        bad = copy.deepcopy(rows); bad[0]['estimate'] = .99; path.write_bytes(f.tsv_bytes(bad))
        with self.assertRaisesRegex(ValueError, 'differs'): f.validate_tsv(path, rows)
    def test_nonfinite_and_bool(self):
        for value in (float('nan'), True):
            self.source.write_text(json.dumps({'a': {'mean': value, 'lo': .4, 'hi': .8, 'n': 4}}))
            self.spec['sources']['raw']['sha256'] = o.digest(self.source)
            with self.assertRaisesRegex(ValueError, 'numeric'): self.build()


class O3Tests(unittest.TestCase):
    def test_complete_pairs_and_determinism(self):
        for kind, seed in [('tail', None), ('high', None), ('paired_random', 7)]:
            pairs, coord = o.coordinates(kind, seed)
            self.assertEqual(len(set(coord)), 32)
            np.testing.assert_array_equal(coord[16:], pairs + 64)
            np.testing.assert_array_equal(coord, o.coordinates(kind, seed)[1])
        self.assertFalse(np.array_equal(o.coordinates('paired_random', 7)[1], o.coordinates('paired_random', 8)[1]))
    def test_gqa_softmax_order(self):
        q = np.zeros((2, 128), np.float32); k = np.zeros((2, 1, 128), np.float32)
        q[:, 0] = [10, -1]; k[1, 0, 0] = 1
        heads, actual = o.probabilities(q, k)
        logits = np.array([[0, 10], [0, -1]]) / np.sqrt(128)
        ref = np.exp(logits); ref /= ref.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(actual[0], ref.mean(axis=0), rtol=1e-6)
        wrong = np.exp(logits.mean(axis=0)); wrong /= wrong.sum()
        self.assertGreater(float(np.max(np.abs(actual[0] - wrong))), .001)
    def test_ties_clamp_and_mass(self):
        q = np.zeros((4, 128), np.float32); k = np.zeros((3, 2, 128), np.float32)
        row = o.measure(q, k, 2, o.coordinates('tail')[1])[0]
        self.assertEqual(row['selected_ids'], [0, 1]); self.assertAlmostEqual(row['retained_mass'], 2/3, places=6)
        self.assertEqual(o.measure(q, k, 1024, o.coordinates('tail')[1])[0]['actual_k'], 3)
    def test_trace_rejections(self):
        data = {'q': np.zeros((2, 4, 128)), 'k': np.zeros((3, 2, 128)), 'qpos': np.array([1, 2])}
        o.check_trace(data)
        for positions in ([1, 1], [1, 3], [1., 2.]):
            bad = dict(data, qpos=np.array(positions))
            with self.assertRaises(ValueError): o.check_trace(bad)
        data['q'][0,0,0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'): o.check_trace(data)
    def test_document_not_head_bootstrap(self):
        rows = [dict(arm_id='tail', document_id='a', retained_mass=0.)] * 20
        rows += [dict(arm_id='tail', document_id='b', retained_mass=1.)]
        result = o.summarize(rows, dict(bootstrap_seed=4, bootstrap_repeats=100, run_id='test'))[0]
        self.assertEqual(result['estimate'], .5); self.assertEqual(result['n_independent_seq'], 2)



class O3IntegrationTests(unittest.TestCase):
    def fixture(self, root):
        rng = np.random.default_rng(10)
        traces = []
        for i in range(2):
            path = root / f'trace{i}.npz'
            np.savez(path, q=rng.normal(size=(2, 4, 128)).astype('float32'),
                     k=rng.normal(size=(9, 2, 128)).astype('float32'), qpos=np.array([4, 8]))
            traces.append(dict(path=path.name, sha256=o.digest(path), sample_id=f's{i}', document_id=f'd{i}',
                               input_ids_sha256=hashlib.sha256(f'synthetic-input-{i}'.encode()).hexdigest(), layer=0, query_indices=[0, 1]))
        receipt = root / 'rope.json'; receipt.write_text('{"synthetic_test_only": true}')
        p = dict(data_kind='synthetic', run_id='synthetic-test-only', model_revision='synthetic', dataset_revision='synthetic',
                 producer_commit='synthetic', split='test-fixture', rope_receipt_path='rope.json',
                 rope_receipt_sha256=o.digest(receipt), rope_layout='split_half_128_descending_frequency',
                 query_phase='prefill', support='all_causal_global_topk', quantization='minmax4_k_float32',
                 budget=3, bootstrap_repeats=100, bootstrap_seed=42, pair_seeds=[7, 8], traces=traces)
        protocol = root / 'protocol.json'; protocol.write_text(json.dumps(p))
        return protocol, p
    def test_end_to_end_repeatability_and_bad_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); protocol, p = self.fixture(root)
            a, b = root/'a', root/'b'
            o.run(protocol, a); o.run(protocol, b)
            self.assertEqual((a/'figure_data.tsv').read_bytes(), (b/'figure_data.tsv').read_bytes())
            expected, receipt = f.build_rows(a/'figure_bindings.json')
            self.assertEqual(f.validate_tsv(a/'figure_data.tsv', expected), 4)
            self.assertEqual(json.loads((a/'validation.json').read_text())['status'], 'pending-independent-review')
            with self.assertRaisesRegex(ValueError, 'output exists'): o.run(protocol, a)
            p['traces'][0]['sha256'] = '0'*64; protocol.write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): o.run(protocol, root/'bad')
            self.assertFalse((root/'bad').exists())
    def test_old_meta_and_floating_protocol_refused(self):
        with self.assertRaisesRegex(ValueError, 'run_id'): o.check_protocol(dict(S=32768, layers=[1]))
        with tempfile.TemporaryDirectory() as tmp:
            protocol, p = self.fixture(Path(tmp)); p['model_revision'] = 'UNFROZEN'
            with self.assertRaisesRegex(ValueError, 'model_revision'): o.check_protocol(p)



class ReviewRegressionTests(unittest.TestCase):
    fixture = O3IntegrationTests.fixture
    def test_duplicate_trace_or_input_cannot_inflate_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, p = self.fixture(Path(tmp))
            p['traces'][1]['sha256'] = p['traces'][0]['sha256']
            with self.assertRaisesRegex(ValueError, 'duplicate trace'): o.check_protocol(p)
            _, p = self.fixture(Path(tmp))
            p['traces'][1]['input_ids_sha256'] = p['traces'][0]['input_ids_sha256']
            with self.assertRaisesRegex(ValueError, 'different documents'): o.check_protocol(p)
    def test_placeholder_and_dtype_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, p = self.fixture(Path(tmp))
            for value in ('UNKNOWN', 'pending', 'main', ' ', 'unfrozen'):
                bad = dict(p, model_revision=value)
                with self.assertRaises(ValueError): o.check_protocol(bad)
            data = o.load_trace(Path(tmp)/'trace0.npz')
            for dtype in ('int32', 'complex64'):
                bad = dict(data, q=data['q'].astype(dtype))
                with self.assertRaisesRegex(ValueError, 'real floating'): o.check_trace(bad)
    def test_snapshot_redaction_and_unverified_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); protocol, p = self.fixture(root)
            p['private_note'] = 'never-export-this'; protocol.write_text(json.dumps(p))
            original = protocol.read_bytes(); original_load = o.load_trace
            def change_protocol(*args):
                protocol.write_text('{}')
                return original_load(*args)
            with mock.patch.object(o, 'load_trace', side_effect=change_protocol): o.run(protocol, root/'out')
            summary = json.loads((root/'out'/'protocol_summary.json').read_text())
            self.assertEqual(summary['run_id'], p['run_id'])
            self.assertNotIn('private_note', summary); self.assertNotIn('path', summary['traces'][0])
            self.assertNotIn('rope_receipt_path', summary)
            receipt = json.loads((root/'out'/'analysis_receipt.json').read_text())
            self.assertEqual(receipt['protocol_sha256'], hashlib.sha256(original).hexdigest())
            self.assertEqual(receipt['identity_status'], 'unverified-declarations')
    def test_failed_stage_not_published_and_no_overwrite_race(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); protocol, _ = self.fixture(root)
            write = Path.write_text
            def fail(path, *args, **kwargs):
                if path.name == 'validation.json': raise OSError('injected')
                return write(path, *args, **kwargs)
            with mock.patch.object(Path, 'write_text', fail):
                with self.assertRaises(OSError): o.run(protocol, root/'out')
            self.assertFalse((root/'out').exists())
            self.assertTrue(list(root.glob('.o3-pending-*')))
            stage = root/'stage'; stage.mkdir(); (stage/'value').write_text('ours')
            destination = root/'destination'; destination.mkdir()
            with self.assertRaises(FileExistsError): o.publish_directory(stage, destination)
            self.assertEqual(list(destination.iterdir()), [])
            self.assertTrue(stage.exists())

if __name__ == '__main__': unittest.main()
