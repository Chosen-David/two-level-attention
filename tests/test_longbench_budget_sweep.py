"""CPU-only runner tests: mocked NVIDIA query and harmless subprocesses, no models."""
import contextlib
import ast
import types
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.LongBench import run_budget_sweep as r

GPU = 'GPU-12345678-1234-1234-1234-123456789abc'
IDLE = {GPU: {'free': 50000, 'used': 0, 'util': 0, 'busy': False}}


def xml(processes='', mig='Disabled', free='50000 MiB', used='0 MiB', util='0 %'):
    return f'<nvidia_smi_log><gpu><uuid>{GPU}</uuid><mig_mode><current_mig>{mig}</current_mig></mig_mode><fb_memory_usage><free>{free}</free><used>{used}</used></fb_memory_usage><utilization><gpu_util>{util}</gpu_util></utilization><processes>{processes}</processes></gpu></nvidia_smi_log>'


class RunnerTests(unittest.TestCase):
    def test_nominal_fixed_flags(self):
        m = {'k1': 128, 'jssa_basis_path': '/basis', 'asynctls_channel_file': '/channels'}
        for k2 in (512, 1024, 2048):
            flags = r.flags('LoDA', k2, m)
            self.assertEqual(flags[flags.index('--tia_level1_topk') + 1], '128')
            self.assertEqual(flags[flags.index('--tia_level2_topk') + 1], str(k2))
            self.assertEqual(flags[flags.index('--tli_enable_layer_skip') + 1], 'false')
        self.assertEqual(r.flags('Full', None, m), ['--method', 'none'])

    def test_query_idle_and_graphics_busy(self):
        with patch.object(r.subprocess, 'check_output', return_value=xml()):
            self.assertEqual(r.query_gpus(), IDLE)
        with patch.object(r.subprocess, 'check_output', return_value=xml('<process_info><type>G</type></process_info>')):
            self.assertTrue(r.query_gpus()[GPU]['busy'])
        with patch.object(r.subprocess, 'check_output', return_value=xml('N/A')):
            self.assertTrue(r.query_gpus()[GPU]['busy'])

    def test_unknown_malformed_missing_command_fail_closed(self):
        for text in (xml(free='N/A'), '<broken>', xml().replace('<processes></processes>', '')):
            with self.subTest(text=text), patch.object(r.subprocess, 'check_output', return_value=text):
                with self.assertRaises(Exception): r.query_gpus()
        for error in (FileNotFoundError(), subprocess.CalledProcessError(1, 'nvidia-smi'), subprocess.TimeoutExpired('nvidia-smi', 10)):
            with patch.object(r.subprocess, 'check_output', side_effect=error):
                with self.assertRaises(Exception): r.query_gpus()
        for mig in ('Enabled', 'N/A', ''):
            with patch.object(r.subprocess, 'check_output', return_value=xml(mig=mig)):
                self.assertNotIn(GPU, r.query_gpus())

    def test_lock_never_deleted_reacquire_and_contention(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'lock'
            with r.lock(p):
                ino = p.stat().st_ino
                with self.assertRaises(BlockingIOError):
                    with r.lock(p): pass
            with r.lock(p): self.assertEqual(p.stat().st_ino, ino)
            link = Path(d) / 'link'; link.symlink_to(p)
            with self.assertRaises(OSError):
                with r.lock(link): pass

    def test_gpu_lock_then_stable_recheck(self):
        with tempfile.TemporaryDirectory() as d:
            seen = []
            def query():
                with self.assertRaises(BlockingIOError):
                    with r.lock(Path(d) / (GPU + '.lock')): pass
                seen.append(1)
                return IDLE
            with patch.object(r, 'query_gpus', side_effect=query):
                with r.acquire_gpu([GPU], d, .3, .03, .01, 40000, 256, 0) as (gpu, fd):
                    self.assertEqual(gpu, GPU); self.assertGreaterEqual(len(seen), 3)
            with r.lock(Path(d) / (GPU + '.lock')): pass

    def test_gpu_busy_then_idle_and_bounded_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            busy = {GPU: dict(IDLE[GPU], busy=True)}
            with patch.object(r, 'query_gpus', side_effect=[busy] + [IDLE] * 20):
                with r.acquire_gpu([GPU], d, .3, .02, .01, 40000, 256, 0): pass
            for field, value in [('busy', True), ('free', 100), ('used', 2000), ('util', 10)]:
                with patch.object(r, 'query_gpus', return_value={GPU: dict(IDLE[GPU], **{field: value})}):
                    with self.assertRaises(TimeoutError):
                        with r.acquire_gpu([GPU], d, .03, .01, .01, 40000, 256, 0): pass

    def test_environment_drops_effectful_inheritance(self):
        with patch.dict(os.environ, {'TLI_FINE_GATHER': '1', 'CUDA_VISIBLE_DEVICES': '0', 'PYTHONPATH': '/bad', 'LD_PRELOAD': '/bad'}):
            env = r.environment()
        self.assertNotIn('TLI_FINE_GATHER', env); self.assertNotIn('CUDA_VISIBLE_DEVICES', env)
        self.assertNotIn('LD_PRELOAD', env); self.assertEqual(env['PYTHONPATH'], str(r.ROOT))
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')

    def test_prepare_and_drift(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ('model', 'data', 'config'): (root / name).mkdir()
            (root / 'model/config.json').write_text(json.dumps({'model_type': 'qwen3', 'num_hidden_layers': 36, 'hidden_size': 4096, 'num_attention_heads': 32, 'num_key_value_heads': 8, 'head_dim': 128}))
            (root / 'model/tokenizer.json').write_text('{}')
            (root / 'model/model.safetensors').write_text('fixture, not real weights')
            (root / 'data/qasper.jsonl').write_text(json.dumps({'answers': ['x'], 'context': 'a'}) + '\n')
            for name, data in [('dataset2prompt', {'qasper': '{context}'}), ('dataset2maxlen', {'qasper': 128}), ('model2maxlen', {'Qwen3-8B': 31500, 'Qwen3-32B': 31500})]:
                (root / 'config' / (name + '.json')).write_text(json.dumps(data))
            m = {'model_path': str(root / 'model'), 'dataset_path': str(root / 'data'), 'config_path': str(root / 'config'), 'tasks': ['qasper']}
            with patch.object(r, 'code_identity', return_value={'git_sha': 'test', 'files': {}}):
                plan = r.prepare(m)
                self.assertEqual(len(plan['jobs']), 13)
                self.assertEqual(sum(j['method'] == 'Full' for j in plan['jobs']), 1)
                self.assertTrue(all(j['blocked'] for j in plan['jobs'] if j['method'] in ('JSSA', 'AsyncTLS')))
                r.verify_plan(plan)
                config_path = root / 'model/config.json'; original_config = config_path.read_text()
                config_path.write_text(json.dumps({'model_type': 'qwen3', 'num_hidden_layers': 64, 'hidden_size': 5120, 'num_attention_heads': 64, 'num_key_value_heads': 8, 'head_dim': 128}))
                with self.assertRaisesRegex(ValueError, 'architecture mismatch'): r.prepare(m)
                plan32 = r.prepare(dict(m, model='Qwen3-32B'))
                self.assertEqual(plan32['manifest']['model'], 'Qwen3-32B')
                self.assertNotEqual(plan32['id'], plan['id']); config_path.write_text(original_config)
                added = root / 'model/tokenizer_config.json'; added.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'inventory drift'): r.verify_plan(plan)
                added.unlink()
                (root / 'data/qasper.jsonl').write_text('{}\n')
                with self.assertRaisesRegex(ValueError, 'input drift'): r.verify_plan(plan)
            for bad in ({'k1': True}, {'k1': 1}, {'k2': []}, {'k2': [512, 512]}, {'tasks': ['lbv2']}, {'methods': ['unknown']}, {'model': 'Qwen3-14B'}, {'model': 'Qwen3-32B'}, {'typo': 1}):
                with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                    r.prepare(dict(m, **bad))

    def test_invalid_score(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d); (a / 'pred').mkdir(); pred = a / 'pred/x.jsonl'
            for value in (float('nan'), 101, -1, True, '90'):
                (a / 'pred/result.json').write_text(json.dumps({'x.jsonl': {'score': value}}))
                with self.assertRaises(ValueError): r.validate_score(a, pred)
            (a / 'pred/result.json').write_text(json.dumps({'x.jsonl': {'score': 90}, '_meta': {}}))
            self.assertEqual(r.validate_score(a, pred), 90)
            (a / 'pred/result.json').write_text('{}')
            with self.assertRaises(ValueError): r.validate_score(a, pred)

    def test_restart_ignores_failure_missing_or_corrupt_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); a = root / 'attempt-1'; a.mkdir()
            (a / 'failure.json').write_text('{}')
            self.assertIsNone(r.completed(root, {'id': 'j'}, {'id': 'p'}, {}))
            (a / 'complete.json').write_text('{broken')
            self.assertIsNone(r.completed(root, {'id': 'j'}, {'id': 'p'}, {}))

    def test_complete_receipt_roundtrip_and_corruption(self):
        from benchmark.LongBench.run_cap_manifest import build_run_cap_core, finalize_run_cap_manifest
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); a = root / 'attempt-1'; (a / 'pred').mkdir(parents=True)
            source = [{'answers': ['yes'], 'context': 'question'}]
            (root / 'qasper.jsonl').write_text(json.dumps(source[0]) + '\n')
            ident = r.build_lbv1_input_identity(source, '{context}')
            plan = {'id': 'p' * 64, 'manifest': {'model': 'Qwen3-8B', 'model_path': str(root), 'dataset_path': str(root)}, 'code': {'git_sha': 'test'}}
            job = {'id': 'j', 'task': 'qasper', 'method': 'Full', 'rows': 1, 'cap': 31500, 'max_gen': 128, 'input_identity': ident}
            pred = a / 'pred/qasper-none-test.jsonl'
            pred.write_text(json.dumps({'_id': 'qasper:' + ident['source_fingerprint'] + ':0', 'answers': ['yes'], 'pred': 'yes'}) + '\n')
            core = build_run_cap_core('Qwen3-8B', str(root), 'none', 'qasper', ident['source_fingerprint'], plan['id'][:16], 31500, None, 128,
                {'git_commit': 'test', 'script_sha256': r.sha(r.ROOT / 'benchmark/LongBench/pred.py')}, input_identity=ident)
            meta = finalize_run_cap_manifest(core, 'complete', 1, [1], [False], pred.name, r.sha(pred), 1, input_ids_sha256=['0' * 64])
            side = Path(str(pred) + r.RUN_CAP_SIDECAR_SUFFIX); r.atomic(side, meta)
            r.atomic(a / 'pred/result.json', {pred.name: {'score': 100}, '_meta': {}})
            expected_treatment = {'filename': pred.name, 'treatment': None}
            self.assertEqual(r.validate_prediction(a, job, plan, expected_treatment), pred)
            with self.assertRaises(ValueError):
                r.validate_prediction(a, job, plan, {'filename': 'wrong-budget.jsonl', 'treatment': None})
            receipt = {'plan_id': plan['id'], 'job_id': 'j', 'hashes': {str(p.relative_to(a)): r.sha(p) for p in (pred, side, a / 'pred/result.json')}}
            r.atomic(a / 'complete.json', receipt)
            self.assertEqual(r.completed(root, job, plan, expected_treatment)['score'], 100)
            for key, wrong in [('rows', 2), ('task', 'hotpotqa'), ('method', 'Quest'), ('cap', 16384)]:
                self.assertIsNone(r.completed(root, dict(job, **{key: wrong}), plan, expected_treatment))
            saved = side.read_bytes()
            meta['status'] = 'generating'; meta['result_sha256'] = None; r.atomic(side, meta)
            self.assertIsNone(r.completed(root, job, plan, expected_treatment)); side.write_bytes(saved)
            (a / 'pred/result.json').write_text('{}')
            self.assertIsNone(r.completed(root, job, plan, expected_treatment))
            r.atomic(a / 'pred/result.json', {pred.name: {'score': 100}, '_meta': {}})
            pred.write_text('{}\n')
            self.assertIsNone(r.completed(root, job, plan, expected_treatment))

    def test_cli_dry_run_never_queries_gpu_or_runs_child(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'plan.json'; r.atomic(p, {'id': 'p', 'jobs': []})
            with patch.object(sys, 'argv', ['runner', '--plan', str(p)]), patch.object(r, 'query_gpus', side_effect=AssertionError('GPU query')), patch.object(r, 'run_child', side_effect=AssertionError('child')), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(r.main(), 0)

    def test_actual_argument_and_identity_functions_without_tensor_imports(self):
        # Execute unchanged AST helpers with only unused heavy imports excluded.
        # This checks actual CLI/snapshot/naming semantics, not Torch integration.
        info_path = r.ROOT / 'sparse_attn/info.py'
        tree = ast.parse(info_path.read_text())
        tree.body = [n for n in tree.body if not (isinstance(n, ast.Import) and any(a.name == 'torch' for a in n.names))]
        info = types.ModuleType('sparse_attn.info'); info.__file__ = str(info_path)
        runtime_path = r.ROOT / 'sparse_attn/ops/far_skip_runtime.py'
        rt = ast.parse(runtime_path.read_text())
        rt.body = [n for n in rt.body if isinstance(n, ast.FunctionDef) and n.name in ('_config_supports_compact', '_args_support_compact', 'resolve_far_skip_modes')]
        runtime = types.ModuleType('sparse_attn.ops.far_skip_runtime')
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules, {'sparse_attn.info': info, 'sparse_attn.ops.far_skip_runtime': runtime}):
            exec(compile(tree, str(info_path), 'exec'), info.__dict__)
            exec(compile(rt, str(runtime_path), 'exec'), runtime.__dict__)
            root = Path(d); (root / 'config.json').write_text(json.dumps({'model_type': 'qwen3', 'head_dim': 128}))
            (root / 'basis.pt').write_bytes(b'identity-only fixture'); (root / 'channels.json').write_text('{}')
            m = {'k1': 128, 'model_path': str(root), 'jssa_basis_path': str(root / 'basis.pt'), 'asynctls_channel_file': str(root / 'channels.json')}
            jobs = [{'id': str(i), 'method': method, 'task': 'qasper', 'blocked': [], 'flags': r.flags(method, 512, m) + ['--tli_enable_layer_skip', 'false', '--tli_enable_kmeans', 'false']} for i, method in enumerate(r.METHODS)]
            result = r.inspect_treatments({'id': 'p' * 64, 'manifest': m, 'jobs': jobs})
            self.assertEqual(len(result), 5)
            tli = result['1']['treatment']
            self.assertEqual(tli['tia_level1_topk'], 128); self.assertEqual(tli['tia_level2_topk'], 512)
            self.assertFalse(tli['tli_enable_layer_skip']); self.assertFalse(tli['tli_enable_kmeans'])
            self.assertIn('_512_', result['2']['filename'])
            self.assertIn('_k512_', result['3']['filename']); self.assertIn('_k512_', result['4']['filename'])
            old = result['1']['filename']
            jobs[1]['flags'][jobs[1]['flags'].index('--tia_level2_topk') + 1] = '1024'
            self.assertNotEqual(r.inspect_treatments({'id': 'p' * 64, 'manifest': m, 'jobs': jobs})['1']['filename'], old)

    def test_real_two_process_lock_contention(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'gpu.lock'
            code = ('from pathlib import Path; from benchmark.LongBench import run_budget_sweep as r; '
                    f'p=Path({str(p)!r}); ctx=r.lock(p); ctx.__enter__(); print("acquired")')
            with r.lock(p):
                child = subprocess.run([sys.executable, '-c', code], cwd=r.ROOT, capture_output=True, timeout=10)
                self.assertNotEqual(child.returncode, 0); self.assertIn(b'BlockingIOError', child.stderr)
            child = subprocess.run([sys.executable, '-c', code], cwd=r.ROOT, capture_output=True, timeout=10)
            self.assertEqual(child.returncode, 0); self.assertIn(b'acquired', child.stdout)

    def test_execute_failure_retry_skip_and_partial_status(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            job = {'id': 'j', 'task': 'qasper', 'method': 'Full', 'k2': None, 'rows': 1, 'flags': ['--method', 'none'], 'blocked': []}
            plan = {'id': 'plan', 'jobs': [job], 'protocol': 'test', 'budget_semantics': 'nominal', 'python': sys.executable,
                    'packages': {'torch': 'fixture'}, 'environment': r.environment(),
                    'manifest': {'model': 'Qwen3-8B', 'k1': 128, 'model_path': '/model', 'dataset_path': '/data', 'config_path': '/config'}}
            args = SimpleNamespace(output=str(root), allow_partial=False, gpus=[GPU], lock_dir=str(root), wait_seconds=1,
                idle_seconds=.01, poll_seconds=.01, min_free_mib=1, max_used_mib=0, max_util_percent=0, job_timeout_seconds=1, score_timeout_seconds=1)
            calls = []
            fail = [True]
            @contextlib.contextmanager
            def gpu(*a):
                with r.lock(root / 'gpu.lock') as fd: yield GPU, fd
            def child(argv, env, timeout, fd, attempt, label):
                calls.append(label)
                (attempt / 'pred').mkdir(exist_ok=True)
                (attempt / 'pred/qasper.jsonl').write_text('{}')
                (attempt / ('pred/qasper.jsonl' + r.RUN_CAP_SIDECAR_SUFFIX)).write_text('{}')
                if label == 'scoring' and fail[0]: raise RuntimeError('scorer failed')
                (attempt / 'pred/result.json').write_text('{}')
            with patch.object(r, 'verify_plan'), patch.object(r, 'protocol_preflight', return_value={'j': {}}), patch.object(r, 'acquire_gpu', side_effect=gpu), patch.object(r, 'run_child', side_effect=child), patch.object(r, 'validate_prediction', side_effect=lambda a,*x: a / 'pred/qasper.jsonl'), patch.object(r, 'validate_score', return_value=80):
                with self.assertRaisesRegex(RuntimeError, 'scorer failed'): r.execute(plan, args)
                self.assertEqual(len(list(root.glob('plan/j/attempt-*/failure.json'))), 1)
                self.assertFalse(list(root.glob('plan/j/attempt-*/complete.json')))
                fail[0] = False
                self.assertEqual(r.execute(plan, args), 0)
                self.assertEqual(len(calls), 4)
                self.assertEqual(r.execute(plan, args), 0); self.assertEqual(len(calls), 4)
                # A previously completed score changed: do not reuse it.
                complete = next(root.glob('plan/j/attempt-*/complete.json'))
                (complete.parent / 'pred/result.json').write_text('{"changed":true}')
                self.assertEqual(r.execute(plan, args), 0); self.assertEqual(len(calls), 6)
                blocked = dict(job, id='blocked', method='JSSA', blocked=['missing adapter'])
                plan['jobs'].append(blocked)
                with self.assertRaisesRegex(ValueError, 'blocked arms'): r.execute(plan, args)
                args.allow_partial = True
                self.assertEqual(r.execute(plan, args), 2)
                status = r.read(root / 'plan.summary.json')
                self.assertEqual(status['status'], 'incomplete')
                self.assertEqual(status['jobs']['blocked']['status'], 'blocked')
                plan['packages']['torch'] = None
                with self.assertRaisesRegex(ValueError, 'missing runtime'): r.execute(plan, args)

    def test_child_success_failure_timeout_and_lock_release(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with r.lock(root / 'gpu.lock') as fd:
                r.run_child([sys.executable, '-c', 'print("ok")'], r.environment(), 5, fd, root, 'good')
                with self.assertRaises(RuntimeError):
                    r.run_child([sys.executable, '-c', 'raise SystemExit(7)'], r.environment(), 5, fd, root, 'bad')
                with self.assertRaises(RuntimeError):
                    r.run_child(['/nonexistent-command'], r.environment(), 5, fd, root, 'missing')
                with self.assertRaises(RuntimeError):
                    r.run_child([sys.executable, '-c', 'import time; time.sleep(30)'], r.environment(), .1, fd, root, 'timeout')
            with r.lock(root / 'gpu.lock'): pass

    def test_exited_leader_descendant_is_cleaned(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            descendant = 'import time; time.sleep(60)'
            worker = ('import subprocess,sys,pathlib; child=subprocess.Popen([sys.executable,"-c",' + repr(descendant) + ']); '
                      'pathlib.Path(' + repr(str(root / 'pid')) + ').write_text(str(child.pid))')
            with r.lock(root / 'gpu.lock') as fd:
                r.run_child([sys.executable, '-c', worker], r.environment(), 10, fd, root, 'descendant')
            pid = int((root / 'pid').read_text())
            state = Path(f'/proc/{pid}/stat')
            self.assertTrue(not state.exists() or state.read_text().split()[2] == 'Z')
            with r.lock(root / 'gpu.lock'): pass

    def test_guardian_parent_death_cleans_worker(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # CPU child writes its PID then sleeps; kill only our launcher, never other jobs.
            worker = 'import os,time,pathlib; pathlib.Path(' + repr(str(root / 'pid')) + ').write_text(str(os.getpid())); time.sleep(60)'
            launcher = ('import pathlib,sys; from benchmark.LongBench import run_budget_sweep as r; '
                        f'p=pathlib.Path({str(root)!r}); '\
                        'ctx=r.lock(p/"gpu.lock"); fd=ctx.__enter__(); '\
                        f'r.run_child([sys.executable,"-c",{worker!r}],r.environment(),60,fd,p,"death")')
            parent = subprocess.Popen([sys.executable, '-c', launcher], cwd=r.ROOT)
            try:
                deadline = time.monotonic() + 10
                while not (root / 'pid').exists() and time.monotonic() < deadline: time.sleep(.02)
                self.assertTrue((root / 'pid').exists())
                pid = int((root / 'pid').read_text())
                parent.kill(); parent.wait()
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        with r.lock(root / 'gpu.lock'): break
                    except BlockingIOError: time.sleep(.05)
                else: self.fail('guardian did not release lease after parent death')
                stat = Path(f'/proc/{pid}/stat')
                self.assertTrue(not stat.exists() or stat.read_text().split()[2] == 'Z')
            finally:
                if parent.poll() is None: parent.terminate(); parent.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
