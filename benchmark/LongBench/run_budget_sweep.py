"""Bounded, offline LongBench-v1 nominal-budget runner. See docs/longbench_budget_sweep.md.

Only --execute launches prediction/scoring. Preparation uses CPU metadata subprocesses;
no model/GPU imports in preparation or dry-run.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import json
import importlib.metadata
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

from benchmark.LongBench.run_cap_manifest import (
    RUN_CAP_SIDECAR_SUFFIX, build_lbv1_input_identity, compute_result_binding,
    read_run_cap_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
TASKS = ('narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report '
         'qmsum multi_news triviaqa passage_retrieval_en '
         'lcc repobench-p').split()
METHODS = {'Full': 'none', 'LoDA': 'tli', 'Quest': 'quest_vendor',
           'JSSA': 'jssa_vendor', 'AsyncTLS': 'asynctls_vendor'}
ADAPTERS = {m: ROOT / 'exp/vendor_baselines' / f'{m.lower()}_vendor' /
            f'qwen3_{m.lower()}_patch.py' for m in ('Quest', 'JSSA', 'AsyncTLS')}
UUID_PATTERN = r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
SOURCE_DIRS = ('benchmark', 'sparse_attn', 'tls_attn', 'exp/vendor_baselines', 'exp/paper')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def inventory(directory):
    directory = Path(directory).resolve(strict=True)
    files = sorted(p for p in directory.rglob('*') if p.is_file()
                   and '.cache' not in p.parts and '__pycache__' not in p.parts)
    if not files:
        raise ValueError(f'empty inventory: {directory}')
    return {str(p): sha(p) for p in files}


def code_identity():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                     text=True, timeout=30).strip()
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'],
                   cwd=ROOT, check=True, timeout=30)
    # Include remote-local/ignored adapter dependencies as well as tracked code.
    files = {}
    for directory in SOURCE_DIRS:
        for p in sorted((ROOT / directory).rglob('*')):
            if p.is_file() and p.suffix in ('.py', '.json', '.so') and '__pycache__' not in p.parts:
                files[str(p)] = sha(p)
    return {'git_sha': commit, 'files': files}


def packages():
    names = ('torch', 'transformers', 'datasets', 'numpy', 'einops', 'accelerate', 'triton', 'jieba', 'rouge', 'flash-attn', 'tqdm', 'safetensors')
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def worker_packages(env):
    return json.loads(subprocess.check_output(
        [sys.executable, '-m', 'benchmark.LongBench.run_budget_sweep', '--_packages'],
        cwd=ROOT, env=env, text=True, timeout=30))


def environment():
    # Explicitly do not inherit TLI_*, CUDA_*, PYTHONPATH or LD_PRELOAD switches.
    env = {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH')
           if k in os.environ}
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               PYTHONPATH=str(ROOT), PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', TLI_SPARSE_PREFILL='0', TLI_DECODE_DENSE='0')
    return env


def flags(method, k2, manifest):
    if method == 'Full':
        return ['--method', 'none']
    if method == 'LoDA':
        return ['--method', 'tli', '--tia_block_size', '64',
                '--tia_level1_topk', str(manifest['k1']), '--tia_level2_topk', str(k2),
                '--tli_far_method', 'minmax', '--tli_near_method', 'avg',
                '--tli_near_pages', '32', '--tli_near_tokens', 'off', '--tli_near_len', '4096',
                '--tli_subspace', 'tail', '--tia_level2_cmp_ratio', '4',
                '--tli_sink_tokens', '4', '--tli_swa_size', '128', '--tli_enable_layer_skip', 'false']
    if method == 'Quest':
        return ['--method', 'quest_vendor', '--quest_token_budget', str(k2), '--quest_chunk_size', '16']
    if method == 'JSSA':
        return ['--method', 'jssa_vendor', '--jssa_budget', str(k2), '--jssa_rank', '32',
                '--jssa_basis_path', manifest['jssa_basis_path']]
    return ['--method', 'asynctls_vendor', '--asynctls_token_budget', str(k2),
            '--asynctls_block_size', '64', '--asynctls_top_kb', '128',
            '--asynctls_channel_file', manifest['asynctls_channel_file']]


def prepare(manifest):
    m = dict(manifest)
    allowed = {'model_path', 'dataset_path', 'config_path', 'tasks', 'methods', 'k1', 'k2',
               'jssa_basis_path', 'asynctls_channel_file', 'model'}
    if set(m) - allowed:
        raise ValueError(f'unknown manifest keys: {set(m) - allowed}')
    m.setdefault('model', 'Qwen3-8B')
    if m['model'] not in ('Qwen3-8B', 'Qwen3-32B'):
        raise ValueError('model must be Qwen3-8B or Qwen3-32B')
    m.setdefault('tasks', TASKS); m.setdefault('methods', list(METHODS))
    m.setdefault('k1', 128); m.setdefault('k2', [512, 1024, 2048])
    positive(m['k1'], 'k1')
    if m['k1'] < 32:
        raise ValueError('fixed near_pages=32 requires k1 >= 32')
    for key, universe in [('tasks', TASKS), ('methods', METHODS)]:
        if not isinstance(m[key], list) or not m[key] or len(set(m[key])) != len(m[key]) or set(m[key]) - set(universe):
            raise ValueError(f'invalid/duplicate {key}')
    if not isinstance(m['k2'], list) or not m['k2'] or len(set(m['k2'])) != len(m['k2']):
        raise ValueError('k2 must be a nonempty unique list')
    for k in m['k2']:
        positive(k, 'k2')
        if not 132 <= k <= m['k1'] * 64:
            raise ValueError('k2 outside protected-token/candidate range')
    for key in ('model_path', 'dataset_path', 'config_path'):
        m[key] = str(Path(m[key]).resolve(strict=True))
    model = inventory(m['model_path'])
    if not (Path(m['model_path']) / 'config.json').is_file() or not any(
            p.endswith(('.safetensors', '.bin')) for p in model):
        raise ValueError('model inventory requires config and local weights')
    cfg = read(Path(m['model_path']) / 'config.json')
    dims = {'Qwen3-8B': (36, 4096, 32), 'Qwen3-32B': (64, 5120, 64)}[m['model']]
    expected = dict(zip(('num_hidden_layers', 'hidden_size', 'num_attention_heads'), dims))
    expected.update(model_type='qwen3', num_key_value_heads=8, head_dim=128)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('model label/config architecture mismatch')
    if not any(Path(p).name in ('tokenizer.json', 'tokenizer.model') for p in model):
        raise ValueError('local tokenizer is required')
    config = inventory(m['config_path'])
    prompts = read(Path(m['config_path']) / 'dataset2prompt.json')
    maxgen = read(Path(m['config_path']) / 'dataset2maxlen.json')
    cap = read(Path(m['config_path']) / 'model2maxlen.json')[m['model']]
    frozen = {**model, **config}
    assets = {}
    blocked = {}
    for method in m['methods']:
        reasons = []
        if method in ADAPTERS and not ADAPTERS[method].is_file():
            reasons.append(f'missing adapter {ADAPTERS[method]} (may be remote-local; no substitute)')
        key = {'JSSA': 'jssa_basis_path', 'AsyncTLS': 'asynctls_channel_file'}.get(method)
        if key:
            path = Path(m.get(key, '/nonexistent'))
            if not path.is_file():
                reasons.append(f'missing {key}')
                m[key] = str(path)
            else:
                m[key] = str(path.resolve()); assets[m[key]] = sha(path)
        if reasons:
            blocked[method] = reasons
    frozen.update(assets)
    jobs = []
    for task in m['tasks']:
        datafile = Path(m['dataset_path']) / (task + '.jsonl')
        rows = [json.loads(line) for line in datafile.read_text().splitlines()]
        if not rows or any(not isinstance(r, dict) or not isinstance(r.get('answers'), list) for r in rows):
            raise ValueError(f'invalid/empty dataset {task}')
        frozen[str(datafile)] = sha(datafile)
        identity = build_lbv1_input_identity(rows, prompts[task])
        for method in m['methods']:
            for k2 in ([None] if method == 'Full' else m['k2']):
                job = {'task': task, 'method': method, 'k2': k2, 'input_identity': identity,
                       'rows': len(rows), 'cap': cap, 'max_gen': maxgen[task],
                       'flags': flags(method, k2, m) + ['--tli_enable_layer_skip', 'false', '--tli_enable_kmeans', 'false'], 'blocked': blocked.get(method, [])}
                job['id'] = digest(job)
                jobs.append(job)
    plan = {'schema': 1, 'protocol': 'LongBench-v1-LB13-current-main-split-prompt; seed=42; bf16',
            'budget_semantics': 'nominal K2; not verified equal effective/protected token caps',
            'manifest': m, 'files': frozen, 'model_inventory': model, 'config_inventory': config, 'code': code_identity(),
            'python': str(Path(sys.executable).resolve()), 'python_sha256': sha(sys.executable),
            'packages': worker_packages(environment()), 'environment': environment(), 'jobs': jobs}
    plan['id'] = digest(plan)
    return plan


def verify_plan(plan):
    unsigned = {k: v for k, v in plan.items() if k != 'id'}
    if plan.get('schema') != 1 or digest(unsigned) != plan.get('id'):
        raise ValueError('plan identity mismatch')
    if code_identity() != plan['code'] or environment() != plan['environment']:
        raise ValueError('code/environment drift; prepare a new plan')
    if (str(Path(sys.executable).resolve()) != plan['python'] or sha(sys.executable) != plan['python_sha256'] or worker_packages(plan['environment']) != plan['packages']):
        raise ValueError('Python interpreter drift')
    for key, directory in [('model_inventory', 'model_path'), ('config_inventory', 'config_path')]:
        if inventory(plan['manifest'][directory]) != plan[key]:
            raise ValueError(f'input inventory drift: {directory}')
    for path, expected in plan['files'].items():
        if sha(path) != expected:
            raise ValueError(f'input drift: {path}')


def inspect_treatments(plan):
    """CPU-only imports through the actual producer's argument/config helpers."""
    from types import SimpleNamespace
    from sparse_attn.arguments import add_sparse_attn_args
    from sparse_attn.info import (get_method_name_with_info, get_treatment_manifest_json,
                                  resolve_treatment_snapshot, truncate_output_name_keep_hash)
    from sparse_attn.ops.far_skip_runtime import resolve_far_skip_modes
    output = {}
    for job in plan['jobs']:
        if job['blocked']:
            continue
        parser = add_sparse_attn_args(argparse.ArgumentParser())
        args = parser.parse_args(job['flags'])
        resolve_far_skip_modes(args, config=SimpleNamespace(**read(Path(plan['manifest']['model_path']) / 'config.json')))
        snapshot = resolve_treatment_snapshot(args) if args.method == 'tli' else None
        method = get_method_name_with_info(args, snapshot)
        prefix = job['task'].split('-')[0]
        name = f"{prefix}-{method}-{plan['id'][:16]}".replace(' ', '').replace("'", '')
        if len(name) > 230:
            name = truncate_output_name_keep_hash(prefix, method, plan['id'][:16]).replace(' ', '').replace("'", '')
        output[job['id']] = {'filename': name + '.jsonl',
            'treatment': json.loads(get_treatment_manifest_json(args, snapshot)) if args.method == 'tli' else None}
    return output


def protocol_preflight(plan):
    # No visible CUDA device, model loading, tokenizer loading or generation here.
    env = {**plan['environment'], 'CUDA_VISIBLE_DEVICES': ''}
    result = subprocess.run([plan['python'], '-m', 'benchmark.LongBench.run_budget_sweep', '--_inspect'],
                            cwd=ROOT, env=env, input=json.dumps(plan), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=True)
    return json.loads(result.stdout)


def query_gpus():
    raw = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=10)
    root = ET.fromstring(raw)
    result = {}
    def number(g, key, unit):
        text = g.findtext(key, '')
        if not re.fullmatch(r'\d+ ' + re.escape(unit), text):
            raise ValueError(f'unknown GPU telemetry: {key}={text!r}')
        return int(text.split()[0])
    for gpu in root.findall('gpu'):
        ident = gpu.findtext('uuid', '')
        if not re.fullmatch(UUID_PATTERN, ident) or ident in result:
            raise ValueError('invalid/duplicate GPU UUID')
        if gpu.findtext('mig_mode/current_mig') != 'Disabled':
            continue  # Unknown and MIG-enabled physical GPUs are never candidates.
        processes = gpu.find('processes')
        if processes is None:
            raise ValueError('missing process telemetry')
        result[ident] = {'free': number(gpu, 'fb_memory_usage/free', 'MiB'),
                         'used': number(gpu, 'fb_memory_usage/used', 'MiB'),
                         'util': number(gpu, 'utilization/gpu_util', '%'),
                         'busy': bool(list(processes)) or bool((processes.text or '').strip())}
    return result


@contextlib.contextmanager
def lock(path):
    # Stable inode: never unlink, truncate, PID-delete or destructively unlock.
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o660)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('lock is not a regular file')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally:
        # close (not LOCK_UN): inherited guardian/worker FDs keep lease after parent death.
        os.close(fd)


@contextlib.contextmanager
def acquire_gpu(allowed, directory, wait, stable, poll, min_free, max_used, max_util):
    directory = Path(directory).resolve(strict=True)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        for ident in allowed:
            try:
                with lock(directory / (ident + '.lock')) as fd:
                    since = None
                    while time.monotonic() < deadline:
                        g = query_gpus().get(ident)
                        idle = g and not g['busy'] and g['free'] >= min_free and g['used'] <= max_used and g['util'] <= max_util
                        if not idle:
                            break
                        now = time.monotonic()
                        since = now if since is None else since
                        if now - since >= stable:
                            yield ident, fd
                            return
                        time.sleep(min(poll, max(0, deadline - now)))
            except BlockingIOError:
                continue
        time.sleep(min(poll, max(0, deadline - time.monotonic())))
    raise TimeoutError('no permitted GPU passed locked stable idle window before deadline')


def stop_group(child, grace=2):
    # Even if the leader exited, terminate its remaining non-daemonizing descendants.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            time.sleep(grace)
    child.wait(timeout=10)


def interrupted(signum, frame):
    raise InterruptedError(f'signal {signum}')


def guard(spec_path, fd, parent):
    # A separate Linux guardian owns the lease and the managed worker lifecycle.
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), 'prctl PDEATHSIG failed')
    if os.getppid() != parent:
        raise InterruptedError('parent exited before guardian registration')
    spec = read(spec_path)
    child = None
    try:
        blocked = {signal.SIGTERM, signal.SIGINT, signal.SIGHUP}
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        try:
            child = subprocess.Popen(spec['argv'], cwd=ROOT, env=spec['env'],
                                     start_new_session=True, pass_fds=(fd,),
                                     preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask))
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        return child.wait(timeout=spec['timeout'])
    finally:
        # Ignore repeated termination during cleanup; no other jobs are signalled.
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        if child is not None:
            stop_group(child)


def run_child(argv, env, timeout, fd, attempt, label):
    spec = attempt / (label + '.command.json')
    atomic(spec, {'argv': argv, 'env': env, 'timeout': timeout})
    with (attempt / (label + '.log')).open('xb') as log:
        child = subprocess.Popen([sys.executable, '-m', 'benchmark.LongBench.run_budget_sweep',
                                  '--_guard', str(spec), str(fd), str(os.getpid())],
                                 cwd=ROOT, env=env, pass_fds=(fd,), stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = child.wait(timeout=timeout + 30)
            if rc:
                raise RuntimeError(f'{label} failed rc={rc}; see {log.name}')
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=20)


def validate_prediction(attempt, job, plan, expected_treatment):
    files = list((attempt / 'pred').glob('*.jsonl'))
    if len(files) != 1:
        raise ValueError('expected exactly one prediction file')
    path = files[0]
    if path.name != expected_treatment['filename']:
        raise ValueError('actual budget/config filename differs from resolved producer flags')
    if expected_treatment['treatment'] is not None:
        if read(str(path) + '.tli_manifest.json') != expected_treatment['treatment']:
            raise ValueError('actual TLI K1/K2/treatment mismatch')
    meta = read_run_cap_manifest(str(path) + RUN_CAP_SIDECAR_SUFFIX)
    actual_sha, lines = compute_result_binding(str(path))
    expected = {'status': 'complete', 'provenance': 'inline', 'dataset': job['task'],
                'method': METHODS[job['method']], 'model_name': plan['manifest']['model'],
                'model_path': plan['manifest']['model_path'], 'effective_cap': job['cap'],
                'max_new_tokens': job['max_gen'], 'n_samples': job['rows'],
                'run_timestamp': plan['id'][:16],
                'result_lines': job['rows'], 'result_file': path.name, 'result_sha256': actual_sha}
    if lines != job['rows'] or any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('prediction receipt mismatch')
    if meta['producer']['git_commit'] != plan['code']['git_sha'] or meta['producer']['script_sha256'] != sha(ROOT / 'benchmark/LongBench/pred.py'):
        raise ValueError('producer code mismatch')
    identity = meta.get('input_identity', {})
    for key in ('schema', 'source_sha256', 'sample_count', 'prompt_template_sha256'):
        if identity.get(key) != job['input_identity'].get(key):
            raise ValueError(f'input identity mismatch: {key}')
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    source = [json.loads(s) for s in (Path(plan['manifest']['dataset_path']) / (job['task'] + '.jsonl')).read_text().splitlines()]
    for i, (row, src) in enumerate(zip(rows, source)):
        expected_id = f"{job['task']}:{job['input_identity']['source_fingerprint']}:{i}"
        if row.get('_id') != expected_id or row.get('answers') != src['answers']:
            raise ValueError('prediction sample/answers mismatch')
    return path


def validate_score(attempt, prediction):
    data = read(attempt / 'pred/result.json')
    if set(data) - {'_meta'} != {prediction.name}:
        raise ValueError('score coverage mismatch')
    value = data[prediction.name]['score']
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError('invalid quality score')
    return value


def result_files(attempt, pred):
    paths = [pred, Path(str(pred) + RUN_CAP_SIDECAR_SUFFIX), attempt / 'pred/result.json']
    treatment = Path(str(pred) + '.tli_manifest.json')
    if treatment.is_file():
        paths.append(treatment)
    return paths


def completed(jobdir, job, plan, expected_treatment):
    for attempt in sorted(jobdir.glob('attempt-*'), reverse=True):
        try:
            receipt = read(attempt / 'complete.json')
            if receipt['plan_id'] != plan['id'] or receipt['job_id'] != job['id']:
                continue
            pred = validate_prediction(attempt, job, plan, expected_treatment)
            score = validate_score(attempt, pred)
            if receipt['hashes'] != {str(p.relative_to(attempt)): sha(p) for p in result_files(attempt, pred)}:
                continue
            return {'attempt': str(attempt), 'score': score}
        except (OSError, ValueError, KeyError, TypeError, SystemExit):
            continue
    return None


def execute(plan, args):
    verify_plan(plan)
    missing = [name for name, version in plan['packages'].items() if version is None]
    if missing:
        raise ValueError(f'missing runtime/scorer packages before GPU acquisition: {missing}')
    blocked = [j for j in plan['jobs'] if j['blocked']]
    if blocked and not args.allow_partial:
        raise ValueError('blocked arms; inspect dry-run, repair prerequisites and re-prepare, or explicitly --allow-partial')
    protocols = protocol_preflight(plan)
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    status = {'plan_id': plan['id'], 'status': 'incomplete', 'protocol': plan['protocol'],
              'budget_semantics': plan['budget_semantics'], 'k1': plan['manifest']['k1'],
              'jobs': {j['id']: {'status': 'pending', 'task': j['task'], 'method': j['method'], 'k2': j['k2']} for j in plan['jobs']}}
    # Same plan/output namespace cannot be dispatched twice, even to different GPUs.
    with lock(output / (plan['id'] + '.lock')):
        atomic(output / (plan['id'] + '.summary.json'), status)
        for job in plan['jobs']:
            key = job['id']
            if job['blocked']:
                status['jobs'][key].update(status='blocked', reasons=job['blocked']); continue
            jobdir = output / plan['id'] / key; jobdir.mkdir(parents=True, exist_ok=True)
            expected_treatment = protocols[key]
            previous = completed(jobdir, job, plan, expected_treatment)
            if previous:
                status['jobs'][key].update(status='complete', **previous); continue
            verify_plan(plan)
            attempt = jobdir / ('attempt-' + uuid.uuid4().hex); attempt.mkdir()
            atomic(attempt / 'identity.json', {'plan': plan, 'job': job})
            try:
                with acquire_gpu(args.gpus, args.lock_dir, args.wait_seconds,
                                 args.idle_seconds, args.poll_seconds, args.min_free_mib,
                                 args.max_used_mib, args.max_util_percent) as (gpu, fd):
                    env = {**plan['environment'], 'CUDA_VISIBLE_DEVICES': gpu}
                    m = plan['manifest']
                    argv = [plan['python'], '-m', 'benchmark.LongBench.pred', '--model', m['model'],
                            '--model_path', m['model_path'], '--dataset-path', m['dataset_path'],
                            '--config-path', m['config_path'], '--task', job['task'],
                            '--output-dir', str(attempt), '--t', plan['id'][:16], *job['flags']]
                    run_child(argv, env, args.job_timeout_seconds, fd, attempt, 'prediction')
                    pred = validate_prediction(attempt, job, plan, expected_treatment)
                    run_child([plan['python'], '-m', 'benchmark.LongBench.eval', '--output-dir', str(attempt),
                               '--expect-count', str(job['rows']), '--require-runcap'],
                              env, args.score_timeout_seconds, fd, attempt, 'scoring')
                    score = validate_score(attempt, pred)
                    verify_plan(plan)
                    receipt = {'plan_id': plan['id'], 'job_id': key, 'gpu_uuid': gpu, 'score': score,
                               'hashes': {str(p.relative_to(attempt)): sha(p) for p in result_files(attempt, pred)}}
                    atomic(attempt / 'complete.json', receipt)
                    status['jobs'][key].update(status='complete', attempt=str(attempt), score=score)
                    atomic(output / (plan['id'] + '.summary.json'), status)
            except (Exception, SystemExit) as exc:
                atomic(attempt / 'failure.json', {'error': str(exc), 'plan_id': plan['id'], 'job_id': key})
                status['jobs'][key].update(status='failed', attempt=str(attempt), error=str(exc))
                atomic(output / (plan['id'] + '.summary.json'), status)
                raise
        if all(j['status'] == 'complete' for j in status['jobs'].values()):
            status['status'] = 'complete'
        atomic(output / (plan['id'] + '.summary.json'), status)
    return 0 if status['status'] == 'complete' else 2


def main():
    if sys.argv[1:] == ['--_packages']:
        print(json.dumps(packages())); return 0
    if sys.argv[1:] == ['--_inspect']:
        print(json.dumps(inspect_treatments(json.load(sys.stdin)))); return 0
    if len(sys.argv) == 5 and sys.argv[1] == '--_guard':
        return guard(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', metavar='MANIFEST'); p.add_argument('--plan', required=True)
    p.add_argument('--execute', action='store_true'); p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--output'); p.add_argument('--gpus', nargs='+'); p.add_argument('--lock-dir')
    for name, default in [('wait-seconds', 600), ('idle-seconds', 30), ('poll-seconds', 5),
                          ('job-timeout-seconds', 21600), ('score-timeout-seconds', 1800),
                          ('min-free-mib', None), ('max-used-mib', 256), ('max-util-percent', 0)]:
        p.add_argument('--' + name, type=int, default=default)
    args = p.parse_args()
    if args.prepare:
        if args.execute or Path(args.plan).exists():
            p.error('prepare is separate from execute and never overwrites a plan')
        plan = prepare(read(args.prepare)); atomic(args.plan, plan)
    else:
        plan = read(args.plan)
    if not args.execute:
        print(json.dumps({'plan_id': plan['id'], 'mode': 'dry-run', 'jobs': [
            {k: j[k] for k in ('task', 'method', 'k2', 'blocked', 'flags')} for j in plan['jobs']]}, indent=2))
        return 0
    if not args.output or not args.gpus or not args.lock_dir:
        p.error('execute requires --output, --gpus UUID..., and shared local --lock-dir')
    if len(set(args.gpus)) != len(args.gpus) or any(not re.fullmatch(UUID_PATTERN, g) for g in args.gpus):
        p.error('only unique full physical GPU UUIDs are accepted')
    if args.min_free_mib is None:
        if plan['manifest']['model'] == 'Qwen3-32B':
            p.error('Qwen3-32B requires an explicit --min-free-mib capacity threshold')
        args.min_free_mib = 40000
    for name in ('wait_seconds', 'idle_seconds', 'poll_seconds', 'job_timeout_seconds', 'score_timeout_seconds', 'min_free_mib'):
        positive(getattr(args, name), name)
    if not 0 <= args.max_util_percent <= 100 or args.max_used_mib < 0 or args.idle_seconds >= args.wait_seconds:
        p.error('invalid idle thresholds/window')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    return execute(plan, args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (Exception, SystemExit) as error:
        if isinstance(error, SystemExit):
            raise
        print(f'[SWEEP-FAIL] {error}', file=sys.stderr)
        sys.exit(1)
