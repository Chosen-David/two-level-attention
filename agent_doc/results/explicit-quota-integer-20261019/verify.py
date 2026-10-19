"""Host constructor differential; run from any cwd, no tensor/GPU claim."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
BASE = 'bf9b07994b187cdebfd5070d94b80d27292e73ec'
sys.path.insert(0, str(ROOT / 'tests'))
import test_explicit_near_page_budget as loader


def state(cls, changes):
    try:
        obj = cls(loader.config(**changes))
    except Exception as exc:
        return {'error': type(exc).__name__, 'message': str(exc)}
    names = ('nt_near_pages', 'nt_near_tokens', 'nt_near_len',
             'alpha', 'beta', 'gamma', 'sink_tokens', 'sliding_window_size')
    return {key: getattr(obj, key) for key in names}


with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, TLI_E151C_ORACLE_QUOTA=''):
    old_root = Path(tmp)
    for name in ('base.py', 'tia_indexer.py', 'tli_indexer.py'):
        rel = 'sparse_attn/indexer/' + name
        path = old_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(['git', 'show', BASE + ':' + rel], cwd=ROOT))
    with patch.object(loader, 'ROOT', old_root):
        old = loader.load_constructors()
    new = loader.load_constructors()
    cases = []
    for name in ('tli_near_pages', 'tli_near_tokens', 'tli_near_len'):
        for value in (-.5, .5):
            args = dict(tli_near_pages=0, tli_near_tokens=0, tli_near_len=1024)
            args[name] = value
            before, after = state(old, args), state(new, args)
            if 'error' in before or after.get('error') != 'ValueError' or name not in after['message']:
                raise RuntimeError((args, before, after))
            cases.append({'input': args, 'before': before, 'after': after})
    controls = []
    for bs, k1, nb, nt in itertools.product((32, 64, 128), (1, 4, 128), (0, 1, 4), (0, 32, 'off')):
        args = dict(tia_block_size=bs, tia_level1_topk=k1, tli_near_pages=nb,
                    tli_near_tokens=nt, tli_near_len=nb * bs, tli_swa_size=bs)
        before, after = state(old, args), state(new, args)
        if before != after:
            raise RuntimeError((args, before, after))
        controls.append({'input': args, 'accepted': 'error' not in after})
    for a, b, g in itertools.product((0., .25, 1.), (0., .5, 1.), (0., 1., 'off')):
        args = dict(tli_near_pages=None, tli_near_tokens=None, tli_near_len=None,
                    tli_alpha=a, tli_beta=b, tli_gamma=g)
        if state(old, args) != state(new, args):
            raise RuntimeError(args)
        controls.append({'input': args, 'accepted': True})
    encoded = json.dumps([row['input'] for row in cases + controls], sort_keys=True,
                         separators=(',', ':')).encode()
    print(json.dumps({
        'baseline': BASE, 'scope': 'complete host constructor bodies; no tensor work',
        'python': sys.version, 'platform': platform.platform(),
        'source_sha256': hashlib.sha256((ROOT / 'sparse_attn/indexer/tli_indexer.py').read_bytes()).hexdigest(),
        'input_sha256': hashlib.sha256(encoded).hexdigest(),
        'regressions': cases, 'controls': controls,
        'summary': {'truncated_fractional_inputs_before': len(cases),
                    'rejected_fractional_inputs_after': len(cases),
                    'identical_controls': len(controls),
                    'accepted_controls': sum(row['accepted'] for row in controls)},
    }, ensure_ascii=False, indent=2))
