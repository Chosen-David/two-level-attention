"""Compare complete host constructor states with the frozen main parent."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from test_explicit_near_page_budget import ROOT, config, load_constructors

BASE = '50abf698a16c506f30fe72a3fbb1f0a4df080d15'
PATH = 'sparse_attn/indexer/tli_indexer.py'


def main():
    candidate = load_constructors()
    source = subprocess.check_output(['git', 'show', BASE + ':' + PATH],
                                     cwd=ROOT, text=True)
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.ClassDef) and n.name == 'TLIIndexer')
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef)
                 and n.name == '__init__']
    scope = dict(candidate.__init__.__globals__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), PATH, 'exec'), scope)
    baseline = scope['TLIIndexer']
    cases = []
    for bs in (16, 32, 64, 128):
        for nb in (0, 1, 2, 4):
            for nt in (0, 'off') if nb == 0 else (0, 16, 'off'):
                for bypass in (False, True):
                    for quant in (False, True):
                        cases.append((None, dict(tia_block_size=bs,
                            tli_swa_size=bs, tli_near_pages=nb,
                            tli_near_tokens=nt, tli_near_len=nb * bs,
                            tli_boundary_bypass=bypass, tli_l2_quant=quant)))
    for quota in (None, [0], [32, 128], [0.5, 1.5, 2.5]):
        for gamma in (0, .5, 1):
            for far in ('4bit', 'cluster', 'sim_greedy'):
                cases.append((quota, dict(tli_near_pages=None,
                    tli_near_tokens=None, tli_near_len=None,
                    tli_alpha=.5, tli_beta=.5, tli_gamma=gamma,
                    tli_far_select=far)))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'quota.json'
        for quota, case in cases:
            path.write_text(json.dumps(quota))
            with patch.dict(os.environ, TLI_E151C_ORACLE_QUOTA=(
                    '' if quota is None else str(path))):
                old, new = baseline(config(**case)), candidate(config(**case))
            if vars(old) != vars(new):
                raise RuntimeError('legal constructor state changed: ' + repr(case))
        rejected = 0
        conflicts = []
        for nt in (0, 32, 128):
            for quota in ([0], [32], [128]):
                conflicts.append((nt, quota))
                path.write_text(json.dumps(quota))
                with patch.dict(os.environ, TLI_E151C_ORACLE_QUOTA=str(path)):
                    baseline(config(tli_near_tokens=nt))
                    try:
                        candidate(config(tli_near_tokens=nt))
                    except ValueError as exc:
                        if 'E151c' not in str(exc):
                            raise
                        rejected += 1
                    else:
                        raise RuntimeError('conflicting quota accepted')
    print(json.dumps(dict(base=BASE, legal_state_equal=len(cases),
        baseline_conflicts_accepted=len(conflicts), candidate_conflicts_rejected=rejected,
        input_sha256=hashlib.sha256(json.dumps([cases, conflicts], sort_keys=True).encode()).hexdigest(),
        scope='host constructors only; no tensor/GPU/performance claim'), indent=2))


if __name__ == '__main__':
    main()
