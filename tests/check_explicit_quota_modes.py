"""Host differential check against the frozen parent; not a GPU oracle."""
import ast
import hashlib
import json
import os
import subprocess
from unittest.mock import patch

from test_explicit_near_page_budget import ROOT, config, load_constructors
from test_explicit_quota_modes import ALTERNATE_MODES

BASE = '927b7279bd66efcf2c90d23b2c5c4bbaf9ddc8e1'
PATH = 'sparse_attn/indexer/nfa_indexer.py'


def main():
    candidate = load_constructors()
    source = subprocess.check_output(['git', 'show', BASE + ':' + PATH],
                                     cwd=ROOT, text=True)
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.ClassDef) and n.name == 'NFAIndexer')
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef)
                 and n.name == '__init__']
    scope = dict(candidate.__init__.__globals__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), PATH, 'exec'), scope)
    baseline = scope['NFAIndexer']
    cases = []
    for bs in (16, 32, 64, 128):
        for nb in (0, 1, 2, 4):
            for nt in (0, 'off') if nb == 0 else (0, 16, 'off'):
                for bypass in (False, True):
                    for quant in (False, True):
                        cases.append(dict(tia_block_size=bs, nfa_swa_size=bs,
                            nfa_near_pages=nb, nfa_near_tokens=nt,
                            nfa_near_len=nb * bs, nfa_boundary_bypass=bypass,
                            nfa_l2_quant=quant))
    cases += [dict(nfa_near_pages=None, nfa_near_tokens=None,
                   nfa_near_len=None, **mode) for mode in ALTERNATE_MODES]
    with patch.dict(os.environ, NFA_E151C_ORACLE_QUOTA=''):
        for case in cases:
            old, new = baseline(config(**case)), candidate(config(**case))
            if vars(old) != vars(new):
                raise RuntimeError('constructor state changed: ' + repr(case))
        rejected = 0
        for mode in ALTERNATE_MODES:
            for nt in (0, 32, 'off'):
                baseline(config(nfa_near_tokens=nt, **mode))
                try:
                    candidate(config(nfa_near_tokens=nt, **mode))
                except ValueError as exc:
                    if 'fixed geometry' not in str(exc):
                        raise
                    rejected += 1
                else:
                    raise RuntimeError('alternate mode accepted')
    print(json.dumps(dict(base=BASE, legal_state_equal=len(cases),
        baseline_conflicts_accepted=15, candidate_conflicts_rejected=rejected,
        input_sha256=hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        scope='complete constructor state only; no tensor/GPU test'), indent=2))


if __name__ == '__main__':
    main()
