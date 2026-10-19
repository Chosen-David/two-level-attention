"""Source-bound HOST geometry regression; not a tensor/GPU selector test.

Run before and after the pending TL-EXPLICIT-NEAR-WIDTH patch. A nonzero exit
means production host expressions still expand an explicit sub-page width.
The complete production constructor chain is executed by the existing loader;
only the named integer-geometry branches are extracted below, not whole tensor
methods. This limited scope is intentional and is printed in every result.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_explicit_near_page_budget import config, load_constructors


def main():
    path = ROOT / "sparse_attn/indexer/tli_indexer.py"
    source = path.read_text()
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == "TLIIndexer")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    predicate = ast.dump(ast.parse("self.nt_near_len is not None", mode="eval").body)
    names = ("_maybe_build_kmeans", "compute_score", "compute_mask",
             "calibrate_dyn_dgate")
    branches = {}
    for name in names:
        matches = [n for n in ast.walk(methods[name])
                   if isinstance(n, ast.If) and ast.dump(n.test) == predicate]
        if len(matches) != 1:
            raise RuntimeError(f"{name}: expected exactly one explicit-width branch")
        branches[name] = compile(ast.Module(body=matches, type_ignores=[]), str(path), "exec")
    boundary = [n for n in methods["compute_mask"].body
                if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "near_blks" for t in n.targets)]
    if len(boundary) != 1:
        raise RuntimeError("compute_mask near boundary must be re-audited")
    boundary_code = compile(ast.Module(body=boundary, type_ignores=[]), str(path), "exec")
    indexer = load_constructors()
    property_scope = {}
    exec(compile(ast.Module(body=[methods["_sink_pool_lo_blk"]], type_ignores=[]),
                 str(path), "exec"), property_scope)
    indexer._sink_pool_lo_blk = property_scope["_sink_pool_lo_blk"]
    cases, failures = [], []
    with patch.dict(os.environ, TLI_E151C_ORACLE_QUOTA=""):
        for bs in (32, 64, 128):
            for nl in (0, 1, bs - 1, bs, bs + 1, 2 * bs, 512):
                for nt in (0, "off"):
                    for length in (0, 2 * bs, 2 * bs + 1, 1024, 4097):
                        case = dict(bs=bs, nl=nl, nt=nt, length=length)
                        cases.append(case)
                        obj = indexer(config(
                            tia_block_size=bs, tli_sink_tokens=bs, tli_swa_size=bs,
                            tia_level2_topk=2 * bs + 128,
                            tli_near_pages=min(2, nl // bs),
                            tli_near_tokens=nt, tli_near_len=nl))
                        for name in names:
                            scope = dict(self=obj, bs=bs, _bs=bs, sink_tok=bs,
                                         mid_len=length - 2 * bs,
                                         _mid_len=max(0, length - 2 * bs))
                            exec(branches[name], scope)
                            if name == "calibrate_dyn_dgate":
                                got, want = scope["far_hi"], bs + max(0, length - 2 * bs - nl)
                            else:
                                got = scope.get("near_len_dyn", scope.get("_near_len_dyn"))
                                want = nl
                            if got != want:
                                failures.append(dict(case, method=name, got=got, want=want))
    # Aligned, non-overlapping protection bands avoid page rounding/bypass ambiguity.
    obj = indexer(config(tli_near_pages=0, tli_near_tokens=0, tli_near_len=0))
    scope = dict(self=obj, bs=64, mid_len=896, near_base=960)
    exec(branches["compute_mask"], scope)
    exec(boundary_code, scope)
    example = dict(S=1024, bs=64, sink=64, swa=64, nb=0, nt=0, nl=0,
                   selector_far_hi=scope["near_blks"] * 64, expected_far_hi=960)
    print(json.dumps(dict(
        scope="host constructor and extracted integer branches only; no tensor/GPU run",
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        input_sha256=hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        cases=len(cases), branch_checks=len(cases) * len(names),
        failures=len(failures), first_failures=failures[:3], example=example,
    ), ensure_ascii=False, indent=2))
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
