"""Host-only source-bound guard differential; not a tensor benchmark."""
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests'))
from test_tli_debug_contract import guards, scope, OFF, ON

BASE = '0b661c477fdc2fb98aad56b4ee314ca5f312e251'
PATH = 'sparse_attn/indexer/tli_indexer.py'
old = subprocess.check_output(['git', 'show', BASE + ':' + PATH], cwd=ROOT).decode()
new = (ROOT / PATH).read_text()


def evaluate(text, value, layer):
    answers = []
    with patch.dict(os.environ):
        os.environ.pop('TLI_DEBUG', None)
        if value is not None:
            os.environ['TLI_DEBUG'] = value
        for node, prelude in guards(text):
            env = scope(layer)
            exec(compile(ast.fix_missing_locations(ast.Module(
                body=prelude, type_ignores=[])), '<prelude>', 'exec'), env)
            answers.append(bool(eval(compile(ast.Expression(node.test), '<guard>', 'eval'), env)))
    return answers


def without_guards(text):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and (
                'TLI_DEBUG' in ast.unparse(node.test) or '_dbg' in ast.unparse(node.test)):
            node.test = ast.Constant(value=True)
    return ast.dump(tree, include_attributes=False)


records = []
for value in OFF + ON:
    for layer in (None, 0, 1):
        records.append(dict(value=value, layer=layer,
                            before=evaluate(old, value, layer),
                            after=evaluate(new, value, layer)))
off_records = [r for r in records if r['value'] in OFF]
off_before = sum(sum(r['before']) for r in off_records)
off_after = sum(sum(r['after']) for r in off_records)
controls = all(r['before'] == r['after'] for r in records if r['value'] in ON)
unchanged = without_guards(old) == without_guards(new)
if (off_before, off_after, controls, unchanged) != (54, 0, True, True):
    raise RuntimeError('guard differential contract failed')
print(json.dumps(dict(
    baseline=BASE, python=sys.version, optimize=sys.flags.optimize, platform=platform.platform(),
    baseline_sha256=hashlib.sha256(old.encode()).hexdigest(),
    candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),
    input_sha256=hashlib.sha256(json.dumps([OFF, ON, [None, 0, 1]]).encode()).hexdigest(),
    off_guard_evaluations=len(off_records) * 7,
    erroneous_debug_entries_before=off_before,
    erroneous_debug_entries_after=off_after,
    enabled_guard_controls=len(ON) * 3 * 7,
    enabled_controls_unchanged=controls,
    entire_module_ast_except_seven_debug_predicates_unchanged=unchanged,
    records=records), indent=2))
