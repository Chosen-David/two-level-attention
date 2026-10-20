"""Observe the unresolved v1 source-identity gap; no model or producer import."""
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'benchmark/LongBench/pred.py'
TEMPLATES = ROOT / 'benchmark/LongBench/config/dataset2prompt.json'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))


def assignment(name, contains):
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
               and contains in ast.unparse(node.value)]
    if len(matches) != 1:
        raise RuntimeError('production fingerprint extraction changed: ' + name)
    return matches[0]


fingerprint_nodes = [assignment('_fp_src', 'obj.get'),
                     assignment('data_fp', '_hashlib.sha256')]
fingerprint_code = compile(ast.Module(body=fingerprint_nodes, type_ignores=[]),
                           str(SOURCE), 'exec')
id_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Subscript) and ast.unparse(t) == "record['_id']"
                    for t in node.targets) and 'data_fp' in ast.unparse(node.value)]
if len(id_nodes) != 1:
    raise RuntimeError('production v1 ID extraction changed')
id_code = compile(ast.Expression(id_nodes[0].value), str(SOURCE), 'eval')
prompt_node = assignment('prompt', 'prompt_format.format(**json_obj)')
prompt_code = compile(ast.Expression(prompt_node.value), str(SOURCE), 'eval')
template = json.loads(TEMPLATES.read_text(encoding='utf-8'))['qasper']


def observe(data):
    env = {'json': json, '_hashlib': hashlib, 'data': data}
    exec(fingerprint_code, env)
    fp = env['data_fp']
    prompts = [eval(prompt_code, {'prompt_format': template, 'json_obj': row}) for row in data]
    ids = [eval(id_code, {'dataset': 'qasper', 'data_fp': fp, 'row_idx': i})
           for i in range(len(data))]
    return {'fingerprint': fp, 'ids': ids,
            'prompt_sha256': [hashlib.sha256(p.encode()).hexdigest() for p in prompts]}


base = [dict(input='Which event?', context='alpha evidence', answers=['yes'],
             length=100, all_classes=[]),
        dict(input='Which event?', context='bravo evidence', answers=['yes'],
             length=100, all_classes=[])]
cases = {
    'context': [dict(base[0], context='other evidence'), base[1]],
    'question': [dict(base[0], input='Where event?'), base[1]],
    'row_swap': base[::-1],
    'unicode_context': [dict(base[0], context='不同正文😀'), base[1]],
}
original = observe(base)
rows = []
for name, data in cases.items():
    changed = observe(data)
    if (changed['prompt_sha256'] == original['prompt_sha256']
            or changed['fingerprint'] != original['fingerprint']
            or changed['ids'] != original['ids']):
        raise RuntimeError('identity-gap premise changed: ' + name)
    rows.append({'case': name, **changed})
controls = {'unchanged': observe(base),
            'answers': observe([dict(base[0], answers=['no']), base[1]]),
            'length': observe([dict(base[0], length=101), base[1]])}
if (controls['unchanged'] != original
        or any(controls[key]['fingerprint'] == original['fingerprint']
               for key in ('answers', 'length'))):
    raise RuntimeError('production fingerprint controls failed')
print(json.dumps({'status': 'confirmed_identity_gap_unfixed',
                  'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  'template_sha256': hashlib.sha256(TEMPLATES.read_bytes()).hexdigest(),
                  'input_sha256': hashlib.sha256(json.dumps(
                      {'base': base, 'cases': cases}, ensure_ascii=False,
                      sort_keys=True).encode()).hexdigest(),
                  'base': original, 'cases': rows, 'controls': controls,
                  'scope': 'source-bound stdlib prompt/fingerprint/ID expressions only; '
                           'no tokenizer, actual prediction, sidecar acceptance or historical contamination proof'},
                 ensure_ascii=False, indent=2))
