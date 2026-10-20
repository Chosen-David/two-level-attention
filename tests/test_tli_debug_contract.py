"""Execute production debug guards without importing Torch; no GPU claim."""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'sparse_attn/indexer/tli_indexer.py'
OFF = (None, '', '0', 'false', 'False', ' FALSE ', ' 0 ', ' \t')
ON = ('1', 'true', 'TRUE', 'yes', 'off', 'diagnostic')


def guards(source):
    tree = ast.parse(source)
    assignments = {n.targets[0].id: n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name)
                   and 'TLI_DEBUG' in ast.unparse(n.value)}
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        condition = ast.unparse(node.test)
        if 'TLI_DEBUG' in condition or '_dbg' in condition:
            prelude = [ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                                  value=copy.deepcopy(value))
                       for name, value in assignments.items() if name in condition]
            result.append((node, prelude))
    return result


def scope(layer):
    return {'os': os, '_os': os, 'self': SimpleNamespace(layer_idx=layer)}


class DebugContractTests(unittest.TestCase):
    def test_all_debug_sites_are_covered(self):
        self.assertEqual(len(guards(SOURCE.read_text())), 7)

    def test_disabled_does_not_execute_original_debug_body(self):
        # Original bodies require tensor/scalar locals deliberately absent here.
        # Any accidental entry raises instead of silently doing device readback.
        for node, prelude in guards(SOURCE.read_text()):
            code = compile(ast.fix_missing_locations(ast.Module(
                body=prelude + [node], type_ignores=[])), str(SOURCE), 'exec')
            for value in OFF:
                for layer in (None, 0, 1):
                    with self.subTest(line=node.lineno, value=value, layer=layer):
                        with patch.dict(os.environ):
                            os.environ.pop('TLI_DEBUG', None)
                            if value is not None:
                                os.environ['TLI_DEBUG'] = value
                            exec(code, scope(layer))

    def test_enabled_preserves_layer_filters_and_legacy_other_strings(self):
        for node, prelude in guards(SOURCE.read_text()):
            for value in ON:
                for layer in (None, 0, 1):
                    with self.subTest(line=node.lineno, value=value, layer=layer):
                        with patch.dict(os.environ, TLI_DEBUG=value):
                            env = scope(layer)
                            exec(compile(ast.fix_missing_locations(ast.Module(
                                body=prelude, type_ignores=[])), '<prelude>', 'exec'), env)
                            actual = eval(compile(ast.Expression(node.test), '<guard>', 'eval'), env)
                            expected = (layer is not None if 'is not None' in
                                        ast.unparse(node.test) else layer == 1)
                            self.assertEqual(bool(actual), expected)


if __name__ == '__main__':
    unittest.main()
