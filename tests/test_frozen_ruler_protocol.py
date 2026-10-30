import ast
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


class StopProtocol(unittest.TestCase):
    def run_native(self, stop_after_tokens):
        source = ast.parse((ROOT/'benchmark/LongBench/prefill_frozen_skip.py').read_text())
        native = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_native')
        native.decorator_list = []
        class Output:
            def __init__(self, token):
                self.logits = torch.zeros(1, 1, 4)
                self.logits[0, 0, token] = 10
        class Model:
            config = None
            generation_config = type('Config', (), dict(eos_token_id=2))()
            def __init__(self): self.tokens=iter([2,1,2])
            def __call__(self,*a,**kw): return Output(next(self.tokens))
        namespace = dict(torch=torch, DynamicCache=lambda **kw:None,
                         greedy_token=lambda logits,*a:logits.argmax(-1,keepdim=True))
        exec(compile(ast.Module(body=[native],type_ignores=[]),'<native>', 'exec'),namespace)
        return namespace['generate_native'](Model(),[1],64,'cpu',stop_after_tokens=stop_after_tokens)

    def test_longbench_default_stops_on_first_eos(self):
        self.assertEqual(self.run_native(1),[2])

    def test_ruler_ignores_first_eos_without_masking_it(self):
        self.assertEqual(self.run_native(2),[2,1,2])

    def test_official_match_fraction(self):
        from benchmark.RULER.frozen_region_compare import score
        self.assertEqual(score('Found AbC', ['abc','xyz']),.5)
        with self.assertRaises(ValueError): score('text',[])


if __name__=='__main__':unittest.main()
