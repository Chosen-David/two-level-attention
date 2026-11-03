"""Host capacity regression; optional real cache freeze on CPU/CUDA."""
import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('NFA_CACHE_SOURCE', ROOT / 'sparse_attn/cache/paged_region_cache.py'))

def bound_method():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GraphReadyPagedRegionCache')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_max_resident_pages_for_layer')
    scope = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope[fn.name]

class SeedPageCapacity(unittest.TestCase):
    def native_freeze(self, device):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('PyTorch unavailable')
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        spec=importlib.util.spec_from_file_location('seed_capacity_cache',SOURCE)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        from sparse_attn.policies.region_geometry import RegionGeometry
        cache=module.GraphReadyPagedRegionCache(1,RegionGeometry(),max_requests=1,
            max_physical_pages=600,max_blocks=600,max_visible_len=32768,
            max_query_tokens=1,num_kv_heads=1,head_dim_k=8,head_dim_v=8,
            dtype=torch.float32,device=device)
        for _ in range(2):
            rid=cache.add_request(0)
            kv=torch.arange(32768,device=device,dtype=torch.float32).view(1,1,-1,1).expand(1,1,-1,8)
            cache.write_prefill(0,rid,kv,kv)
            seeds=list(range(8192,12288,64))
            cache.freeze(rid,[True],[False],{0:seeds})
            state=cache.states[rid]
            self.assertEqual(state.max_resident_pages_by_layer,[133])
            for token in seeds:
                physical=state.block_tables[0][token//64]
                self.assertEqual(float(cache.layers[0].kv_pool_k[physical,0,0,token%64,0]),float(token))
            state.first_decode_seeded=True
            cache.pending_lengths={rid:32768}
            cache.commit_step()
            self.assertEqual(state.max_resident_pages_by_layer,[69])
            self.assertTrue(all(t//64 not in state.block_tables[0] for t in seeds))
            if device=='cuda': torch.cuda.synchronize()
            cache.remove_request(rid)

    def test_native_cpu_freeze_release_reuse(self):
        self.native_freeze('cpu')

    def test_native_cuda_freeze_release_reuse(self):
        self.native_freeze('cuda')

    def test_distant_seed_pages_and_consumption(self):
        bound = bound_method()
        geo = NS(page=64,sink=4,near=4096,window=128,max_new_tokens=512)
        state = NS(prompt_length=32768,logical_length=32768,first_decode_seeded=False,
                   skip_far_by_layer=[True],skip_near_by_layer=[False],
                   prefill_topk_ids_by_layer={0:list(range(8192,12288,64))})
        cache = NS(geometry=geo)
        self.assertEqual(bound(cache,state,0),133)
        state.first_decode_seeded=True
        self.assertEqual(bound(cache,state,0),69)

    def test_independent_page_set_and_invalid_ids(self):
        bound = bound_method()
        cases=0
        for length in (7,257,32768,65537):
          for page in (16,64,128):
            for far_skip,near_skip in ((False,False),(False,True),(True,False),(True,True)):
              for consumed in (False,True):
                geo=NS(page=page,sink=4,near=4096,window=128,max_new_tokens=512)
                ids=list(range(0,length,max(1,length//64)))[:64]
                ids += ids[:3]+[-1,length,length+999]
                state=NS(prompt_length=length,logical_length=length,first_decode_seeded=consumed,
                    skip_far_by_layer=[far_skip],skip_near_by_layer=[near_skip],prefill_topk_ids_by_layer={0:ids})
                sink=min(4,length); near=max(sink,length-128)
                far=min(near,max(sink,((length-128-4096)//page)*page))
                resident=set(range(sink))|set(range(near,length))
                if not far_skip: resident.update(range(sink,far))
                if not near_skip: resident.update(range(far,near))
                if not consumed: resident.update(x for x in ids if 0<=x<length)
                required=len({x//page for x in resident})
                value=bound(NS(geometry=geo),state,0)
                self.assertGreaterEqual(value,required)
                state.prefill_topk_ids_by_layer={0:sorted({x for x in ids if 0<=x<length})}
                self.assertEqual(value,bound(NS(geometry=geo),state,0))
                cases+=1
        self.assertEqual(cases,96)

if __name__ == '__main__':
    unittest.main()
