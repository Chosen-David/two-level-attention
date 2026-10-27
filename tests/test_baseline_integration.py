import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from sparse_attn.baseline_runtime import (baseline_preflight, baseline_request,
    ffd_identity, ffd_settings, install_ffd_guard, require_unmasked_decode,
    validate_minference_pattern, verified_vendor)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'exp/vendor_baselines/sparq_vendor'))
from qwen3_sparq_patch import enable_sparq


class FakeCache:
    def __init__(self, **kw):
        self.length = 0
        self.layers = []
        self.__dict__.update(kw)
    def get_seq_length(self): return self.length


class FakeFFD(torch.nn.Module):
    def __init__(self, settings, stats):
        super().__init__()
        self.config = types.SimpleNamespace(num_hidden_layers=2, attn_settings=dict(settings, debug_stats=stats))
        self.stats = stats
        self.wrong_route = False
        self.fail_after_update = False
    def forward(self, input_ids, past_key_values=None, **kw):
        if input_ids.shape[1] > 1: key = 'prefill_calls'
        else: key = 'flash_path_decode_calls' if self.wrong_route else 'sparse_path_calls'
        self.stats[key] = self.stats.get(key, 0) + 2
        if not past_key_values.layers:
            kw = {k:getattr(past_key_values,k) for k in ('BS','k_bits','use_fp8_residual','max_seq_len')}
            past_key_values.layers = [FakeCache(**kw) for _ in range(2)]
        for layer in past_key_values.layers:layer.length += input_ids.shape[1]
        past_key_values.length += input_ids.shape[1]
        if self.fail_after_update:raise RuntimeError('injected model failure')
        return types.SimpleNamespace(past_key_values=past_key_values)


class BaselineTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(7)
    def test_plan_no_gpu_import(self):
        for mode in ([], ['-O']):
            out = subprocess.check_output([sys.executable, *mode, '-m', 'sparse_attn.baseline_runtime'], cwd=ROOT)
            self.assertFalse(json.loads(out)['gpu_started'])
    def test_integrity_and_options(self):
        for name in ('ffd_vendor', 'sparq_vendor'): self.assertTrue(verified_vendor(name).is_dir())
        for delta in (0, -1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError): ffd_settings(types.SimpleNamespace(ffd_delta=delta))
        for cap in (0, 127, 129, True):
            with self.assertRaises(ValueError): ffd_settings(types.SimpleNamespace(ffd_cache_length=cap))
        with self.assertRaises(ValueError): baseline_preflight(types.SimpleNamespace(method='ffd_vendor'))
        with self.assertRaises(ValueError): baseline_preflight(types.SimpleNamespace(method='ffd_vendor',baseline_execute=True,model='Qwen3-8B'))
        baseline_preflight(types.SimpleNamespace(method='ffd_vendor',baseline_execute=True,model='Meta-Llama-3.1-8B-Instruct'))
    def ffd(self):
        args=types.SimpleNamespace(); settings, capacity=ffd_settings(args); stats={}
        model=FakeFFD(settings,stats); install_ffd_guard(model,FakeCache,stats,settings,capacity)
        model._baseline_identity=ffd_identity(args)
        return model
    def test_ffd_real_cache_injection_and_route_gate(self):
        model=self.ffd()
        with baseline_request(model) as receipt:
            out=model(input_ids=torch.ones(1,128,dtype=torch.long),past_key_values=None,use_cache=True)
            self.assertIsInstance(out.past_key_values,FakeCache)
            model(input_ids=torch.ones(1,1,dtype=torch.long),past_key_values=out.past_key_values,use_cache=True)
        self.assertEqual(receipt['routes']['sparse_path_calls'],2)
        self.assertIsNone(receipt['actual_unique_k'])
        with self.assertRaises(ValueError):
            model(input_ids=torch.ones(1,2,dtype=torch.long),past_key_values=out.past_key_values)
        model.wrong_route=True
        with self.assertRaises(RuntimeError):
            with baseline_request(model):
                model(input_ids=torch.ones(1,1,dtype=torch.long),past_key_values=out.past_key_values)
    def test_ffd_external_empty_and_mismatched_caches_rejected(self):
        for length in (1,127,128):
            model=self.ffd()
            external=FakeCache(BS=128,k_bits=2,use_fp8_residual=True,max_seq_len=131200)
            with self.assertRaisesRegex(ValueError, 'external caches'):
                model(input_ids=torch.ones(1,length,dtype=torch.long),past_key_values=external)
            self.assertEqual(model.stats,{})
        # Ownership is per-loader, not merely the same class/configuration.
        first=self.ffd();cache=first(input_ids=torch.ones(1,128)).past_key_values
        with self.assertRaisesRegex(ValueError, 'external caches'):
            self.ffd()(input_ids=torch.ones(1,1),past_key_values=cache)

    def test_ffd_owned_cache_configuration_and_empty_reset(self):
        changes={'BS':256,'k_bits':4,'use_fp8_residual':False,'max_seq_len':256}
        for field,bad in changes.items():
            for layer in (False,True):
                model=self.ffd();cache=model(input_ids=torch.ones(1,128)).past_key_values
                component=cache
                if layer:
                    component=cache.layers[1]
                setattr(component,field,bad)
                before=dict(model.stats)
                with self.assertRaisesRegex(ValueError,'configuration'):
                    model(input_ids=torch.ones(1,1),past_key_values=cache)
                self.assertEqual(model.stats,before)
        model=self.ffd();cache=model(input_ids=torch.ones(1,128)).past_key_values
        cache.length=0
        with self.assertRaisesRegex(ValueError,'empty cache'):
            model(input_ids=torch.ones(1,1),past_key_values=cache)
        with self.assertRaisesRegex(ValueError,'lengths'):
            model(input_ids=torch.ones(1,128),past_key_values=cache)
        fresh=model(input_ids=torch.ones(1,128)).past_key_values
        model(input_ids=torch.ones(1,1),past_key_values=fresh)

    def test_ffd_layer_replacement_lengths_and_failed_cache(self):
        for mutation in ('replace','append','length'):
            model=self.ffd();cache=model(input_ids=torch.ones(1,128)).past_key_values
            if mutation=='replace':cache.layers[1]=copy.copy(cache.layers[1])
            elif mutation=='append':cache.layers.append(copy.copy(cache.layers[0]))
            else:cache.layers[1].length-=1
            with self.assertRaisesRegex(ValueError,'layer'):
                model(input_ids=torch.ones(1,1),past_key_values=cache)
        model=self.ffd();cache=model(input_ids=torch.ones(1,128)).past_key_values
        model.wrong_route=True
        with self.assertRaises(RuntimeError):model(input_ids=torch.ones(1,1),past_key_values=cache)
        model.wrong_route=False
        with self.assertRaisesRegex(ValueError,'failed call'):
            model(input_ids=torch.ones(1,1),past_key_values=cache)
        model=self.ffd();cache=model(input_ids=torch.ones(1,128)).past_key_values
        model.fail_after_update=True
        with self.assertRaisesRegex(RuntimeError,'injected model failure'):
            model(input_ids=torch.ones(1,1),past_key_values=cache)
        model.fail_after_update=False
        with self.assertRaisesRegex(ValueError,'failed call'):
            model(input_ids=torch.ones(1,1),past_key_values=cache)

    def test_clock_and_ruler_limits(self):
        for times in ([0,0],[2,1],[0,float('inf')],[float('nan'),2],[0,1e308]):
            model=self.ffd()
            with patch('sparse_attn.baseline_runtime.time.perf_counter',side_effect=times):
                with self.assertRaises(ValueError):
                    with baseline_request(model):
                        model._baseline_stats['sparse_path_calls']=2
        for length in (65536,131072):
            with self.assertRaises(ValueError):baseline_preflight(types.SimpleNamespace(
                method='ffd_vendor',baseline_execute=True,model='Llama',context_length=length))

    def test_ffd_invalid_inputs_and_no_sparse_receipt(self):
        model=self.ffd()
        for ids in (torch.ones(2,128),torch.ones(1,127),torch.ones(1,0)):
            with self.assertRaises(ValueError): model(input_ids=ids)
        with self.assertRaises(ValueError): model(input_ids=torch.ones(1,128),use_cache=False)
        with self.assertRaises(ValueError): model(input_ids=torch.ones(1,128),attention_mask=torch.zeros(1,128))
        with self.assertRaises(RuntimeError):
            with baseline_request(model): model(input_ids=torch.ones(1,128))
        model.config.attn_settings['pattern_layers']=[]
        with self.assertRaises(ValueError): model(input_ids=torch.ones(1,128))
    def tiny(self):
        c=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                      num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        c._attn_implementation='eager'
        return Qwen3ForCausalLM(c).eval()
    def test_sparq_full_budget_dense_parity_and_routes(self):
        native=self.tiny(); model=copy.deepcopy(native)
        self.assertEqual(enable_sparq(model,k=1024,context_length=32768),2)
        ids=torch.tensor([[1,2,3,4,5]])
        with torch.no_grad(),baseline_request(model) as receipt:
            a=native(input_ids=ids,use_cache=True); b=model(input_ids=ids,use_cache=True)
            torch.testing.assert_close(a.logits,b.logits)
            for token in (6,7):
                ids=torch.tensor([[token]])
                a=native(input_ids=ids,past_key_values=a.past_key_values,use_cache=True)
                b=model(input_ids=ids,past_key_values=b.past_key_values,use_cache=True)
                torch.testing.assert_close(a.logits,b.logits,rtol=1e-5,atol=1e-6)
        self.assertEqual(receipt['routes'],{'native_prefill_calls':2,'sparq_decode_calls':4})
        with self.assertRaises(ValueError): enable_sparq(model)
    def test_sparq_rejects_unsupported_decode_before_mutation(self):
        from sparse_attn.baseline_runtime import BatchCtx, batch_decode_scope
        model=self.tiny();enable_sparq(model,k=2,context_length=16)
        with torch.no_grad():
            out=model(input_ids=torch.tensor([[1,2,3,4]]),use_cache=True)
        attn=model.model.layers[0].self_attn
        cache=out.past_key_values
        before=cache.get_seq_length(0)
        stats=dict(getattr(model,'_baseline_stats',{}))
        base=dict(hidden_states=torch.zeros(1,1,32),position_embeddings=(torch.ones(1,1,8),torch.zeros(1,1,8)),
                  past_key_values=cache,cache_position=torch.tensor([4]))
        for changes in (dict(attention_mask=torch.tensor([[[[True,False,True,True,True]]]])),
                        dict(attention_mask=torch.tensor([[[[0.,-1.,0.,0.,0.]]]])),
                        dict(attention_mask=torch.ones(1,5)), dict(cache_position=None),
                        dict(use_cache=False)):
            with self.subTest(changes=tuple(changes)):
                with self.assertRaises(ValueError):attn(**dict(base,**changes))
                self.assertEqual(cache.get_seq_length(0),before)
                self.assertEqual(getattr(model,'_baseline_stats',{}),stats)
        with batch_decode_scope(BatchCtx(1,[4],[0],[True],0)):
            with self.assertRaisesRegex(ValueError,'batched decode contexts'):
                attn(**base)
        self.assertEqual(cache.get_seq_length(0),before)
        self.assertEqual(getattr(model,'_baseline_stats',{}),stats)
        # main's cache=None contract is single-token mean rebuilding, not an
        # invalid install/decode. It must not append to the existing cache.
        with torch.no_grad():
            value, _ = attn(**dict(base, past_key_values=None,
                                  hidden_states=torch.arange(32).reshape(1,1,32).float()/32))
        self.assertEqual(value.shape, (1,1,32))
        self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(cache.get_seq_length(0),before)
        self.assertEqual(attn.sparq_vmean.n.item(),1)

    def test_sparq_invalid_installation_is_atomic(self):
        for kwargs in (dict(k=0),dict(k=True),dict(context_length=0),
                       dict(context_length=1),dict(context_length=1.5)):
            model=self.tiny()
            before=[m.self_attn.forward for m in model.model.layers]
            with self.assertRaises(ValueError):enable_sparq(model,**kwargs)
            self.assertEqual(before,[m.self_attn.forward for m in model.model.layers])
            self.assertFalse(hasattr(model,'_baseline_stats'))

    def test_quest_real_hf_full_budget_and_position_rejection(self):
        sys.path.insert(0,str(ROOT/'exp/vendor_baselines/quest_vendor'))
        from qwen3_quest_patch import enable_quest
        c=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
                      num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        c._attn_implementation='eager'
        native=Qwen3ForCausalLM(c).eval();model=copy.deepcopy(native);enable_quest(model,32,4)
        with torch.no_grad():
            a=native(input_ids=torch.tensor([[1,2,3,4]]));b=model(input_ids=torch.tensor([[1,2,3,4]]))
            a=native(input_ids=torch.tensor([[5]]),past_key_values=a.past_key_values)
            b=model(input_ids=torch.tensor([[5]]),past_key_values=b.past_key_values)
            torch.testing.assert_close(a.logits,b.logits,atol=1e-6,rtol=1e-5)
            attn=model.model.layers[2].self_attn
            with self.assertRaises(ValueError):
                attn(hidden_states=torch.zeros(1,1,32),position_embeddings=(None,None),past_key_values=b.past_key_values)

    def test_sparq_no_sparse_is_rejected(self):
        model=self.tiny();enable_sparq(model)
        with self.assertRaises(RuntimeError):
            with baseline_request(model):
                model(input_ids=torch.tensor([[1,2,3,4]]))

    def test_sparq_sparse_nontrivial_and_error_inputs(self):
        model=self.tiny();enable_sparq(model,k=2,context_length=16)
        with torch.no_grad():
            b=model(input_ids=torch.tensor([[1,2,3,4,5]]),use_cache=True)
            b=model(input_ids=torch.tensor([[6]]),past_key_values=b.past_key_values,use_cache=True)
            self.assertTrue(torch.isfinite(b.logits).all())
        self.assertEqual(model._baseline_stats['sparq_decode_calls'],2)
        for k,ctx in ((0,16),(2,0),(True,16)):
            with self.assertRaises(ValueError):enable_sparq(self.tiny(),k=k,context_length=ctx)
        with self.assertRaises(ValueError):
            model(input_ids=torch.ones(2,5,dtype=torch.long))
    def test_shared_mask_and_position_guards(self):
        h=torch.zeros(1,1,8); pos=torch.tensor([5]);cache=object()
        require_unmasked_decode(h,torch.zeros(1,1,1,6),cache,pos)
        require_unmasked_decode(h,torch.ones(1,1,1,6,dtype=torch.bool),cache,pos)
        for mask in (torch.ones(1,1,1,6),torch.zeros(1,1,1,6,dtype=torch.bool),torch.zeros(1,6)):
            with self.assertRaises(ValueError):require_unmasked_decode(h,mask,cache,pos)
        with self.assertRaises(ValueError):require_unmasked_decode(h,None,cache,None)
    def test_minference_pattern_no_default_fallback(self):
        pattern=[{'0':['vertical_and_slash',2,3,1]} for _ in range(2)]
        validate_minference_pattern(pattern,2,1,-1)
        for p,start in ((pattern,2),(pattern,-2),(pattern[:1],-1),([{},{}],-1),([{'0':['oops',2,3,1]}]*2,-1)):
            with self.assertRaises(ValueError):validate_minference_pattern(p,2,1,start)
    def test_entrypoint_wiring(self):
        for path in ('benchmark/LongBench/pred.py','benchmark/RULER/pred_ruler.py'):
            text=(ROOT/path).read_text()
            self.assertIn('load_ffd(',text)
            self.assertIn('baseline_request(model) as baseline_receipt',text)
            self.assertIn('baseline_preflight(args)',text)
            self.assertIn('baseline_receipt',text)
        from sparse_attn.arguments import add_sparse_attn_args
        parser=argparse.ArgumentParser();add_sparse_attn_args(parser)
        args=parser.parse_args(['--method','ffd_vendor'])
        self.assertFalse(args.baseline_execute)

if __name__=='__main__': unittest.main()
