"""CPU state-machine contracts; these tests do not validate CUDA transport."""
import copy
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache,DynamicSlidingWindowLayer
from sparse_attn.asynctls_component import checkpoint_files
from sparse_attn.asynctls_offload import LaggedKV,enable_system
from sparse_attn.baseline_runtime import baseline_request

class OffloadTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(312);torch.set_num_threads(1)
    def fixture(self, n=63, h=2):
        return torch.randn(1,h*2,1,8),torch.randn(1,h,n,8),torch.randn(1,h,n,8),torch.tensor([[0,2,5]]*h)
    def test_scalar_lagged_reference_and_tail(self):
        q,k,v,c=self.fixture();e=LaggedKV(q,k,v,c,blocks=1,tokens=5,capacity=70)
        previous=q
        for pos in range(63,68):
            query,nk,nv,_=self.fixture(n=1)
            out,actual=e.step(query,nk,nv,pos)
            for h in range(2):
                # Scalar block oracle uses the PREVIOUS query and historical K only.
                ks=k[0,h].tolist();qs=previous[0,2*h:2*h+2,0].tolist()
                pages=[list(range(i,min(i+64,pos))) for i in range(0,pos,64)]
                scores=[sum(max(qq[d]*min(ks[i][d] for i in page),qq[d]*max(ks[i][d] for i in page)) for qq in qs for d in range(8)) for page in pages]
                ids=pages[max(range(len(pages)),key=lambda j:scores[j])]+[pos]
                full=torch.cat((k,nk),2);vals=torch.cat((v,nv),2)
                weights=[0.]*len(ids)
                for qq in query[0,2*h:2*h+2,0].tolist():
                    logits=[]
                    for i in ids:
                        row=full[0,h,i].tolist();scale=max(abs(row[d]) for d in c[h].tolist())/7
                        logits.append(sum(qq[d]*round(row[d]/scale)*scale for d in c[h].tolist())/math.sqrt(8))
                    exps=[math.exp(x-max(logits)) for x in logits]
                    weights=[a+b/sum(exps)/2 for a,b in zip(weights,exps)]
                selected=[ids[j] for j in sorted(range(len(ids)),key=lambda j:weights[j],reverse=True)[:5]]
                self.assertEqual(sorted(actual[h].tolist()),sorted(selected))
                qq=query[0,2*h:2*h+2,0];keys=full[0,h,selected]
                expected=(qq@keys.T/math.sqrt(8)).softmax(-1)@vals[0,h,selected]
                torch.testing.assert_close(out[0,2*h:2*h+2,0],expected,atol=1e-6,rtol=1e-5)
            k=torch.cat((k,nk),2);v=torch.cat((v,nv),2);previous=query
            torch.testing.assert_close(e.host_views()[0],k)
            torch.testing.assert_close(e.host_views()[1],v)
        self.assertGreater(e.stats['reused_blocks'],0);self.assertGreater(e.stats['tail_updates'],0)
        e.close();self.assertIsNone(e.pending)
    def test_lag_is_not_current_selection(self):
        q=torch.tensor([[[[1.]]]]);k=torch.tensor([[[[5.],[-5.]]]]);v=k.clone()
        e=LaggedKV(q,k,v,torch.tensor([[0]]),blocks=1,tokens=3,block_size=1,capacity=4)
        _,support=e.step(-q,torch.zeros_like(q),torch.zeros_like(q),2)
        self.assertEqual(set(support[0].tolist()),{0,2})
        self.assertEqual(e.pending[1],[[1]])
    def test_full_budget_dense_inference_mode(self):
        for heads in (1,2):
            with torch.inference_mode():
                q,k,v,c=self.fixture(n=1,h=heads);e=LaggedKV(q,k,v,c,blocks=8,tokens=100,capacity=8)
                for pos in range(1,4):
                    q,nk,nv,_=self.fixture(n=1,h=heads);out,_=e.step(q,nk,nv,pos)
                    k=torch.cat((k,nk),2);v=torch.cat((v,nv),2)
                    expected=(q@k.repeat_interleave(2,1).transpose(-1,-2)/8**.5).softmax(-1)@v.repeat_interleave(2,1)
                    torch.testing.assert_close(out,expected,atol=1e-6,rtol=1e-5)
                e.close()
    def test_failure_contracts_and_cleanup(self):
        q,k,v,c=self.fixture(n=1)
        for mutation in ('position','capacity','nonfinite','shape','external','schedule','attention'):
            e=LaggedKV(q,k,v,c,capacity=1 if mutation=='capacity' else 8)
            qq=q.clone();pos=1
            if mutation=='position':pos=2
            if mutation=='nonfinite':qq.fill_(float('nan'))
            if mutation=='shape':qq=qq[:,:3]
            if mutation=='external':e.host_k[0,0,0]=0
            method='_schedule' if mutation=='schedule' else '_attend'
            if mutation in ('schedule','attention'):
                with patch.object(e,method,side_effect=RuntimeError('injected')):
                    with self.assertRaises(RuntimeError):e.step(qq,k,v,pos)
            else:
                with self.assertRaises(ValueError):e.step(qq,k,v,pos)
            self.assertTrue(e.closed);self.assertTrue(e.failed)
            with self.assertRaises(RuntimeError):e.step(q,k,v,1)
        with self.assertRaises(ValueError):LaggedKV(q,k,v,c,backend='cuda')
        with self.assertRaises(ValueError):LaggedKV(q,k,v,torch.tensor([[0,0],[0,0]]))
    def tiny_pair(self,tmp,tokens=1024):
        conf=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        conf._attn_implementation='eager';native=Qwen3ForCausalLM(conf).eval();native.save_pretrained(tmp)
        model=Qwen3ForCausalLM.from_pretrained(tmp,local_files_only=True,attn_implementation='eager').eval()
        data=dict(schema='asynctls-component-calibration-v1',provenance=dict(producer='synthetic CPU fixture; not scientific calibration',held_out_input_sha256='0'*64,score_rule='mean_qheads(max_abs_q * max_abs_k); post-RoPE; held-out'),model_shape={k:getattr(model.config,k) for k in ('model_type','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim')},channels={str(i):[[0,2,5],[1,3,6]] for i in range(2)},checkpoint_files=checkpoint_files(tmp),effective_rope={k:getattr(model.config,k,None) for k in ('rope_theta','rope_scaling')})
        path=Path(tmp)/'cal.json';path.write_text(json.dumps(data))
        args=types.SimpleNamespace(asynctls_calibration=str(path),asynctls_blocks=2,asynctls_tokens=tokens,asynctls_capacity=140,asynctls_backend='cpu')
        enable_system(model,args)
        with self.assertRaises(ValueError):enable_system(model,args)
        return native,model
    def test_real_qwen_dense_parity_reset_receipt(self):
        with tempfile.TemporaryDirectory() as tmp,torch.inference_mode():
            native,model=self.tiny_pair(tmp)
            for _ in range(2):
                with baseline_request(model) as receipt:
                    a=native(input_ids=torch.tensor([[1,2,3]]),use_cache=True)
                    b=model(input_ids=torch.tensor([[1,2,3]]),use_cache=True)
                    for t in (4,5):
                        a=native(input_ids=torch.tensor([[t]]),past_key_values=a.past_key_values,use_cache=True)
                        b=model(input_ids=torch.tensor([[t]]),past_key_values=b.past_key_values,use_cache=True)
                        torch.testing.assert_close(a.logits,b.logits,atol=1e-6,rtol=1e-5)
                self.assertEqual(receipt['routes']['paper_decode_calls'],4)
                self.assertEqual(len(model._asynctls_engines),0)
            with self.assertRaisesRegex(RuntimeError,'never exercised'):
                with baseline_request(model):model(input_ids=torch.tensor([[1,2]]),use_cache=True)
    def test_real_qwen_sparse_and_cache_rejection(self):
        with tempfile.TemporaryDirectory() as tmp,torch.no_grad():
            _,model=self.tiny_pair(tmp,tokens=3)
            for action in ('replace','crop','mutate','padding'):
                b=model(input_ids=torch.tensor([[1,2,3,4]]),use_cache=True);cache=b.past_key_values
                if action=='replace':cache.layers[0].keys=cache.layers[0].keys.clone()
                if action=='crop':cache.crop(2)
                if action=='mutate':cache.layers[0].keys.add_(1)
                kw={'attention_mask':torch.zeros(1,5)} if action=='padding' else {}
                old=cache.get_seq_length()
                with self.assertRaises(ValueError):model(input_ids=torch.tensor([[5]]),past_key_values=cache,use_cache=True,**kw)
                self.assertEqual(cache.get_seq_length(),old);self.assertEqual(len(model._asynctls_engines),0)
            cache=DynamicCache();cache.offloading=True
            with self.assertRaises(ValueError):model(input_ids=torch.tensor([[1,2]]),past_key_values=cache,use_cache=True)
            self.assertEqual(cache.get_seq_length(),0)
            with baseline_request(model) as receipt:
                b=model(input_ids=torch.tensor([[1,2,3,4]]),use_cache=True)
                b=model(input_ids=torch.tensor([[5]]),past_key_values=b.past_key_values,use_cache=True)
            self.assertEqual(receipt['routes']['selected_tokens_sum'],12)

if __name__=='__main__':unittest.main()
