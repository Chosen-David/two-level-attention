import copy
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from sparse_attn.asynctls_component import select_attention,enable_component,frozen_spec,checkpoint_files
from sparse_attn.baseline_runtime import baseline_request

class ComponentTests(unittest.TestCase):
    def test_group_softmax_and_partial_block(self):
        torch.manual_seed(42)
        q=torch.randn(1,4,1,8);k=torch.randn(1,2,67,8);v=torch.randn_like(k)
        c=torch.tensor([[0,2,4],[1,3,5]])
        out,ids=select_attention(q,k,v,c,blocks=2,tokens=67)
        kk=k.repeat_interleave(2,1);vv=v.repeat_interleave(2,1)
        oracle=(q@kk.transpose(-1,-2)/8**.5).softmax(-1)@vv
        torch.testing.assert_close(out,oracle,atol=2e-6,rtol=2e-6)
        self.assertEqual(ids[0].tolist(),list(range(67)))
        out,ids=select_attention(q,k,v,c,blocks=1,tokens=7)
        for h in range(2):
            self.assertEqual(len(ids[h]),min(7,len(ids[h])))
            qq=q[0,h*2:h*2+2,0];keys=k[0,h,ids[h]]
            expected=(qq@keys.T/8**.5).softmax(-1)@v[0,h,ids[h]]
            torch.testing.assert_close(out[0,h*2:h*2+2,0],expected)
    def test_independent_scalar_selection_oracle(self):
        torch.manual_seed(38)
        q=torch.randn(1,4,1,8);k=torch.randn(1,2,130,8);v=torch.randn_like(k)
        c=torch.tensor([[0,2,4],[1,3,5]])
        _,actual=select_attention(q,k,v,c,blocks=2,tokens=7)
        for h in range(2):
            qs=q[0,2*h:2*h+2,0].tolist();ks=k[0,h].tolist()
            blocks=[list(range(a,min(a+64,130))) for a in range(0,130,64)]
            upper=[]
            for ids in blocks:
                upper.append(sum(max(qq[d]*max(ks[i][d] for i in ids),qq[d]*min(ks[i][d] for i in ids)) for qq in qs for d in range(8)))
            chosen=sorted(range(3),key=lambda b:upper[b],reverse=True)[:2]
            candidates=sorted(i for b in chosen for i in blocks[b])
            weights=[0.]*len(candidates)
            for qq in qs:
                logits=[]
                for i in candidates:
                    scale=max(abs(ks[i][d]) for d in c[h].tolist())/7
                    logits.append(sum(qq[d]*round(ks[i][d]/scale)*scale for d in c[h].tolist())/math.sqrt(8))
                exps=[math.exp(x-max(logits)) for x in logits];den=sum(exps)
                weights=[a+b/den/2 for a,b in zip(weights,exps)]
            expected=sorted(candidates[j] for j in sorted(range(len(candidates)),key=lambda j:weights[j],reverse=True)[:7])
            self.assertEqual(actual[h].tolist(),expected)
    def test_bad_inputs(self):
        q=torch.ones(1,2,1,8);k=torch.ones(1,1,3,8)
        for c in (torch.tensor([[0,0]]),torch.tensor([[-1]]),torch.tensor([[8]])):
            with self.assertRaises(ValueError):select_attention(q,k,k,c)
        with self.assertRaises(ValueError):select_attention(q,k,k,torch.tensor([[0]]),tokens=0)
        with self.assertRaises(ValueError):frozen_spec(types.SimpleNamespace())
    def tiny(self):
        c=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        c._attn_implementation='eager'
        return Qwen3ForCausalLM(c).eval()
    def test_real_qwen_prefill_decode_reset_and_rejection(self):
        torch.manual_seed(7);native=self.tiny();model=copy.deepcopy(native)
        shape={k:getattr(model.config,k) for k in ('model_type','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim')}
        data=dict(schema='asynctls-component-calibration-v1',provenance=dict(producer='synthetic fixture; NOT scientific calibration',held_out_input_sha256='0'*64,score_rule='mean_qheads(max_abs_q * max_abs_k); post-RoPE; held-out'),model_shape=shape,channels={str(i):[[0,1,2],[3,4,5]] for i in range(2)})
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            model=Qwen3ForCausalLM.from_pretrained(tmp,local_files_only=True,attn_implementation='eager').eval()
            data['checkpoint_files']=checkpoint_files(tmp)
            data['effective_rope']={key:getattr(model.config,key,None) for key in ('rope_theta','rope_scaling')}
            path=Path(tmp)/'cal.json';path.write_text(json.dumps(data));args=types.SimpleNamespace(asynctls_calibration=str(path),asynctls_blocks=128,asynctls_tokens=1024)
            wrong=copy.deepcopy(data);wrong['checkpoint_files']['config.json']='f'*64
            path.write_text(json.dumps(wrong))
            with self.assertRaises(ValueError):enable_component(model,copy.copy(args))
            path.write_text(json.dumps(data))
            wrong=copy.deepcopy(data);wrong['effective_rope']['rope_theta']=1
            path.write_text(json.dumps(wrong))
            with self.assertRaises(ValueError):enable_component(model,copy.copy(args))
            path.write_text(json.dumps(data))
            self.assertEqual(enable_component(model,args),2)
            with self.assertRaises(ValueError):model(input_ids=torch.tensor([[1,2]]),attention_mask=torch.tensor([[0,1]]),use_cache=True)
            with self.assertRaises(ValueError):model(input_ids=torch.tensor([[1,2],[1,2]]),use_cache=True)
            path.write_text('{}') # Runtime consumes frozen bytes, never rereads mutable file.
            for _ in range(2):
                with torch.no_grad(),baseline_request(model) as receipt:
                    a=native(input_ids=torch.tensor([[1,2,3]]),use_cache=True)
                    b=model(input_ids=torch.tensor([[1,2,3]]),use_cache=True)
                    torch.testing.assert_close(a.logits,b.logits)
                    for t in (4,5):
                        a=native(input_ids=torch.tensor([[t]]),past_key_values=a.past_key_values,use_cache=True)
                        b=model(input_ids=torch.tensor([[t]]),past_key_values=b.past_key_values,use_cache=True)
                        torch.testing.assert_close(a.logits,b.logits,atol=1e-6,rtol=1e-5)
                self.assertEqual(receipt['routes']['component_decode_calls'],4)
                cache=b.past_key_values;before=cache.get_seq_length()
                with self.assertRaises(ValueError):model(input_ids=torch.tensor([[6]]),past_key_values=cache,cache_position=torch.tensor([0]),use_cache=True)
                self.assertEqual(cache.get_seq_length(),before)
                # Even a rejected call conservatively invalidates this cache.
                with self.assertRaises(ValueError):model(input_ids=torch.tensor([[6]]),past_key_values=cache,use_cache=True)
                with self.assertRaises(ValueError):model(input_ids=torch.tensor([[6]]),past_key_values=cache,attention_mask=torch.zeros(1,before+1),use_cache=True)
                self.assertEqual(cache.get_seq_length(),before)

if __name__=='__main__':unittest.main()
