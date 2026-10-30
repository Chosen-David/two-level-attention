from types import MethodType, SimpleNamespace
from contextlib import nullcontext
import pytest
import torch
from transformers import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM
from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.ops.frozen_region_tli import tli_args
from sparse_attn.patches.glm4_moe_lite_attn_patch import glm4_moe_lite_attn_forward
from sparse_attn.ops.frozen_region_gate import GateConfig, profile_regions
from sparse_attn.ops.frozen_region_cache import RegionGeometry
from benchmark.LongBench.prefill_frozen_skip import generate, generate_native

@pytest.mark.parametrize('mode',['full','far','near','dual'])
@torch.inference_mode()
def test_real_glm_class_region_decode(monkeypatch,mode):
    torch.set_num_threads(2)
    # Generic GLM eager adapter scopes CUDA devices; CPU reference has no stream.
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setenv('TLI_FINE_GATHER','0')
    torch.manual_seed(42)
    cfg=Glm4MoeLiteConfig(hidden_size=32,intermediate_size=48,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,q_lora_rank=16,kv_lora_rank=16,
        qk_nope_head_dim=192,qk_rope_head_dim=64,v_head_dim=256,
        first_k_dense_replace=1,n_routed_experts=2,num_experts_per_tok=1,
        vocab_size=32,max_position_embeddings=10000,rope_interleave=True)
    cfg._attn_implementation='sdpa'
    model=Glm4MoeLiteForCausalLM(cfg).eval()
    attn=model.model.layers[0].self_attn
    attn.indexer=TLIIndexer(tli_args());attn.indexer.layer_idx=0
    attn.indexer.head_dim=256
    attn.forward=MethodType(glm4_moe_lite_attn_forward,attn)
    model._frozen_tli=True;model.generation_config.eos_token_id=None
    ids=torch.randint(0,32,(4352,)).tolist()
    tok=SimpleNamespace(decode=lambda ids,**kw:str(ids))
    result=generate(model,tok,ids,3,GateConfig(epsilon=1.,rows=2,mode=mode),'cpu')
    d=result['decisions'][0]
    assert d['skip_near']==(mode in ('near','dual'))
    assert d['skip_far']==(mode in ('far','dual'))
    receipt=result['tli_receipts'][0]
    assert receipt['dimensions']==list(range(208,224))+list(range(240,256))
    assert result['final_kv']['layers'][0]['logical_length']==4354
    if mode=='full':assert result['generated_ids']==generate_native(model,ids,3,'cpu')
    if mode=='near':assert not result['frozen_kv']['layers'][0]['nodes']['near_staging']['attention_visible']

def test_profile_uses_explicit_attention_scale():
    torch.manual_seed(2)
    q=torch.randn(1,2,24,8);k=torch.randn_like(q)
    geom=RegionGeometry(sink=2,window=4,near=8,page=2)
    result=profile_regions(q,k,geom,rows=1,scale=.7)
    probs=(q[:,:,-1:]@k.transpose(-1,-2)*.7).softmax(-1)
    s,f,n=geom.bounds(24)
    assert result['far']['mean']==pytest.approx(probs[...,s:f].sum(-1).mean().item())
    assert result['near']['mean']==pytest.approx(probs[...,f:n].sum(-1).mean().item())
