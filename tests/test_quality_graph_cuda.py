"""Native CUDA capture/replay acceptance using small real HF model classes."""
import copy
from pathlib import Path
import runpy
from types import MethodType

import pytest
import torch
from transformers import DynamicCache

from sparse_attn.ops.quality_graph import from_dynamic, static_decode
from sparse_attn.baseline_runtime import BatchCtx
from sparse_attn.ops.tli_static_cache import TLIStaticCache
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='native CUDA required')


def model_for(kind,gate=False):
    from transformers import Qwen3Config,Qwen3ForCausalLM
    helpers=runpy.run_path(str(Path(__file__).parents[1]/'exp/bench/test_glm47_d256_gates.py'))
    args=helpers['make_prod_args']()
    args.tli_dyn_dgate=gate;args.tli_dyn_dgate_rows=2;args.tli_dyn_dgate_eps=1.0
    if kind=='qwen':
        config=Qwen3Config(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                          num_attention_heads=4,num_key_value_heads=2,head_dim=128,
                          vocab_size=128,max_position_embeddings=40000)
        model=Qwen3ForCausalLM(config)
        from sparse_attn.patches.qwen3_attn_patch import qwen3_attn_forward as forward
        dim=128
    else:
        from transformers import Glm4MoeLiteConfig,Glm4MoeLiteForCausalLM
        config=Glm4MoeLiteConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=20,num_key_value_heads=20,q_lora_rank=16,kv_lora_rank=16,
            qk_nope_head_dim=192,qk_rope_head_dim=64,v_head_dim=256,
            first_k_dense_replace=2,n_routed_experts=2,num_experts_per_tok=1,
            vocab_size=128,max_position_embeddings=202752,rope_interleave=True)
        model=Glm4MoeLiteForCausalLM(config)
        from sparse_attn.patches.glm4_moe_lite_attn_patch import glm4_moe_lite_attn_forward as forward
        dim=256
    config._attn_implementation='sdpa'
    for i,layer in enumerate(model.model.layers):
        attn=layer.self_attn
        attn.indexer=helpers['make_indexer'](dim,args)
        attn.indexer.layer_idx=i
        attn.forward=MethodType(forward,attn)
    return model.cuda().to(torch.bfloat16).eval()


@pytest.mark.parametrize('kind,gate',[('qwen',False),('qwen',True),('glm',False),('glm',True)])
def test_real_hf_graph_and_static_eager_generate_same_tokens(monkeypatch,kind,gate):
    monkeypatch.setenv('TLI_QUALITY_TIMING', '1')
    torch.manual_seed(123)
    model=model_for(kind,gate)
    ids=torch.randint(0,128,(1,16384),device='cuda')
    results=[]
    with torch.inference_mode():
        for policy in ('0','1'):
            monkeypatch.setenv('TLI_CUDA_GRAPH',policy)
            with eager_decoding_request_guard():
                output=model(input_ids=ids,use_cache=True,logits_to_keep=1)
                past=output.past_key_values
                first=output.logits[:,-1].argmax(-1).item()
                del output
                if policy=='1':
                    result=from_dynamic(model,past,first,5,set())
                    assert model._quality_graph_execution['captures']==1
                    assert model._quality_graph_execution['replay_steps']==4
                    layers = model._quality_graph_execution['capture_layers']
                    assert len(layers) == 2
                    for receipt, layer in zip(layers, model.model.layers):
                        assert receipt['normalization_route'] == layer.self_attn.indexer._graph_normalization_route
                        assert receipt['skip_far'] == layer.self_attn.indexer.skip_far
                        if gate:
                            assert receipt['gate_frozen'] is True
                    timing = model._quality_graph_execution['timing']
                    assert timing['decode_steps'] == timing['generated_decode_tokens'] == 4
                    assert len(timing['replay_gpu_ms']) == 4
                    assert all(x > 0 for x in timing['replay_gpu_ms'])
                    assert timing['cache_conversion_wall_ms'] > 0
                    assert timing['decode_wall_ms'] >= sum(timing['step_wall_ms']) > 0
                    assert not past.layers
                else:
                    cache=TLIStaticCache(model.config,16389,1)
                    for target,source in zip(cache.layers,past.layers):
                        target.lazy_initialization(source.keys)
                        target.keys[:,:,:16384].copy_(source.keys)
                        target.values[:,:,:16384].copy_(source.values)
                        target.cumulative_length=16384
                    past.layers.clear()
                    ctx=BatchCtx(1,[16384],[0],[True],0)
                    result=static_decode(model,cache,[[first]],[16384],set(),5,ctx)[0]
                    del cache
                results.append(result)
            if gate:
                assert all(layer.self_attn.indexer.skip_far for layer in model.model.layers)
    assert results[0]==results[1]
    del model,ids
    torch.cuda.empty_cache()


@pytest.mark.parametrize('code',[1,2,3])
def test_cuda_deferred_failure_prevents_publication_for_graph_and_eager(monkeypatch,tmp_path,code):
    from sparse_attn.ops.eager_decoding import _err_flag
    for policy in ('0','1'):
        monkeypatch.setenv('TLI_CUDA_GRAPH',policy)
        with pytest.raises(RuntimeError):
            with eager_decoding_request_guard():
                flag=_err_flag(torch.device('cuda:0'))
                if policy=='1':
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):flag.fill_(code)
                    graph.replay()
                else:flag.fill_(code)
            (tmp_path/'complete').write_text('bad')
        assert not (tmp_path/'complete').exists()


@pytest.mark.parametrize('failure_stage', ['allocation', 'receipt'])
def test_capture_oom_auto_recovers_on_static_cache_and_strict_aborts(monkeypatch, failure_stage):
    torch.manual_seed(321)
    model=model_for('qwen')
    ids=torch.randint(0,128,(1,16384),device='cuda')
    def fail_capture(*args):
        raise torch.cuda.OutOfMemoryError('injected capture allocation failure')
    results=[]
    with torch.inference_mode():
        for policy in ('auto','1'):
            monkeypatch.setenv('TLI_CUDA_GRAPH',policy)
            with eager_decoding_request_guard():
                out=model(input_ids=ids,use_cache=True,logits_to_keep=1)
                past=out.past_key_values;first=out.logits[:,-1].argmax(-1).item()
                del out
                with monkeypatch.context() as patch:
                    if failure_stage == 'allocation':
                        patch.setattr(torch.cuda,'CUDAGraph',fail_capture)
                    else:
                        patch.setattr('sparse_attn.ops.quality_graph._capture_routes', fail_capture)
                    if policy=='1':
                        with pytest.raises(torch.cuda.OutOfMemoryError):
                            from_dynamic(model,past,first,5,set())
                    else:
                        results=from_dynamic(model,past,first,5,set())
                        assert 'capture_failed:OutOfMemoryError' in model._quality_graph_execution['fallback_reason']
                        assert model._quality_graph_execution['actual']=='staged_static'
                        assert model._quality_graph_execution['captures'] == 0
                        assert 'capture_layers' not in model._quality_graph_execution
        monkeypatch.setenv('TLI_CUDA_GRAPH','1')
        with eager_decoding_request_guard():
            out=model(input_ids=ids,use_cache=True,logits_to_keep=1)
            past=out.past_key_values;first=out.logits[:,-1].argmax(-1).item();del out
            assert from_dynamic(model,past,first,5,set())==results
            assert model._quality_graph_execution['replay_steps']==4
    del model,ids
    torch.cuda.empty_cache()
