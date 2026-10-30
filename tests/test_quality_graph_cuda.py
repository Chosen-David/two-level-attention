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


def model_for(kind,gate=False,max_position_embeddings=None):
    from transformers import Qwen3Config,Qwen3ForCausalLM
    helpers=runpy.run_path(str(Path(__file__).parents[1]/'exp/bench/test_glm47_d256_gates.py'))
    args=helpers['make_prod_args']()
    args.tli_dyn_dgate=gate;args.tli_dyn_dgate_rows=2;args.tli_dyn_dgate_eps=1.0
    if kind=='qwen':
        config=Qwen3Config(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                          num_attention_heads=4,num_key_value_heads=2,head_dim=128,
                          vocab_size=128,max_position_embeddings=max_position_embeddings or 40000)
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
@pytest.mark.parametrize('input_mode',['0','1',None])
def test_real_hf_graph_and_static_eager_generate_same_tokens(monkeypatch,kind,gate,input_mode):
    if input_mode is None:
        monkeypatch.delenv('TLI_DEVICE_DECODE_INPUTS', raising=False)
        monkeypatch.delenv('TLI_REUSE_WARMUP_STREAM', raising=False)
    else:
        monkeypatch.setenv('TLI_DEVICE_DECODE_INPUTS',input_mode)
    for key in ('TLI_LIVE_NORMALIZATION', 'TLI_SPARSE_SELECTION', 'TLI_SPARSE_STATS'):
        monkeypatch.delenv(key, raising=False)
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
                    assert model._quality_graph_execution['input_update_route']==(
                        'host_reference' if input_mode=='0' else 'device_row_fused')
                    layers = model._quality_graph_execution['capture_layers']
                    assert len(layers) == 2
                    for receipt, layer in zip(layers, model.model.layers):
                        assert receipt['normalization_width_route'] == 'live_length'
                        assert receipt['normalization_stats_route'] == 'live_length_owner'
                        assert receipt['ranking_route'] == 'candidate_domain_native_1024'
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
    from sparse_attn.ops.cuda_graph_state import graph_state
    assert not graph_state().quality_defaults
    del model,ids
    torch.cuda.empty_cache()


@pytest.mark.parametrize('batch', [1, 2, 4])
def test_real_hf_batch_eager_graph_share_row_incremental_states(monkeypatch, batch):
    from collections import Counter
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    from sparse_attn.ops.static_batch_prefill import consume_prefill_rows

    for key, value in dict(TLI_BATCH_INDEXER='legacy', TLI_OPERATOR_MODE='off',
                           TLI_DEVICE_DECODE_INPUTS='1', TLI_QUALITY_TIMING='1').items():
        monkeypatch.setenv(key, value)
    for key in ('TLI_LIVE_NORMALIZATION', 'TLI_SPARSE_SELECTION', 'TLI_SPARSE_STATS'):
        monkeypatch.delenv(key, raising=False)
    torch.manual_seed(991 + batch)
    model = model_for('qwen')
    lens = [16384 - 3 * b for b in range(batch)]
    ids = torch.randint(0, 128, (batch, max(lens)), device='cuda')
    original = TLIIndexer._prepare_index_base
    calls = []

    def counted(self, key, cu, **kwargs):
        calls.append((id(self), key.data_ptr()))
        return original(self, key, cu, **kwargs)

    monkeypatch.setattr(TLIIndexer, '_prepare_index_base', counted)
    outputs = []
    with torch.inference_mode(), eager_decoding_request_guard():
        for policy in ('0', '1'):
            monkeypatch.setenv('TLI_CUDA_GRAPH', policy)
            rows, first = [], []
            for b, length in enumerate(lens):
                out = model(input_ids=ids[b:b + 1, :length], use_cache=True,
                            logits_to_keep=1)
                rows.append(out.past_key_values)
                first.append(out.logits[0, -1].argmax(-1).item())
                del out
            cache = TLIStaticCache(model.config, max(lens) + 6, batch)
            consume_prefill_rows(cache, rows, lens)
            ctx = BatchCtx(batch, list(lens), [0] * batch, [True] * batch, 0)
            calls.clear()
            result = static_decode(model, cache, [[token] for token in first], lens,
                                   set(), 6, ctx)
            receipt = model._quality_graph_execution
            assert receipt['decode_index_route'] == 'request_row_incremental'
            assert receipt['input_update_route'] == 'device_row_fused'
            assert receipt['captures'] == (1 if policy == '1' else 0)
            counts = Counter(calls)
            for layer in model.model.layers:
                layer_calls = [count for (owner, _), count in counts.items()
                               if owner == id(layer.self_attn.indexer)]
                assert sorted(layer_calls) == [1] * batch
                assert len(ctx._tli_row_index_pools[id(layer.self_attn.indexer)]) == batch
            outputs.append(result)
            del cache, ctx
    assert outputs[0] == outputs[1]
    del model, ids
    torch.cuda.empty_cache()


@pytest.mark.parametrize('online_eos', [True, False])
def test_real_hf_long_context_eager_graph_keep_device_incremental_route(monkeypatch, online_eos):
    """真实模型跨 65536 后仍走同一增量选择器与 tiled 设备输入。"""
    from sparse_attn.ops.static_batch_prefill import consume_prefill_rows
    for key, value in dict(TLI_BATCH_INDEXER='legacy', TLI_OPERATOR_MODE='off',
                           TLI_DEVICE_DECODE_INPUTS='1', TLI_QUALITY_TIMING='1').items():
        monkeypatch.setenv(key, value)
    for key in ('TLI_LIVE_NORMALIZATION', 'TLI_SPARSE_SELECTION', 'TLI_SPARSE_STATS'):
        monkeypatch.delenv(key, raising=False)
    torch.manual_seed(65537 + int(online_eos))
    model = model_for('qwen', max_position_embeddings=131072)
    length = 65537
    ids = torch.randint(0, 128, (1, length), device='cuda')
    outputs = []
    with torch.inference_mode(), eager_decoding_request_guard():
        for policy in ('0', '1'):
            monkeypatch.setenv('TLI_CUDA_GRAPH', policy)
            out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            first = out.logits[0, -1].argmax(-1).item()
            rows = [out.past_key_values]
            del out
            cache = TLIStaticCache(model.config, length + 4, 1)
            consume_prefill_rows(cache, rows, [length])
            ctx = BatchCtx(1, [length], [0], [True], 0)
            result = static_decode(model, cache, [[first]], [length], {2}, 4, ctx,
                                   online_eos=online_eos)[0]
            receipt = model._quality_graph_execution
            assert receipt['input_update_route'] == 'device_row_fused'
            assert receipt['input_update_fallback'] is None
            assert receipt['decode_index_route'] == 'request_row_incremental'
            assert receipt['captures'] == (1 if policy == '1' else 0)
            assert receipt['generation_mode'] == ('online_eos' if online_eos else 'fixed_steps')
            assert receipt['token_readback'] == ('per_step' if online_eos else 'final')
            outputs.append(result)
            del cache, ctx, rows
    assert outputs[0] == outputs[1]
    del model, ids
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
@pytest.mark.parametrize('input_mode',['0','1',None])
def test_capture_oom_auto_recovers_on_static_cache_and_strict_aborts(monkeypatch, failure_stage,input_mode):
    if input_mode is None:
        monkeypatch.delenv('TLI_DEVICE_DECODE_INPUTS', raising=False)
    else:
        monkeypatch.setenv('TLI_DEVICE_DECODE_INPUTS',input_mode)
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
                        assert model._quality_graph_execution['input_update_route']==(
                            'device_row_fused' if input_mode!='0' else 'host_reference')
                        assert model._quality_graph_execution['input_update_fallback'] is None
                        assert model._quality_graph_execution['decode_index_route']=='request_row_incremental'
                        assert 'capture_layers' not in model._quality_graph_execution
        monkeypatch.setenv('TLI_CUDA_GRAPH','1')
        with eager_decoding_request_guard():
            out=model(input_ids=ids,use_cache=True,logits_to_keep=1)
            past=out.past_key_values;first=out.logits[:,-1].argmax(-1).item();del out
            assert from_dynamic(model,past,first,5,set())==results
            assert model._quality_graph_execution['replay_steps']==4
    del model,ids
    torch.cuda.empty_cache()


@pytest.mark.parametrize('row_reference', ['0','1'])
@pytest.mark.parametrize('fail_replay', [False,True])
def test_one_shot_batch_request_releases_exact_owners(monkeypatch,row_reference,fail_replay):
    import gc
    import weakref
    from sparse_attn.ops import exact_linear_batch as exact
    for key,value in dict(TLI_CUDA_GRAPH='on',TLI_BATCH_INDEXER='exact',
            TLI_OPERATOR_MODE='serial',TLI_OPERATOR_BACKEND='simt',
            TLI_OPERATOR_SELECTED_STAGES='1',TLI_SPARSE_SELECTION='1',
            TLI_GRAPH_INC='1',TLI_LIVE_NORMALIZATION='1',TLI_SPARSE_STATS='owner',
            TLI_EXACT_BATCH_REFERENCE=row_reference).items():monkeypatch.setenv(key,value)
    model=model_for('qwen',False)
    states=[]
    original=exact.ExactLinearBatch.__init__
    def observe(self,*args,**kwargs):
        original(self,*args,**kwargs);states.append(weakref.ref(self))
    monkeypatch.setattr(exact.ExactLinearBatch,'__init__',observe)
    if fail_replay:
        def failed(*args,**kwargs):raise RuntimeError('injected replay failure')
        monkeypatch.setattr(torch.cuda.CUDAGraph,'replay',failed)
    for request in range(2):
        cache=TLIStaticCache(model.config,16448,2)
        with torch.inference_mode(),eager_decoding_request_guard():
            for layer in cache.layers:
                layer.lazy_initialization(torch.zeros(2,2,1,128,device='cuda',dtype=torch.bfloat16))
                layer.cumulative_length=16384
            kv=[weakref.ref(layer.keys) for layer in cache.layers]
            ctx=BatchCtx(2,[16384,16380],[0,0],[True,True],0)
            if fail_replay:
                with pytest.raises(RuntimeError,match='injected replay failure'):
                    static_decode(model,cache,[[1],[2]],[16384,16380],set(),3,ctx)
            else:
                result=static_decode(model,cache,[[1],[2]],[16384,16380],set(),3,ctx)
                assert all(len(row)==3 for row in result)
                assert model._quality_graph_execution['captures']==1
            assert all(m.indexer._exact_linear_batch is None for m in model.modules() if hasattr(m,'indexer'))
        del layer,cache
        gc.collect();torch.cuda.synchronize()
        assert states and all(ref() is None for ref in states)
        assert all(ref() is None for ref in kv)
