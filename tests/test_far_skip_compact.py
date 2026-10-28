"""Executable CPU/CUDA correctness tests; no simulated torch/cache objects."""
import argparse
from dataclasses import replace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache, DynamicLayer

from sparse_attn.arguments import _add_tia_args, _add_tli_args
from sparse_attn.indexer.tli_indexer import TLIIndexer
from sparse_attn.ops.far_skip import FarSkipConfig, select_far_skip, attend_far_skip
from sparse_attn.ops.far_skip_cache import FarSkipLayer, evict_far
from sparse_attn.patches.patch import register_patch
from sparse_attn.ops.far_skip_runtime import resolve_far_skip_modes, try_far_skip_decode
from sparse_attn.info import get_treatment_manifest_json, resolve_treatment_snapshot


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def args_for(mode="evict"):
    parser = _add_tli_args(_add_tia_args(argparse.ArgumentParser()))
    return parser.parse_args([
        "--tli_enable_layer_skip", "false", "--tli_enable_kmeans", "false",
        "--tli_near_len", "16", "--tli_near_pages", "2", "--tli_near_tokens", "off",
        "--tia_block_size", "4", "--tia_level2_topk", "12",
        "--tli_sink_tokens", "3", "--tli_swa_size", "4",
        "--tli_far_skip_backend", "compact", "--tli_far_skip_cache", mode])


def dense_selected_reference(q, k, v, positions):
    # Independent full-width masked attention, deliberately expensive.
    groups = q.shape[2] // k.shape[2]
    kh = k[0].repeat_interleave(groups, 1).transpose(0, 1).double()
    vh = v[0].repeat_interleave(groups, 1).transpose(0, 1).double()
    logits = torch.einsum("hd,hsd->hs", q[0, 0].double(), kh) / q.shape[-1] ** .5
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, positions.repeat_interleave(groups, 0), True)
    probabilities = logits.masked_fill(~mask, -torch.inf).softmax(-1)
    return torch.einsum("hs,hsd->hd", probabilities, vh)[None, None].to(q.dtype)


@pytest.mark.parametrize("method", ["avg", "minmax"])
@pytest.mark.parametrize("quantize", [False, True])
@pytest.mark.parametrize("groups", [1, 3])
def test_far_poison_and_compact_attention(method, quantize, groups):
    torch.manual_seed(17)
    cfg = FarSkipConfig(4, 20, 2, 15, 3, 5, method, quantize)
    q = torch.randn(1, 1, 2 * groups, 12)
    k, v = torch.randn(2, 1, 109, 2, 12)
    dims = torch.arange(8)
    sel = select_far_skip(q, k, 109, cfg, dims, dims)
    result = attend_far_skip(q, k, v, sel, cfg)
    torch.testing.assert_close(result, dense_selected_reference(q, k, v, sel.positions))
    # NaNs in excluded far K/V must not reach coarse, fine, or attention.
    poisoned_k, poisoned_v = k.clone(), v.clone()
    poisoned_k[:, cfg.sink_tokens:cfg.near_start(109)] = float("nan")
    poisoned_v[:, cfg.sink_tokens:cfg.near_start(109)] = float("nan")
    sel2 = select_far_skip(q, poisoned_k, 109, cfg, dims, dims)
    assert torch.equal(sel.positions, sel2.positions)
    torch.testing.assert_close(result, attend_far_skip(q, poisoned_k, poisoned_v, sel2, cfg))


@pytest.mark.parametrize("method", ["avg", "minmax"])
def test_random_selection_against_scalar_page_and_dense_probability_oracle(method):
    torch.manual_seed(84)
    cfg = FarSkipConfig(4, 24, 2, 12, 3, 4, method)
    length, groups = 79, 3
    q, k = torch.randn(1, 1, 6, 12), torch.randn(1, length, 2, 12)
    coarse, fine = torch.arange(8), torch.arange(4, 12)
    result = select_far_skip(q, k, length, cfg, coarse, fine)
    protected = set(range(3)) | set(range(length-4, length))
    start = cfg.near_start(length)
    eligible = set(range(max(3, start), length-4))
    pages = [p for p in range(length//4) if set(range(p*4, p*4+4)) <= eligible]
    edge = eligible - {t for p in pages for t in range(p*4, p*4+4)}
    for h in range(2):
        scores = {}
        for p in pages:
            kk = k[0, p*4:p*4+4, h, :8].double()
            per_head = []
            for g in range(groups):
                qq = q[0, 0, h*groups+g, :8].double()
                representative = (kk.mean(0) if method == "avg" else
                                  torch.where(qq >= 0, kk.amax(0), kk.amin(0)))
                per_head.append(float((qq * representative).sum()) / 12**.5)
            scores[p] = sum(per_head) / groups
        winners = sorted(pages, key=lambda p: (-scores[p], p))[:2]
        candidates = edge | {t for p in winners for t in range(p*4, p*4+4)}
        # Dense all-history logits and mask independently define the L2 pool.
        logits = q[0, 0, h*groups:(h+1)*groups, 4:].double() @ k[0, :, h, 4:].double().T / 12**.5
        for t in range(length):
            if t not in candidates | protected:
                logits[:, t] = -torch.inf
        probabilities = logits.softmax(-1).mean(0)
        selected = sorted(candidates, key=lambda t: (-float(probabilities[t]), t))[:12-len(protected)]
        assert result.positions[h].tolist() == sorted(protected | set(selected))


@pytest.mark.parametrize("sink,window", [(0, 0), (0, 5), (3, 0), (3, 5), (8, 8), (17, 2)])
@pytest.mark.parametrize("pages", [0, 1, 4])
def test_all_boundaries_budget_and_ties(sink, window, pages):
    cfg = FarSkipConfig(4, 16, pages, sink + window + 7, sink, window)
    for length in range(1, 85):
        q = torch.zeros(1, 1, 2, 4)
        k = torch.zeros(1, length, 1, 4)
        ids = select_far_skip(q, k, length, cfg, torch.arange(4), torch.arange(4)).positions[0].tolist()
        protected = set(range(min(sink, length))) | set(range(max(0, length-window), length))
        allowed = protected | set(range(max(sink, cfg.near_start(length)), length))
        assert ids == sorted(set(ids))
        assert protected <= set(ids) <= allowed
        assert len(ids) <= min(length, cfg.token_budget)
        # Independent enumeration: full near pages + protection boundary fragments.
        middle = sorted(allowed - protected)
        full_pages = [p for p in range((length + 3)//4)
                      if set(range(p*4, p*4+4)) <= set(middle)]
        boundary = [t for t in middle if t//4 not in full_pages]
        candidates = sorted(boundary + [t for p in full_pages[:pages] for t in range(p*4, p*4+4)])
        expected = sorted(protected | set(candidates[:cfg.token_budget-len(protected)]))
        assert ids == expected


@pytest.mark.parametrize("sink,window", [(0, 0), (3, 5), (16, 0), (0, 9)])
def test_cache_rollover_real_storage_and_logical_positions(sink, window):
    torch.manual_seed(11)
    cfg = FarSkipConfig(4, 16, 3, sink + window + 9, sink, window)
    cache = DynamicCache()
    full_k, full_v = torch.randn(2, 1, 2, 3, 8)
    cache.update(full_k, full_v, 0)
    layer = evict_far(cache, 0, cfg)
    addresses = (layer.keys.data_ptr(), layer.values.data_ptr())
    for length in range(4, 138):
        nk, nv = torch.randn(2, 1, 2, 1, 8)
        full_k, full_v = torch.cat((full_k, nk), -2), torch.cat((full_v, nv), -2)
        k, v = cache.update(nk, nv, 0)
        tail = cfg.tail_start(length)
        pos = list(range(min(sink, length))) + list(range(tail, length))
        assert cache.get_seq_length() == length
        assert cache.get_mask_sizes(torch.tensor([length]), 0) == (length + 1, 0)
        capacity = 16 + window + 3
        physical = [t if t < sink else sink + (t-sink) % capacity for t in pos]
        torch.testing.assert_close(k[..., physical, :], full_k[..., pos, :])
        torch.testing.assert_close(v[..., physical, :], full_v[..., pos, :])
        assert (k.data_ptr(), v.data_ptr()) == addresses
        for tensor in (k, v):
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        assert k.shape[-2] <= sink + window + 16 + 3
        q = torch.randn(1, 1, 4, 8)
        dims = torch.arange(8)
        retained = select_far_skip(q, full_k.transpose(1, 2), length, cfg, dims, dims)
        packed = select_far_skip(q, k.transpose(1, 2), length, cfg, dims, dims, layout=layer.layout)
        assert torch.equal(retained.positions, packed.positions)
        if packed.positions.shape[-1]:
            actual = attend_far_skip(q, k.transpose(1, 2), v.transpose(1, 2), packed, cfg, layout=layer.layout)
            expected = dense_selected_reference(q, full_k.transpose(1, 2), full_v.transpose(1, 2), retained.positions)
            torch.testing.assert_close(actual, expected)
    assert layer.keys.untyped_storage().nbytes() < full_k.untyped_storage().nbytes()
    with pytest.raises(ValueError, match="geometry"):
        evict_far(cache, 0, replace(cfg, near_length=20))
    with pytest.raises(ValueError, match="rewind"):
        cache.crop(10)
    with pytest.raises(ValueError, match="absolute positions"):
        cache.to_legacy_cache()
    cache.reset()
    assert cache.get_seq_length() == 0
    cache.update(full_k, full_v, 0)
    assert cache.get_seq_length() == full_k.shape[-2]
    assert cache.layers[0].config is None


def make_model(mode, dynamic=True):
    torch.manual_seed(32)
    config = Qwen3Config(vocab_size=32, hidden_size=256, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=128, max_position_embeddings=1024,
                        attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval()
    args = args_for(mode)
    args.tli_dyn_dgate = dynamic
    args.tli_dyn_dgate_eps = 1.0  # real calibration will skip all finite far masses
    args.tli_dyn_dgate_rows = 2
    args.method = "tli"
    snapshot = resolve_treatment_snapshot(args)
    assert register_patch(model, args, snapshot=snapshot) == 2
    for decoder in model.model.layers:
        attn = decoder.self_attn
        if not dynamic:
            attn.indexer._skip_ids = {0, 1}
    return model


@pytest.mark.parametrize("dynamic", [True, False])
@torch.inference_mode()
def test_real_qwen_prefill_decode_and_cache_reuse(dynamic):
    retained, evicted = make_model("retain", dynamic), make_model("evict", dynamic)
    for prompt_length in (39, 35 if dynamic else 7):  # short prompts have no dynamic far gate
        caches = [DynamicCache(), DynamicCache()]
        ids = torch.arange(prompt_length)[None] % 32
        for model, cache in zip((retained, evicted), caches):
            model(input_ids=ids, past_key_values=cache, use_cache=True)
        for step in range(21):
            token = torch.tensor([[step % 32]])
            outputs = [model(input_ids=token, past_key_values=cache, use_cache=True).logits
                       for model, cache in zip((retained, evicted), caches)]
            torch.testing.assert_close(*outputs, atol=2e-6, rtol=2e-5)
            length = prompt_length + step + 1
            assert caches[1].get_seq_length() == length
            for layer in caches[1].layers:
                assert isinstance(layer, FarSkipLayer)
                assert layer.keys.shape[-2] <= 3 + 16 + 4 + 3


@torch.inference_mode()
def test_unsupported_mask_fails_before_cache_mutation():
    model = make_model("evict")
    cache = DynamicCache()
    model(input_ids=torch.ones(1, 39, dtype=torch.long), past_key_values=cache)
    original = cache.layers[0].keys
    with pytest.raises(ValueError, match="attention_mask=None"):
        model(input_ids=torch.ones(1, 1, dtype=torch.long), past_key_values=cache,
              attention_mask=torch.cat((torch.zeros(1, 1), torch.ones(1, 39)), -1))
    assert cache.get_seq_length() == 39
    assert cache.layers[0].keys is original


def test_config_rejects_unsupported_geometry():
    args = args_for()
    args.tli_near_tokens = 4
    with pytest.raises(ValueError, match="near_tokens=off"):
        TLIIndexer(args)
    args = args_for()
    args.tli_far_skip_backend = "legacy"
    with pytest.raises(ValueError, match="requires compact"):
        TLIIndexer(args)


def test_auto_resolution_and_unresolved_snapshot_guard():
    parser = _add_tli_args(_add_tia_args(argparse.ArgumentParser()))
    args = parser.parse_args([])
    args.method = "tli"
    assert (args.tli_far_skip_backend, args.tli_far_skip_cache) == ("auto", "auto")
    with pytest.raises(ValueError, match="before treatment snapshot"):
        resolve_treatment_snapshot(args)
    qwen = Qwen3Config(hidden_size=256, num_attention_heads=2, head_dim=128,
                       num_hidden_layers=1, num_key_value_heads=1)
    assert resolve_far_skip_modes(args, config=qwen) == ("legacy", "retain")
    manifest = get_treatment_manifest_json(args)
    assert 'tli_far_skip_backend' not in manifest
    assert 'tli_far_skip_cache' not in manifest

    compact_args = parser.parse_args([
        "--tli_near_len", "4096", "--tli_near_pages", "32",
        "--tli_near_tokens", "off",
    ])
    compact_args.method = "tli"
    assert resolve_far_skip_modes(compact_args, config=qwen) == ("compact", "evict")
    manifest = get_treatment_manifest_json(compact_args)
    assert '"tli_far_skip_backend": "compact"' in manifest
    assert '"tli_far_skip_cache": "evict"' in manifest
    assert "auto" not in manifest

    legacy_selection_args = parser.parse_args([
        "--tli_near_len", "4096", "--tli_near_pages", "32",
        "--tli_near_tokens", "off", "--tli_per_q_head", "true",
    ])
    legacy_selection_args.method = "tli"
    assert resolve_far_skip_modes(legacy_selection_args, config=qwen) == ("legacy", "retain")

    explicit_args = parser.parse_args([])
    explicit_args.method = "tli"
    explicit_args.tli_far_skip_backend = "compact"
    with pytest.raises(ValueError, match="fixed near_len/pages"):
        resolve_far_skip_modes(explicit_args, config=qwen)

    from transformers import LlamaConfig
    llama_args = parser.parse_args([])
    llama_args.method = "tli"
    llama = LlamaConfig(hidden_size=256, num_attention_heads=2, num_hidden_layers=1)
    assert resolve_far_skip_modes(llama_args, config=llama) == ("legacy", "retain")
    legacy = get_treatment_manifest_json(llama_args)
    assert 'tli_far_skip_backend' not in legacy and 'tli_far_skip_cache' not in legacy
    llama_args.tli_far_skip_backend = "compact"
    with pytest.raises(ValueError, match="only TLI Qwen3"):
        resolve_far_skip_modes(llama_args, config=llama)
    llama_args.tli_far_skip_backend = "auto"
    llama_args.tli_far_skip_cache = "evict"
    with pytest.raises(ValueError, match="only TLI Qwen3"):
        resolve_far_skip_modes(llama_args, config=llama)


def test_treatment_identity_default_compatibility_and_modes():
    args = args_for("retain")
    args.method = "tli"
    compact = get_treatment_manifest_json(args)
    args.tli_far_skip_cache = "evict"
    evict = get_treatment_manifest_json(args)
    args.tli_far_skip_backend, args.tli_far_skip_cache = "legacy", "retain"
    legacy = get_treatment_manifest_json(args)
    assert len({compact, evict, legacy}) == 3
    assert 'tli_far_skip_backend' not in legacy
    assert 'tli_far_skip_cache' not in legacy
    del args.tli_far_skip_backend, args.tli_far_skip_cache
    assert get_treatment_manifest_json(args) == legacy


@torch.inference_mode()
def test_far_on_layer_is_not_evicted_and_frozen_gate_cannot_reverse():
    model = make_model("evict", dynamic=False)
    module = model.model.layers[0].self_attn
    cache = DynamicCache()
    k, v = torch.randn(2, 1, 1, 39, 128)
    cache.update(k, v, 0)
    q = torch.randn(1, 2, 1, 128)
    nk, nv = torch.randn(2, 1, 1, 1, 128)
    module.indexer._skip_ids = set()
    original = cache.layers[0].keys
    assert try_far_skip_decode(module, q, nk, nv, cache, None, None, None) is None
    assert type(cache.layers[0]) is DynamicLayer and cache.layers[0].keys is original
    module.indexer._skip_ids = {0}
    assert try_far_skip_decode(module, q, nk, nv, cache, None, None, None) is not None
    assert cache.get_seq_length() == 40
    module.indexer._skip_ids = set()
    with pytest.raises(ValueError, match="re-enable far"):
        try_far_skip_decode(module, q, nk, nv, cache, None, None, None)
    assert cache.get_seq_length() == 40


@torch.inference_mode()
def test_hf_generate_uses_logical_positions():
    a, b = make_model("retain"), make_model("evict")
    ids = torch.arange(39)[None] % 32
    generated = [model.generate(ids, attention_mask=torch.ones_like(ids),
                               max_new_tokens=12, do_sample=False,
                               eos_token_id=None, pad_token_id=0)
                 for model in (a, b)]
    assert torch.equal(*generated)


def test_autograd_cannot_keep_evicted_far_storage_alive():
    cache = DynamicCache()
    cache.update(torch.randn(1, 1, 39, 8, requires_grad=True), torch.randn(1, 1, 39, 8), 0)
    original = cache.layers[0]
    with pytest.raises(ValueError, match="autograd graph"):
        evict_far(cache, 0, FarSkipConfig(4, 16, 2, 12, 3, 4))
    assert cache.layers[0] is original
    model = make_model("evict")
    with pytest.raises(ValueError, match="no_grad/inference_mode"):
        model(input_ids=torch.ones(1, 39, dtype=torch.long), past_key_values=DynamicCache())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA hardware")
@torch.inference_mode()
def test_cuda_bfloat16_retained_vs_evicted():
    a, b = make_model("retain").cuda().bfloat16(), make_model("evict").cuda().bfloat16()
    caches = [DynamicCache(), DynamicCache()]
    for model, cache in zip((a, b), caches):
        model(input_ids=torch.arange(129, device="cuda")[None] % 32, past_key_values=cache)
    for step in range(17):
        result = [model(input_ids=torch.tensor([[step]], device="cuda"), past_key_values=cache).logits
                  for model, cache in zip((a, b), caches)]
        torch.testing.assert_close(*result, atol=1e-2, rtol=1e-2)
