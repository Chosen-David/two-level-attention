# -*- coding: utf-8 -*-
import os, sys, json, importlib.util, hashlib, argparse

os.environ['TLI_SHADOW_PKG'] = os.path.expanduser('~/tli_ako/solution')
os.environ['TLI_SPARSE_PREFILL'] = '0'
os.environ['PYTHONUNBUFFERED'] = '1'

REPO = os.path.expanduser('~/two-level-attention')
sys.path.insert(0, REPO)

_spec = importlib.util.spec_from_file_location(
    'e152v4', os.path.join(REPO, 'exp/bench/e152v4_graph_latency.py'))
e152 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e152)

from sparse_attn.arguments import add_sparse_attn_args
parser = argparse.ArgumentParser()
add_sparse_attn_args(parser)
parser.add_argument('-L', '--context-length', type=int, required=True)
parser.add_argument('--model-path', type=str, default=os.path.expanduser('~/Qwen3-8B'))
parser.add_argument('--max-gen', type=int, default=8)

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from benchmark.RULER.yarn_receipt import resolve_yarn_config

CHAMP = [
    '--method', 'tli', '--tli_far_method', 'minmax', '--tli_near_method', 'avg',
    '--tli_near_pages', '32', '--tli_near_tokens', 'off', '--tli_near_len', '4096',
    '--tli_enable_kmeans', 'false', '--tli_enable_layer_skip', 'false',
    '--tli_subspace', 'tail', '--tia_level2_cmp_ratio', '4',
    '--tli_sink_tokens', '4', '--tli_swa_size', '128',
    '--tia_level1_topk', '128', '--tia_level2_topk', '1024',
]
MODEL = os.path.expanduser('~/Qwen3-8B')

results = {}
sys.argv = ['e170a.py']
for L in [32768, 65536, 131072]:
    print(f'\n[E170a] L={L} start', flush=True)
    args = parser.parse_args(CHAMP + ['-L', str(L), '--max-gen', '8', '--model-path', MODEL])
    use_yarn = L > e152.QWEN3_NATIVE_MPE
    factor, rope_scaling = resolve_yarn_config(use_yarn, None, L, e152.YARN_AUTO, e152.QWEN3_NATIVE_MPE)
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    if use_yarn:
        config.rope_scaling = rope_scaling
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
    snap = e152.resolve_treatment_snapshot(args)
    n = e152.register_patch(model, args, snap)
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    neg_inf = torch.finfo(dtype).min
    pad_id = getattr(tokenizer, 'pad_token_id', None) or tokenizer.eos_token_id
    ids = e152.build_prompt_tokens(tokenizer, L)
    S_cap = len(ids) + args.max_gen + 8
    static_cache = e152.TLIStaticCache(model.config, S_cap, 1)
    runner = e152.GraphRunner(model, static_cache, neg_inf, S_cap, pad_id, dev)
    e152.enable_graph(1)
    gs = e152.graph_state()
    gs.pos_buf = runner.pos_buf
    gs.valid_len_buf = runner.vl_buf
    rec = e152.timed_request(model, runner, ids, args.max_gen, dev, pad_id, True)
    toks = rec['gen_tokens']
    md5 = hashlib.md5(','.join(str(t) for t in toks).encode()).hexdigest()
    results[L] = {
        'tpot_ms': round(rec.get('tpot_ms', rec.get('decode_ms', 0)), 3),
        'md5': md5, 'n_toks': len(toks), 'first5': toks[:5]
    }
    print(f'[E170a] L={L}: tpot={results[L]["tpot_ms"]:.3f}ms  md5={md5[:12]}', flush=True)
    del model; torch.cuda.empty_cache()

v = {'experiment': 'e170a_bitexact', 'gpu': 'H20', 'results': results}
out = os.path.expanduser('~/tli_ako/e170a_gpt_patch_bitexact_verdict.json')
with open(out, 'w') as f: json.dump(v, f, indent=2)
print(f'\n[E170a] DONE -> {out}', flush=True)
