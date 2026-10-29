# -*- coding: utf-8 -*-
"""E152v4：CUDA Graph decode TPOT 采集器（graph-vs-graph 公平口径）。

背景（e152v3 判决，2026-10-23）：e152_e2e_latency.py 是纯 eager decode，
不消费 TLI_CUDA_GRAPH——TLI 冠军臂无 graph 裸跑，32K TPOT 129.8ms vs
dense 25.9ms（反慢 5.0×，launch/Python 链主导）。用户拍板：转 CUDA
Graph 下比较。生产 graph 机器（pred_ruler_batch 阶段 2a/2c-5：
TLIStaticCache + cuda_graph_state + 整步 capture）已由 T6 大 bs 战役
验证（sink4 verify 8/8 逐字节一致），本脚本把它接到 e152 协议上，且
**两臂都上 graph**（dense 臂 method=none 复用同一 TLIStaticCache——其
update 只依赖 batch_decode_ctx + pos_buf，与 attention 方法无关）。

协议（e152 v2 契约 + v4 扩展）：
  - 每进程一臂 × 一 L × (warmup+repeats)；同一 prompt 逐位相同（确定性
    needle，input_ids sha256 落盘）；
  - prefill 两臂均指定后端的原生 dense（默认 FlashAttention2）（TLI_SPARSE_PREFILL=0），DynamicCache
    → fill_static_from_dynamic 搬入进程级 TLIStaticCache（有效在前，
    S_cap = len(ids)+max_gen+8；fill 是一次性 setup，单独计时
    fill_static_ms，不进 TTFT/TPOT——serving 中静态池同口径）；
  - 线性 cache 首轮 capture 后复用；compact cache 每请求重建布局与
    workspace，释放旧 graph 后重新 capture，capture 成本单独记录；
  - 步进注入：nxt/pid/pos/vl/m4 五 buffer copy_（graph_state 模式）；
    decode 步 s 喂 gen[s]（首 token 来自 prefill argmax），写列 L+s，
    seq_lens = L+s+1；
  - TPOT = decode wall（fill 完成后起算）/(max_gen-1)，每步 replay +
    argmax + .item() 读回同步（与 v2 eager 同为 per-step-device-
    synchronized 口径，可直接对照）；
  - --graph-mode eager：同一 TLIStaticCache 上的 staged 逐步 forward
    （graph_state 关闭，无 capture），仅用于生成 token 逐位对拍门禁，
    不作计时口径；
  - fail-loud：capture 失败 / warmup 后 graph 被 disable / tli 臂挂载
    0 个 attention → SystemExit（bench 不允许静默回退错标臂）。

输出 JSON schema "e152v4-graph-latency"：runs[] 含 per-step ms 列表与
gen_tokens（对拍用）。
"""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# 影子包钩子（复用 AKO --pkg 同款机制）：TLI_SHADOW_PKG 指向含 sparse_attn/
# 的目录（如 ~/tli_ako/solution）。插入位置在 REPO 之前 → 影子包生效。
# 当前影子包与主树仅 indexer/tli_indexer.py 一个文件不同（AKO iter-22
# 优化版，md5 cc487ed3，bitexact 100% 双门），其余 53 文件逐位一致。
_shadow = os.environ.get("TLI_SHADOW_PKG")
if _shadow:
    sys.path.insert(0, os.path.abspath(os.path.expanduser(_shadow)))

from sparse_attn.arguments import add_sparse_attn_args  # noqa: E402
from sparse_attn.patches import register_patch  # noqa: E402
from sparse_attn.info import resolve_treatment_snapshot  # noqa: E402
from sparse_attn.baseline_runtime import BatchCtx, batch_decode_scope  # noqa: E402
from sparse_attn.ops.tli_static_cache import (  # noqa: E402
    TLIStaticCache, fill_static_from_dynamic,
    finalize_direct_prefill, prepare_direct_prefill)
from sparse_attn.ops.cuda_graph_state import (  # noqa: E402
    enable_graph, disable_graph, graph_state)
from sparse_attn.ops.far_skip_runtime import resolve_far_skip_modes  # noqa: E402

QWEN3_NATIVE_MPE = 40960
YARN_AUTO = {65536: 2.0, 131072: 4.0}

NEEDLE = ("The special magic number for the hidden vault is 4198302. "
          "Remember this number. ")
FILLER_WORDS = ("the quick silver fox jumps over quiet hills while distant "
                "lanterns glow softly under autumn skies and travelers "
                "rest near ancient stone bridges where rivers flow calmly "
                "past forgotten villages and orchards ").split()
QUESTION = ("What is the special magic number for the hidden vault mentioned "
            "in the document? Reply with only the number.")


def build_prompt_tokens(tokenizer, L, q_frac=0.02):
    """Build exactly L deterministic token IDs without decode/re-tokenize drift."""
    prefix = tokenizer("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    suffix = tokenizer("<|im_end|>\n<|im_start|>assistant\n\n\n\n",
                       add_special_tokens=False)["input_ids"]
    question = tokenizer(QUESTION, add_special_tokens=False)["input_ids"]
    needle = tokenizer(NEEDLE, add_special_tokens=False)["input_ids"]
    filler = tokenizer(" ".join(FILLER_WORDS), add_special_tokens=False)["input_ids"]
    remaining = L - len(prefix) - len(suffix) - len(question) - len(needle)
    if remaining < 0 or not filler:
        raise ValueError("context length cannot contain the fixed chat/needle/question tokens")
    fill = (filler * ((remaining + len(filler) - 1) // len(filler)))[:remaining]
    midpoint = remaining // 2
    ids = prefix + fill[:midpoint] + needle + fill[midpoint:] + question + suffix
    if len(ids) != L:
        raise RuntimeError("exact context length invariant failed")
    return ids


def sha(b):
    return hashlib.sha256(b).hexdigest()


def _memory_snapshot():
    """Absolute process/device counters; no reserved-as-savings substitution."""
    torch.cuda.synchronize()
    process_mib = None
    try:
        lines = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits"], text=True).splitlines()
        process_mib = sum(int(line.split(",")[1]) for line in lines
                          if int(line.split(",")[0]) == os.getpid())
        if process_mib == 0:
            process_mib = None  # Some virtualized drivers hide per-process usage.
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "process_used_mib": process_mib,
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
    }


def _storage_snapshot(cache, model):
    """Per-layer/row physical KV ownership, including direct-prefill staging."""
    modules = {m.layer_idx: m for m in model.modules()
               if getattr(m, "indexer", None) is not None}
    staging = getattr(cache, "_direct_prefill", None)
    entries = []
    for li, layer in enumerate(cache.layers):
        idx = getattr(modules.get(li), "indexer", None)
        if staging is not None:
            state = staging.states[li]
            logical_length = staging.length if state.written else 0
            owners = (("staging_linear", state.linear, None, idx.skip_far),
                      ("staging_ring", state.ring, None, idx.skip_far))
        elif getattr(layer, "rows", None) is not None:
            owners = tuple(
                ("ring" if type(row).__name__ == "FarSkipStaticLayer" else "linear",
                 row, ri, layer.indexers[ri].skip_far)
                for ri, row in enumerate(layer.rows))
        else:
            role = "ring" if type(layer).__name__ == "FarSkipStaticLayer" else "linear"
            owners = ((role, layer, None, getattr(idx, "skip_far", None)),)
        for role, owner, row, skip_far in owners:
            if owner is None or owner.keys is None:
                continue
            logical_length = (cache._hybrid_lengths[row] if row is not None
                              else sum(cache._hybrid_lengths)
                              if getattr(cache, "_hybrid_batch", False)
                              else staging.length if staging is not None
                              else layer.cumulative_length)
            entries.append({
                "layer": li,
                "row": row,
                "role": role,
                "skip_far": skip_far,
                "logical_tokens": logical_length,
                "physical_tokens_per_row": owner.keys.shape[-2],
                "keys_bytes": owner.keys.numel() * owner.keys.element_size(),
                "values_bytes": owner.values.numel() * owner.values.element_size(),
            })
    totals = {}
    for entry in entries:
        totals[entry["role"]] = totals.get(entry["role"], 0) + entry["keys_bytes"] + entry["values_bytes"]
    logical = {(entry["layer"], entry["row"]): entry["logical_tokens"] for entry in entries}
    gates = {(entry["layer"], entry["row"]): entry["skip_far"] for entry in entries}
    return {
        "total_bytes": sum(totals.values()),
        "bytes_by_role": totals,
        "logical_tokens": sum(logical.values()),
        "n_skip_layers": sum(value is True for value in gates.values()),
        "n_non_skip_layers": sum(value is False for value in gates.values()),
        "layers": entries,
    }


def _dynamic_storage_snapshot(dyn):
    if dyn is None:
        return None
    entries = [{"layer": li,
                "logical_tokens": layer.keys.shape[-2],
                "physical_tokens_per_row": layer.keys.shape[-2],
                "keys_bytes": layer.keys.numel() * layer.keys.element_size(),
                "values_bytes": layer.values.numel() * layer.values.element_size()}
               for li, layer in enumerate(dyn.layers)]
    return {"total_bytes": sum(e["keys_bytes"] + e["values_bytes"] for e in entries),
            "layers": entries}


def _append_record(path, record):
    """与 e152 相同的 sidecar 锁 + 临时文件原子替换追加协议。"""
    import fcntl
    import stat
    import tempfile

    path = os.path.realpath(os.fspath(path))
    with open(path + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = []
        mode = None
        if os.path.exists(path):
            with open(path) as stream:
                existing = json.load(stream)
            mode = stat.S_IMODE(os.stat(path).st_mode)
        if not isinstance(existing, list):
            raise ValueError("E152v4 output must be a JSON array; refusing overwrite")
        existing.append(record)
        fd, temporary = tempfile.mkstemp(
            prefix="." + os.path.basename(path) + ".", suffix=".tmp",
            dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w") as stream:
                if mode is not None:
                    os.fchmod(stream.fileno(), mode)
                json.dump(existing, stream, indent=1)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


class GraphRunner:
    """进程级 graph 机器：五 buffer + 一次性 capture + 逐步 replay。

    B=1 固定。buffer 与 pred_ruler_batch._decode_batch_graph 同名同义：
      nxt_buf [1,1] 本步喂入 token；pid_buf [1,1] RoPE 绝对位置；
      pos_buf [1] KV 写列；vl_buf [1] 有效长（TLI valid_length 输入）；
      m4_buf [1,1,1,S_cap] dense 层 additive mask（-inf，可见列 0）。
    """

    def __init__(self, model, static_cache, neg_inf, S_cap, pad_id, dev):
        self.model = model
        self.cache = static_cache
        self.S_cap = S_cap
        self.dev = dev
        self.neg_inf = neg_inf
        self.nxt_buf = torch.full((1, 1), pad_id, dtype=torch.long, device=dev)
        self.pid_buf = torch.zeros((1, 1), dtype=torch.long, device=dev)
        self.pos_buf = torch.zeros(1, dtype=torch.long, device=dev)
        self.vl_buf = torch.zeros(1, dtype=torch.long, device=dev)
        self.m4_buf = torch.full((1, 1, 1, S_cap), neg_inf,
                                 dtype=next(model.parameters()).dtype, device=dev)
        self.graph = None
        self.out = None

    def inject(self, token, seq_len_after_inc):
        """步进注入：token = 本步喂入 token；seq_len_after_inc = 含本步
        新 token 的有效长（= 写列 pos+1 = 可见列数）。"""
        self.nxt_buf.copy_(torch.tensor([[token]], dtype=torch.long,
                                        device=self.dev))
        self.pid_buf.copy_(torch.tensor([[seq_len_after_inc - 1]],
                                         dtype=torch.long, device=self.dev))
        self.pos_buf.copy_(torch.tensor([seq_len_after_inc - 1], dtype=torch.long,
                                        device=self.dev))
        self.vl_buf.copy_(torch.tensor([seq_len_after_inc], dtype=torch.long,
                                       device=self.dev))
        self.m4_buf.fill_(self.neg_inf)
        self.m4_buf[0, 0, 0, :seq_len_after_inc] = 0

    def forward(self):
        return self.model(input_ids=self.nxt_buf, position_ids=self.pid_buf,
                          cache_position=self.pos_buf,
                          attention_mask=self.m4_buf, past_key_values=self.cache,
                          use_cache=True)

    def fill(self, dyn, length):
        """Refill one request; compact gates/workspace/pointers require recapture."""
        compact = getattr(self.cache, '_far_skip_prepared', False) or any(
            getattr(getattr(m, 'indexer', None), 'far_skip_backend', 'legacy') == 'compact'
            for m in self.model.modules())
        if compact:
            self.graph = self.out = None
        fill_static_from_dynamic(self.cache, dyn, [length], model=self.model)
        if getattr(self.cache, '_far_skip_prepared', False):
            # This caller owns the unpadded mask and updates it with position.
            self.cache._far_skip_attention_mask = self.m4_buf

    def prepare_prefill(self, length):
        """Start direct partitioned prefill and invalidate request-local graphs."""
        self.graph = self.out = None
        return prepare_direct_prefill(self.cache, self.model, length)

    def finalize_prefill(self, length):
        finalize_direct_prefill(self.cache, self.model, length)
        self.cache._far_skip_attention_mask = self.m4_buf

    def warmup_and_capture(self, token, seq_len_after_inc):
        """2 步主 stream 热跑（惰性分配/autotune 出图外）+ 1 步 capture。

        热跑写列被 replay 首步同位置同 token 覆写（确定性），无污染
        （pred_ruler 同款论证）。返回 capture wall ms。"""
        t0 = time.perf_counter()
        self.inject(token, seq_len_after_inc)
        for _ in range(2):
            self.forward()
        gs = graph_state()
        if not gs.enabled:
            raise RuntimeError(
                "warmup 期间 compute_mask_graph 关闭了 graph（配置不受支持）")
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self.forward()
        return (time.perf_counter() - t0) * 1e3


def timed_request(model, runner, ids, max_gen, dev, pad_id, do_capture,
                  prefill_mode="conversion"):
    """一个 repeat：direct 或 DynamicCache conversion → [capture] → decode。"""
    if prefill_mode not in ("auto", "conversion", "direct"):
        raise ValueError("invalid prefill mode")
    compact = any(getattr(getattr(m, "indexer", None), "far_skip_backend", "legacy") == "compact"
                  for m in model.modules())
    direct = prefill_mode == "direct" or (prefill_mode == "auto" and compact)
    if prefill_mode == "direct" and not compact:
        raise ValueError("direct prefill requires compact Qwen3 attention")
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    memory_before = _memory_snapshot()
    L = len(ids)
    input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
    ctx = BatchCtx(batch=1, seq_lens=[L], gen_lens=[0], alive=[True],
                   pad_id=pad_id)
    capture_ms = None

    with torch.no_grad(), batch_decode_scope(ctx):
        if direct:
            runner.prepare_prefill(L)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=True,
                    past_key_values=runner.cache if direct else None)
        dyn = None if direct else out.past_key_values
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1e3
        first_token = int(out.logits[0, -1, :].argmax(dim=-1).item())
        torch.cuda.synchronize()
        first_output = time.perf_counter()
        del out
        prefill_memory = _memory_snapshot()
        prefill_storage = {
            "static": _storage_snapshot(runner.cache, model),
            "dynamic_source": _dynamic_storage_snapshot(dyn),
        }

        t_fill = time.perf_counter()
        if direct:
            runner.finalize_prefill(L)
        else:
            runner.fill(dyn, L)
        torch.cuda.synchronize()
        fill_ms = (time.perf_counter() - t_fill) * 1e3
        finalize_ms = fill_ms if direct else None
        del dyn
        post_finalize_memory = _memory_snapshot()
        decode_cache_allocated_gib = post_finalize_memory["allocated_bytes"] / 2**30
        kv_storage = _storage_snapshot(runner.cache, model)
        kv_storage_bytes = kv_storage["total_bytes"]

        if do_capture or (graph_state().enabled and runner.graph is None):
            # enable_graph 在 main 里于进程启动时已开（graph 模式）；
            # buffer 绑定在 main 里完成。
            capture_ms = runner.warmup_and_capture(first_token, L + 1)
        post_capture_memory = _memory_snapshot()

        gen = [first_token]
        step_ms = []
        torch.cuda.synchronize()
        decode_t0 = time.perf_counter()
        if runner.graph is not None:
            for s in range(max_gen - 1):
                ctx.seq_lens[0] = L + s + 1
                runner.inject(gen[-1], ctx.seq_lens[0])
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                runner.graph.replay()
                pred = int(runner.out.logits[0, -1, :].argmax(dim=-1).item())
                torch.cuda.synchronize()
                completed = time.perf_counter()
                step_ms.append((completed - t1) * 1e3)
                gen.append(pred)
        else:
            # eager staged 参照臂：graph_state 保持关闭（update 回退
            # Python pos 构造），逐步直接 forward——仅对拍，不计时口径。
            if graph_state().enabled:
                raise SystemExit("[E152v4 GATE-FAIL] eager 参照臂必须在 "
                                 "graph_state 关闭下运行")
            for s in range(max_gen - 1):
                ctx.seq_lens[0] = L + s + 1
                runner.inject(gen[-1], ctx.seq_lens[0])
                t1 = time.perf_counter()
                out_s = runner.forward()
                pred = int(out_s.logits[0, -1, :].argmax(dim=-1).item())
                torch.cuda.synchronize()
                completed = time.perf_counter()
                step_ms.append((completed - t1) * 1e3)
                gen.append(pred)
                del out_s
        last_output = time.perf_counter()

    decode_wall_ms = (last_output - decode_t0) * 1e3
    steady_decode_memory = _memory_snapshot()
    peak_delta_gib = (steady_decode_memory["peak_allocated_bytes"] - base_mem) / 2**30
    return {
        "prefill_ms": prefill_ms,
        "prefill_mode": "direct" if direct else "conversion",
        "fill_static_ms": fill_ms,
        "finalize_ms": finalize_ms,
        "capture_ms": capture_ms,
        "decode_ms": sum(step_ms),
        "ttft_ms": (first_output - t0) * 1e3,
        "decode_after_fill_wall_ms": decode_wall_ms,
        "request_wall_ms": (last_output - t0) * 1e3,
        "tpot_ms": decode_wall_ms / len(step_ms) if step_ms else None,
        "step_ms": step_ms,
        "peak_mem_delta_gib": peak_delta_gib,
        "prefill_peak_allocated_gib": prefill_memory["peak_allocated_bytes"] / 2**30,
        "prefill_peak_delta_gib": (prefill_memory["peak_allocated_bytes"] - base_mem) / 2**30,
        "decode_cache_allocated_gib": decode_cache_allocated_gib,
        "kv_storage_bytes": kv_storage_bytes,
        "kv_storage": kv_storage,
        "prefill_storage": prefill_storage,
        "memory": {
            "before_prefill": memory_before,
            "after_prefill": prefill_memory,
            "after_finalize_or_fill": post_finalize_memory,
            "after_capture": post_capture_memory,
            "steady_decode": steady_decode_memory,
        },
        "n_input_tokens": L,
        "n_generated": max_gen,
        "gen_tokens": gen,
    }


def _validate_protocol(args):
    if args.method not in ("none", "tli"):
        raise ValueError("E152v4 requires --method none or tli")
    for name, minimum in (("context_length", 16384), ("max_gen", 4),
                          ("repeats", 1), ("warmup", 0)):
        value = getattr(args, name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if args.method == "tli" and os.environ.get("TLI_SPARSE_PREFILL", "0") != "0":
        raise ValueError("E152v4 是 decode-only 口径：TLI_SPARSE_PREFILL 必须为 0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-L", "--context-length", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-gen", type=int, default=128)
    parser.add_argument("--model-path", type=str,
                        default=os.path.expanduser("~/Qwen3-8B"))
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--attention-backend", choices=["flash_attention_2", "sdpa"], default="flash_attention_2")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--graph-mode", choices=["graph", "eager"],
                        default="graph",
                        help="graph=整步 capture+replay 计时臂；eager=同 "
                             "TLIStaticCache 上的 staged 参照臂（仅对拍）")
    parser.add_argument("--prefill-mode", choices=["auto", "conversion", "direct"],
                        default="auto",
                        help="auto=compact 用 direct，其余用 DynamicCache conversion")
    add_sparse_attn_args(parser)
    args = parser.parse_args()

    try:
        _validate_protocol(args)
    except ValueError as exc:
        parser.error(str(exc))
    os.environ.setdefault("TLI_SPARSE_PREFILL", "0")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from benchmark.RULER.yarn_receipt import resolve_yarn_config

    use_yarn = args.context_length > QWEN3_NATIVE_MPE
    factor, rope_scaling = resolve_yarn_config(
        use_yarn, None, args.context_length, YARN_AUTO, QWEN3_NATIVE_MPE)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if use_yarn:
        config.rope_scaling = rope_scaling
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation=args.attention_backend).eval()

    resolve_far_skip_modes(args, model=model)
    snapshot = None
    if args.method == "tli":
        snapshot = resolve_treatment_snapshot(args)
    n_patched = register_patch(model, args, snapshot)
    if args.method != "none" and n_patched == 0:
        raise SystemExit("[E152v4 GATE-FAIL] tli 臂挂载 0 个 attention 模块"
                         "——拒绝计时")

    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    neg_inf = torch.finfo(dtype).min
    pad_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id
    ids = build_prompt_tokens(tokenizer, args.context_length)
    L = len(ids)
    S_cap = L + args.max_gen + 8
    if S_cap < 16384:
        raise SystemExit(f"[E152v4 GATE-FAIL] S_cap={S_cap} < 16384（Triton "
                         "gather 批量路径下界）")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=REPO, text=True).strip()

    # 进程级静态 cache + runner；graph 模式全程 enable_graph（capture 前
    # 绑定 pos/vl buffer，TLIStaticCache.update 走 scatter_ 张量索引）。
    static_cache = TLIStaticCache(model.config, S_cap, 1)
    runner = GraphRunner(model, static_cache, neg_inf, S_cap, pad_id, dev)
    if args.graph_mode == "graph":
        enable_graph(1)
        gs = graph_state()
        gs.pos_buf = runner.pos_buf
        gs.valid_len_buf = runner.vl_buf

    runs = []
    for r in range(args.warmup + args.repeats):
        do_capture = (args.graph_mode == "graph" and runner.graph is None)
        try:
            rec = timed_request(model, runner, ids, args.max_gen, dev,
                                pad_id, do_capture, args.prefill_mode)
        except SystemExit:
            raise
        except Exception as e:
            if do_capture:
                import traceback
                traceback.print_exc()
                disable_graph()
                raise SystemExit(f"[E152v4 GATE-FAIL] graph 臂 capture/decode "
                                 f"失败（{type(e).__name__}: {e}）")
            raise
        selector_routes = []
        for name, module in model.named_modules():
            idx = getattr(module, "indexer", None)
            constants = getattr(idx, "_graph_const_cache", None)
            if constants is not None:
                selector_routes.append({"module": name,
                    "sparse_selection": constants[0][-1] is True})
        if os.environ.get("TLI_SPARSE_SELECTION", "0") == "1":
            if len(selector_routes) != n_patched or not all(x["sparse_selection"] for x in selector_routes):
                raise RuntimeError("sparse benchmark fell back to a dense selector")
        rec["selector_routes"] = selector_routes
        rec["n_patched_attention_layers"] = n_patched
        rec.update({
            "run": r, "phase": "warmup" if r < args.warmup else "measured",
            "method": args.method, "L": args.context_length,
            "yarn_factor": factor, "graph_mode": args.graph_mode,
        })
        runs.append(rec)
        tpot = "N/A" if rec["tpot_ms"] is None else f"{rec['tpot_ms']:.2f}ms"
        print(f"[E152v4] method={args.method} L={args.context_length} "
              f"mode={args.graph_mode} run={r} "
              f"prefill={rec['prefill_ms']:.1f}ms fill={rec['fill_static_ms']:.1f}ms "
              f"tpot={tpot} decodewall={rec['decode_after_fill_wall_ms']:.1f}ms "
              f"peakΔ={rec['peak_mem_delta_gib']:.2f}GiB "
              f"first_tok={rec['gen_tokens'][0]}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    if args.graph_mode == "graph":
        disable_graph()

    layers = model.config.num_hidden_layers
    kvh = model.config.num_key_value_heads
    hd = (model.config.head_dim or
          model.config.hidden_size // model.config.num_attention_heads)
    kv_gib = 2 * L * layers * kvh * hd * 2 / 2**30

    record = {
        "schema": "e152v4-graph-latency",
        "caliber": {
            "commit": commit,
            "model": args.model_path,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "max_gen": args.max_gen,
            "early_stop": False,
            "graph_mode": args.graph_mode,
            "S_cap": S_cap,
            "timer_api": "time.perf_counter (timestamps in seconds)",
            "wall_boundary": ("after input H2D+sync, prompt forward -> "
                               "last token argmax+sync"),
            "decode_boundary": ("decode wall from after fill_static sync to "
                                "last token argmax+sync（fill 一次性 setup "
                                "排除，serving 静态池同口径）"),
            "tokenization_included": False,
            "input_h2d_included": False,
            "sampling_included": "greedy argmax, no text decode/transport",
            "instrumentation_mode": ("per-step-device-synchronized "
                                     "(graph: inject+replay+argmax+item)"),
            "backend_verified": False,
            "prefill_backend": args.attention_backend,
            "decode_backend": (
                "TLI sparse, whole-step CUDA Graph" if
                (args.method == "tli" and args.graph_mode == "graph") else
                "native dense, whole-step CUDA Graph" if
                (args.method == "none" and args.graph_mode == "graph") else
                "eager staged on TLIStaticCache (token gate ref)"),
            "treatment_manifest": (json.loads(snapshot.manifest_json)
                                   if snapshot is not None else None),
            "input_ids_sha256_16": sha(torch.tensor([ids]).numpy()
                                       .tobytes())[:16],
            "kv_theory_gib": round(kv_gib, 2),
            "cache_impl": f"TLIStaticCache(S_cap={S_cap})",
            "label": args.label,
        },
        "runs": runs,
    }
    _append_record(args.out, record)
    print(f"[E152v4] appended to {args.out}")


if __name__ == "__main__":
    main()
