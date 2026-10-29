# -*- coding: utf-8 -*-
"""E152：decode-only 请求时延诊断采集器（E2 候选，尚需 GPU/低扰动验收）。

问题：dense-P/dense-D vs dense-P/sparse-D（TLI decode 稀疏）在完整 36 层请求上
是否真正缩短请求——TTFT / TPOT / request wall / KV 内存。补 E127/E128 单层合成
稳态不能当 36 层 TPOT 的缺口（evaluation.md E2「当前证据限制」）。

协议（evaluation.md E2 契约）：
  - 请求边界 host wall（入口到最后一个输出完成并同步）；阶段分解 CUDA sync 边界；
  - TTFT = prompt 前向（material+question）wall + 首 token 处理；
    两臂 prefill 均走指定后端的原生 dense（默认 FlashAttention2）（qwen3_attn_patch L100 attention_interface，
    TLI_SPARSE_PREFILL=0）；实际派发尚未由本脚本核验；
  - TPOT = 首 token 后 decode wall / 其余实际生成 token 数；固定输出长度
    max_gen=128 无 early-stop（等工作量诊断口径，eval.md 允许；自然生成长度
    另由质量链承载）；
  - 成对交替：编排脚本按 ABBA 交错两臂进程，消 cache/频率漂移；
  - 峰值显存：每请求前 reset_peak_memory_stats，记录 max_memory_allocated 增量
    + 理论 KV 字节数（2·L·layers·kv_heads·head_dim·bf16）；
  - 输入：合成 needle prompt（固定 seed，逐 token 确定性），两臂逐位相同
    input_ids（sha256 落盘回执）；
  - yarn：L>40960 时 Qwen3 官方配方（resolve_yarn_config，64K→2.0/128K→4.0），
    加载前写 config.rope_scaling（与 pred_ruler 同口径）；
  - tli 臂 patch 计数 fail-closed（0 挂载即 SystemExit，同 pred.py B3）。

每进程跑一臂×一 L×(warmup+repeats)，JSON 追加写 out；编排器负责交错。
本地 POSIX 同路径写者持久 sidecar 锁串行追加，临时文件完成后原子替换。
v2 保留逐步同步诊断，连续 wall 包含同步扰动，不是自然 serving 时延。
v1 的分段和不能无损换算为 v2 request wall/TTFT，旧记录须保留并重采。
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
sys.path.insert(0, REPO)

from sparse_attn.arguments import add_sparse_attn_args  # noqa: E402
from sparse_attn.patches import register_patch  # noqa: E402
from sparse_attn.info import resolve_treatment_snapshot  # noqa: E402

QWEN3_NATIVE_MPE = 40960
YARN_AUTO = {65536: 2.0, 131072: 4.0}

# 固定 needle 语料（确定性，无随机性依赖）
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


def timed_request(model, input_ids, max_gen, dev, static_cache_len=None):
    """连续模型入口→末 token 完成 wall；不含分词、输入 H2D 或清理。

    保留逐步同步诊断，不能把本采集器当成低扰动 serving benchmark。
    prefill_ms/decode_ms 是局部分段读数，不用于合成 request_wall_ms。
    static_cache_len 非 None 时每请求新建 StaticCache（E134-pre 臂：
    消 HF DynamicCache 每层每步 2 次 cat 全量拷贝 = 72 cat/步），
    max_cache_len = L + max_gen + 8。
    """
    if type(max_gen) is not int or max_gen < 1:
        raise ValueError("max_gen must be a positive integer")
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=dev)
    cache = None
    if static_cache_len is not None:
        from transformers import StaticCache
        cache = StaticCache(config=model.config, max_batch_size=1,
                            max_cache_len=static_cache_len, device=dev,
                            dtype=torch.bfloat16)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # dynamic（cache=None）首步省略 past_key_values kwarg 保持 v2 协议
        # 调用签名不变（test_e152_request_timing 断言）；static 臂显式传入
        if cache is None:
            out = model(input_ids=input_ids, use_cache=True)
        else:
            out = model(input_ids=input_ids, past_key_values=cache,
                        use_cache=True)
        cache = out.past_key_values
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1e3
        token = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        torch.cuda.synchronize()
        generated_tokens = [token]
        first_output = time.perf_counter()
        last_output = first_output
        step_ms = []
        per_token_wall_ms = []
        for _ in range(max_gen - 1):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            out = model(input_ids=token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            token = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            torch.cuda.synchronize()
            completed = time.perf_counter()
            generated_tokens.append(token)
            step_ms.append((completed - t1) * 1e3)
            per_token_wall_ms.append((completed - last_output) * 1e3)
            last_output = completed
    wall_ms = (last_output - t0) * 1e3
    decode_wall_ms = (last_output - first_output) * 1e3
    peak_delta_gib = (torch.cuda.max_memory_allocated() - base_mem) / 2**30
    return {
        "prefill_ms": prefill_ms,
        "decode_ms": sum(step_ms),
        "ttft_ms": (first_output - t0) * 1e3,
        "decode_after_first_wall_ms": decode_wall_ms,
        "tpot_ms": decode_wall_ms / len(step_ms) if step_ms else None,
        "request_wall_ms": wall_ms,
        "request_start_timestamp": t0,
        "first_output_timestamp": first_output,
        "last_output_timestamp": last_output,
        "per_token_wall_ms": per_token_wall_ms,
        "peak_mem_delta_gib": peak_delta_gib,
        "n_input_tokens": int(input_ids.shape[1]),
        "n_generated": max_gen,
        "gen_tokens": torch.cat(generated_tokens, dim=1).cpu().tolist(),
    }


def _validate_protocol(args):
    """Reject mislabeled/empty E152 runs before loading a model, also under -O."""
    if args.method not in ("none", "tli"):
        raise ValueError("E152 requires --method none or tli")
    for name, minimum in (("context_length", 1), ("max_gen", 1),
                          ("repeats", 1), ("warmup", 0)):
        value = getattr(args, name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if args.method == "tli" and os.environ.get("TLI_SPARSE_PREFILL", "0") != "0":
        raise ValueError("E152 是 decode-only 口径：TLI_SPARSE_PREFILL 必须为 0")


def _append_record(path, record):
    """Cooperative local-POSIX writers; readers see a complete old or new JSON.

    Keep the sidecar inode: unlinking it after unlock would let waiting and new
    writers lock different inodes. This is not distributed locking or idempotent
    retry after an uncertain publication; no power-loss guarantee is claimed.
    """
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
            raise ValueError("E152 output must be a JSON array; refusing overwrite")
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
            # Ordinary errors clean up; a killed process may leave a private
            # orphan .tmp, which is never selected as the published output.
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # --method 由 add_sparse_attn_args 提供（含 tli/none），此处不重复定义
    parser.add_argument("-L", "--context-length", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-gen", type=int, default=128)
    parser.add_argument("--model-path", type=str,
                        default=os.path.expanduser("~/Qwen3-8B"))
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--attention-backend", choices=["flash_attention_2", "sdpa"], default="flash_attention_2")
    parser.add_argument("--profile-last-request", action="store_true",
                        help="CUDA profiler API range on final request; separate from timing evidence")
    parser.add_argument("--static-cache", action="store_true",
                        help="E134-pre 臂：StaticCache 替代 DynamicCache"
                             "（消每层每步 2 次 cat 全量拷贝）")
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
        torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation=args.attention_backend).eval()

    if model.config._attn_implementation != args.attention_backend:
        raise RuntimeError("attention backend mismatch")
    snapshot = None
    if args.method == "tli":
        snapshot = resolve_treatment_snapshot(args)
    n_patched = register_patch(model, args, snapshot)
    if args.method != "none" and n_patched == 0:
        raise SystemExit("[E152 GATE-FAIL] tli 臂挂载 0 个 attention 模块，"
                         "实际将 dense 运行——拒绝计时")

    dev = next(model.parameters()).device
    ids = build_prompt_tokens(tokenizer, args.context_length)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=REPO, text=True).strip()

    runs = []
    # 容量按实际 prompt 长度算（build_prompt_tokens 有 ~57 token 模板开销，
    # 用 L 近似会 index_copy_ 越界 device-side assert——冒烟实测教训）
    static_cache_len = (len(ids) + args.max_gen + 8
                        if args.static_cache else None)
    for r in range(args.warmup + args.repeats):
        profile = args.profile_last_request and r == args.warmup + args.repeats - 1
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        try:
            rec = timed_request(model, ids, args.max_gen, dev,
                                static_cache_len=static_cache_len)
        finally:
            if profile:
                torch.cuda.cudart().cudaProfilerStop()
        rec.update({
            "run": r, "phase": "warmup" if r < args.warmup else "measured",
            "method": args.method, "L": args.context_length,
            "yarn_factor": factor,
        })
        runs.append(rec)
        tpot = "N/A" if rec["tpot_ms"] is None else f"{rec['tpot_ms']:.2f}ms"
        print(f"[E152] method={args.method} L={args.context_length} "
              f"run={r} prefill={rec['prefill_ms']:.1f}ms "
              f"ttft={rec['ttft_ms']:.1f}ms tpot={tpot} "
              f"wall={rec['request_wall_ms']:.1f}ms "
              f"peakΔ={rec['peak_mem_delta_gib']:.2f}GiB", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    # 理论 KV 字节（bf16，全 L token）
    layers = model.config.num_hidden_layers
    kvh = model.config.num_key_value_heads
    hd = model.config.head_dim or model.config.hidden_size // model.config.num_attention_heads
    kv_gib = 2 * args.context_length * layers * kvh * hd * 2 / 2**30

    record = {
        "schema": "e152-e2e-latency-v2",
        "caliber": {
            "commit": commit,
            "model": args.model_path,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "max_gen": args.max_gen,
            "early_stop": False,
            "timer_api": "time.perf_counter (timestamps in seconds)",
            "wall_boundary": "after input H2D+sync, before prompt forward -> last token argmax+sync",
            "tokenization_included": False,
            "input_h2d_included": False,
            "sampling_included": "greedy argmax, no text decode/transport",
            "instrumentation_mode": "per-step-device-synchronized",
            "sync_policy": "device sync before request, after prefill, after first argmax, before/after each decode step",
            "diagnostic_fields": ["prefill_ms", "decode_ms"],
            "backend_verified": False,
            "prefill_backend": args.attention_backend,
            "backend_config_checked": model.config._attn_implementation,
            "backend_execution_verified": False,
            "script_sha256": sha(open(__file__, "rb").read()),
            "operator_mode": os.environ.get("TLI_OPERATOR_MODE", "off"),
            "profiled_run": args.profile_last_request,
            "actual_prompt_tokens": len(ids),
            "decode_backend": ("TLI sparse (configured treatment; route unverified)"
                               if args.method == "tli" else "native dense"),
            "treatment_manifest": (json.loads(snapshot.manifest_json)
                                   if snapshot is not None else None),
            "input_ids_sha256_16": sha(torch.tensor([ids]).numpy()
                                       .tobytes())[:16],
            "kv_theory_gib": round(kv_gib, 2),
            "cache_impl": "StaticCache" if args.static_cache else "DynamicCache",
            "label": args.label,
        },
        "runs": runs,
    }
    _append_record(args.out, record)
    print(f"[E152] appended to {args.out}")


if __name__ == "__main__":
    main()
