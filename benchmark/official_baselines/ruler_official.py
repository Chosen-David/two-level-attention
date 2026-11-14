"""RULER 的独立官方 Quest/JSSA 入口。

Quest 使用官方 CUDA kernel 加本地原生 GQA/接口适配；JSSA 使用固定提交的
patched-vLLM 官方 Triton backend。本文件不导入 NFA、sparse_attn 或旧 vendor。
预测通过现有不可变 generation + 单指针 + producer-yarn-config-v2 回执提交。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

from benchmark.RULER.yarn_receipt import (
    _file_sha256,
    acquire_output_lock,
    build_yarn_receipt,
    commit_yarn_generation,
    generation_pointer_path,
    release_output_lock,
    resolve_yarn_config,
    stage_yarn_generation,
    stage_yarn_receipt,
    validate_committed_generation,
)
from benchmark.official_baselines.longbench_official import OfficialEngine
from benchmark.rope_policy import configure_rope, verify_runtime_rope

REPO = Path(__file__).resolve().parents[2]
RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt",
]
RULER_MAX_NEW_TOKENS = {
    **{task: 128 for task in RULER_TASKS if task.startswith("niah_")},
    "cwe": 120,
    "fwe": 50,
    "vt": 30,
}
MIN_NEW_TOKENS = 1
SEED = 42
QWEN3_NATIVE_MPE = 40960
YARN_FACTOR_AUTO = {65536: 2.0, 131072: 4.0}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--method", choices=("quest", "jssa"), required=True)
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", choices=RULER_TASKS, required=True)
    parser.add_argument(
        "--context-length", type=int,
        choices=(4096, 8192, 16384, 32768, 65536, 131072), required=True,
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--t", default="ruler run")
    parser.add_argument("--pred-postfix", default="")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--max-num", type=int, default=500)
    parser.add_argument("--yarn", action="store_true")
    parser.add_argument("--yarn-factor", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--quest-page-size", type=int, default=16)
    parser.add_argument("--quest-max-seq-len", type=int, default=40960)
    parser.add_argument("--jssa-rank", type=int, default=32)
    parser.add_argument("--jssa-basis-path", default=str(REPO / "bases/qwen3_8b.pt"))
    parser.add_argument("--jssa-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def load_rows(args):
    path = Path(args.data_root) / str(args.context_length) / f"{args.task}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"RULER 输入不存在: {path}")
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if args.skip < 0 or args.max_num < 0:
        raise ValueError("skip/max-num 必须非负")
    rows = rows[args.skip:]
    if args.max_num:
        rows = rows[:args.max_num]
    if not rows:
        raise ValueError("当前 RULER 分片没有样本，拒绝提交空 prediction receipt")
    return rows, hashlib.sha256(raw).hexdigest(), path


def resolve_rope(args, config):
    if args.model != "Qwen3-8B" or config.model_type != "qwen3":
        raise ValueError("当前官方 RULER 入口仅验收 Qwen3-8B；32B/TP2 不得套用")
    native_mpe = int(config.max_position_embeddings)
    if args.context_length > QWEN3_NATIVE_MPE and not args.yarn:
        raise ValueError("超过 Qwen3 原生 40960 的 RULER 档位必须显式启用 YaRN")
    factor, rope_scaling = resolve_yarn_config(
        args.yarn, args.yarn_factor, args.context_length,
        YARN_FACTOR_AUTO, native_mpe,
    )
    policy = configure_rope(
        config, args.context_length, RULER_MAX_NEW_TOKENS[args.task],
        yarn=args.yarn, factor=args.yarn_factor,
    )
    preflight = verify_runtime_rope(config)
    return native_mpe, factor, rope_scaling, policy, preflight


def output_path(args):
    if args.method == "quest":
        label = f"quest_official_b{args.budget}_p{args.quest_page_size}"
    else:
        label = f"jssa_official_b{args.budget}_r{args.jssa_rank}"
    directory = (
        Path(args.output_dir) / f"L{args.context_length}"
        / f"pred{args.pred_postfix}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{args.task}-{label}-{args.t}.jsonl"


def generate_ruler(
    args, engine, rows, *, native_mpe, factor, rope_scaling,
    rope_policy, rope_preflight, runtime_rope, data_sha256, data_path, eos_ids,
):
    """生成完整任务并通过未修改的 v2 消费端校验后原子提交。"""
    max_new_tokens = RULER_MAX_NEW_TOKENS[args.task]
    out_path = output_path(args)
    attempt_id = (
        datetime.now().strftime("%Y%m%d%H%M%S")
        + f"-{os.getpid()}-{random.randint(1000, 9999)}"
    )
    lock_fd = acquire_output_lock(str(out_path))
    try:
        tmp_pred = stage_yarn_generation(str(out_path), attempt_id)
        with open(tmp_pred, "w", encoding="utf-8") as handle:
            for row in rows:
                ids = engine.tokenizer(
                    row["input"], truncation=False, return_tensors="pt"
                ).input_ids[0].tolist()
                if len(ids) + max_new_tokens > runtime_rope["runtime_max_position_embeddings"]:
                    raise ValueError("实际 RULER 输入加输出超过已验证 RoPE 容量")
                generated, _ = engine.generate(
                    ids, max_new_tokens, ignore_eos=False,
                    stop_token_ids=None, min_new_tokens=MIN_NEW_TOKENS,
                )
                if len(generated) < MIN_NEW_TOKENS:
                    raise RuntimeError("RULER 首 token EOS 抑制未生效")
                record = {
                    "pred": engine.tokenizer.decode(
                        generated, skip_special_tokens=True
                    ),
                    "answers": row["outputs"],
                    "generated_token_ids": generated,
                    "length": row["length"],
                    "budget": args.budget,
                    "execution": {
                        "requested": "official-" + args.method,
                        "actual": "official-" + args.method,
                    },
                    "implementation": args.method + "_official",
                    "input": {
                        "input_tokens": len(ids),
                        "input_ids_sha256": hashlib.sha256(
                            json.dumps(ids, separators=(",", ":")).encode("utf-8")
                        ).hexdigest(),
                    },
                    "_id": f"{args.task}:{row['index']}",
                    "_answers_sha": hashlib.sha256(json.dumps(
                        row["outputs"], ensure_ascii=False, sort_keys=True
                    ).encode("utf-8")).hexdigest()[:16],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{args.method}:{args.task}] index={row['index']} "
                    f"in={len(ids)} out={len(generated)}", flush=True,
                )

        pred_sha = _file_sha256(tmp_pred)
        with open(tmp_pred, "rb") as prediction_file:
            pred_lines = sum(1 for _ in prediction_file)
        model_config = Path(args.model_path) / "config.json"
        receipt = build_yarn_receipt(
            yarn_enabled=args.yarn,
            effective_factor=factor,
            yarn_factor_cli=args.yarn_factor,
            rope_scaling=rope_scaling,
            context_length=args.context_length,
            task=args.task,
            model_path=str(Path(args.model_path).resolve()),
            model_config_sha256=(
                _file_sha256(str(model_config)) if model_config.is_file() else None
            ),
            native_mpe=native_mpe,
            generation_params={
                "max_gen": max_new_tokens,
                "min_new_tokens": MIN_NEW_TOKENS,
                "eos_token_ids": list(eos_ids),
                "max_num": args.max_num,
                "seed": SEED,
                "method": args.method,
                "pred_postfix": args.pred_postfix,
                "t": args.t,
                "budget": args.budget,
                "official_backend": True,
            },
            producer_script_path="benchmark/official_baselines/ruler_official.py",
            producer_script_sha256=_file_sha256(str(Path(__file__).resolve())),
            run_id=attempt_id,
            prediction_basename=out_path.name,
            prediction_sha256=pred_sha,
            prediction_lines=pred_lines,
        )
        receipt.update({
            "official_runtime": engine.runtime,
            "rope_policy": rope_policy,
            "rope_preflight": rope_preflight,
            "runtime_rope_verification": runtime_rope,
            "ruler_input": {
                "path": str(data_path),
                "sha256": data_sha256,
                "rows": len(rows),
                "skip": args.skip,
                "max_num": args.max_num,
            },
        })
        tmp_receipt = stage_yarn_receipt(str(out_path), receipt, attempt_id)
        # 与正式 scorer 共用同一严格校验器；失败时绝不切换 generation 指针。
        validate_committed_generation(tmp_pred, tmp_receipt)
        pointer = commit_yarn_generation(tmp_pred, tmp_receipt, str(out_path))
        print(f"[pred] 提交：指针 {pointer}", flush=True)
        print(f"saved -> {out_path} (via generation pointer)", flush=True)
        return out_path
    finally:
        release_output_lock(lock_fd)


def main():
    args = parse_args()
    if args.budget <= 0 or args.jssa_rank <= 0:
        raise ValueError("budget/rank 必须为正")
    if args.method == "quest" and args.budget % args.quest_page_size:
        raise ValueError("Quest budget 必须整除 page-size")
    random.seed(SEED)
    rows, data_sha256, data_path = load_rows(args)

    from transformers import AutoConfig, GenerationConfig

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    native_mpe, factor, rope_scaling, rope_policy, rope_preflight = resolve_rope(
        args, config
    )
    generation_config = GenerationConfig.from_pretrained(args.model_path)
    configured_eos = generation_config.eos_token_id
    if configured_eos is None:
        configured_eos = ()
    elif not isinstance(configured_eos, (list, tuple)):
        configured_eos = (configured_eos,)
    eos_ids = tuple(dict.fromkeys(
        token for token in configured_eos
        if isinstance(token, int) and not isinstance(token, bool) and token >= 0
    ))

    max_new_tokens = RULER_MAX_NEW_TOKENS[args.task]
    args.model_config = config
    args.quest_rope_mode = "hf-position-embeddings" if args.yarn else "native"
    args.hf_overrides = {
        "rope_scaling": getattr(config, "rope_scaling", None),
        "max_position_embeddings": int(config.max_position_embeddings),
    }
    engine = OfficialEngine(args, args.context_length + max_new_tokens + 16)
    if args.method == "quest":
        runtime_rope = verify_runtime_rope(engine.model)
    else:
        # 从 vLLM 实际构造后的 ModelConfig 读取 effective HF config，再构造
        # CPU rotary 验证频率/缩放；不是仅复述 CLI 或原始 config.json。
        runtime_config = engine.model.llm_engine.vllm_config.model_config.hf_config
        runtime_rope = verify_runtime_rope(runtime_config)
    out_path = generate_ruler(
        args, engine, rows,
        native_mpe=native_mpe, factor=factor, rope_scaling=rope_scaling,
        rope_policy=rope_policy, rope_preflight=rope_preflight,
        runtime_rope=runtime_rope, data_sha256=data_sha256,
        data_path=data_path, eos_ids=eos_ids,
    )
    print(f"[RULER DONE] {generation_pointer_path(str(out_path))}", flush=True)


if __name__ == "__main__":
    main()
