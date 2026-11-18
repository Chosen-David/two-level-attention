"""将官方 Quest CUDA 的完整进程 nsys 轨迹归因到三个 decode 阶段。

生产入口仍使用 long_context_official.py；本脚本只消费其输出与 nsys SQLite。
profiler 下的 TPOT 永久隔离，不与 clean-e2e 混排或相减推导 host 开销。
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark.efficiency.measure_flashinfer_operator import _kernel_instances


CATEGORY_TOKENS = {
    "estimate": "MaxPossibleSampleWithPagedKVCacheKernel",
    "topk_raft": "raft::matrix::detail::select::radix::impl::radix_topk_one_block_kernel",
    "topk_canonicalize": "canonicalize_topk_kernel",
    "sparse_partition": "BatchDecodeWithPagedKVCacheKernel",
    "sparse_merge": "VariableLengthMergeStatesKernel",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _timing(values, steps):
    return {
        "calls_total": len(values),
        "calls_per_step": len(values) / steps,
        "total_ms": sum(values),
        "mean_ms_per_step": sum(values) / steps,
        "mean_instance_ms": statistics.mean(values),
        "median_instance_ms": statistics.median(values),
        "min_instance_ms": min(values),
        "max_instance_ms": max(values),
    }


def finalize(result, sqlite_path, nsys_rep):
    if result.get("method") != "quest" or result.get("status") != "complete":
        raise ValueError("不是完整官方 Quest 运行回执")
    if result.get("warmup_requests") != 0 or result.get("repeats") != 1:
        raise ValueError("官方 operator 口径固定为 0 warmup + 1 repeat")
    if result.get("max_new_tokens") != 33:
        raise ValueError("官方 operator 口径固定为 33 token")
    if not result.get("runtime", {}).get("implementation", "").startswith(
        "official Quest CUDA"
    ) or result["runtime"].get("dense_or_torch_fallback") is not False:
        raise ValueError("回执不是 fail-closed 官方 Quest CUDA runtime")

    steps = result["max_new_tokens"] - 1
    batch_size = result["batch_size"]
    expected_counts = {
        "estimate": 34 * steps,
        # 官方 RAFT 按 batch 行各启动一个 one-block kernel；其余五类按层调用。
        "topk_raft": 34 * steps * batch_size,
        "topk_canonicalize": 34 * steps,
        "sparse_partition": 36 * steps,
        "sparse_merge": 36 * steps,
    }
    recorded = result["quest_call_counts"]
    if recorded != {
        "prefill_forward": 72 * result["batch_size"],
        "decode_estimate": 34 * steps,
        "decode_topk": 34 * steps,
        "decode_sparse_attn": 36 * steps,
        "dense_or_torch_fallback": 0,
    }:
        raise ValueError(f"官方 Quest 调用计数不符合单次 T(1)+T(33): {recorded}")

    instances = _kernel_instances(Path(sqlite_path))
    categories = defaultdict(list)
    members = defaultdict(set)
    for name, values in instances.items():
        for category, token in CATEGORY_TOKENS.items():
            if token in name:
                categories[category].extend(values)
                members[category].add(name)
    for category, expected in expected_counts.items():
        actual = len(categories[category])
        if actual != expected:
            raise RuntimeError(
                f"{category} kernel 调用数 {actual} != {expected}，拒绝错误归一化"
            )

    combined = {
        "decode_estimate": categories["estimate"],
        "decode_topk": categories["topk_raft"] + categories["topk_canonicalize"],
        "decode_sparse_attn": categories["sparse_partition"] + categories["sparse_merge"],
    }
    result["measurement_mode"] = "nsys-cuda-full-process-operator-profile"
    result["nsys_profile_finalized"] = True
    result["profiler_timing_quarantined"] = True
    result["profiled_decode_steps"] = steps
    result["operator_summary"] = {
        **{f"{name}_ms": _timing(values, steps) for name, values in combined.items()},
        "topk_raft_ms": _timing(categories["topk_raft"], steps),
        "topk_canonicalize_ms": _timing(categories["topk_canonicalize"], steps),
        "sparse_partition_ms": _timing(categories["sparse_partition"], steps),
        "sparse_merge_ms": _timing(categories["sparse_merge"], steps),
        "members": {key: sorted(value) for key, value in members.items()},
        "scope": (
            "Quest is a single-level selector: decode_estimate plus RAFT/input-order "
            "Top-K is the selector; sparse-attn includes partition attention and "
            "VariableLengthMergeStates. Prefill and KV append are excluded."
        ),
        "implementation": "official Quest CUDA _kernels; no Torch/dense fallback",
    }
    result["selected_kernel_summary"] = {
        category: _timing(values, steps)
        for category, values in sorted(categories.items())
    }
    result["nsys_receipt"] = {
        "version": subprocess.check_output(["nsys", "--version"], text=True).strip(),
        "trace": "cuda full process; unique decode kernel names and exact call counts",
        "report_path": str(Path(nsys_rep).resolve()),
        "report_sha256": _sha256(nsys_rep),
        "sqlite_path": str(Path(sqlite_path).resolve()),
        "sqlite_sha256": _sha256(sqlite_path),
        "observer_effect": (
            "Profiler timings are observed kernel times only; request TPOT in this "
            "receipt is quarantined and must not enter clean-e2e rankings."
        ),
    }
    result["verification"] = "pending-independent-verification"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--nsys-rep", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    result = finalize(json.loads(output.read_text()), args.sqlite, args.nsys_rep)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "profiled_decode_steps": result["profiled_decode_steps"],
        "operator_summary": {
            key: value["mean_ms_per_step"]
            for key, value in result["operator_summary"].items()
            if key.endswith("_ms") and isinstance(value, dict)
        },
    }, indent=2))


if __name__ == "__main__":
    main()
