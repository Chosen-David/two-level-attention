#!/usr/bin/env python3
"""GLM NFA r2 正式分片只读状态汇总。"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path("/home/wangyuanshuo02/ak_verify_results/glm_longbench_16en_nfa_eps010_graph_20261112_r2")
now = datetime.now(timezone.utc).timestamp()
summary = {"time": datetime.now().astimezone().isoformat(), "ranks": {}}
for rank in map(int, sys.argv[1:]):
    heartbeat_path = root / f"heartbeat_rank{rank}.json"
    done_path = root / f"worker_done_rank{rank}.json"
    manifest_path = root / f"manifest_rank{rank}.json"
    heartbeat = json.loads(heartbeat_path.read_text()) if heartbeat_path.is_file() else None
    log_path = Path(f"/tmp/glm_nfa_eps010_graph_lb16r2_rank{rank}.log")
    lines = log_path.read_text(errors="replace").splitlines() if log_path.is_file() else []
    relevant = [line for line in lines if "Loading weights:" not in line]
    summary["ranks"][str(rank)] = {
        "manifest": manifest_path.is_file(),
        "done": done_path.is_file(),
        "heartbeat": heartbeat,
        "heartbeat_age_s": round(now - heartbeat["time"], 1) if heartbeat else None,
        "log_tail": relevant[-4:],
    }
summary["request_files"] = len(list(root.glob("requests/*/*.json")))
meminfo = Path("/proc/meminfo").read_text().splitlines()
summary["mem_available_mib"] = next(int(line.split()[1]) // 1024 for line in meminfo if line.startswith("MemAvailable:"))
print(json.dumps(summary, ensure_ascii=False))
