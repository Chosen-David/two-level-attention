#!/usr/bin/env bash
# 本机 BS1 decode-stage e2e 补格监督器：每 1h 推进一轮，全部基线格闭集后汇总并推送 origin/main。
set -uo pipefail
RUNNER=/tmp/run_bs1_decode_only_abba_local_20261117.py
REPO=/tmp/nfa-bs1-attribution-20261117
OUT=/tmp/nfa-bs1-attribution-20261117/agent_doc/results/nfa_bs1_decode_only_20261117
MAIN=/home/wangyuanshuo02/two-level-attention
INTERVAL=3600

missing_cells() {
  OUT="$OUT" /usr/bin/python3 - <<'PY_MISSING'
import os
out=os.environ["OUT"]
methods=("asynctls","jssa","quest")  # NFA 32K/64K natural gate 全空，fail-closed，见 diagnostics
lengths=(32768,65536)
passes=("forward","reverse")
for L in lengths:
    for m in methods:
        for pn in passes:
            rp=os.path.join(out,"clean",f"{m}_L{L}_{pn}.receipt.json")
            if not os.path.exists(rp):
                print(f"{m} {L} {pn}")
PY_MISSING
}

run_gpu_chain() {
  local gpu=$1 length=$2
  echo "$(date -Is) gpu$gpu length=$length baselines start"
  /usr/bin/python3 "$RUNNER" --repo "$REPO" --output "$OUT" --gpu "$gpu" \
    --lengths "$length" --passes both --methods asynctls jssa quest
}

while true; do
  missing=$(missing_cells)
  echo "==== $(date -Is) missing baseline cells: $(echo "$missing" | tr '\n' ' ')"
  if [[ -z "$missing" ]]; then
    echo "ALL_BS1_32K64K_BASELINE_CELLS_DONE $(date -Is)"
    break
  fi
  gpu0_busy=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  gpu1_busy=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  need32=$(echo "$missing" | grep -c ' 32768 ' || true)
  need64=$(echo "$missing" | grep -c ' 65536 ' || true)
  if [[ "$need32" -gt 0 && -z "$gpu0_busy" ]]; then
    (run_gpu_chain 0 32768 > /tmp/bs1_local_gpu0_32768.supervisor.log 2>&1 &)
  fi
  if [[ "$need64" -gt 0 && -z "$gpu1_busy" ]]; then
    (run_gpu_chain 1 65536 > /tmp/bs1_local_gpu1_65536.supervisor.log 2>&1 &)
  fi
  sleep "$INTERVAL"
done

# 汇总 32K/64K baseline clean cells
OUT="$OUT" /usr/bin/python3 - <<'PY_SUMMARY'
import json, os
from pathlib import Path
out=Path(os.environ["OUT"])
rows=[]
for L in (32768,65536):
    for m in ("asynctls","jssa","quest"):
        for pn in ("forward","reverse"):
            rp=out/"clean"/f"{m}_L{L}_{pn}.receipt.json"
            if rp.exists():
                rows.append(json.loads(rp.read_text()))
summary={"schema":"bs1-decode-e2e-32k64k-baseline-v1",
         "verification":"pending-independent-verification",
         "cells":len(rows),"rows":rows}
(out/"bs1_32k64k_baseline_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print("SUMMARY_DONE",len(rows))
PY_SUMMARY
echo "BUILD_SUMMARY_DONE $(date -Is)"

# 归档到主仓并推送 origin/main（只提交本轮结果与任务文档）
ARCH="$MAIN/agent_doc/results/nfa_bs1_decode_only_20261117"
mkdir -p "$(dirname "$ARCH")"
rsync -a --exclude _dispatch "$OUT"/ "$ARCH"/
python3 - <<'PY_PUSH'
from pathlib import Path
p=Path('/home/wangyuanshuo02/two-level-attention/agent_doc/task/TASK.md')
s=p.read_text()
marker='## 2026-11-17'
entry=('- [ ] [TL-QUEST-BATCH-RULER-PERF-20261116/BS1-DECODE-ONLY] 32K/64K BS1 decode-stage e2e 补格：'
       'Quest/JSSA/AsyncTLS 已闭集，NFA 32K/64K natural gate 全空 fail-closed；'
       '结果归档 `agent_doc/results/nfa_bs1_decode_only_20261117/`，保持 pending-independent-verification。\n')
if entry not in s:
    s=s.replace(marker, entry+marker, 1)
    p.write_text(s)
PY_PUSH
cd "$MAIN"
git add agent_doc/results/nfa_bs1_decode_only_20261117 agent_doc/task/TASK.md \
  agent_doc/task/task_details/TL-QUEST-BATCH-RULER-PERF-20261116.md
git commit -m "feat(results): add BS1 decode-stage e2e 32K/64K baseline round (pending verification)" \
  && git push origin main
echo "PUSHED_TO_ORIGIN_MAIN $(date -Is)"
