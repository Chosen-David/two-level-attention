#!/bin/bash
# E155d T2 收割：五远程机 + 本机 → ~/tmp_e155d_collect/<host>/ 快照。
# 幂等可重跑（tar 覆盖式快照）；在飞格的部分文件照收，完整性门在
# analyze_e155d_grid.py 侧判（5 任务 × 行数门 200/500）。
# 排除零字节 .runcap_gate.lock；包含 .jsonl / .runcap.json / .tli_manifest.json。
set -u
export SSHPASS='zHWXp=eK5JvX~I|v'
ROOT=$HOME/tmp_e155d_collect
mkdir -p "$ROOT"

collect_remote () {  # <ip> <name>
  local IP=$1 NAME=$2
  echo "===== collect $NAME ($IP) ====="
  mkdir -p "$ROOT/$NAME"
  timeout 120 sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    -p 8416 wangyuanshuo02@$IP \
    "cd ~/tmp_e155_grid 2>/dev/null && \
     tar czf - --exclude='*.runcap_gate.lock' \$(ls -d pred_* 2>/dev/null) 2>/dev/null" \
    | tar xzf - -C "$ROOT/$NAME" 2>/dev/null && echo "$NAME OK" \
    || echo "$NAME FAILED/EMPTY"
}

collect_remote 10.238.122.187 pc02
collect_remote 10.238.139.251 pc8100
collect_remote 10.245.58.57   pc31
collect_remote 10.245.84.251  pc32
collect_remote 10.238.9.187   dot187

# 本机（直接快照，不动原目录）
echo "===== collect local ====="
mkdir -p "$ROOT/local"
(cd ~/tmp_e155_grid && tar czf - --exclude='*.runcap_gate.lock' \
   $(ls -d pred_* 2>/dev/null)) | tar xzf - -C "$ROOT/local" && echo "local OK"

echo "===== 快照统计 ====="
for d in "$ROOT"/*/; do
  n=$(ls "$d" 2>/dev/null | grep -c '^pred_')
  echo "$(basename "$d"): $n 格"
done
echo "done $(date '+%F %T')"
