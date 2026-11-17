#!/usr/bin/env bash
set -euo pipefail
WT="$HOME/tla-lb-k2-20261117-v2"
EXPECTED_HEAD=8736a965e903fd461ef5591347933205e18adee0
ROOT="$HOME/ak_verify_results/qwen3_lb_k2_ablation_8736a965_20261117"
DATA="$HOME/sparse-bench/third_party/KVCache-Factory/data/LongBench"
mkdir -p "$ROOT/_logs" "$ROOT/_status"
if [[ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED_HEAD" ]]; then
  printf 'unexpected source HEAD\n' >&2
  exit 3
fi
if [[ -n "$(git -C "$WT" status --short)" ]]; then
  printf 'source worktree is not clean\n' >&2
  exit 3
fi
run_cell() {
  local gpu="$1" task="$2" k="$3"
  local out="$ROOT/${task}_k2_${k}"
  local log="$ROOT/_logs/${task}_k2_${k}_gpu${gpu}.log"
  local lock="$ROOT/_status/${task}_k2_${k}_gpu${gpu}.lock"
  local donef="$ROOT/_status/${task}_k2_${k}_gpu${gpu}.done"
  exec 9>"$lock"
  if ! flock -n 9; then
    printf 'cell %s k2=%s gpu=%s already locked\n' "$task" "$k" "$gpu" >&2
    return 0
  fi
  if [[ -e "$donef" || -e "$out/quality_manifest_s0of1.json" ]]; then
    printf 'skip %s k2=%s existing complete\n' "$task" "$k" >&2
    return 0
  fi
  printf 'START task=%s k2=%s gpu=%s pid=%s time=%s\n' "$task" "$k" "$gpu" "$$" "$(date -Is)" >"$log"
  env -C "$WT" PYTHONPATH="$WT" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" \
    /usr/bin/python -m benchmark.efficiency.benchmark_longbench_quality \
    --method nfa --execution eager --nfa-subspace full --nfa-subspace-dim 128 \
    --nfa-gate off --nfa-level2-topk "$k" \
    --model-path "$HOME/Qwen3-8B" --model-name Qwen3-8B \
    --data-root "$DATA" --output-dir "$out" \
    --tasks "$task" --device cuda:0 --visible-len 8192 >>"$log" 2>&1
  if [[ ! -f "$out/quality_manifest_s0of1.json" ]]; then
    printf '%s k2=%s exited without manifest\n' "$task" "$k" >&2
    exit 6
  fi
  printf 'DONE task=%s k2=%s gpu=%s time=%s\n' "$task" "$k" "$gpu" "$(date -Is)" >"$donef"
}
run_cell 0 gov_report 256 &
run_cell 0 hotpotqa 256 &
run_cell 0 hotpotqa 512 &
run_cell 1 gov_report 512 &
run_cell 1 hotpotqa 1024 &
run_cell 1 hotpotqa 2048 &
run_cell 2 gov_report 1024 &
run_cell 2 gov_report 2048 &
wait
