#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: GPU RANK}"
RANK="${2:?usage: GPU RANK}"
WORLD=6
EXPECTED_HEAD=8ec4e66de07a4f84432da0f8461e45cbfbc05040
EXPECTED_KERNEL_SHA=de7d6e32052fd63206c7ba20b54d9040c40f8dc6617eb2a392c80e0bcc1b6b1f
REPO=/tmp/official-lbv2-quest-8ec4e66d-20261114
OUT=/home/wangyuanshuo02/ak_verify_results/quest_lbv2_official_8ec4e66d_20261115
KERNEL="$REPO/third-party/Quest/quest/_kernels.cpython-311-x86_64-linux-gnu.so"

actual_head="$(git -C "$REPO" rev-parse HEAD)"
[[ "$actual_head" == "$EXPECTED_HEAD" ]]
actual_kernel_sha="$(sha256sum "$KERNEL" | cut -d' ' -f1)"
[[ "$actual_kernel_sha" == "$EXPECTED_KERNEL_SHA" ]]
[[ -f /home/wangyuanshuo02/datasets/LongBench/data/lbv2.json ]]
[[ -d /home/wangyuanshuo02/Qwen3-8B ]]
cd "$REPO"
mkdir -p "$OUT/_logs" "$OUT/_status" "$OUT/rank$RANK"

printf '===== START host=%s gpu=%s rank=%s world=%s head=%s time=%s =====\n' \
  "$(hostname)" "$GPU" "$RANK" "$WORLD" "$actual_head" "$(date --iso-8601=seconds)"
if CUDA_VISIBLE_DEVICES="$GPU" /home/fluentllmenv/bin/python \
  -m benchmark.official_baselines.longbench_official \
  --method quest \
  --mode quality \
  --model-path /home/wangyuanshuo02/Qwen3-8B \
  --model-name Qwen3-8B \
  --data-root /home/wangyuanshuo02/datasets/LongBench/data \
  --config-path "$REPO/benchmark/LongBench/config" \
  --output-dir "$OUT/rank$RANK" \
  --tasks lbv2 \
  --shard-index "$RANK" \
  --num-shards "$WORLD" \
  --device cuda:0 \
  --budget 1024 \
  --quest-page-size 16; then
  printf '===== DONE host=%s gpu=%s rank=%s time=%s =====\n' \
    "$(hostname)" "$GPU" "$RANK" "$(date --iso-8601=seconds)"
  date --iso-8601=seconds > "$OUT/_status/rank$RANK.done"
else
  rc=$?
  printf '===== FAILED host=%s gpu=%s rank=%s rc=%s time=%s =====\n' \
    "$(hostname)" "$GPU" "$RANK" "$rc" "$(date --iso-8601=seconds)"
  printf '%s\n' "$rc" > "$OUT/_status/rank$RANK.failed"
  exit "$rc"
fi
