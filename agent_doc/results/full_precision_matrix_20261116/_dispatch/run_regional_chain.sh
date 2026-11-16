#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: $0 <GPU> <c2|c3>}"
ARM="${2:?usage: $0 <GPU> <c2|c3>}"
EXPECTED_HEAD="${EXPECTED_HEAD:?set EXPECTED_HEAD to the clean producer commit}"
WT="${WT:-/tmp/nfa-dac-ruler-regional-${EXPECTED_HEAD:0:8}}"
RUNNER="$WT/agent_doc/results/nfa_dac_experiments_20261115/_dispatch/run_ruler_regional_methods.sh"
case "$ARM" in
  c2) CELLS=(c2_32768 c2_65536 c2_131072) ;;
  c3) CELLS=(c3_32768 c3_65536 c3_131072) ;;
  *) printf 'arm must be c2 or c3\n' >&2; exit 3 ;;
esac
[[ -x "$RUNNER" ]] || { printf 'missing executable regional runner %s\n' "$RUNNER" >&2; exit 3; }
for CELL in "${CELLS[@]}"; do
  for SHARD in 0 1; do
    bash "$RUNNER" "$GPU" "$CELL" "$SHARD"
  done
done
