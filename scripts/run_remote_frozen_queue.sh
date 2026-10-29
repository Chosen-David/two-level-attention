#!/usr/bin/env bash
set -u
# This host-side watchdog survives SSH logout. Worker state is durable on disk.
python_bin="$1"
source_root="$2"
plan="$3"
host_key="$4"
export TLI_SCORER_BACKEND=difflib
export PYTHONPATH="$source_root${PYTHONPATH:+:$PYTHONPATH}"
while true; do
  "$python_bin" "$source_root/scripts/remote_frozen_queue.py" "$plan" --host "$host_key"
  code=$?
  if [ "$code" -eq 0 ]; then exit 0; fi
  sleep 60
done
