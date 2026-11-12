#!/usr/bin/env python3
"""在冻结 /tmp 快照上运行原生 LongBench 分片；仅登记原 worker 的 commit。"""
import os
from pathlib import Path
import sys

REPO = Path(os.environ.get("GLM_REPO", "/tmp/tli-glm-jssa-prod-20261106"))
sys.path.insert(0, str(REPO))
import benchmark.LongBench.prefill_frozen_skip as target  # noqa: E402

git_head = os.environ["GIT_HEAD"]
expected_identity = os.environ.get("GLM_EXPECT_IDENTITY")
identity_only = os.environ.get("GLM_IDENTITY_ONLY") == "1"
if not identity_only and not expected_identity:
    raise ValueError("GLM_EXPECT_IDENTITY is required for real workers")

original_check_output = target.subprocess.check_output


def check_output_with_snapshot_head(command, *args, **kwargs):
    if command == ["git", "rev-parse", "HEAD"]:
        return git_head + "\n" if kwargs.get("text") else (git_head + "\n").encode()
    return original_check_output(command, *args, **kwargs)


target.subprocess.check_output = check_output_with_snapshot_head
original_atomic_json = target.atomic_json


def checked_atomic_json(path, value):
    if Path(path).name.startswith("manifest_"):
        identity = value.get("identity")
        if identity_only:
            print(f"GLM_IDENTITY={identity}", flush=True)
            raise SystemExit(0)
        if identity != expected_identity:
            raise RuntimeError(
                f"16-task identity mismatch: actual={identity} expected={expected_identity}")
    original_atomic_json(path, value)


target.atomic_json = checked_atomic_json
target.main()
