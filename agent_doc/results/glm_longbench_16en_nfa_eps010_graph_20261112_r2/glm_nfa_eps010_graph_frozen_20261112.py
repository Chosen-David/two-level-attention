#!/usr/bin/env python3
"""冻结快照的 identity 门禁；真实 worker 必须匹配预登记 identity。"""
import os
from pathlib import Path
import sys

REPO = Path(os.environ.get("GLM_REPO", "/tmp/tli-glm-nfa-eps010-graph-20261112"))
sys.path.insert(0, str(REPO))
import benchmark.LongBench.prefill_frozen_skip as target  # noqa: E402

expected_identity = os.environ.get("GLM_EXPECT_IDENTITY")
identity_only = os.environ.get("GLM_IDENTITY_ONLY") == "1"
if not identity_only and not expected_identity:
    raise ValueError("GLM_EXPECT_IDENTITY is required for real workers")

original_atomic_json = target.atomic_json


def checked_atomic_json(path, value):
    if Path(path).name.startswith("manifest_"):
        identity = value.get("identity")
        if identity_only:
            print(f"GLM_IDENTITY={identity}", flush=True)
            raise SystemExit(0)
        if identity != expected_identity:
            raise RuntimeError(
                f"NFA identity mismatch: actual={identity} expected={expected_identity}")
    original_atomic_json(path, value)


target.atomic_json = checked_atomic_json
target.main()
