# Static vs dynamic calibration plan

Goal: verify whether task-aware online prefill calibration + decode freeze (static/frozen region gates) loses accuracy relative to per-layer dynamic far-skip calibration, for GLM and Qwen, on RULER `niah_multikey_2` and LongBench AVG5.

Current code facts (clean origin/main `42821f81...`):
- Static frozen gates are available for Qwen3 and GLM via `prefill_frozen_request` with `GateConfig(mode='full|far|near|dual')`.
- Qwen3 has legacy `dyn_dgate` support in `qwen3_attn_patch.py`; RULER `pred_ruler.py` can run non-frozen TLI with `--tli_dyn_dgate`.
- GLM dynamic path is intentionally disabled in current PagedRegionCache path (`glm4_moe_lite_attn_patch.py` raises when `dyn_dgate` with region cache); needs a separate legacy DynamicCache runner or careful patch restore before it can be measured.

Planned arms:
- Qwen3-8B static LongBench/RULER: existing frozen-region runners.
- Qwen3-8B dynamic LongBench/RULER: run `pred_ruler.py --tli_dyn_dgate` for RULER; add a small manual `DynamicCache` LongBench runner for Qwen.
- GLM static: `glm_subspace_runner.py --gate-mode dual` (records decisions).
- GLM dynamic: restore/implement legacy GLM dynamic path, then same paired AVG5/RULER subset.

Resources: queue after current GLM subspace screen wave, using freed H20 GPUs; local two GPUs untouched.
