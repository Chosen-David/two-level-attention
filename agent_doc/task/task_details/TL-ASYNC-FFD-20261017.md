# TL-ASYNC-FFD-20261017

User scope: archive AsyncTLS/FFD papers in existing other_paper directory; reuse official code, otherwise labeled paper reproduction and benchmark entry. Paper metadata may publish main; engineering only gpt_dev/paper-aligned-runtime-20261015. No GPU, paid execution, model download, SGLang or DAC changes.

Acceptance: source/license and exact bytes when present; no fabricated PDF; actual model/phase entry; fail-closed calibration/contracts; CPU normal/-O and independent review; no claim of real-model quality or full AsyncTLS where absent.

Method: reuse existing native FFD Llama vendor; synchronous AsyncTLS §4.1 Qwen3 component explicitly separate from system; immutable calibration/checkpoint identity. See [implementation and missing inputs](../../advice/2026-10-17_asynctls_ffd_integration_by_gpt.md).

## Progress

- Main metadata 36285019bf39b22ba58d767c8ff46a400ad69531 independently reviewed; AsyncTLS existing PDF unchanged, FFD PDF download HTTP403 and original still missing. Readback matches.
- Official FFD vendored source hashes match. Independent review found missing append position guard; fixed before cache updates. Requirements separately documented, vendor unchanged.
- Component candidate wired to both benchmark loaders with explicit method identity. Independent review required checkpoint binding, prefill padding guard, failed-cache protection; added before final validation.
- Remaining: FFD original bytes; real held-out calibration and model/GPU acceptance; full asynchronous system unimplemented. Not a completed scientific baseline. No autonomous collection or experiment expansion.
- Final independent review `/root/independent_review`: usable-with-scope, no remaining blocker; independently ran component 4+4 and FFD 6+6 normal/-O. Source/inputs/environment binding and logs in `agent_doc/results/asynctls-ffd-20261017/contract.json`. Real scientific acceptance remains open.

## Scope revision v2 — 2026-10-17

- User authorized full paper-derived AsyncTLS §4.2 work; synchronous component retained. Added previous-step resident selection, fixed host store, delta transfers, CUDA stream/event lifecycle and fail-closed contracts. Explicit CUDA opt-in remains unvalidated; no full-system acceptance claim. Existing TIA audited: mask lag exists, transport does not; fixed-channel/quantization/SWA differences retained rather than silently replacing the historical route.
- Producer and independent reviewer: CPU offload 6 normal + 6 optimized; affected component 4+4 and FFD 6+6 pass. No GPU jobs or model downloads. Source/evidence in `agent_doc/results/asynctls-offload-20261017/contract.json`.
- FFD original publisher PDF subsequently archived main57b30397; prior missing-PDF statement is historical.
- Latest quality scope is RULER32K + existing LongBench lengths; existing64/128K negative data retained for supplement. Kernel32/64/128K; E2E32/64/96K. No1M/new-model/long-quality matrix and no128K-lossless claim from32K quality.
- Real held-out calibration, GPU transport/numerics/performance and scientific acceptance remain open. Engineering publishes only original dev; user owns main merge.

## Bounded priority clarification

Dry-run output separates `ruler_quality_lengths=[32768]` from LongBench actual existing task lengths. Prioritize a few frontier baselines: representation discussion JSSA/FASA/ours; full-method discussion Quest/AsyncTLS/ours + FullKV. Complete JSSA is a potential competitor only, not implementation authorization. No further SparQ/HISA integration; preserve existing code/history and known regression failures, and do not stop host jobs. FFD remains backup: no Qwen port or new Llama matrix. This clarification starts no experiment or new integration.
