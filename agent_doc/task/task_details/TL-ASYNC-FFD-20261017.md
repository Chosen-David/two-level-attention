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
