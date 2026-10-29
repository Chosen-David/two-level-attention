# Partial attention address width

SIMT and Tensor Core partial attention now promote the program row and physical
KV token index to int64 before multiplying by tensor strides. Merge row offsets
use int64 as well. Valid large batch or noncontiguous KV layouts can exceed
2^31 elements; widening after multiplication would preserve an already-overflowed
address. Floating-point attention arithmetic is unchanged.

This change was transplanted onto main `5fff867` as `e756261`; tests are in
`d9a9677`. Final-source native validation passed 62 tests on reserved H20 GPU1,
including four one-token closed-form probes: B64/S65528/Hkv8 and token stride
2^30, each with SIMT and TC. CPU-only collection passed 14 and skipped 48 CUDA
cases. Independent review checked the final delta, source hashes and raw results.
Independent original-source reruns passed all four address probes and one strict
SIMT token-stride memcheck with zero reported errors; that memcheck was not rerun
on final main and does not certify unrelated experimental ring/TC changes.

Local evidence: `agent_doc/results/operator_partial_address_main_20261030/`
and `agent_doc/results/operator_partial_address_20261030/`. This is a correctness
fix, with no new performance or model-quality claim. Geometry performance numbers
in `operator_geometry_capture.md` remain tied to their stated measured revision.
