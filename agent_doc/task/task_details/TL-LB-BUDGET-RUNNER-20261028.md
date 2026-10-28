# TL-LB-BUDGET-RUNNER-20261028

## Scope / decision

Implement the requested remote benchmark orchestration, not attention algorithm
changes or a resumption of unrelated audit patches. Final user scope: reference
LongBench13, explicitly selected Qwen3-8B/32B, Full/LoDA/Quest/JSSA/AsyncTLS;
LoDA K1 fixed (128 default), nominal K2 512/1024/2048. Current LongBench producer,
scorer and run-cap receipt APIs are reused. No real GPU jobs launched here.
Latest explicit publication instruction permits a normal direct push to main;
no force push, merge of unrelated PRs, SGLang or workflow-repository modifications.

Base inspected/fetched: `e6f0cc9e79f1212d04c3f91577f7ff33255dee1c`.
Read CLAUDE, human GUIDE (unchanged), current narrative, task index, producer,
scorer, receipt/lock APIs, launch scripts, JSSA calibration note and open PRs.
Historical LongBench summary paths/row counts do not prove this sweep's matching
code/model/input/config identity. They are not automatically skipped or relabelled.

## Implementation

- `benchmark/LongBench/run_budget_sweep.py`
- `docs/longbench_budget_sweep.md`
- `docs/longbench_budget_sweep.example.json`
- `tests/test_longbench_budget_sweep.py`

Preparation/dry-run are separate from explicit execution. Model/tokenizer/config
inventories, data and calibration bytes, code revision/local source, Python and
worker environment/dependency versions are bound. A pre-GPU CPU protocol check
uses actual argument/snapshot/naming helpers. Successful reuse requires matching
complete receipt, source IDs/answers, actual treatment/name, prediction/run-cap and
score hashes. Failed/incomplete attempts remain recoverable. Shared-local UUID
flock, locked stable NVIDIA XML idle window, bounded waits/runtime and a Linux
parent-death guardian cover cooperative admission/managed process cleanup.
Advisory locks do not exclude noncooperating processes; no hardware exclusivity or
GPU latency claim. Defaults never install dependencies or start daemon schedules.

Current committed tree lacks AsyncTLS/JSSA adapter files referenced by producer;
remote-local copies may exist (checked and hashed on execution host). Missing
selected arms remain blocked, never substituted. Explicit partial execution stays
incomplete with exit 2. RULER10 and tensor-parallel execution are out of scope.

## Independent review

Separate host agent `review_plan` approved revised plan before implementation.
Initial implementation review found two blockers: newly added model/config files
were not detected; actual K1/K2 treatment was not bound. Both fixed and covered by
regressions. Subsequent review required truthful model architecture/label and
removing the unused baseline default skip-mask naming dependency; both resolved.
Final review: **approved for publication of CPU-tested orchestration only**, no
unresolved blocker. Independently reran normal 18/18 and optimized 18/18 tests,
plus py_compile. The reviewer also verified real exited-leader descendant cleanup.
A final comment clarifies duplicate-plan exclusion is per shared output namespace.

Final runner SHA256:
`6c5af28cbc52fb3c01cc4c83d102d777c14d7186e40470a96cfdfaa6ec629480`.
Tests SHA256:
`346c0a88ffa09f1699d2d73c2ae07314df0ba6fb4355da3386ed13b8da9394a1`.

## Actual validation

Environment: Python 3.13, standard library; no Torch/pytest installed. No dependency
installation and no model downloads. Commands (from checkout root):

```sh
python -m unittest discover -s tests -p test_longbench_budget_sweep.py -v
python -O -m unittest discover -s tests -p test_longbench_budget_sweep.py -v
python -m unittest discover -s tests -p 'test_lbv1*.py' -v
python -m unittest discover -s tests -p test_longbench_prompt_split.py -v
python -m unittest discover -s tests -p test_longbench_empty_handoff.py -v
# Also repeat the preceding three existing suites with python -O.
python -m compileall -q benchmark/LongBench/run_budget_sweep.py tests/test_longbench_budget_sweep.py
git diff --check
```

- New runner: **18 passed** per mode; includes mocked NVIDIA errors, stable recheck,
  real two-process lock contention, timeout/missing command, parent death, exited
  leader descendants, strict real-schema completion/corruption, failed scorer and
  restart, partial status, model/config additions and explicit model scopes.
- Existing adjacent tests: **17 + 5 + 6 = 28 passed** per mode.
- Actual CLI/snapshot/naming/far-skip helper AST contracts tested without executing
  heavy imports. This is not a real Torch integration pass.
- Existing `run_cap_manifest.compute_result_binding` emits a ResourceWarning for
  an unclosed read handle; no unrelated production fix included.
- Compile and whitespace checks passed.

**Unrun, not passed:** real producer/scorer dependency integration, NVIDIA driver
admission, pretrained LongBench quality, GPU execution/performance, and repository
GPU hard gates `exp/bench/test_e127_gates.py` / `test_tli_batch_gates.py`.
No completed benchmark results, effective-budget equivalence or speedup claimed.
