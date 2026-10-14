# E1 paired quality CPU candidate — Opus handoff

Base: `2433b5209512536bd9d4d6e5d364c375b120059b`, including sink protection fix after `0288ed5`.
CPU diagnostic tooling only; no production conclusion. S-T019/S-T022 remain pending;
baselines remain DeepSeek's responsibility. No dynamic research or new model/GPU jobs.

## Existing evidence and exact missing inputs

E134 sub/sub and `exp/trace/results/e123_trial_raw/pred_fullkv` contain five
matching-ID/answers/length task pairs: qasper/hotpotqa/gov_report/musique (200
samples each), repobench (500). These are available historical predictions,
not absent measurements. Existing `_id` binds answers/length fingerprint plus
row index, not actual prompt/input tokens or independent document identity.
E123 dense lacks inline run-cap sidecars. Do not fabricate these or use row_idx
as document_id. No main-text paired document CI is currently accepted.

Recover original document IDs, final input token hashes, dense runtime cap,
model/tokenizer/data revisions, phase, dtype, generation/stop settings, protection,
producer commit/files and actual L1/L2 configuration receipts. Recover scorer
backend/hash and preregistration provenance. The sink fix must not be silently
mixed across runs. Current tool deliberately requires equal producer identity;
legitimate differing producer revisions require separate reviewed evidence,
not weakening the gate. Check existing completed records before rerunning.
RULER has formal raw/manifest/receipt chains, but E119's identity is not by itself
proof of sub/sub phase/dtype/document identity. This entry supports LongBench
only; continue RULER through existing E119, do not translate aggregate scores.

## Commands after data recovery

```bash
python -m unittest discover -s exp/figures/tests -v
python -O -m unittest discover -s exp/figures/tests -v
MPLCONFIGDIR=/tmp/tla-mpl XDG_CACHE_HOME=/tmp/tla-cache \
python exp/figures/paired_quality.py \
  --protocol /approved/private/e1/protocol.json \
  --out /approved/private/e1/paired-01 --plot
python exp/figures/figure_data_contract.py \
  --bindings /approved/private/e1/paired-01/figure_bindings.json \
  --check /approved/private/e1/paired-01/figure_data.tsv
```

Paths are placeholders. Use the existing CPU environment with numpy/jieba/rouge;
matplotlib needed only for plot. New output directory required; Linux atomic
publication reuses existing figure helper. Failed stages remain hidden.

Input schema is `paired-longbench-v1`, status frozen, run_id, run_mode natural,
bootstrap_draws (100–100000), integer seed, arms {dense,sub_sub} with selections
`dense_fullkv`/`sparse_sub_sub`. Each identity supplies all string fields in
`paired_quality.IDENTITY`; complex effective configurations should be canonical
serialized strings from original evidence, never guessed constants.
`scorer_sha256` hashes concatenated bytes of eval.py, metrics.py, lbv2_choice.py
in that order. Each tasks[official_task] contains dense/sub_sub source objects
(path, sha256, runcap_sha256) and samples keyed by original `_id`. Each sample
has document_id, input_ids_sha256 and record_sha256_dense/sub_sub (SHA256 of
json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False). Independent
source verification of the token/document mapping is still required. JSON field
hashes are bindings, not authentication. Cross-task shared documents are rejected
because this implementation resamples independently by task.

`noninferiority_delta_pp` defaults null. If supplied, it must map every task to
nonnegative finite percentage-point tolerances, with preregistration {path,
sha256} pointing to JSON containing matching noninferiority_delta_pp and
frozen_before_run evidence reference. Arithmetic lower-bound checks are reported,
but chronology cannot be proven by this self-declared file: independent review
must verify it predates confirmation. No winner or noninferiority acceptance is
automatically emitted, even when all arithmetic checks pass. Do not enlarge
thresholds after seeing results.

Official score logic is factored into eval.sample_score and reused by existing
scorer (unchanged rounding/normalization). Bootstrap follows E100 task-stratified
macro aggregation, correcting its positional/answers matching and using whole
document clusters; sum/count ratio preserves within-task sample weighting.
Report per-task point estimate, marginal percentile 95% CI, independent document
N, planned/valid/excluded counts, excluded IDs/reasons, worst point-estimate task,
and macro difference. Missing/failed samples or <2 docs block all reported CIs
instead of silently conditioning the main result on successes. No zero imputation.
CI crossing zero does not prove losslessness. Marginal CIs are not simultaneous
intervals. Raw predictions and real values remain unchanged.

Existing assigned cards may fill only genuinely missing already-authorized cells:
Opus checks allocation/owner/locks/VRAM/dependencies/budget first and parallelizes
ready independent work within that allocation. Performance runs use existing
exclusive locks. No new paid capacity, preemption, occupancy jobs, or repeated
completed cells to fill idle cards.

## Validation scope

36 CPU tests passed in normal and optimized mode; synthetic end-to-end CLI,
TSV binding verification and SVG succeeded. Independent statistical reviewer
executed unequal-size document, macro weighting and missing/failure probes;
engineering reviewer executed full/full, IDs, producer, hash and -O rejection
probes. Initial import-name and missing producer cross-check defects were fixed.
Final independent review additionally caught and fixed nullable/random-entropy
seeds, nullable sidecar hash checks and Python 3.12 `sum` changing the original
scorer's rounding. The official scorer retains its original ordered `+=` sum.
Unknown tasks, empty references and inconsistent task class vocabularies are
rejected. Nonempty `error` and `failed` records block CIs and are explicitly
reported as `error-present` or `failed-flag`, without copying private error text.
The existing LongBench-v2 parser regression passed. Existing E116a gate test
fails identically on the fixed base and candidate because its legacy fixture
has no mandatory run-cap sidecar; no gate or fixture was relaxed. The full E096
historical integration test could not complete in the isolated review snapshot
(requires Git history and a private model snapshot); no pass is claimed.
Synthetic fixtures are software checks only. Actual GPU, model generation,
production document CI, noninferiority and main-paper acceptance: NOT RUN.
