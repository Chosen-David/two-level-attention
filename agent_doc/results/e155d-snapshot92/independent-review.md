# Independent snapshot92 code review — 2026-10-20

Scope: only `Chosen-David/two-level-attention`, isolated `twolevel-audit92` worktree at `b418e220aae5cd86d4ba4ce1833ea79d4ad6bcde`, pending analyzer and two snapshot-test files. Read the pending diff, complete analyzer/tests, relevant actual evaluator read/scorer implementation, and snapshot92 checkpoint. No source/test edits, commit, push, remote jobs, GPU, Torch, or Triton operations. No real main invocation against configured home paths: every executed entry point was redirected into a fresh temporary directory.

## Result

No blocking findings for the bounded fix. The original defect is independently reproduced: aggregate scoring consumes staging copies, whereas baseline `main` supplies original collected paths to `per_example_scores`. A later atomic replacement of original A predictions changes baseline CI despite unchanged ranking.

An independent real-temp-files harness, loading the baseline with `git show`, produced:

| Version | Ranked AVG5 A / B | CI point difference after source A replacement |
| --- | --- | --- |
| Baseline | 75 / 50 | -50 |
| Pending patch | 75 / 50 | +25 |

The evaluator subprocess and individual metric were numeric test doubles. This establishes orchestration/byte-binding behavior, not real LongBench metric, provenance-sidecar, model, or GPU correctness.

## Mechanism verified

- Existing collection hashes now survive into the admitted cell record. Staged JSONL bytes must match before evaluation and after successful evaluation before the result can enter ranking.
- Successful scoring results carry staged paths and expected hashes. `main` uses those results, not collection paths, for CI.
- CI hashes the very decoded lines it consumes, re-encoded as UTF-8. Strict UTF-8 plus `newline=""` preserves byte identity for valid UTF-8 input while retaining untranslated LF, CRLF, and lone-CR delimiters. The digest is checked before the helper returns scores.
- The legacy standalone helper contract without `file_sha256` remains accepted; the production collection/scoring path supplies hashes.
- Stable-input full-entry behavior matches baseline after normalizing timestamp and newly added path/hash metadata.

## Executed verification

- `python -m unittest discover -s tests -p 'test_e155d*.py'`: 41 PASS.
- Same command with `python -O`: 41 PASS.
- `python tests/check_e155d_snapshot.py`: 24 full-entry baseline differentials PASS.
- Same command with `python -O`: 24 PASS.
- Differential input digest: `3aaf366af9095b401cb7517f1767b986d99093e03091adf1b3be7d9f3125f8f9`.
- `git diff --check`: clean at review time.
- Additional independent harness: atomic original-source replacement reproduced the baseline/patched difference above.
- Additional independent harness: LF/CRLF/CR, each with and without a final newline, with Chinese, emoji, and combining Unicode characters, passed exact byte hashing and the no-hash legacy helper (six variants).
- Additional independent harness: overwriting a >8 KB staged prediction file in place after the first per-example score was consumed caused a snapshot-digest rejection.
- Additional independent harness: atomically replacing the staged pathname during CI still read the original open inode and returned its original scores, while the pathname subsequently held different bytes. This is safe consumed-byte binding for that read, not immutable-path persistence.

The committed-test rerun also covers source replacement/deletion after score, source replacement/deletion after admission, staged mutation during evaluation/before CI, and abort before writing a new verdict on CI mismatch.

## Deliberate limits / interpretation

- This is not a concurrent-analyzer-safe or ABA-safe immutable snapshot/archive. Staging is still globally recreated; two analyzers can interfere. A write-and-restore between pre/post checks is not excluded. Added serialized paths can later disappear or refer to changed bytes.
- Collection row counting and collection hashing are separate reads, not an atomic transaction. Sidecar/manifest closure remains delegated to the evaluator; cross-cell answer/input identity is not newly established.
- File disappearance inside an `exists`/copy or `isfile`/hash race may raise rather than produce a per-cell `eval_error`; it does not silently produce a fresh inconsistent verdict. CI failure leaves any previously existing output file untouched, so "no new verdict" must not be represented as "no old verdict exists."
- Exact hash equivalence is conditional on valid strict UTF-8 input, as the legacy parser already requires. Lone CR compatibility was verified for the standalone helper, not claimed for collector row admission or complete strict LongBench evaluation.
- Two extra binary hash scans and per-line UTF-8 encoding/hashing add correctness work. No latency, memory, throughput, or GPU performance improvement was measured or claimed.

## Reviewed content identifiers

SHA-256 at review time:

- Analyzer: `c719089b3c4e279abc1a42430d116802b901929054f936ec52100abbd4f44658`.
- Snapshot unit test: `22adfabe8bb2ac0c17f6c3876e75b103e948b37ee88db3567a7eca1550496b50`.
- Full-entry differential check: `570d2cfcd49ef883e2985f7340a81be9c6243a86dedaefe2f04f007db89dcd34`.
