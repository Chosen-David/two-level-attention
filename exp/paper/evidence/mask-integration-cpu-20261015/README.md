# Complementary prefill/decode mask integration

Base dev dc083b20af02f7a4b8d4d062a3de2ee10fb5fd16.
Integrated main 366fd384ebd18c9caf294f9114be144bb2040fb4.
CPU Python /workspace/twolevel-cpu-env/bin/python, torch 2.14.1+cpu.
OMP_NUM_THREADS=1; no weights, GPU, kernel changes, experimental matrix or collection.

Commands (optimized logs add -O before -m):

- python -m unittest discover -s sparse_attn/ops/tests -q
  normal.log / optimized.log: 37 tests each, OK.
- python -m unittest discover -s tests -p 'test_qwen*.py' -q
  qwen-normal.log / qwen-optimized.log: 14 tests each, OK.

Independent read-only reviewer /root/independent_review: no blocking finding;
normal/-O mask, lifecycle and heterogeneous KV subset: 17 tests each passed.
Reviewed causal None prefill, explicit -inf prefill acceptance, early decode
all-zero guard and Tensor/layout short-circuit before property access.
Reviewer made no edits/commits/pushes. Git merge index resolution is performed
by the main executor after review; no conflict markers remain.

identity.json records source SHA256 and Git blob IDs. Qwen Fig1/archive,
Graph selector and heterogeneous KV consumer are unchanged from base dev.
CPU executes the production forward via AST with model/cache/indexer doubles;
identity RoPE isolates host lifetime/mask semantics. It does not establish actual
HF cache/pretrained/CUDA correctness, long-sequence memory or performance.
Fresh prefill None uses main's O(L^2) boolean causal mask. Decode nonidentity
masks are rejected, not supported/consumed, and value checks may synchronize
CUDA. No performance claim or MLA algorithm research is implied.
