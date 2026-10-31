"""Execute the actual before/after producer method with a NumPy tensor facade.
This checks host routing, values, aliasing and gather count, NOT Torch/CUDA."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).parent
CANDIDATE = ROOT/'candidate/sparse_attn/ops/batch_candidate_producer.py'
if not CANDIDATE.exists():
    CANDIDATE = ROOT.parents[2]/'sparse_attn/ops/batch_candidate_producer.py'
def check(ok, message):
    if not ok:
        raise AssertionError(message)

def rounded(x, dtype):
    x = np.asarray(x, dtype=np.float32)
    if dtype == 'bf16':
        bits = x.copy().view(np.uint32)
        bits = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
        return bits.view(np.float32)
    if dtype == 'fp16':
        return x.astype(np.float16).astype(np.float32)
    return x

class Tensor:
    gathers = 0
    def __init__(self, value, dtype='fp32'):
        self.a = np.asarray(value)
        self.dtype = dtype
    @property
    def shape(self):
        return self.a.shape
    def numel(self):
        return self.a.size
    def __getitem__(self, key):
        if isinstance(key, tuple) and any(isinstance(x, np.ndarray) for x in key):
            Tensor.gathers += 1
            return Tensor(np.ascontiguousarray(self.a[key]), self.dtype)
        return Tensor(self.a[key], self.dtype)
    def __mul__(self, scale):
        return Tensor(rounded(self.a * scale, self.dtype), self.dtype)
    def to(self, dtype):
        return Tensor(self.a.astype(np.float32), dtype)
    def contiguous(self):
        return self if self.a.flags.c_contiguous else Tensor(np.ascontiguousarray(self.a), self.dtype)
    def view(self, *shape):
        return Tensor(self.a.reshape(shape), self.dtype)
    def mean(self, dim):
        return Tensor(rounded(self.a.mean(axis=dim), self.dtype), self.dtype)

def load(path, source=None):
    tree = ast.parse(path.read_text() if source is None else source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BatchedCandidateProducer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')
    captured = []
    def score(*args, **kwargs):
        query = args[-1]
        captured.append(query)
        # Test-only read-only stand-in. Native score kernels are not executed.
        return Tensor(np.repeat(query.a.sum(-1)[..., None], 3, axis=-1), kwargs['out_dtype'])
    def select(coarse, average, *args, **kwargs):
        return coarse, average, 'counts'
    ns = dict(torch=SimpleNamespace(float32='fp32'), score_coarse_batched=score,
              score_average_batched=score, select_candidates=select,
              BatchedCandidateInputs=lambda *xs: xs)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    return ns['__call__'], captured

before, before_seen = load(ROOT/'baseline/sparse_attn/ops/batch_candidate_producer.py')
after, after_seen = load(CANDIDATE)
rng = np.random.default_rng(9183)
cases = 0
saved = 0
for batch in (1, 2, 8, 32):
    for dtype in ('bf16', 'fp16'):
        for strided in (False, True):
            for mode in ('shared', 'full', 'different', 'equal_distinct', 'coarse_full', 'fine_full'):
                coarse = np.arange(96, 128)
                fine = coarse
                if mode == 'full':
                    coarse = fine = None
                elif mode == 'different':
                    fine = np.arange(0, 32)
                elif mode == 'equal_distinct':
                    fine = coarse.copy()
                elif mode == 'coarse_full':
                    coarse = None
                elif mode == 'fine_full':
                    fine = None
                state = SimpleNamespace(heads=2, pages=Tensor(np.arange(3)), coarse_dims=coarse,
                    fine_dims=fine, offsets=None, capacity=192, bs=64, k1=128, k2=1024,
                    near_pages=32, far_method='minmax', near_method='avg', sink=4, swa=128,
                    geometry=lambda lengths: SimpleNamespace(lower=1, upper=2, quota=3))
                previous = None
                for step in range(2):
                    values = rng.normal(size=(batch, 1, 8, 256 if strided else 128)).astype(np.float32)
                    if strided:
                        values = values[..., ::2]
                    q = Tensor(rounded(values, dtype), dtype)
                    original = q.a.copy()
                    outputs = []
                    for fn, seen in ((before, before_seen), (after, after_seen)):
                        seen.clear()
                        Tensor.gathers = 0
                        out = fn(state, q, 'fine_arena',
                                 dict(k_min=1, k_max=2, k_avg=3), None, 128**-.5)
                        outputs.append((out, Tensor.gathers, list(seen)))
                    old, new = outputs
                    for i in (1, 2, 3):
                        check(np.array_equal(old[0][i].a, new[0][i].a, equal_nan=True), 'value mismatch')
                    check(np.array_equal(q.a, original), 'query mutated')
                    share = coarse is fine
                    check((new[0][1] is new[2][0]) == share, 'unexpected projection alias')
                    check(new[2][0] is new[2][1], 'coarse/average query differs')
                    expected_saved = int(share and coarse is not None)
                    check(old[1] - new[1] == expected_saved, 'gather count mismatch')
                    saved += expected_saved
                    if previous is not None:
                        check(new[0][1] is not previous, 'projection reused across calls')
                    previous = new[0][1]
                cases += 1
print(f'PASS {cases} actual-method host pairs, two invocations each; {saved} duplicate gathers removed in shared-index cases; no native GPU claim')

# Execute an unsafe unconditional-sharing mutant on the final fine-full case.
path = CANDIDATE
bad_source = path.read_text().replace('self.fine_dims is self.coarse_dims', 'True')
mutant, _ = load(path, bad_source)
bad = mutant(state, q, 'fine_arena', dict(k_min=1, k_max=2, k_avg=3), None, 128**-.5)
check(bad[1].shape != new[0][1].shape, 'unconditional-sharing negative not detected')
print('PASS unsafe-sharing mutant rejected; BF16/FP16 are NumPy rounding models only')

