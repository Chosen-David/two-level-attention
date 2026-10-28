"""CPU cast-storage probe, deliberately not a GPU memory/performance benchmark."""
import hashlib
import inspect
import json

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sparse_attn.indexer.tli_indexer import TLIIndexer
from exp.bench.prefill_pair_tail_compare import baseline, candidate


class CastStorage(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.records = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if func == torch.ops.aten._to_copy.default and kwargs.get('dtype') == torch.float32:
            self.records.append(dict(shape=list(args[0].shape),
                                     backing_bytes=out.untyped_storage().nbytes()))
        return out


def main():
    torch.manual_seed(784)
    q = torch.randn(2, 4097, 4, 128, dtype=torch.bfloat16).transpose(1, 2)
    data = dict(torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                scope='CPU backing storage of FP32 cast only; no CUDA allocator or latency',
                main='17e56e8fe30cf8037caf3ce7b38040e754f0521e',
                candidate_source_sha256=hashlib.sha256(inspect.getsource(TLIIndexer.observe_prefill_q).encode()).hexdigest())
    outputs = []
    for name, fn in [('baseline', baseline), ('candidate', candidate)]:
        with CastStorage() as mode:
            outputs.append(fn(q))
        data[name] = mode.records
    assert torch.equal(*outputs)
    data['pair_indices_equal'] = True
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
