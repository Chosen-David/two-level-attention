"""Isolated CUDA assertion gate. Never imported as a pytest test module."""
import os
import torch
from test_exact_linear_batch_gpu import CONFIG, _case


def main():
    os.environ.update(CONFIG)
    with torch.no_grad(), _case(2) as case:
        for _ in range(3):case.run()
        torch.cuda.synchronize();graph,out=case.capture()
        old=case.source._exact_linear_batch
        case.source.clear()
        case.k=case.k.clone();case.v=case.v.clone()
        for _ in range(3):case.run()
        torch.cuda.synchronize()
        assert case.source._exact_linear_batch is not old
        assert any(item is old for item in graph._tli_address_owners)
        assert not hasattr(case.source,'_retired_exact_linear_batches')
        assert not bool(old.ready.cpu())
        print('EXPECT_OLD_READY_FAILURE',flush=True)
        graph.replay();torch.cuda.synchronize()
        raise AssertionError('stale graph unexpectedly replayed successfully')


if __name__=='__main__':main()
