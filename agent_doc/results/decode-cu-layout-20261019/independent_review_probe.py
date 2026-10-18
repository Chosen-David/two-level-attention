"""CPU host ABI inspection only; no numerical/device-kernel claims."""
import json
from unittest.mock import patch
import torch
from sparse_attn.ops import eager_decoding as ed


class Launch:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        return self.fn


def main():
    q = torch.ones(1, 4, 8)
    k = torch.zeros(16384, 2, 8)
    v = torch.zeros_like(k)
    mask = torch.zeros(2, 16384, dtype=torch.bool)
    mask[:, 0] = True
    checked = []
    for dtype in (torch.int32, torch.int64):
        for stride in (1, 2, 3, 7, 31):
            for offset in (0, 1, 5):
                backing = torch.full((offset + stride + 2,), -123, dtype=dtype)
                cu = backing.as_strided((2,), (stride,), offset)
                cu.copy_(torch.tensor([0, 16384], dtype=dtype))
                original = backing.clone()
                captured = []

                def expand(m, idx, cnt, flag, cuk, *args, **kwargs):
                    captured.append(cuk)
                    assert cuk.dtype == torch.int32
                    assert cuk.stride(0) == 1
                    assert cuk.as_strided((2,), (1,)).tolist() == [0, 16384]
                    if dtype == torch.int32 and stride == 1:
                        assert cuk is cu
                    else:
                        assert cuk.data_ptr() != cu.data_ptr()
                    flag.zero_()

                def attn(idx, cnt, qs, keys, vals, out, *args, **kwargs):
                    out.zero_()

                with patch.object(ed, '_eager_expand_kernel', Launch(expand), create=True), \
                        patch.object(ed, '_eager_attn_kernel', Launch(attn), create=True), \
                        patch.object(ed, '_ERR_FLAG', {}):
                    ed._eager_gather_triton(q, k, v, mask, False, 2, 32, cu, .5)
                    ed.eager_decoding_check_errors()
                assert len(captured) == 1
                assert torch.equal(backing, original), 'input storage mutated'
                checked.append({'dtype': str(dtype), 'stride': stride, 'offset': offset})
    print(json.dumps({'cases': len(checked), 'passed': len(checked),
                      'scope': 'CPU real wrapper + unit-stride pointer ABI double; GPU untested',
                      'torch': torch.__version__, 'layouts': checked}, indent=2))


if __name__ == '__main__':
    main()
