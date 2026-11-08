import torch

def prepare_pad_mask(block_size: int, cu_seqlens: torch.Tensor, total: int = None):
    """块对齐 pad mask + pad cu_seqlens。

    total（可选）：pad 后总 token 数的静态上界/精确值。给出时走纯张量
    路径（无 GPU→host 同步）——CUDA Graph capture 期间，下方逐元素
    `new_ones(x)`/`new_zeros(y-x)` 与 `zip(seqlens, ...)` 迭代会触发
    隐式 `.item()`（cudaStreamSynchronize → "operation not permitted
    when stream is capturing"）。数值与逐元素路径逐位相同（同一整数
    算术 + 同一前缀布局）；未给 total 时保持原实现零改动。
    """
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    if total is not None:
        pad_seqlens = (seqlens + block_size - 1) // block_size * block_size
        pad_cu_seqlens = torch.cat((pad_seqlens.new_zeros(1),
                                    pad_seqlens.cumsum(dim=0)))
        rel = (torch.arange(total, device=seqlens.device, dtype=seqlens.dtype)
               .view(1, -1) - pad_cu_seqlens[:-1].view(-1, 1))
        pad_mask = ((rel >= 0) & (rel < seqlens.view(-1, 1))).reshape(-1)
        return pad_mask, pad_cu_seqlens
    pad_seqlens = torch.stack([(x + block_size - 1) // block_size * block_size for x in seqlens])
    pad_cu_seqlens = torch.cat((pad_seqlens.new_tensor([0]), pad_seqlens.cumsum(dim=0)))
    pad_mask = torch.cat([torch.cat((seqlens.new_ones(x), seqlens.new_zeros(y - x))) for x, y in zip(seqlens, pad_seqlens)]).to(torch.bool)
    return pad_mask, pad_cu_seqlens

def pad_tensor(x: torch.Tensor, pad_mask: torch.Tensor):
    pad_x = x.new_zeros((1, pad_mask.size(0)) + x.shape[2:])
    pad_x[:, pad_mask] = x
    return pad_x

def prepare_seqlens(cu_seqlens: torch.Tensor):
    return cu_seqlens[1:] - cu_seqlens[:-1]
