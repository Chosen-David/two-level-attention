"""Device-side input preparation outside the model's captured forward.

一个 row×tile CTA 独立从旧元数据推导本步长度；tile 0 写元数据，所有 tile
只写自己的 mask 区间，因此不存在元数据 store/read 竞争，也不再受单个
Triton block 最大 65536 元素限制。采样仍是 torch.argmax；在线 EOS 的逐步
读回与离线固定步数的最终读回由调用方分别处理。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _inject(Seed, Previous, Counts, Tokens, Positions, Pos, Lengths, Alive, Mask,
            B: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr,
            FIRST: tl.constexpr, PAD: tl.constexpr, EOS: tl.constexpr,
            MIN_NEW: tl.constexpr, NEG: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    token = tl.load(Previous + row)
    if FIRST:
        position = tl.load(Seed + B + row)
        length = tl.load(Seed + row)
        count = tl.load(Seed + 2 * B + row)
        live = tl.load(Seed + 3 * B + row) != 0
    else:
        position = tl.load(Positions + row) + 1
        length = tl.load(Lengths + row)
        count = tl.load(Counts + row)
        live = tl.load(Alive + row)
        stop = tl.full((), False, tl.int1)
        for eos in tl.static_range(len(EOS)):
            stop = stop | (token == EOS[eos])
        live = live & ~(stop & (count >= MIN_NEW))
    length += live.to(tl.int64)
    count += live.to(tl.int64)
    # 所有 tile 从同一份旧值推导，只有 tile 0 更新共享元数据。
    if tile == 0:
        tl.store(Tokens + row, tl.where(live, token, PAD))
        tl.store(Positions + row, position)
        tl.store(Pos + row, length - 1)
        tl.store(Lengths + row, length)
        tl.store(Alive + row, live)
        tl.store(Counts + row, count)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    tl.store(Mask + row * CAP + cols, tl.where(cols < length, 0., NEG), cols < CAP)


class DeviceDecodeInputs:
    def __init__(self, tokens, positions, pos, lengths, alive, mask,
                 seq_lens, prompt_lens, generated_lengths, initial_alive,
                 pad_id, eos_ids, min_new_tokens):
        self.buffers = tokens, positions, pos, lengths, alive, mask
        self.batch, self.capacity = tokens.shape[0], mask.shape[-1]
        if not tokens.is_cuda or not all(t.is_contiguous() for t in self.buffers):
            raise ValueError('device decode inputs require contiguous CUDA buffers')
        self.seed = torch.tensor([seq_lens, prompt_lens, generated_lengths, initial_alive],
                                 device=tokens.device, dtype=torch.int64)
        self.counts = torch.empty_like(lengths)
        self.pad, self.eos, self.minimum = pad_id, tuple(sorted(eos_ids)), min_new_tokens

    def inject(self, step, previous, *, block=1024):
        if block < 1 or block & (block - 1):
            raise ValueError('device input block must be a positive power of two')
        grid = (self.batch, triton.cdiv(self.capacity, block))
        _inject[grid](
            self.seed, previous, self.counts, *self.buffers,
            B=self.batch, CAP=self.capacity, BLOCK=block,
            FIRST=step == 0, PAD=self.pad, EOS=self.eos, MIN_NEW=self.minimum,
            NEG=torch.finfo(self.buffers[-1].dtype).min, num_warps=4)
