"""Device-side input preparation outside the model's captured forward.

两个有序 kernel 分工：每行一个 CTA 先推进并提交元数据，随后 row×tile CTA
只读取已提交长度并写自己的 mask 区间。不同 tile 不再读写共享元数据，既消除
跨 CTA store/read 竞争，也不受单个 Triton block 最大 65536 元素限制。采样
仍是 torch.argmax；在线 EOS 的逐步读回与离线固定步数的最终读回由调用方
分别处理。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _advance(Seed, Previous, Counts, Tokens, Positions, Pos, Lengths, Alive,
             B: tl.constexpr, FIRST: tl.constexpr, PAD: tl.constexpr,
             EOS: tl.constexpr, MIN_NEW: tl.constexpr):
    row = tl.program_id(0)
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
    tl.store(Tokens + row, tl.where(live, token, PAD))
    tl.store(Positions + row, position)
    tl.store(Pos + row, length - 1)
    tl.store(Lengths + row, length)
    tl.store(Alive + row, live)
    tl.store(Counts + row, count)


@triton.jit
def _write_mask(Lengths, Mask, CAP: tl.constexpr, BLOCK: tl.constexpr,
                NEG: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    length = tl.load(Lengths + row)
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
        tokens, positions, pos, lengths, alive, mask = self.buffers
        # 同一 CUDA stream 上先提交唯一行元数据，再让各 tile 读取新长度。
        # 单 kernel 内 tile 0 与其他 tile 无同步，不能安全地承担这两个阶段。
        _advance[(self.batch,)](
            self.seed, previous, self.counts, tokens, positions, pos, lengths, alive,
            B=self.batch, FIRST=step == 0, PAD=self.pad, EOS=self.eos,
            MIN_NEW=self.minimum, num_warps=1)
        _write_mask[(self.batch, triton.cdiv(self.capacity, block))](
            lengths, mask, CAP=self.capacity, BLOCK=block,
            NEG=torch.finfo(mask.dtype).min, num_warps=4)
        # Triton 原地写入不会推进 PyTorch _version；为 eager 几何缓存提供
        # 每步单调修订号，保留同一步跨层复用且禁止跨步 stale hit。
        lengths._tli_device_revision = getattr(lengths, '_tli_device_revision', 0) + 1
