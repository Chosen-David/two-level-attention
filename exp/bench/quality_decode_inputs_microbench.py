"""DeviceDecodeInputs tiled kernel versus the former host mask/input update."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from sparse_attn.ops.quality_decode_inputs import DeviceDecodeInputs


def measure(function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    gpu_ms, wall_ms = [], []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        begin.record()
        function()
        end.record()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - start) * 1000)
        gpu_ms.append(begin.elapsed_time(end))
    return dict(gpu_ms=gpu_ms, wall_ms=wall_ms,
                gpu_median_ms=statistics.median(gpu_ms),
                wall_median_ms=statistics.median(wall_ms))


def run(args):
    device = torch.device('cuda')
    batch, capacity = args.batch, args.capacity
    dtype = torch.bfloat16
    prompt_lens = [9000 + 3 * b for b in range(batch)]
    seq_lens = [length + args.generated for length in prompt_lens]
    generated = [args.generated] * batch
    initial_alive = [True] * batch
    previous = torch.randint(0, 128, (batch,), device=device)
    pad_id = 0

    def buffers():
        return (torch.empty(batch, 1, device=device, dtype=torch.long),
                torch.empty(batch, 1, device=device, dtype=torch.long),
                torch.empty(batch, device=device, dtype=torch.long),
                torch.empty(batch, device=device, dtype=torch.long),
                torch.empty(batch, device=device, dtype=torch.bool),
                torch.empty(batch, 1, 1, capacity, device=device, dtype=dtype))

    owned = buffers()
    inputs = DeviceDecodeInputs(*owned, seq_lens, prompt_lens, generated,
                                initial_alive, pad_id, set(), 1)

    def device_update():
        # FIRST 每次从不可变 seed 重置，避免重复计时把长度推出容量。
        inputs.inject(0, previous, block=args.block)

    reference = buffers()
    tokens, positions, pos, lengths, alive, mask = reference
    reference_alive = torch.tensor(initial_alive, device=device)
    reference_lengths = torch.tensor(seq_lens, device=device) + reference_alive
    negative = torch.finfo(dtype).min

    def host_update():
        tokens.copy_(previous.view(batch, 1))
        positions.copy_(torch.tensor(prompt_lens, device=device).view(batch, 1))
        lengths.copy_(reference_lengths)
        pos.copy_(lengths - 1)
        alive.copy_(reference_alive)
        mask.fill_(negative)
        for row, length in enumerate(seq_lens):
            mask[row, 0, 0, :length + 1] = 0

    device_update()
    host_update()
    torch.cuda.synchronize()
    for actual, expected in zip(owned, reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    result = dict(schema='quality-decode-inputs-microbench', batch=batch,
                  capacity=capacity, block=args.block, repeats=args.repeats,
                  warmup=args.warmup, dtype=str(dtype), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(device),
                  scope='step-zero reset; full mask write; includes no model forward',
                  device=measure(device_update, args.warmup, args.repeats),
                  host_reference=measure(host_update, args.warmup, args.repeats))
    result['wall_speedup_host_over_device'] = (
        result['host_reference']['wall_median_ms'] / result['device']['wall_median_ms'])
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(text)
    print(text, end='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--capacity', type=int, default=131072)
    parser.add_argument('--generated', type=int, default=1)
    parser.add_argument('--block', type=int, default=1024)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--output')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('native CUDA is required')
    if args.batch < 1 or args.capacity < 16384 or args.repeats < 1 or args.warmup < 0:
        parser.error('batch>=1, capacity>=16384, repeats>=1 and warmup>=0 required')
    if max(9000 + 3 * b for b in range(args.batch)) + args.generated + 1 > args.capacity:
        parser.error('seed lengths exceed capacity')
    run(args)


if __name__ == '__main__':
    main()
