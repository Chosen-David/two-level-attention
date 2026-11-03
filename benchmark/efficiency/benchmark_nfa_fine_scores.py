"""iter106：fine-score int32 地址特化与原实现逐位对拍。"""

import json
import statistics

import numpy as np
import torch
import triton
import triton.language as tl
from triton.testing import do_bench



@triton.jit
def _fine_scores_kernel(
    KV_PTRS, BLOCK_TABLE, VALID_REQ, DIMS, Q, CAND, KEEP, OWNERS,
    GENERATIONS, LOGITS, LOW, HIGH, QUOTA, KV_REF, R,
    stride_qm, stride_qh, stride_ch, stride_cr,
    S: tl.constexpr, MAX_BLOCKS: tl.constexpr, PAGE: tl.constexpr,
    HK: tl.constexpr, HEAD_DIM: tl.constexpr, DIM_NUMEL: tl.constexpr,
    D_BLOCK: tl.constexpr, HKV: tl.constexpr, B: tl.constexpr,
    TILE: tl.constexpr, MAX_REQ: tl.constexpr, LAYER_OFFSET: tl.constexpr,
    PACK_KEY: tl.constexpr, ADDRESS32: tl.constexpr,
):
    """独立参考/候选：ADDRESS32=False 复现 iter104 通用路径。"""
    row = tl.program_id(0)
    t = tl.program_id(1)
    m = row // HKV
    kv = row % HKV
    layer = m // B
    req = m % B
    req_id = tl.load(VALID_REQ + req).to(tl.int32)
    kv_addr = tl.load(KV_PTRS + layer)
    kv_ptr = kv_addr.to(tl.pointer_type(KV_REF.dtype.element_ty))
    offs_d = tl.arange(0, D_BLOCK)
    dim_mask = offs_d < DIM_NUMEL
    dim = tl.load(DIMS + offs_d, mask=dim_mask, other=0)
    if not ADDRESS32:
        dim = dim.to(tl.int64)
    offs_r = t * TILE + tl.arange(0, TILE)
    mask_r = offs_r < R
    keep = tl.load(KEEP + row * stride_ch + offs_r * stride_cr,
                   mask=mask_r, other=False)
    read_r = mask_r & keep
    cand = tl.load(CAND + row * stride_ch + offs_r * stride_cr,
                   mask=read_r, other=0)
    in_range = (cand >= 0) & (cand < S)
    owner = tl.load(OWNERS + row * S + cand,
                    mask=read_r & in_range, other=-1)
    generation = tl.load(GENERATIONS + row)
    unique = read_r & in_range & (owner == generation * R + offs_r)
    safe = tl.where(read_r, cand, 0)
    lp = safe // PAGE
    off = safe % PAGE
    lp_valid = lp < MAX_BLOCKS
    phys = tl.load(
        BLOCK_TABLE + (layer + LAYER_OFFSET) * MAX_REQ * MAX_BLOCKS
        + req_id * MAX_BLOCKS + lp,
        mask=read_r & lp_valid, other=0)
    if not ADDRESS32:
        phys = phys.to(tl.int64)
    src = (kv_ptr
           + phys[:, None] * HK * PAGE * HEAD_DIM
           + kv * PAGE * HEAD_DIM
           + off[:, None] * HEAD_DIM
           + dim[None, :])
    k = tl.load(src, mask=(read_r & lp_valid)[:, None] & dim_mask[None, :],
                other=0.0)
    q = tl.load(Q + m * stride_qm + kv * stride_qh + offs_d,
                mask=dim_mask, other=0.0)
    score = tl.sum(k.to(tl.float32) * q[None, :], axis=1)
    score_bf16 = score.to(tl.bfloat16)
    score = score_bf16.to(tl.float32)
    good = (unique & lp_valid
            & (cand >= tl.load(LOW + m))
            & (cand < tl.load(HIGH + m))
            & (tl.load(QUOTA + m) > 0)
            & (score == score))
    score = tl.where(good, score, float("-inf"))
    if PACK_KEY:
        bits = score_bf16.to(tl.uint16, bitcast=True).to(tl.int32) & 0xFFFF
        negative = (bits & 0x8000) != 0
        ordered = tl.where(
            negative, (~bits) & 0xFFFF, bits | 0x8000) - 0x8000
        packed_good = good & (score != float("-inf"))
        delta = S - cand.to(tl.int32)
        invalid = -32768 << 16
        key = tl.where(packed_good, (ordered << 16) + delta, invalid)
        tl.store(LOGITS + row * R + offs_r, key, mask=mask_r)
    else:
        tl.store(LOGITS + row * R + offs_r, score.to(tl.bfloat16), mask=mask_r)


def _launch(kernel, case, out, num_warps, optimized):
    kwargs = dict(
        S=case['capacity'], MAX_BLOCKS=case['max_blocks'], PAGE=case['page'],
        HK=8, HEAD_DIM=128, DIM_NUMEL=32, D_BLOCK=32, HKV=8,
        B=case['m'], TILE=128, MAX_REQ=case['m'], LAYER_OFFSET=0,
        PACK_KEY=True, ADDRESS32=optimized, num_warps=num_warps)
    kernel[(case['m'] * 8, triton.cdiv(case['r'], 128))](
        case['kv_ptrs'], case['block_table'], case['valid_req'], case['dims'],
        case['q'], case['cand'], case['keep'], case['owners'],
        case['generations'], out, case['lower'], case['upper'], case['quota'],
        case['kv'], case['r'], case['q'].stride(0), case['q'].stride(1),
        case['cand'].stride(1), case['cand'].stride(2), **kwargs)


def _make_case(seed, m=3, hkv=8, r=8452, capacity=32768, page=64):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    max_blocks = capacity // page
    kv = torch.randn(
        (max_blocks, hkv, page, 128), generator=generator,
        dtype=torch.bfloat16, device='cuda').contiguous()
    block_table = torch.stack([
        torch.randperm(max_blocks, generator=generator, device='cuda')
        for _ in range(m)
    ]).to(torch.int32).contiguous()
    valid_req = torch.randperm(m, generator=generator, device='cuda').to(
        torch.int64).contiguous()
    dims = torch.tensor(
        list(range(48, 64)) + list(range(112, 128)),
        dtype=torch.int64, device='cuda')
    q = torch.randn(
        (m, hkv, 32), generator=generator, dtype=torch.float32,
        device='cuda').contiguous()
    cand = torch.stack([
        torch.randperm(capacity, generator=generator, device='cuda')[:r]
        for _ in range(m * hkv)
    ]).view(m, hkv, r).to(torch.int64).contiguous()
    # 覆盖重复 token、keep 洞、无效项、quota 和尾 tile。
    cand[:, :, 640:768] = cand[:, :, 128:256]
    cand[:, :, 512:520] = capacity
    keep = torch.rand(
        (m, hkv, r), generator=generator, device='cuda') > 0.03
    keep[:, :, :132] = True
    generations = (1 + torch.arange(m * hkv, device='cuda') % 3).to(
        torch.int32).contiguous()
    # owner 取每个 token 的第一次出现位置；与生产 generation 去重契约一致。
    owners_np = np.full((m * hkv, capacity), -1, dtype=np.int32)
    cand_np = cand.view(m * hkv, r).cpu().numpy()
    for row in range(m * hkv):
        generation = int(generations[row])
        for pos, token in enumerate(cand_np[row]):
            if 0 <= token < capacity and owners_np[row, token] < 0:
                owners_np[row, token] = generation * r + pos
    owners = torch.from_numpy(owners_np).cuda()
    lower = torch.tensor([128, 0, 0][:m], dtype=torch.int64, device='cuda')
    upper = torch.tensor([8000, 64, 4096][:m], dtype=torch.int64, device='cuda')
    quota = torch.tensor([1024, 300, 0][:m], dtype=torch.int64, device='cuda')
    kv_ptrs = torch.tensor([kv.data_ptr()], dtype=torch.int64, device='cuda')
    return dict(m=m, hkv=hkv, r=r, capacity=capacity, page=page,
                max_blocks=max_blocks, kv=kv, block_table=block_table,
                valid_req=valid_req, dims=dims, q=q, cand=cand, keep=keep,
                owners=owners, generations=generations, lower=lower,
                upper=upper, quota=quota, kv_ptrs=kv_ptrs)


def _graph_time(launch, inner=16, replays=100, reps=5):
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            launch()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / (inner * replays))
    return statistics.median(samples)


def main():
    torch.cuda.set_device(0)
    all_ok = True
    for seed in range(5):
        case = _make_case(20261105 + seed)
        shape = case['cand'].shape
        reference_out = torch.empty(shape, dtype=torch.int32, device='cuda')
        candidate_out = torch.empty_like(reference_out)
        # Graph/eager 两种生产 warp 配置都必须逐位一致。
        for warps in (1, 4):
            _launch(_fine_scores_kernel, case, reference_out,
                    warps, optimized=False)
            _launch(_fine_scores_kernel, case, candidate_out,
                    warps, optimized=True)
            torch.cuda.synchronize()
            ok = torch.equal(reference_out, candidate_out)
            all_ok &= ok
            print(f'CASE seed={seed} warps={warps} bit_exact={ok}')
            if not ok:
                print('mismatched_keys', (reference_out != candidate_out).sum().item())
                return 1

    case = _make_case(20261115, m=1)
    ref_out = torch.empty(case['cand'].shape, dtype=torch.int32, device='cuda')
    new_out = torch.empty_like(ref_out)
    reference_eager = lambda: _launch(
        _fine_scores_kernel, case, ref_out, 4, optimized=False)
    candidate_eager = lambda: _launch(
        _fine_scores_kernel, case, new_out, 4, optimized=True)
    reference_graph = lambda: _launch(
        _fine_scores_kernel, case, ref_out, 1, optimized=False)
    candidate_graph = lambda: _launch(
        _fine_scores_kernel, case, new_out, 1, optimized=True)
    reference_eager()
    candidate_eager()
    reference_graph()
    candidate_graph()
    torch.cuda.synchronize()
    eager_ref = do_bench(reference_eager, warmup=25, rep=100)
    eager_new = do_bench(candidate_eager, warmup=25, rep=100)
    graph_ref = _graph_time(reference_graph)
    graph_new = _graph_time(candidate_graph)
    graph_exact = torch.equal(ref_out, new_out)
    all_ok &= graph_exact
    print(json.dumps({
        'shape': [1, 8, case['r']], 'dims': 32,
        'eager_reference_ms': eager_ref, 'eager_candidate_ms': eager_new,
        'eager_speedup': eager_ref / eager_new,
        'graph_reference_ms': graph_ref, 'graph_candidate_ms': graph_new,
        'graph_speedup': graph_ref / graph_new,
        'graph_bit_exact': graph_exact, 'all_ok': all_ok}))
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
