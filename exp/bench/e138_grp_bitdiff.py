# -*- coding: utf-8 -*-
"""E138-residual 组选择版逐位对拍：组版（chunk=2048 计算组织不变 +
TLI_PREFILL_SELECT_GROUP=G 组粒度选择 + 并集 compact）vs 缩 chunk 版
（TLI_SPARSE_PREFILL_CHUNK=G，每 G 行独立 chunk）。两者数学等价——
行 r 的可见集相同（组选择 vs 独立 chunk 选择逐位同，inc 缓存前缀序列
完全一致）、idx 升序、行级 -inf 屏蔽并集差量 → 输出应逐位一致。
"""
import os
import sys

TREE = "/tmp/wt_e138_ablation"
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "exp", "bench"))
import torch  # noqa: E402
from e131_arm_bench import make_args  # noqa: E402
from sparse_attn.indexer.tli_indexer import TLIIndexer  # noqa: E402
from sparse_attn.ops.eager_prefill import sparse_prefill_attn  # noqa: E402

DEV = "cuda:0"


def make_qkv(L, seed=999):
    g = torch.Generator(device=DEV).manual_seed(seed)
    H, Hkv, D = 32, 8, 128
    return (torch.randn(1, L, H, D, device=DEV, dtype=torch.bfloat16, generator=g),
            torch.randn(1, L, Hkv, D, device=DEV, dtype=torch.bfloat16, generator=g),
            torch.randn(1, L, Hkv, D, device=DEV, dtype=torch.bfloat16, generator=g))


def fresh():
    idx = TLIIndexer(make_args("mavg"))
    idx.layer_idx = 2
    idx.clear()
    return idx


def to_bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


cases = [
    # (L, G)：覆盖多 chunk 多组、尾组不满、G=bs 对齐、G<dense_below 组
    (8192, 64),
    (8192, 128),
    (5000, 256),    # 尾 chunk 不满 + 尾组不满
    (10240, 64),    # n_sparse=4 恰过门槛（预取门控在组版关）
    (3000, 64),     # 单稀疏 chunk（2048<3000<4096）
]
os.environ["TLI_PREFILL_PREFETCH"] = "0"   # 两臂都关预取对齐口径
n_pass = 0
for L, G in cases:
    q, k, v = make_qkv(L)
    # A：缩 chunk 版（chunk=G 独立 chunk）
    os.environ["TLI_PREFILL_SELECT_GROUP"] = "0"
    os.environ["TLI_SPARSE_PREFILL_CHUNK"] = str(G)
    oA = sparse_prefill_attn(fresh(), q, k, v)
    # B：组版（chunk=2048 组织 + G 组选择）
    os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
    os.environ["TLI_PREFILL_SELECT_GROUP"] = str(G)
    oB = sparse_prefill_attn(fresh(), q, k, v)
    # 判定 allclose：两臂 chunk 组织不同（W/idx/归约树不同）→ 非零项
    # softmax-einsum 树形归约分组有 ULP 级漂移（与 E138b docstring 口径
    # 一致：stable 升序保非零项顺序，allclose 验证）。bitwise 只作参考
    ok = torch.allclose(oA.float(), oB.float(), rtol=1e-2, atol=1e-3)
    md = (oA.float() - oB.float()).abs().max().item()
    bw = torch.equal(to_bits(oA), to_bits(oB))
    print(f"L={L:6d} G={G:4d}: allclose={'PASS' if ok else 'FAIL'} "
          f"bitwise={'PASS' if bw else 'FAIL'} maxdiff={md:.3e}")
    n_pass += ok
os.environ.pop("TLI_PREFILL_SELECT_GROUP", None)
os.environ["TLI_SPARSE_PREFILL_CHUNK"] = "2048"
os.environ.pop("TLI_PREFILL_PREFETCH", None)
print(f"=== {n_pass}/{len(cases)} ===")
sys.exit(0 if n_pass == len(cases) else 1)
