import os

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

# ---------------------------------------------------------------------------
# 【E127 iter3】GPU 端错误标志缓冲：decode 全程零 host 同步的配套机制。
# 旧口径的两次当场同步（counts 读回取 W + B7 空行检查）每次排空整个
# launch 队列——36 层模型每 token 36 次，CPU 无法跨层跑在 GPU 前面。
# 现把契约违规（B7 空行 / W 超 TLI_E127_W_UPPER 上界）以 GPU 端原地点并
# （copy_ + maximum，无同步）记录到本模块全局标志，生成结束由
# eager_decoding_check_errors() 一次性读回校验——fail-closed 语义保持
# （错误仍然显式 raise、整次生成作废），仅 raise 时机后移到生成结束，
# 与 iter1 metrics 惰性化同一精神。
# ---------------------------------------------------------------------------
_ERR_FLAG = {}  # device -> [1] int32 GPU 缓冲（0=ok / 1=B7 空行 / 2=W 超上界）


def _err_flag(device: torch.device) -> torch.Tensor:
    buf = _ERR_FLAG.get(device)
    if buf is None:
        buf = torch.zeros(1, dtype=torch.int32, device=device)
        _ERR_FLAG[device] = buf
    return buf


def eager_decoding_check_errors(device=None, clear: bool = True):
    """【E127 iter3】生成结束一次性错误校验（读取端唯一同步点）。

    TLI_E127_FAST=1 的 gather 路径不再当场 raise（避免每层每步同步），
    而是把契约违规记录到 GPU 标志；生产接线上层在生成结束后调用本函数
    一次性校验。code=1 → B7 全 False 行（同原报错文案）；code=2 → 某行
    mask 选中数超过 TLI_E127_W_UPPER（漏选保护，fail-closed——提示调大
    上界或回退 env=0）。clear=True 校验后清零（连续多次生成各查各的）。
    """
    if device is None:
        for dev, buf in list(_ERR_FLAG.items()):
            code = int(buf.item())
            if clear:
                buf.zero_()
            _raise_for_code(code)
        return
    buf = _ERR_FLAG.get(torch.device(device))
    if buf is None:
        return
    code = int(buf.item())
    if clear:
        buf.zero_()
    _raise_for_code(code)


def _raise_for_code(code: int):
    if code == 1:
        raise RuntimeError(
            "eager_decoding_attn: 选择 mask 存在全 False 行——softmax 将"
            "产生 NaN。indexer 契约应保证 sink/swa 强制区使每行至少一个"
            "有效 token；请检查 indexer 的 mask 输出。"
        )
    if code == 2:
        raise RuntimeError(
            "eager_decoding_attn: mask 某行选中数超过 TLI_E127_W_UPPER"
            "上界——gather 路径已 fail-closed 截断（本提示应在校验点"
            "raise，整次生成作废）。请调大 TLI_E127_W_UPPER 或设"
            "TLI_E127_FAST=0 回退稠密路径。"
        )
    if code == 3:
        raise RuntimeError(
            "eager_decoding_attn: cu_seqlens_k 契约破坏——gather 路径的"
            "零同步长度推导假设 cu == [0, S]（prepare_cu_seqlens 构造保证、"
            "B=1 生产路径恒成立），实测不符。本提示在校验点 raise，整次"
            "生成作废；请检查调用方传入的 cu_seqlens_k。"
        )


def eager_decoding_attn(
    q: torch.Tensor, # [1, tq, hq, d]
    k: torch.Tensor, # [1, tk, h, d]
    v: torch.Tensor, # [1, tk, h, d]
    mask: torch.Tensor,
    block_size: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float = None,
):
    """【E127 iter2+3】稀疏 attention 消费端入口（原实现原样保留为 _dense 版）。

    TLI_E127_FAST=1 时走 gather 路径：原实现对全序列 S 个 token 全量计算
    fp32 分数再 mask（稀疏性在消费端被完全丢弃——S=64K 时 k.to(float32)
    物化 + 全长 einsum + 全长 softmax ≈ 2.2ms/层步）；gather 路径只取
    每行 mask 选中的 token（~K2+sink+swa ≈ 1280）做 attention，带宽
    从 O(S·Hkv·D) 降到 O(R·W·D)。数学语义不变：softmax(全 S 行、未选
    位 -inf) ≡ softmax(选中子集)（exp(-inf)=0 精确），数值上仅归约顺序
    不同（输出 ULP 级漂移，allclose(1e-2) 门禁）。

    【iter3】gather 路径零 host 同步：W 上界化（TLI_E127_W_UPPER，契约
    上界 K2+sink+swa+bs≈1344，默认 2048）+ B7/W 校验 GPU 端并入全局错误
    标志（不再当场 raise）——生成结束由 eager_decoding_check_errors()
    一次性校验（fail-closed 语义保持，raise 时机后移）。
    """
    if os.environ.get("TLI_E127_FAST", "0") == "1":
        return _eager_decoding_attn_gather(
            q, k, v, mask, block_size, cu_seqlens_k, softmax_scale)
    return _eager_decoding_attn_dense(
        q, k, v, mask, block_size, cu_seqlens_k, softmax_scale)


def _eager_decoding_attn_dense(
    q: torch.Tensor, # [1, tq, hq, d]
    k: torch.Tensor, # [1, tk, h, d]
    v: torch.Tensor, # [1, tk, h, d]
    mask: torch.Tensor,
    block_size: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float = None,
):
    if softmax_scale is None:
        softmax_scale = k.shape[-1] ** -0.5
    q, k, v, mask = (x.squeeze(0) for x in (q, k, v, mask))
    o = q.new_zeros(q.shape[:-1] + (v.shape[-1],))
    G = q.shape[-2] // k.shape[-2]
    bos_k, eos_k = cu_seqlens_k[0], cu_seqlens_k[1]
    # ---- E103：kv-head 共享消融消费端 ----
    # 共享口径：mask [Hkv, tk]（kv-head 级），repeat 后 [Hkv, tk*bs] 直接广播到组内
    # 全部 G 个 q-head（b_mask[:, None, :]）——「同组共享同一份选择」的落点；
    # per_q_head 口径：mask [H, tk]（q-head 级），按 G 拆回 [Hkv, G, tk*bs]
    # 逐元素对应——每个 q-head 用自己的选择，不做组内广播。
    m0 = mask[0]
    per_qh = m0.shape[0] != k.shape[-2]
    # B05 修复（GPT 审查 2026-10-08）：原版 [:, :eos_k-bos_k] 切的是 dim1
    # （per_qh 的 G 轴 / shared 的 Hkv 轴）而非最后一维 token 轴——无 padding
    # 时恰好是 no-op 故长期未暴露；投影路径 Tpad>T 时会广播失败或掩码不裁。
    # 改为裁 token 轴；无 padding（tk*bs == eos_k-bos_k）时与原版逐位等价。
    if per_qh:
        g_m = m0.shape[0] // k.shape[-2]
        b_mask = repeat(
            m0, '(h g) tk -> h g (tk bs)', g=g_m, bs=block_size
        )[..., :eos_k - bos_k]
    else:
        b_mask = repeat(mask[0], 'h tk -> h (tk bs)', bs=block_size)[..., :eos_k - bos_k]

    b_q = rearrange(q[0] * softmax_scale, '(h g) d -> h g d', g=G).to(torch.float32)
    b_k = k.to(torch.float32)
    b_v = v
    b_s = einsum(b_q, b_k, 'h g d, t h d -> h g t')
    if per_qh:
        # E103：b_mask 已是 [Hkv, G, T] 与 b_s 同形，逐元素对应（每 q-head 自己的选择）
        b_s = torch.where(b_mask, b_s, float('-inf'))
    else:
        b_s = torch.where(b_mask[:, None, :], b_s, float('-inf'))
    # 【B7 修复（kimi3 清单 F12，2026-10-08）】全 False 行防线（消费端）：
    # 任一 head 行全 False 时 softmax 对全 -inf 行产生 NaN 输出。indexer
    # 契约上由 sink/swa 正交强制保证不可达，此为防御性 fail-closed（契约
    # 破损时显式报错优于 NaN 静默扩散）。代价 = 每次调用一次 host 同步
    # （any().all() → bool）；本函数是 HF 参考路径，非 sglang 生产热路径。
    if not bool(b_mask.any(dim=-1).all()):
        raise RuntimeError(
            "eager_decoding_attn: 选择 mask 存在全 False 行——softmax 将"
            "产生 NaN。indexer 契约应保证 sink/swa 强制区使每行至少一个"
            "有效 token；请检查 indexer 的 mask 输出。"
        )
    b_p = F.softmax(b_s, dim=-1).to(q.dtype)
    b_o = rearrange(einsum(b_p, b_v, 'h g t, t h d -> h g d'), 'h g d -> (h g) d')
    o[0] = b_o
    return o.unsqueeze(0)


def _eager_decoding_attn_gather(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    block_size: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float = None,
):
    """E127 gather 路径（数学语义 = _dense，仅归约顺序不同）。

    行定义（与 _dense 的 b_mask 一致）：
      共享口径：R = Hkv 行，行 h = kv-head h，组内 G 个 q-head 广播同一选择；
      per_q_head：R = Hkv*G 行，行 (h,g) = q-head g 用 kv-head h。
    【iter3】零同步：topk 用 host 已知上界 W_upper（≥ 每行真实选中数 →
    全部命中 + 补齐位 valid 掩掉）；B7 空行 / W 超上界契约违规 GPU 端并
    入 _err_flag，生成结束 eager_decoding_check_errors() 校验 raise。
    """
    if softmax_scale is None:
        softmax_scale = k.shape[-1] ** -0.5
    q, k, v, mask = (x.squeeze(0) for x in (q, k, v, mask))
    G = q.shape[-2] // k.shape[-2]
    # 【iter3】零同步长度推导：cu_seqlens_k 是 GPU 张量，原 bos_k/eos_k 的
    # 隐式 .item() 切片（b_mask 裁剪 + KV 切段）是每层步一次的存量 host
    # 同步（_dense 同样存在）。B=1 无 padding 生产路径下 cu 恒 [0, S]
    # （prepare_cu_seqlens 构造保证），故用 host 元数据判等：
    #   mask 块宽 × block_size == k 长度 → 无 padding，bos=0/eos=S 直接
    #   推导，不再切片；判等失败（投影 padding 等）→ 回退 _dense
    #   （同步次数与 env=0 持平，行为一致）。cu 契约本身由下方 GPU 端
    #   校验兜底（code=3，尾部 raise——原 _dense 对异常 cu 的「静默切
    #   子段」语义在生产 B=1/prepare_cu_seqlens 下不可达，fail-closed
    #   显式报错优于静默语义分歧）。
    full_len = mask.shape[-1] * block_size
    if full_len != k.shape[0]:
        return _eager_decoding_attn_dense(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            mask.unsqueeze(0), block_size, cu_seqlens_k, softmax_scale)
    m0 = mask[0]
    per_qh = m0.shape[0] != k.shape[-2]
    # 与 _dense 相同的 b_mask 构造（repeat 到 token 级；无 padding 时不裁剪，
    # 与 [..., :eos_k-bos_k] 在 cu=[0,S] 契约下逐位等价）
    if per_qh:
        g_m = m0.shape[0] // k.shape[-2]
        b_mask = repeat(
            m0, '(h g) tk -> h g (tk bs)', g=g_m, bs=block_size
        )
    else:
        b_mask = repeat(mask[0], 'h tk -> h (tk bs)', bs=block_size)

    # 统一为 2D [R, T] 行视图
    if per_qh:
        bm2 = b_mask.reshape(-1, b_mask.shape[-1])        # [Hkv*G, T]
        head_of_row = torch.arange(m0.shape[0], device=q.device) // G
    else:
        bm2 = b_mask                                     # [Hkv, T]
        head_of_row = None                              # 行号即 kv-head 号

    T = bm2.shape[-1]
    # 小序列路由：gather 的固定开销（topk + ~15 个小 op）在 T < 16K 时
    # 得不偿失（S=8K 实测 gather 0.48ms > dense 0.28ms），且 T 是 host
    # 已知量——本分支零同步直接回退，8K 档与 baseline 逐位同路径。
    if T < 16384:
        return _eager_decoding_attn_dense(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            mask.unsqueeze(0), block_size, cu_seqlens_k, softmax_scale)
    counts = bm2.sum(dim=-1)                            # [R]（GPU int64，无同步）

    # 【iter3】零同步化：W 用 host 已知的契约上界（indexer 契约：每行选中
    # ≈ K2(1024)+sink(128)+swa(128)+当前块(64) ≈ 1344 token，上界取 2048
    # 留 50% 安全垫，env TLI_E127_W_UPPER 可调）。topk k=W_upper 时真实
    # counts ≤ W_upper 的行全部命中 + 补齐位由 valid 掩掉（与原精确 W 路径
    # 同一掩码机制）；某行 counts > W_upper 时漏选保护 fail-closed（下方
    # code=2 记入 GPU 标志，生成结束校验 raise）。
    # 路由等价性：原逻辑用真实 W 判 W*8>T，生产 mask W≈1344 与 W_upper=2048
    # 在 16K/32K/64K 档同向（都走 gather），行为零变化。
    W_upper = min(int(os.environ.get("TLI_E127_W_UPPER", "2048")), T)
    if W_upper * 8 > T:
        return _eager_decoding_attn_dense(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            mask.unsqueeze(0), block_size, cu_seqlens_k, softmax_scale)

    # 【iter3】GPU 端契约校验（无同步，替代原 stats 读回 + 当场 raise）：
    #   counts.min()==0   → B7 全 False 行（code=1，文案与 _dense 一致）
    #   counts.max()>W_up → 漏选保护（code=2）
    #   cu ≠ [0, S]       → 长度推导契约破坏（code=3）
    # 原地点并入全局标志（copy_，中间张量每步释放不长计算图），
    # 读取端 eager_decoding_check_errors() 生成结束一次校验。
    flag = _err_flag(bm2.device)
    cu_ok = (cu_seqlens_k[0] == 0) & \
        ((cu_seqlens_k[1] - cu_seqlens_k[0]) == full_len)
    code = torch.where(
        counts.min() > 0,
        torch.where(counts.max() <= W_upper,
                    torch.where(cu_ok,
                                flag.new_zeros(()), flag.new_full((), 3)),
                    flag.new_full((), 2)),
        flag.new_full((), 1))
    flag.copy_(torch.maximum(flag, code))

    bm_int = bm2.to(torch.int32)
    idx = torch.topk(bm_int, W_upper, dim=-1).indices   # [R, W_upper]
    valid = torch.gather(bm_int, 1, idx).to(torch.bool)  # [R, W] 补齐位 False

    # KV gather：kv_t [Hkv, T, D]；行 → head 映射（共享口径行号即 head 号）
    # 【iter3】无 padding 契约下 k 全量即 [bos:eos] 段——不再 GPU 标量切片
    kv_t = k.permute(1, 0, 2)                            # [Hkv, T, D]
    vv_t = v.permute(1, 0, 2)
    if head_of_row is None:
        idx3 = idx.unsqueeze(-1).expand(*idx.shape, kv_t.shape[-1])
        gk = torch.gather(kv_t, 1, idx3)                 # [Hkv, W, D]
        gv = torch.gather(vv_t, 1, idx3)
    else:
        h_idx = head_of_row.unsqueeze(-1).expand_as(idx)  # [R, W]
        gk = kv_t[h_idx, idx, :]                         # [R, W, D] 高级索引 gather
        gv = vv_t[h_idx, idx, :]

    if per_qh:
        b_q = (q[0] * softmax_scale).to(torch.float32)   # [H=R, D]
        b_s = einsum(b_q, gk.to(torch.float32), 'r d, r w d -> r w')
        b_s = torch.where(valid, b_s, float('-inf'))
        b_p = F.softmax(b_s, dim=-1).to(q.dtype)
        b_o = einsum(b_p, gv, 'r w, r w d -> r d')       # [H, D]
    else:
        b_q = rearrange(q[0] * softmax_scale, '(h g) d -> h g d', g=G).to(torch.float32)
        b_s = einsum(b_q, gk.to(torch.float32), 'h g d, h w d -> h g w')
        b_s = torch.where(valid[:, None, :], b_s, float('-inf'))
        b_p = F.softmax(b_s, dim=-1).to(q.dtype)
        b_o = rearrange(einsum(b_p, gv, 'h g w, h w d -> h g d'),
                        'h g d -> (h g) d')
    o = q.new_zeros(q.shape[:-1] + (v.shape[-1],))
    o[0] = b_o
    return o.unsqueeze(0)
