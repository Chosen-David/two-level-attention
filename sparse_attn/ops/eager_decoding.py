import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

# 【E127 iter7】Triton 可选依赖：eager_decoding 是生产必经模块（env=0 也
# import），triton 缺失时只禁用 iter7 融合路径、回退 torch gather（值同
# 原路径），不破坏模块可用性。
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

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


@contextmanager
def eager_decoding_request_guard(device=None):
    """【E127 095 修复（GPT 审计 2026-10-12_0527，2026-10-12）】请求级
    fail-closed 生命周期 guard——deferred error flag 的统一消费点。

    缺陷背景：iter3 把三类契约违规（B7 空行=1 / W 超上界=2 / cu 破坏=3）
    延迟到 GPU 端 _ERR_FLAG 以换取 decode 零 host 同步，设计声明由
    eager_decoding_check_errors() 在「生成结束」一次读回显式 raise、整次
    生成作废——但修复前该函数唯一调用者在 exp/bench/{e127_stage_bench,
    test_e127_gates}，三条生产入口（benchmark/LongBench/pred.py、
    benchmark/RULER/pred_ruler.py、benchmark/RULER/llm_eval.py）均未消
    费：TLI_E127_FAST=1 下触发契约违规时，零输出/截断注意力照常完成生成
    并写盘 JSONL，fail-closed 没落到消费者。

    语义：进入时清空设备错误标志（防跨请求继承旧 flag——上一请求未消费
    的违规不得误杀/污染下一请求），退出时 check_errors() 一次性校验，
    有错显式 raise（整次生成作废由调用方传播路径保证：raise 必须先于任
    何完成外观产物的落盘）。

    生产接线（三条入口均已接入，见各文件「095 修复」注释）：
      benchmark/LongBench/pred.py    —— 每样本生成段（record append 前）
      benchmark/RULER/pred_ruler.py  —— 每行生成段（fout.write 前；raise
        后 gen 目录不 commit、.tli_gen 指针不切，best-file 口径不变）
      benchmark/RULER/llm_eval.py    —— simple_evaluate 整段（save_results 前）

    【限制（如实登记）】_ERR_FLAG 是设备级全局单例，本 guard 不做同进程
    并发请求隔离：两个并发请求共享同一 flag，A 的违规会被先退出的 B 的
    guard 捕获（fail-closed 方向正确，但报错归因到 B 的请求）。当前生
    产口径为单请求串行 benchmark（B=1、逐样本串行生成），该限制不可
    达；将来引入同进程并发请求时须改为 per-request flag 槽
    （如 dict[request_id]）再撤此登记。
    """
    for _, _buf in list(_ERR_FLAG.items()):
        _buf.zero_()
    try:
        yield
    finally:
        eager_decoding_check_errors(device)


# ---------------------------------------------------------------------------
# 【E127 iter7】eager gather Triton 融合主路径（原型 exp/bench/
# _scratch_triton_eager.py 已验证：6 配置 allclose(1e-2) 全绿、
# 0.10-0.29ms vs torch gather 0.41-0.66ms、flag 语义逐码一致）。
#
# 原 torch gather 路径 = repeat + counts + topk(W) + gather×2 + einsum×2 +
# softmax + where 等 ~15 个小 op（每 op 一次 launch + 中间张量往返 HBM）；
# 融合为两个 kernel：
#   kernel A（mask 展开）：token 级 bool mask → 每行紧凑选中索引 IDX [R, W]
#     + 行计数 COUNT [R]，chunk 循环（TP=1024）+ running cumsum；iter3 的
#     三类契约校验（B7 空行=1 / W 超上界=2 / cu 破坏=3）原地点并入——
#     atomic_max 进全局错误标志，语义/文案与 torch 路径逐码一致。
#   kernel B（gather attention）：program = (行 r × 组内头 j) 展平，单头
#     标量 online-softmax（m/l 全标量 + acc[D]，规避 Triton 不支持块内
#     标量索引赋值）；BW=256 块循环 gather K/V，cnt 超上界 clamp 到 W
#     （与 torch topk 只取 W_upper 同语义，漏选 flag=2 由 kernel A 记录）。
# 数学语义：softmax(全 S 行、未选位 -inf) ≡ softmax(选中子集)（exp(-inf)=0
# 精确）；online-softmax 与全长 softmax 仅归约顺序不同 → ULP 级漂移，
# allclose(1e-2) 门禁。fail-closed：编译/执行异常 → 置 _EAGER_TRITON_OK
# 下方标志 False，本次即回退 torch gather 路径（值同原路径）。
# ---------------------------------------------------------------------------
_EAGER_TRITON_OK = {"ok": True}

if _HAS_TRITON:

    @triton.jit
    def _eager_expand_kernel(
        MASK, IDX, COUNT, FLAG, CUK,  # MASK [R, T] bool；IDX [R, W] int32(-1)；COUNT [R]；FLAG 全局；CUK cu[2]
        T, stride_mr,
        W: tl.constexpr, TP: tl.constexpr,
    ):
        """行 r：token 级 mask → 紧凑选中索引（chunk 循环 + running cumsum）。
        校验并入：count==0 → flag 1（B7）；槽 ≥ W → flag 2（漏选保护）；
        cu ≠ [0, T] → flag 3（长度契约）。"""
        r = tl.program_id(0)
        cu0 = tl.load(CUK)
        cu1 = tl.load(CUK + 1)
        f = tl.where((cu0 == 0) & ((cu1 - cu0) == T), 0, 3)
        base = tl.zeros((), tl.int32)
        for c0 in range(0, T, TP):
            offs = c0 + tl.arange(0, TP)
            tm = offs < T
            sel = tl.load(MASK + r * stride_mr + offs, mask=tm, other=0).to(tl.int32)
            csum = tl.cumsum(sel, axis=0)
            pos = base + csum - sel                  # exclusive 位置
            w = (sel != 0) & tm & (pos < W)
            tl.store(IDX + r * W + tl.where(w, pos, 0), tl.where(w, offs, 0),
                     mask=w)
            over = (sel != 0) & tm & (pos >= W)
            f = tl.maximum(f, tl.where(tl.sum(over, axis=0) > 0, 2, 0))
            base += tl.sum(sel, axis=0)
        tl.store(COUNT + r, base)
        f = tl.maximum(f, tl.where(base == 0, 1, 0))
        tl.atomic_max(FLAG, f)

    @triton.jit
    def _eager_attn_kernel(
        IDX, COUNT, Q, K, V, OUT,
        stride_kt, stride_kh, stride_qh, stride_oh,
        G: tl.constexpr, D: tl.constexpr,
        PER_QH: tl.constexpr, BW: tl.constexpr, W: tl.constexpr,
    ):
        """单头 online-softmax gather attention。
        共享口径（PER_QH=0）：program = 行 r × 组内头 j 展平（grid=R*G），
        行 h 的组内 G 个 q-head 广播同一选择。
        per_qh 口径（PER_QH=1）：【097 修复】grid=(R,)，一 program 一行，
        pid 即行号 r——修复前固定 grid=(R*G,) 而 PER_QH 分支 j 维完全未
        用，Qwen3（H=32, G=4）时 128 个 program 对 32 行输出各重复计算/
        同址非原子 store 竞争写 4 次（同输入派生同值故数值未错，冗余 +
        写竞争 inconclusive）。
        K/V 每 program 重读（G× 读冗余，W×D×2×2B×G ≈ 4MB/行，可接受）——
        换取全标量 m/l + acc [D]，无 G 维广播/索引。"""
        pid = tl.program_id(0)
        if PER_QH:
            r = pid                # 097：per_qh 一行一 program，j 维不再参与
            kvh = r // G
            qh = r
        else:
            j = pid % G
            r = pid // G
            kvh = r
            qh = r * G + j
        cnt = tl.load(COUNT + r)
        cnt = tl.minimum(cnt, W)   # 超上界时只取前 W（漏选保护 flag=2 由 expand 记录）
        offs_d = tl.arange(0, D)
        qv = tl.load(Q + qh * stride_qh + offs_d)          # [D] fp32（已乘 scale）
        m = float('-inf')
        l = 0.0
        acc = tl.zeros((D,), tl.float32)
        for w0 in range(0, cnt, BW):
            offs_w = w0 + tl.arange(0, BW)
            wm = offs_w < cnt
            tok = tl.load(IDX + r * W + offs_w, mask=wm, other=0)
            kk = tl.load(K + tok[:, None] * stride_kt + kvh * stride_kh + offs_d[None, :],
                         mask=wm[:, None], other=0.0).to(tl.float32)
            vv = tl.load(V + tok[:, None] * stride_kt + kvh * stride_kh + offs_d[None, :],
                         mask=wm[:, None], other=0.0).to(tl.float32)
            s = tl.sum(kk * qv[None, :], axis=1)
            s = tl.where(wm, s, float('-inf'))
            m_new = tl.maximum(m, tl.max(s, axis=0))
            m_safe = tl.where(m_new == float('-inf'), 0.0, m_new)
            m_old = tl.where(m == float('-inf'), 0.0, m)
            alpha = tl.exp(m_old - m_safe)
            p = tl.exp(s - m_safe)
            p = tl.where(wm, p, 0.0)
            l = l * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * vv, axis=0)
            m = m_new
        o = acc / tl.where(l == 0.0, 1.0, l)
        tl.store(OUT + qh * stride_oh + offs_d, o.to(OUT.dtype.element_ty))


def _attn_grid_size(R: int, G: int, per_qh: bool) -> int:
    """【E127 097 修复】kernel B 的 program 数（独立纯函数供门禁直断）：
    per_qh 口径一行一 program（R 个——修复前误为 R*G，Qwen3 G=4 时
    128→32）；共享口径行×组内头展平（R*G 个，语义不变）。"""
    return R if per_qh else R * G


# ---------------------------------------------------------------------------
# 【101 修复（GPT 审计 2026-10-12_0739 TL-E127-GQA-SHAPE-101，P2）】host 侧
# GQA/shape 边界门——在任何分配/launch 之前拒绝非法输入。
#
# 缺陷背景：gather 路径用 G=H//Hkv 且仅以 mask 行数是否等于 Hkv 推断
# per-q-head，没有校验 Hkv>0、H%Hkv==0、K/V 头数一致或 R∈{Hkv,H}。审计
# 三例（无 GPU sanitize 下纯 Triton 推演 + 本机 GPU 实测复现）：
#   H=10,Hkv=3,R=10：per_qh 分支 kvh=r//G=3 ≥ Hkv=3 → 越界读 K/V
#     （连续布局下读进下一 token 的 head 0——不崩溃、静默错值，最危险）；
#   H=10,Hkv=3,R=3：共享 grid=R*G=9 而 H=10 → qh=9 从未写入（uninitialized）；
#   H=32,Hkv=8,R=9：被误判 per_qh（R≠Hkv），grid=9 → 23 个输出头未初始化。
# 正常 Qwen3 配置 H=32,Hkv=8 合法不触发；历史 E127 归档零污染——本门是
# 面向未来 kernel 接入/非法调用的内存安全 fail-closed（launch 前拒绝）。
# ---------------------------------------------------------------------------
def _validate_gqa_shape(q, k, v, mask, block_size):
    """101 host 边界门（纯 host 检查，零 GPU 同步、零分配）。

    入参为 eager_decoding_attn 的原始入参形状：q [..,H,D] / k,v [..,T,Hkv,D]
    / mask [..,R,blocks]。拒绝四类非法输入（ValueError，文案带完整参数）：
      ① Hkv<=0 或 H<=0（零头数——空 KV 头维会让一切行映射失去意义）；
      ② H % Hkv != 0（GQA 契约外，G=H//Hkv 截断——审计 H=10,Hkv=3 两例）；
      ③ K/V 头数不一致（共享口径行号直接当 head 索引，错位读）；
      ④ mask 行数 R ∉ {Hkv, H}（R=9 @ H=32,Hkv=8 误判 per_qh 例）。
    block_size 同时做正整数检查（mask 块宽 × block_size 的长度推导前提）。
    """
    def _fail(msg):
        raise ValueError(
            f"eager_decoding_attn: 非法 GQA/shape 输入（101 fail-closed，"
            f"host 侧 launch 前拒绝）——{msg}。契约：H%Hkv==0、K/V 头数"
            f"一致、mask 行数 R∈{{Hkv,H}}（R==Hkv=共享选择，R==H=per-q-"
            f"head 选择）、block_size≥1。请检查调用方传入的 q/k/v/mask "
            f"形状。"
        )
    H = q.shape[-2]
    Hkv = k.shape[-2]
    if H <= 0 or Hkv <= 0:
        _fail(f"零/负头数 H={H} Hkv={Hkv}")
    if H % Hkv != 0:
        _fail(f"H={H} 不能被 Hkv={Hkv} 整除（G={H // Hkv} 截断，per_qh 行"
              f"映射 kvh=r//G 将越界）")
    if k.shape[-2] != v.shape[-2]:
        _fail(f"K/V 头数不一致 k_hkv={k.shape[-2]} v_hkv={v.shape[-2]}")
    if int(block_size) != block_size or int(block_size) < 1:
        _fail(f"block_size={block_size} 非正整数")
    if mask.dim() < 2:
        _fail(f"mask 维度 {mask.dim()} < 2（须为 [..,R,blocks]）")
    R = mask.shape[-2]
    if R != Hkv and R != H:
        _fail(f"mask 行数 R={R} ∉ {{Hkv={Hkv}, H={H}}}（既非共享选择也非"
              f"per-q-head 选择——H=32,Hkv=8,R=9 类输入会被误判 per_qh，"
              f"grid 只写 R 行、其余输出头 uninitialized）")
    return True


def _eager_gather_triton(q, k, v, m0, per_qh, G, W_upper,
                         cu_seqlens_k, softmax_scale):
    """【E127 iter7】Triton 融合 launch 封装（原型已验证参数：
    kernel A TP=1024/num_warps=8；kernel B BW=256/num_warps=4）。
    q [1,H,D]；k/v [T,Hkv,D]；m0 [R,T] bool（token 级，block_size==1 前置
    已保证）。异常向上抛（调用方置标志回退 torch gather）。"""
    # 生产 cu 恒 int64（prepare_cu_seqlens 的 cumsum 整数提升）；kernel A
    # 原型验证口径为 int32——此处确定性 cast（2 元素 tiny kernel，长度
    # < 2^31 无溢出），值逐位同。
    if cu_seqlens_k.dtype != torch.int32:
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)
    R = m0.shape[0]
    H = q.shape[-2]
    D = k.shape[-1]
    T = k.shape[0]
    dev = q.device
    idx = torch.full((R, W_upper), -1, device=dev, dtype=torch.int32)
    cnt = torch.empty(R, device=dev, dtype=torch.int32)
    flag = _err_flag(dev)
    _eager_expand_kernel[(R,)](
        m0, idx, cnt, flag, cu_seqlens_k,
        T, m0.stride(0),
        W=W_upper, TP=1024, num_warps=8)
    # q*scale 后转 fp32（与 torch 路径的 b_q 同口径：bf16 乘 scale 舍入
    # 后升精度——数值与原型逐位同）
    qs = (q[0] * softmax_scale).to(torch.float32)   # [H, D]
    o = torch.empty((H, D), device=dev, dtype=q.dtype)
    _eager_attn_kernel[(_attn_grid_size(R, G, per_qh),)](
        idx, cnt, qs, k, v, o,
        k.stride(0), k.stride(1), qs.stride(0), o.stride(0),
        G=G, D=D, PER_QH=per_qh, BW=256, W=W_upper, num_warps=4)
    return o[None, None]


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
    【101】入口先过 _validate_gqa_shape host 边界门（Hkv<=0 / H%Hkv!=0 /
    K,V 头数不等 / mask 行数 R∉{Hkv,H} 一律 launch/分配前 ValueError
    拒绝——修复前三例非法输入静默越界读/未初始化输出）。
    """
    _validate_gqa_shape(q, k, v, mask, block_size)
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
    # 101：直呼本函数同样受 host 边界门保护（公共入口之外的第二防线）
    _validate_gqa_shape(q, k, v, mask, block_size)
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
    # 101：per_qh 判定显式化——validator 已保证 R∈{Hkv,H}，故「R==H 且
    # 非 MHA（H≠Hkv）才 per-q-head」与旧推断「R != Hkv」在全部合法输入上
    # 布尔等价（含 G=1 时 R==H==Hkv 走共享的历史语义），合法路径逐位不变。
    per_qh = m0.shape[0] == q.shape[-2] and q.shape[-2] != k.shape[-2]
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
    【101】直呼本函数同样受 _validate_gqa_shape host 边界门保护（公共入口
    之外的第二防线——Triton kernel 无 kvh<Hkv 边界检查，非法 shape 的
    越界读必须在此前拦下）。
    """
    _validate_gqa_shape(q, k, v, mask, block_size)
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
    # 101：per_qh 判定显式化（与 _dense 同口径——validator 保证 R∈{Hkv,H}
    # 下与旧推断「R != Hkv」布尔等价，合法路径逐位不变）。
    per_qh = m0.shape[0] == q.shape[-2] and q.shape[-2] != k.shape[-2]

    # 【iter7】Triton 融合主路径：T ≥ 16K（同 torch gather 路由）、
    # block_size==1（TLI 生产 mask 即 token 级，kernel A 直接读 m0 免
    # repeat）、布局/dtype 契约满足时，两个 kernel 替代下方 ~15 op 链。
    # 路由条件全部 host 已知量（零同步判走）；W_upper 判定与 torch 路径
    # 同口径（W*8>T 回退 dense，生产 W≈1344 在 16K+ 档同向）。cu 接受
    # int32/int64（生产 prepare_cu_seqlens 的 cumsum 整数提升恒为 int64，
    # helper 内确定性 cast 到 kernel 验证口径 int32）。
    # fail-closed：异常 → 置全局标志 False 并落入下方 torch gather 路径
    # （本次即回退，值同原路径，仅性能回退）。
    T_tok = full_len
    if (
        T_tok >= 16384
        and block_size == 1
        and _HAS_TRITON
        and _EAGER_TRITON_OK["ok"]
        and m0.dtype == torch.bool
        and q.shape[0] == 1
        and (k.shape[-1] & (k.shape[-1] - 1)) == 0   # D 须为 2 的幂（tl.arange）
        and m0.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
        and cu_seqlens_k.dtype in (torch.int32, torch.int64)
    ):
        W_up_tri = min(int(os.environ.get("TLI_E127_W_UPPER", "2048")), T_tok)
        if W_up_tri * 8 <= T_tok:
            try:
                return _eager_gather_triton(
                    q, k, v, m0, per_qh, G, W_up_tri,
                    cu_seqlens_k, softmax_scale)
            except Exception:
                _EAGER_TRITON_OK["ok"] = False
    # ---- 以下原 torch gather 路径（回退/契约外布局兜底，值同 iter3）----

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
