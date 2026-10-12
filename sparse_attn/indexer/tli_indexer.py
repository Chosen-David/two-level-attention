"""TLI (Two-Level Indexer)：TIA 下一代，组合三个实测 Go 的创新点。

创新点（全部有 2026-09-23 真实 trace 实测支撑，见 exp/trace/results/）：
  A  position-stable subspace L1：块 min/max 只在低频尾维 d'=32 上算
     （E3：d'=32 mass recall 0.729 ≈ 全维 0.732；random/highfreq 崩溃）
  B  far/near 分区 L2 预算：远端与近端独立 top-K2 池，防远端被近端高分挤出；
     远端用 TIA 4bit token 级精筛（E4c：≈oracle，L03 0.999）。原「远端 kmeans
     聚类代表」方案为 E4c negative result（E4b 有整簇超选 bug 虚高 4-10×，
     严格预算下 km_blk 0.09-0.39 / km_tok 0.45-0.79 均无一致优势），
     cluster 模式保留作消融（--tli_far_select cluster）
  D' 层自适应跳过：离线校准的静态层掩码，far 质量低的层把远端块从 L1 剔除
     （E6：precision 0.92-1.00，far 质量损失 <0.3%；在线信号版已否定）

继承 TIAIndexer，复用其 L2（4bit 部分维 token 精筛 + 滑窗强制）语义。
"""
import json
import os

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

from .tia_indexer import TIAIndexer
from ..info import DEFAULT_LAYER_SKIP_MASK_PATH

# 【E127 inplace 合并（用户 2026-10-12 指令）】原 TLIIndexerFast 子类
# （tli_indexer_fast.py）语义并入基类：Triton 依赖惰性导入——CPU 环境
# （无 GPU / 无 triton）不破模块导入，_HAS_TRITON=False 时三处 fast
# 分支守卫短路，全部走原版 repeat+einsum 表达式（值逐位同原版）。
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:   # triton 缺失或导入异常：fast path 整体退役
    triton = None
    tl = None
    _HAS_TRITON = False

# 081（TL-E121-OUTPUT-SNAPSHOT-081）：默认 D′ 掩码路径定义上移到
# sparse_attn/info.py（DEFAULT_LAYER_SKIP_MASK_PATH，单一事实源）——
# manifest 侧（resolve_treatment_snapshot）与运行时侧展开的必须是同
# 一个文件，此前 info 侧无从得知该路径 → 默认配置 manifest 恒 null
# 而可读名含 D（081 身份缺口 ①）。本别名保留旧引用名（既有代码/
# exp 脚本零改动），realpath 与旧定义逐位一致。
DEFAULT_MASK = DEFAULT_LAYER_SKIP_MASK_PATH

# ---- E113b：Triton 贪心 kernel（sim_greedy 臂的 Python 循环 → GPU kernel）----
#   开关：SGLANG_TLI_GREEDY_KERNEL（默认 1 = 用 kernel；0 = 回退 Python 循环，
#   供对拍/回退）。kernel 本体在 sparse_attn/indexer/greedy_triton.py（与
#   exp/trace/e113_greedy_triton.py 单一事实源），两路径语义逐位一致
#   （exp/trace/test_e113_greedy_triton.py T1-T5 对拍 + E110/E109a 回归零差异）。
#   惰性加载：非 sim_greedy 臂不 import triton（对正在跑的扫描臂零影响）。
_GREEDY_TRITON_FN = None   # None=未尝试；False=不可用（回退）；其余=greedy_pass_triton


def _load_greedy_triton():
    global _GREEDY_TRITON_FN
    if _GREEDY_TRITON_FN is None:
        try:
            from .greedy_triton import greedy_pass_triton
            _GREEDY_TRITON_FN = greedy_pass_triton
        except Exception as e:   # triton 缺失/导入失败 → 回退 Python 循环
            print(f"[TLI] E113b: Triton 贪心 kernel 加载失败（{type(e).__name__}: {e}），"
                  f"回退 Python 循环")
            _GREEDY_TRITON_FN = False
    return _GREEDY_TRITON_FN if _GREEDY_TRITON_FN else None


class TLIIndexer(TIAIndexer):
    """A+B+D' 全开为 method='tli'；参数可单独关闭做消融。"""

    def __init__(self, args, snapshot=None) -> None:
        # 081（TL-E121-OUTPUT-SNAPSHOT-081）：snapshot = 生产入口在模型
        # 加载前冻结的 TreatmentSnapshot（sparse_attn.info.
        # resolve_treatment_snapshot 产出）。提供时 D′ 段直接消费
        # snapshot.skip_ids、投影基段直接消费 snapshot.basis_tensor——
        # 不再 open/torch.load 任何路径（旧路径 = patch.py 对每个
        # attention 层各建一个实例、每实例各读一次文件，单模型跨层
        # 混代可达：patch 中途换文件 → 层间 skip 集合/基不一致）。
        # snapshot=None 保留旧路径读取行为（既有非 benchmark 入口不破），
        # 但生产入口（benchmark pred.py / pred_ruler.py）必须注入。
        super().__init__(args)
        self.layer_idx = None            # 由 register_patch 注入
        # ---- A：子空间维度选择（2026-09-29 用户指令：参数化 rope/nope/full，默认 full）----
        #   full = 全 128 维（默认）；rope = 前 64 旋转维；nope = 后 64 非旋转维；
        #   tail = 旧口径低频尾维 32（E71/B7/C0 主表口径，向后兼容保留）
        self.subspace = getattr(args, "tli_subspace", "full")
        assert self.subspace in ("full", "rope", "nope", "tail", "random", "highfreq"), \
            f"tli_subspace 非法取值: {self.subspace}"
        self.enable_subspace = getattr(args, "tli_enable_subspace", True)
        if self.subspace == "full":
            self.enable_subspace = False      # 全维（用户默认）
        elif self.subspace in ("rope", "nope"):
            self.enable_subspace = True       # 显式子空间段
        # ---- E134：L2 细筛子空间独立开关（子空间矩阵第三臂 sub+full）----
        #   same = L2 与 L1 同一子空间（现状，E85f 口径：粗筛+细筛同特征）
        #   full = L2 细筛 k_qat 全 128 维量化（L1 仍由 tli_subspace 决定）
        self.l2_subspace = getattr(args, "tli_l2_subspace", "same")
        assert self.l2_subspace in ("same", "full"), \
            f"tli_l2_subspace 非法取值: {self.l2_subspace}"
        # ---- B：远端 kmeans（far_select='cluster' 时才构建，默认 4bit 分区）----
        self.enable_kmeans = getattr(args, "tli_enable_kmeans", True)
        self.far_select = getattr(args, "tli_far_select", "4bit")
        self.far_clusters = getattr(args, "tli_far_clusters", 256)
        self.far_niter = getattr(args, "tli_far_niter", 10)
        self.far_blocks = getattr(args, "tli_far_blocks", 16)   # 远端块预算 K1_far
        self.far_tokens = getattr(args, "tli_far_tokens", 512)  # 远端 token 预算 K2_far
        self.near_len = 2048                                      # 近端窗（D'/B 共用）
        # ---- E64j method 组合参数化（用户命名: mminmax/cavg/aavg 等）----
        self.far_method = getattr(args, "tli_far_method", "minmax")
        self.near_method = getattr(args, "tli_near_method", "avg")
        # ---- E110：ccluster / sim_greedy（用户 2026-10-07 定义，TASK.md「关于cluster的方法」节）----
        #   far_select/near_select 独立选择 L2 侧「簇代表打分」选择方式：
        #     ccluster     = (far=cluster,    near=cluster)
        #     cavg_sim     = (far=sim_greedy, near=4bit)  —— 与 cavg 只差聚类方式
        #     ccluster_sim = (far=sim_greedy, near=sim_greedy)
        #   默认双双 '4bit'（现状，回归保护：不开新 flag 行为逐位不变）。
        #   注意：cluster/sim_greedy 臂须 --tli_enable_kmeans true（沿用 cavg 惯例）。
        self.near_select = getattr(args, "tli_near_select", "4bit")
        assert self.near_select in ("4bit", "cluster", "sim_greedy"), \
            f"tli_near_select 非法取值: {self.near_select}"
        self.sim = getattr(args, "tli_sim", 0.9)
        self.sim_dims = getattr(args, "tli_sim_dims", "subspace")
        assert self.sim_dims in ("subspace", "nope", "tail", "full"), \
            f"tli_sim_dims 非法取值: {self.sim_dims}"
        # ---- E87：top-σ 选择（far/near/mid 侧不做两级，细筛分 ≥ sink max − σ 即选中）----
        self.sigma_select = getattr(args, "tli_sigma_select", "none")
        self.moba_gate = getattr(args, "tli_moba", False)
        self.sigma = getattr(args, "tli_sigma", 8.0)
        self.sink_blocks = 2                                      # sink = 前 2 块（128 tok）
        if self.near_select != "4bit":
            # E110：near 侧簇分路径与 σ/MoBA 均绕过/改写两级选择语义，显式互斥
            assert self.sigma_select == "none", \
                "--tli_near_select 与 --tli_sigma_select 互斥（σ 路径绕过两级选择）"
            assert not self.moba_gate, "--tli_near_select 与 --tli_moba 互斥"
        # ---- E103：kv-head 共享消融（审稿 MAJOR）----
        #   论文 §2.2 口径：L1/L2 分数先在 GQA 组内 mean 聚合到 kv-head 级再 topk
        #   （32 q-head → 8 kv-head，索引量省 4 倍）。本开关打开时旁路全部聚合点，
        #   per-q-head 分数直接进 topk（topk 沿最后一维，H 维天然独立），
        #   mask 从 [1,1,Hkv,T] 变 [1,1,H,T]（消费端 eager_decoding 按组拆回）。
        #   互斥守卫：cluster/moba/σ 路径的簇分数与 gate 均为 kv-head 级语义，
        #   与 per-q-head 组合无定义，显式禁止（E103 臂不经过这些路径）。
        self.per_q_head = getattr(args, "tli_per_q_head", False)
        if self.per_q_head:
            assert not self.moba_gate, "--tli_per_q_head 与 --tli_moba 互斥（gate 是 kv-head 级语义）"
            assert self.far_select == "4bit", \
                "--tli_per_q_head 与 --tli_far_select cluster/sim_greedy 互斥（簇分数是 kv-head 级语义）"
            assert self.near_select == "4bit", \
                "--tli_per_q_head 与 --tli_near_select cluster/sim_greedy 互斥（簇分数是 kv-head 级语义）"
            assert self.sigma_select == "none", \
                "--tli_per_q_head 与 --tli_sigma_select 互斥（E103 臂不经过 σ 路径）"
        # ---- D'：静态层掩码 ----
        self.enable_layer_skip = getattr(args, "tli_enable_layer_skip", True)
        self.skip_far = False
        self._skip_ids: set[int] | None = None
        if snapshot is not None:
            # 081：注入已解析对象——掩码内容已在 resolve 阶段单次 open
            # 解析（同一批 bytes 出 sha256 + skip 集合），此处零文件 IO。
            if self.enable_layer_skip:
                if snapshot.skip_ids is None:
                    # resolve 阶段对不可读掩码已 fail closed（[GATE-FAIL]
                    # 079 SystemExit），到达此处即 snapshot 与 args 失配
                    # ——不允许静默关 D'（079 静默退化口径在注入路径同样
                    # 不放行）。
                    raise SystemExit(
                        "[GATE-FAIL] 081: snapshot.skip_ids 为 None 但 "
                        "tli_enable_layer_skip=True——注入的快照与 args "
                        "失配（掩码不可用应在 resolve_treatment_snapshot "
                        "阶段 fail closed 退出），拒绝运行")
                self._skip_ids = set(snapshot.skip_ids)
        else:
            # 旧路径（非 benchmark 入口兼容）：每实例各 open 一次。
            mask_path = getattr(args, "tli_layer_skip_path", None) or DEFAULT_MASK
            if self.enable_layer_skip:
                try:
                    with open(mask_path) as f:
                        self._skip_ids = set(json.load(f)["skip"])
                except (OSError, KeyError, ValueError):
                    print(f"[TLI] 警告: 层掩码 {mask_path} 不可读, D' 关闭")
                    self._skip_ids = None
        # B 的聚类缓存（prefill 后一次；decode 期 far 区不变）
        self._km_centroids = None    # [Hkv, K_c, d']
        self._km_token_assign = None  # [Hkv, Tfar]
        self._km_far_lo = None       # far 区起点（token idx）
        self._km_far_blk_lo = None   # far 区块起点
        self._km_blk_ids = None      # [Hkv, Tfar] 每个 far token 的块 id
        self._km_dims = None         # E110：far 侧建簇时实际用的维度索引（打分端对齐用）
        # E110：near 侧聚类缓存（ccluster；near 区左右缘 decode 期都会动 → 按 key 全量重建）
        self._km_near_centroids = None   # [Hkv, K_c, d']
        self._km_near_assign = None      # [Hkv, Tn]（相对 _km_near_lo 的 token 偏移）
        self._km_near_lo = -1            # near 簇覆盖区起点（token idx，块对齐）
        self._km_near_hi = -1            # near 簇覆盖区终点（token idx，块对齐）
        self._km_near_key = None         # (far_hi, near_hi) 重建 key
        self._km_near_dims = None        # near 侧建簇维度索引
        # E110：far 侧 sim_greedy 增量状态（贪心是时序过程，可对新增段精确续跑）
        self._km_greedy_sums = None      # [Hkv, Kcap, d']  簇成员和（簇心 = sums/cnt）
        self._km_greedy_cnt = None       # [Hkv, Kcap]
        self._km_greedy_sq = None        # [Hkv, Kcap]  ||sums||^2 增量维护（免每步全量 norm）
        self._km_greedy_klive = None     # [Hkv] 活簇数
        # 【E127 inplace 合并（用户 2026-10-12 指令）】原 TLIIndexerFast
        # 子类语义并入基类：decode 增量缓存状态 + Triton fast path 健康标志
        # （异常置 False 后本实例永久回退原版表达式，fail-closed）+
        # quant indices 缓存。_inc=None 表示无缓存（首步全量建）。
        self._inc = None
        self._triton_ok = True
        self._qat_idx_cache = None
        # ---- E64 框架参数化：α/β/γ 分区 + sup_wsvd 投影基（B 配置帕累托点）----
        self.alpha = getattr(args, "tli_alpha", 0.0)
        self.beta = getattr(args, "tli_beta", 0.0)
        # E122：tli_gamma 可为 None（CLI 'off'，大小写不敏感）= 自由竞争模式
        # （取消 L2 near/far 配额分割，mid 单池 topk，见 compute_mask use_partition
        # 分支的 γ=off 段）；数值照旧 float。缺省 1.0 兼容不传该参数的旧 harness。
        self.gamma = getattr(args, "tli_gamma", 1.0)
        self._basis_all = None
        bp_path = getattr(args, "tli_proj_basis", None)
        if bp_path:
            if self.subspace != "tail":
                raise ValueError(
                    f"tli_proj_basis 定义在 tail-32 子空间上，与 tli_subspace={self.subspace} "
                    "不兼容（投影基的输入是 32 维特征）；请用 --tli_subspace tail"
                )
            if snapshot is not None:
                # 081：注入已解析的投影基（resolve 阶段单次 open，SHA 与
                # tensor 同一批 bytes）——零文件 IO；快照缺基即与 args
                # 失配，fail closed。
                if snapshot.basis_tensor is None:
                    raise SystemExit(
                        "[GATE-FAIL] 081: snapshot.basis_tensor 为 None 但 "
                        "args.tli_proj_basis 已声明——注入的快照与 args 失配"
                        "（快照必须在 tli_proj_basis 声明的同一 args 上"
                        "resolve），拒绝运行")
                self._basis_all = snapshot.basis_tensor
            else:
                # 旧路径（非 benchmark 入口兼容）
                self._basis_all = torch.load(bp_path, map_location="cpu", weights_only=True).float()
            self.enable_subspace = True  # 投影基定义在子空间内，强制开启
            print(f"[TLI] 投影基已加载 {bp_path} shape={list(self._basis_all.shape)}")
        self._basis = None   # 当前层子空间基 [Hkv, 32, r]（惰性提取于 prepare_index）
        # ---- E85f：per-layer 静态 pair 选取（e2e 判决）----
        #   E85e 重放：prefill 尾 256 q 平均 |q| 的 pair 幅值 |q̄_j|+|q̄_{j+64}|
        #   top-16 完整对，层固定、可索引化，recall 0.849 vs tail32 0.811。
        #   E85f 判决问题：该增益在 e2e（decode q 分布漂移下）是否保住
        #   （E66 投影基 trace 成立 e2e 崩的前车之鉴）。pair 未定前回退 tail32。
        self.static_pair = getattr(args, "tli_static_pair", False)
        self._pair_idx = None
        if self.static_pair:
            if self.subspace != "tail":
                raise ValueError(
                    f"--tli_static_pair 需 --tli_subspace tail（当前 {self.subspace}）；"
                    "静态 pair 是 tail32 的同宽度替换，非独立子空间段"
                )
            self.enable_subspace = True

    # ---------------- E85f：prefill q 统计 → per-layer 静态 pair ---------------- #

    def observe_prefill_q(self, q):
        """prefill 分支旁路采集（qwen3_attn_patch 调用）：post-RoPE q [B,H,S,D]。
        取尾 256 行全体 head 展平的平均 |q|，按 pair 幅值 |q̄_j|+|q̄_{j+64}|
        取 top-16 完整旋转对（与 E85e 重放口径一致）。只算一次（首个完整 prefill）。
        """
        if self._pair_idx is not None or not self.static_pair:
            return
        qm = q.detach().float()[:, :, -256:].reshape(-1, q.shape[-1]).abs().mean(0)  # [128]
        pair_mag = qm[:64] + qm[64:]                     # [64]（频率 j）
        top_pairs = torch.topk(pair_mag, 16).indices     # 频率 j
        self._pair_idx = torch.cat([top_pairs, top_pairs + 64]).sort().values.to(q.device)
        if os.environ.get("TLI_DEBUG") and self.layer_idx is not None:
            print(f"[E85f] L{self.layer_idx} static pairs (freq j): "
                  f"{sorted(top_pairs.tolist())}", flush=True)

    # ---------------- D' ---------------- #

    def _resolve_skip(self):
        if self._skip_ids is not None and self.layer_idx is not None:
            self.skip_far = self.layer_idx in self._skip_ids

    # ---------------- A：子空间 L1 ---------------- #

    def _subspace_indices(self, device):
        # 子空间选择（用户指令参数化）：rope=前 64 旋转维 / nope=后 64 非旋转维 /
        # tail=低频尾维（旧口径，cmp_ratio 控制宽度）/ full 不进此函数
        # E85f：静态 pair 已定（prefill q 统计）→ 用 per-layer 完整旋转对替换 tail32
        if self.static_pair and self._pair_idx is not None:
            return self._pair_idx.to(device)
        if self.subspace == "rope":
            return torch.arange(0, 64, device=device)
        if self.subspace == "nope":
            return torch.arange(64, 128, device=device)
        # E90 消融臂（对齐 E3 mass 口径）：random=固定 seed 随机 32 维 /
        # highfreq=头 16+头 16（RoPE 旋转对前元素，最高频对）
        if self.subspace == "random":
            g = torch.Generator().manual_seed(20261002)
            return torch.randperm(128, generator=g)[:32].sort().values.to(device)
        if self.subspace == "highfreq":
            delta = 64 // self.cmp_ratio
            return torch.tensor(
                list(range(0, delta)) + list(range(64, 64 + delta)), device=device)
        delta = 64 // self.cmp_ratio
        return torch.tensor(
            list(range(64 - delta, 64)) + list(range(128 - delta, 128)), device=device
        )

    def prepare_index(self, k, cu_seqlens_k):
        """【E127 inplace 合并（用户 2026-10-12 指令）】decode 热路径增量缓存
        入口（原 TLIIndexerFast.prepare_index 语义原样并入，env 门退役——
        fast 成为唯一路径）：k append-only 且前置条件满足时尾段增量重算，
        否则全量。逐位一致性为数学保证（块内独立归约 + quant 逐 token 独立）。
        契约（fail-closed）：
          - k append-only（HF DynamicCache decode 语义）；S_now ≤ S_prev
            或任一前置不符 → 全量重建缓存。
          - clear() 使 _inc 失效 → 新请求首步全量。
          - 投影基/moba/异形 D≠128 一律全量（值同原版路径，逐位不变）。
        """
        inc = getattr(self, "_inc", None)
        S_now = k.shape[1]
        if (
            inc is not None
            and S_now > inc["S"]
            and self._basis is None
            and self._basis_all is None
            and not self.moba_gate
            and k.shape[-1] == 128
            and inc["sub_key"] == self._sub_key()
            and (not self._need_avg_score() or inc["k_avg"] is not None)
        ):
            return self._prepare_index_inc(k, cu_seqlens_k, inc)
        return self._prepare_index_full(k, cu_seqlens_k)

    def _sub_key(self):
        """子空间一致性 key：跨步 idx_sub 不变才可增量（static_pair 的
        _pair_idx 在 prefill 后冻结；rope/nope/tail 为常量分支）。"""
        if not self.enable_subspace:
            return ("off", 0)
        if self.static_pair and self._pair_idx is not None:
            return ("pair", int(self._pair_idx.numel()))
        return (self.subspace, self.cmp_ratio)

    def _prepare_index_full(self, k, cu_seqlens_k):
        d = self._prepare_index_base(k, cu_seqlens_k)
        # 建缓存（k_qat 仅非投影路径可作增量 buffer——投影路径 k_qat 为
        # 全维量化特征，散布语义不同；_basis 非 None 时 can_inc 已挡，
        # 此处再防御一次）
        k_qat_buf = None
        if self._basis is None and not self.moba_gate and k.shape[-1] == 128:
            k_qat_buf = d["k_qat"].squeeze(0).clone() if d["k_qat"].dim() == 4 \
                else None
        self._inc = {
            "S": k.shape[1],
            "k_min": d["k_min"],
            "k_max": d["k_max"],
            "k_avg": d.get("k_avg"),
            "k_qat_buf": k_qat_buf,
            "k_qat_cap": k.shape[1] if k_qat_buf is not None else 0,
            "sub_key": self._sub_key(),
        }
        return d

    def _prepare_index_inc(self, k, cu_seqlens_k, inc):
        """【E127 iter4】增量缓存：复用块 [0, fb) + 尾段重算（pad 语义
        与全量版一致）。b_k_qat 物化缓存 + pad_cu 向量化 + quant indices
        缓存（原每步 H2D 构造）均为逐位等价替换。"""
        self._resolve_skip()   # 与原版头部同构（幂等；影响 _maybe_build_kmeans 条件）
        bs = self.args.tia_block_size
        S_prev, S_now = inc["S"], k.shape[1]
        fb = (S_prev - 1) // bs        # 第一个可能变更的块（含新 token 的块）
        lo = fb * bs                   # 复用前缀 = token [0, lo)（块 [0, fb)）
        idx_sub = self._subspace_indices(k.device) if self.enable_subspace else None

        # ---- 尾段 [lo, S_now) 重算（pad 语义与全量版一致：块对齐、pad 0）----
        # 【iter5】pad_cu 向量化：原 prepare_pad_mask 对 seqlens 逐元素
        # Python 迭代（GPU 张量迭代 = 隐式 .item() host 同步，每步一次）；
        # 整数算术逐位等价：pad = (x+bs-1)//bs*bs，cumsum 前接 0。
        _seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        _pad_seqlens_k = (_seqlens_k + bs - 1) // bs * bs
        pad_cu_seqlens_k = torch.cat(
            (_pad_seqlens_k.new_zeros((1,)), _pad_seqlens_k.cumsum(dim=0)))
        tail_k = k[:, lo:S_now]                              # [1, n, Hkv, D]
        # 尾段块数与 pad：与全量版一致 = ceil(S_now/bs)*bs 覆盖下的尾块集
        # （lo 为块边界，前缀块已对齐——尾段 pad 量 = 全量 pad 量）
        pad_total = (bs - S_now % bs) % bs
        tail_pad = tail_k.new_zeros((1, S_now - lo + pad_total) + tail_k.shape[2:])
        tail_pad[:, : S_now - lo] = tail_k
        t_coarse = rearrange(
            tail_pad[..., idx_sub] if idx_sub is not None else tail_pad,
            "b (t bs) h d -> b t bs h d", bs=bs,
        )
        k_min = torch.cat([inc["k_min"][:, :fb], t_coarse.amin(dim=2)], dim=1)
        k_max = torch.cat([inc["k_max"][:, :fb], t_coarse.amax(dim=2)], dim=1)
        k_avg = None
        if self._need_avg_score():
            k_avg = torch.cat([inc["k_avg"][:, :fb], t_coarse.mean(dim=2)], dim=1)

        # ---- k_qat 增量：预分配 buffer 步进写（尾段 zero_ + scatter， ----
        # 与全量版 zeros_like+scatter 逐位同构）
        # E134：l2_subspace=full 时尾段全维量化直写（buffer [cap,Hkv,128]
        # 整段赋值，与全量版全维量化逐位同构）。
        indices = idx_sub if idx_sub is not None else self._qat_indices(k.device)
        buf, cap = inc["k_qat_buf"], inc["k_qat_cap"]
        if S_now > cap:   # 扩容（翻倍摊销）
            new_cap = max(2 * cap, S_now)
            new_buf = buf.new_zeros((new_cap,) + buf.shape[1:])
            new_buf[:S_prev] = buf[:S_prev]
            inc["k_qat_buf"], inc["k_qat_cap"] = new_buf, new_cap
            buf = new_buf
        if self.l2_subspace == "full":
            buf[lo:S_now] = self.min_max_per_token_quant(
                k[0, lo:S_now])                             # 全 128 维量化
        else:
            buf[lo:S_now].zero_()
            buf[lo:S_now, :, indices] = self.min_max_per_token_quant(
                k[0, lo:S_now, :, indices])
        k_qat = buf[:S_now].unsqueeze(0)                     # [1, S, Hkv, D] 视图

        index_dict = {
            "k_min": k_min,
            "k_max": k_max,
            "k_avg": k_avg,
            "k_avg_full": None,
            "k_qat": k_qat,
            "cu_seqlens_k_coarse": pad_cu_seqlens_k // bs,
            "cu_seqlens_k_fine": cu_seqlens_k,
            "valid_length": k.shape[1],
        }
        if (self.enable_kmeans and not self.skip_far
                and (self.far_select != "4bit" or self.near_select != "4bit")):
            self._maybe_build_kmeans(k, index_dict)

        # 缓存滚动（cat 产物作为新缓存；k_qat buffer 原地复用）
        inc["S"] = S_now
        inc["k_min"], inc["k_max"], inc["k_avg"] = k_min, k_max, k_avg
        return index_dict

    def _qat_indices(self, device):
        """【iter5】非子空间 quant indices 缓存（原每步 list→tensor 的
        pageable H2D 构造；cmp_ratio 为实例级常量 → 命中后零构造零拷贝）。
        GPU 端 arange 构造（纯 kernel 零 H2D，int64 与原 torch.tensor(list)
        同 dtype，值逐位同）。"""
        key = (self.cmp_ratio, device)
        cache = getattr(self, "_qat_idx_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        delta = 64 // self.cmp_ratio
        idx = torch.cat([torch.arange(64 - delta, 64, device=device),
                         torch.arange(128 - delta, 128, device=device)])
        self._qat_idx_cache = (key, idx)
        return idx

    def _prepare_index_base(self, k, cu_seqlens_k):
        from .utils import prepare_pad_mask, pad_tensor
        assert k.shape[0] == 1
        self._resolve_skip()
        idx_sub = self._subspace_indices(k.device) if self.enable_subspace else None

        pad_mask, pad_cu_seqlens_k = prepare_pad_mask(self.args.tia_block_size, cu_seqlens_k)
        pad_k = pad_tensor(k, pad_mask)
        # ---- 投影基惰性提取（每层一次）：full-D [128,r] → 子空间 [32,r] ----
        if self._basis_all is not None and self._basis is None and self.layer_idx is not None:
            li = min(self.layer_idx, self._basis_all.shape[0] - 1)
            # 校准时 W[:, D2I, :] = B_sub，故 [:, idx_sub, :] 精确还原子空间基
            idx_cpu = idx_sub.cpu() if idx_sub is not None else slice(None)
            self._basis = self._basis_all[li][:, idx_cpu, :].to(k.device)  # [Hkv, 32, r]
        k_for_coarse = pad_k[..., idx_sub] if idx_sub is not None else pad_k
        if self._basis is not None:
            # 投影路径（E64f 协议）：粗筛+细筛都在 d 维投影特征上
            kp = torch.einsum("bshd,hde->bshe", k_for_coarse.float(), self._basis)
            k_coarse = rearrange(kp, "b (t bs) h d -> b t bs h d", bs=self.args.tia_block_size)
            k_min = k_coarse.amin(dim=2)
            k_max = k_coarse.amax(dim=2)
            cu_seqlens_k_coarse = pad_cu_seqlens_k // self.args.tia_block_size
            # L2 4bit 精筛：投影特征逐 token 量化（任意 last-dim 通用）
            k_qat = self.min_max_per_token_quant(kp)
            index_dict = {
                "k_min": k_min,
                "k_max": k_max,
                "k_qat": k_qat,
                "cu_seqlens_k_coarse": cu_seqlens_k_coarse,
                "cu_seqlens_k_fine": cu_seqlens_k,
                # 【075 修复 2026-10-10（GPT 审计 TL-E121-PROJ-PAD-S-075）】投影
                # 路径 k_qat 从 pad_k 派生 → score_fine 宽度 = Tpad（pad 后），
                # 真实长度必须显式携带（= pad 前 k 的 token 数；B=1 下与
                # cu_seqlens_k[-1]-cu_seqlens_k[0] 相等），供 compute_mask 把
                # near/far 边界与 SWA 强制区落在真实域。
                "valid_length": k.shape[1],
            }
            if self._need_avg_score():
                # near/far 粗筛 avg 分数（E64f：near=avg / far=minmax；far=avg 单池也需）
                index_dict["k_avg"] = k_coarse.mean(dim=2)
            # ---- B：远端/近端聚类（cluster/sim_greedy 消融模式构建；默认 4bit 分区无需聚类）----
            if (self.enable_kmeans and not self.skip_far
                    and (self.far_select != "4bit" or self.near_select != "4bit")):
                self._maybe_build_kmeans(k, index_dict)
            return index_dict
        k_coarse = rearrange(
            k_for_coarse, "b (t bs) h d -> b t bs h d", bs=self.args.tia_block_size
        )
        k_min = k_coarse.amin(dim=2)
        k_max = k_coarse.amax(dim=2)
        if self._need_avg_score():
            k_avg = k_coarse.mean(dim=2)
        else:
            k_avg = None
        # E89 MoBA 臂：全维 chunk-mean（gate 分数用完整 128 维 pad_k，严格 MoBA 口径）
        k_avg_full = None
        if self.moba_gate:
            kc_full = rearrange(
                pad_k, "b (t bs) h d -> b t bs h d", bs=self.args.tia_block_size
            )
            k_avg_full = kc_full.mean(dim=2)
        cu_seqlens_k_coarse = pad_cu_seqlens_k // self.args.tia_block_size

        assert k.shape[-1] == 128
        # E85f 口径修正：L2 细筛 k_qat 与 L1 粗筛用同一子空间（idx_sub）——
        # E85e 重放口径是粗筛+细筛同特征；tail 臂 idx_sub 与旧硬编码 tail32
        # 逐位相同（cmp_ratio=2），仅 rope/nope 探索臂与静态 pair 臂行为改变
        # E134：--tli_l2_subspace full = L2 细筛全 128 维量化（L1 仍由 idx_sub
        # 决定）——子空间矩阵第三臂 sub+full 的消费端，score_fine 宽度 = 全维。
        if self.l2_subspace == "full":
            k_qat = self.min_max_per_token_quant(k)   # [1, S, Hkv, 128] 全维量化
        else:
            delta = 64 // self.cmp_ratio
            indices = idx_sub if idx_sub is not None else torch.tensor(
                list(range(64 - delta, 64)) + list(range(128 - delta, 128))
            ).to(k.device)
            k_qat = torch.zeros_like(k)
            k_qat[..., indices] = self.min_max_per_token_quant(k[..., indices])
        index_dict = {
            "k_min": k_min,
            "k_max": k_max,
            "k_avg": k_avg,
            "k_avg_full": k_avg_full,
            "k_qat": k_qat,
            "cu_seqlens_k_coarse": cu_seqlens_k_coarse,
            "cu_seqlens_k_fine": cu_seqlens_k,
            # 075：非投影路径 k_qat 从真实 k 派生，valid_length == score_fine
            # 宽度 → compute_mask 行为逐位不变（对称携带，接口统一）
            "valid_length": k.shape[1],
        }
        # ---- B：远端/近端聚类（cluster/sim_greedy 消融模式构建；默认 4bit 分区无需聚类）----
        if (self.enable_kmeans and not self.skip_far
                and (self.far_select != "4bit" or self.near_select != "4bit")):
            self._maybe_build_kmeans(k, index_dict)
        return index_dict

    def _maybe_build_kmeans(self, k, index_dict):
        S = k.shape[1]
        # far_lo 对齐 sink 边界（sink_blocks×block_size），与 compute_mask 的远端块区间一致
        far_lo = self.sink_blocks * self.args.tia_block_size
        # 修复（2026-09-29 E72 预检）：far_hi 判据块对齐——decode 时 S 每步 +1，
        # 若用 token 级 far_hi = S - near_len 则每步每层重建 kmeans（实测 61s/step）。
        # far 区真正参与选择的是 [far_lo, near_blks) 块区间（compute_mask 口径），
        # 新 token 落在 near 区不改变 far 块集合 → 块对齐后缓存跨 step 命中。
        # E7 已证：far 区内容不变时复用聚类无损（增量 assign recall 衰减 ≤0.03）。
        # near_len_dyn 与 compute_mask 同源（α>0 时 = α·mid_len），保证簇 token
        # 覆盖范围与消费端 far_tok_hi = near_blks·bs 一致。
        bs = self.args.tia_block_size
        swa_tok = self.sliding_window_size
        # 【B10 修复 2026-10-10】与 compute_mask 同源分支化：e64 分区臂
        # far_hi 从 swa 起点（S − swa_tok）推（near 区实际宽 = α·mid_len），
        # 保持簇覆盖范围与消费端 far_tok_hi = near_blks·bs 一致；(0,0)/老逻辑
        # 分支保持旧式（S − near_len_dyn）逐位不变。
        if self.alpha > 0 and self.beta > 0:
            sink_tok = self.sink_blocks * bs
            mid_len = max(0, S - sink_tok - swa_tok)
            near_len_dyn = max(bs, int(self.alpha * mid_len))
            near_base = S - swa_tok
        elif self.far_select != "4bit" and self.alpha == 0 and self.beta == 0:
            # TASK.md L231 严格口径：cluster 组合 (0,0) 单池点 = 纯 cluster 全 mid
            # （near 区空；swa 仍正交强制不占预算）
            near_len_dyn = swa_tok
            near_base = S
        else:
            near_len_dyn = self.near_len
            near_base = S
        far_hi_blk = max(self.sink_blocks + 1, (near_base - near_len_dyn) // bs)
        far_hi = far_hi_blk * bs
        # ---- E110：far/near 两侧各自独立构建 ----
        #   far=cluster（cavg 原路径，原样保留）/ far=sim_greedy（增量贪心，见下）
        #   near=cluster/sim_greedy（ccluster，复用 far 侧逻辑，见 _update_near_cluster）
        if self.far_select == "sim_greedy":
            self._update_far_greedy(k, far_lo, far_hi)
        elif self.far_select == "cluster":
            # 已缓存且 far 块区间未增长（decode 新 token 全在近端）→ 复用
            if not (self._km_centroids is not None and self._km_far_hi_cached == far_hi):
                self._build_far_kmeans(k, far_lo, far_hi)
        if self.near_select in ("cluster", "sim_greedy"):
            self._update_near_cluster(k, far_hi)

    def _build_far_kmeans(self, k, far_lo, far_hi):
        """far 侧 kmeans 建簇（cavg 原逻辑抽出，行为逐位不变；E110 加 _km_dims 记录）。"""
        self._km_far_hi_cached = far_hi
        k_f = k[0, far_lo:far_hi].float()  # [Tfar, Hkv, D]
        idx_sub = self._subspace_indices(k.device)
        k_sub = k_f[..., idx_sub]           # [Tfar, Hkv, d']
        Hkv = k_sub.shape[1]
        Tfar = k_sub.shape[0]
        if Tfar < self.far_clusters * 4:
            self._km_centroids = None
            return
        centroids, assign = [], []
        for h in range(Hkv):
            c, a = self._gpu_kmeans(k_sub[:, h, :], self.far_clusters, self.far_niter)
            centroids.append(c)
            assign.append(a)
        self._km_centroids = torch.stack(centroids)      # [Hkv, K_c, d']
        self._km_token_assign = torch.stack(assign)      # [Hkv, Tfar]
        self._km_far_lo = far_lo
        self._km_far_blk_lo = far_lo // self.args.tia_block_size
        self._km_dims = idx_sub                            # E110：打分端 q 取同维（对齐建簇口径）
        # 每个 far token 的块 id（用于簇分数 → 块分数的 scatter-max）
        tok_blk = torch.arange(Tfar, device=k.device) // self.args.tia_block_size
        self._km_blk_ids = tok_blk.unsqueeze(0).expand(Hkv, Tfar)

    def _update_far_greedy(self, k, far_lo, far_hi):
        """E110：far 侧 sim_greedy（增量贪心聚类，E108 probe greedy_cluster_assign
        的 e2e 移植，语义逐位一致）。

        贪心是时序过程：token i 与「i 时刻的簇心」（running mean）比较。decode 期
        far_hi 块对齐前移时只对新增段 [old_far_hi, far_hi) 续跑增量指派——新 token
        与当前簇心比较恰为严格时序语义，故续跑 ≡ 全量重放（逐位一致）。
        这使 decode 期贪心代价 = 新增段长度（~1 块），而非全 far 区重放。
        """
        if self._km_greedy_sums is not None and self._km_far_hi_cached == far_hi:
            return  # 缓存命中（decode 每步 far_hi 不动时零成本）
        idx = self._sim_dims_indices(k.device)
        if (self._km_greedy_sums is None or far_hi <= self._km_far_hi_cached
                or far_hi - far_lo < 128):
            # 冷启动 / far 收缩（clear 后跨请求）/ far 区过小 → 全量重建
            self._km_greedy_sums = self._km_greedy_cnt = self._km_greedy_sq = None
            self._km_greedy_klive = None
            self._km_token_assign = None
            if far_hi - far_lo < 128:
                self._km_centroids = None
                self._km_far_hi_cached = far_hi
                return
            x = k[0, far_lo:far_hi].float()[..., idx]      # [Tfar, Hkv, dd]
            H, Tfar, dd = x.shape[1], x.shape[0], x.shape[2]
            sums = x.new_zeros(H, Tfar, dd)                # Kcap = Tfar（最坏每 token 一簇）
            cnt = x.new_zeros(H, Tfar)
            sq = x.new_zeros(H, Tfar)
            k_live = torch.zeros(H, dtype=torch.long, device=x.device)
            sums, cnt, sq, k_live, assign = self._greedy_cluster_pass(
                x, self.sim, sums, cnt, sq, k_live)
            self._km_token_assign = assign
        else:
            # 增量续跑：只处理新增段（严格时序语义的精确延续）
            x = k[0, self._km_far_hi_cached:far_hi].float()[..., idx]
            sums, cnt, sq, k_live, assign_new = self._greedy_cluster_pass(
                x, self.sim, self._km_greedy_sums, self._km_greedy_cnt,
                self._km_greedy_sq, self._km_greedy_klive)
            self._km_token_assign = torch.cat(
                [self._km_token_assign, assign_new], dim=1)
        self._km_greedy_sums, self._km_greedy_cnt = sums, cnt
        self._km_greedy_sq, self._km_greedy_klive = sq, k_live
        self._km_far_hi_cached = far_hi
        self._km_far_lo = far_lo
        self._km_dims = idx
        # 簇代表 = 成员算术均值（零簇行无害：assign 永不指向空槽）
        self._km_centroids = sums / cnt.clamp(min=1).unsqueeze(-1)

    def _update_near_cluster(self, k, far_hi):
        """E110：near 侧聚类（ccluster，复用 far 侧逻辑：kmeans 或 sim_greedy）。

        near 区 = [far_hi, near_hi)，near_hi 取块对齐 swa 起点（(S-swa)//bs·bs）。
        与 far 侧不同：near 区 decode 期右缘每 token 前移、左缘随 far_hi 前移，
        均值簇心无法精确撤销成员 → near 侧不做增量，按 key=(far_hi, near_hi)
        变化全量重建（near 区 ~α·mid 较小，重建代价可控；重建间隙的右缘新 token
        由消费端用细筛原始分回退覆盖——细筛分质量高于簇代表分，语义无损偏保守）。
        """
        S = k.shape[1]
        bs = self.args.tia_block_size
        near_hi = max(far_hi, (S - self.sliding_window_size) // bs * bs)
        if self._km_near_key == (far_hi, near_hi) and self._km_near_centroids is not None:
            return
        self._km_near_key = (far_hi, near_hi)
        self._km_near_centroids = None
        self._km_near_assign = None
        Tn = near_hi - far_hi
        if Tn < 256:   # near 区过小（< 4 块）不值得聚类，消费端回退细筛分
            return
        # 【near kmeans 4 倍样本守卫（DeepSeek 审计 round1 ③，2026-10-11，
        #   主 AI 派单确认）】far 侧 _build_far_kmeans 有同口径守卫
        #   （Tfar < far_clusters*4 → 放弃建簇 _km_centroids=None 回退细筛
        #   分），near 侧 kmeans 分支复用 far_clusters 作簇数（256 簇固定），
        #   却只有 Tn<256 下限：Tn ∈ [256, far_clusters*4)（far_clusters=256
        #   时 [256,1024)）时簇数逼近/≥样本数 → kmeans 退化为一簇一样本/
        #   大量空簇，聚类完全失效且静默继续（LB 短序列 near 区可落入，
        #   Tn=300 实测每头 256 簇几乎全被单样本占据）。补与 far 侧同口径
        #   守卫：kmeans 分支样本不足 4 倍簇数时跳过更新，沿用本函数
        #   Tn<256 的跳过惯例（_km_near_centroids 保持 None，消费端回退
        #   细筛分——细筛分质量高于簇代表分，语义无损偏保守）。
        #   sim_greedy 分支不受影响：贪心簇数自适应（Kcap=Tn，活簇随数据
        #   增长），不存在簇数≥样本数退化，与 far 侧 greedy 无 4 倍守卫
        #   口径一致。far_clusters=64 时阈值=256 与既有下限重合，行为不变。
        if self.near_select == "cluster" and Tn < self.far_clusters * 4:
            return
        idx = (self._sim_dims_indices if self.near_select == "sim_greedy"
               else self._subspace_indices)(k.device)
        x = k[0, far_hi:near_hi].float()[..., idx]   # [Tn, Hkv, dd]
        Hkv = x.shape[1]
        if self.near_select == "sim_greedy":
            T, dd = x.shape[0], x.shape[2]
            sums = x.new_zeros(Hkv, T, dd)
            cnt = x.new_zeros(Hkv, T)
            sq = x.new_zeros(Hkv, T)
            k_live = torch.zeros(Hkv, dtype=torch.long, device=x.device)
            sums, cnt, sq, k_live, assign = self._greedy_cluster_pass(
                x, self.sim, sums, cnt, sq, k_live)
            centroids = sums / cnt.clamp(min=1).unsqueeze(-1)
        else:
            centroids, assign = [], []
            for h in range(Hkv):
                c, a = self._gpu_kmeans(x[:, h, :], self.far_clusters, self.far_niter)
                centroids.append(c)
                assign.append(a)
            centroids = torch.stack(centroids)   # [Hkv, K_c, dd]
            assign = torch.stack(assign)         # [Hkv, Tn]
        self._km_near_centroids = centroids
        self._km_near_assign = assign            # 相对 near_lo 的 token 偏移
        self._km_near_lo = far_hi
        self._km_near_hi = near_hi
        self._km_near_dims = idx

    @staticmethod
    def _greedy_cluster_pass(x, sim, sums, cnt, sq, k_live):
        """E113b：贪心一趟的调度器 —— CUDA 上默认走 Triton kernel，否则回退 Python。

        开关 SGLANG_TLI_GREEDY_KERNEL（默认 1；0 = 强制 Python 循环，供对拍/回退）。
        回退条件：env=0 / x 在 CPU / triton 不可用 / dd 非 2 的幂（tl.arange 编译
        硬约束；现有 sim_dims 口径全为 2 的幂，此为防御性回退）。
        两条路径语义逐位一致（exp/trace/test_e113_greedy_triton.py 对拍 +
        test_e113b_kernel_integration.py 开关双跑对拍）。
        【E131-S2】开关 TLI_E131_GREEDY_V2（默认 1）：fn 换 SPLIT 协作 v2
        kernel（greedy_triton.greedy_pass_triton_v2，grid=(H,SPLIT) 同头多
        program 均分活簇扫描 + 两阶段软件 barrier；小 T/探测失败 wrapper
        内部回退 v1）。与 v1/Python 逐位一致（exp/bench/e131s2_v2_validate.py
        V1-V5 全 PASS）。E131 实测（H20 randn 最坏）：far 冷建 8077→958ms
        （8.4×）、大 k_live 续跑 9.5×；v2 不可用 fail-closed 保持 v1。
        """
        if (os.environ.get("SGLANG_TLI_GREEDY_KERNEL", "1") not in ("0", "false", "False")
                and x.is_cuda):
            fn = _load_greedy_triton()
            T, H, dd = x.shape
            if fn is not None and T > 0 and dd > 0 and (dd & (dd - 1)) == 0:
                if (os.environ.get("TLI_E131_GREEDY_V2", "1")
                        not in ("0", "false", "False")):
                    try:
                        from .greedy_triton import greedy_pass_triton_v2
                        fn = greedy_pass_triton_v2
                    except Exception:
                        pass   # fail-closed：保持 v1
                return TLIIndexer._greedy_cluster_pass_triton(x, sim, sums, cnt, sq,
                                                              k_live, fn)
        return TLIIndexer._greedy_cluster_pass_python(x, sim, sums, cnt, sq, k_live)

    @staticmethod
    def _greedy_cluster_pass_triton(x, sim, sums, cnt, sq, k_live, fn):
        """E113b：Triton kernel 路径。长段分 chunk=16384（E113 microbench 实测最优
        chunk，launch 间隙效应 1.75×）；分段不改语义——贪心时序跨段精确延续，
        对拍见 test_e113_greedy_triton.py T3（分段续跑 ≡ 全量重放，逐位）。"""
        T, H, dd = x.shape
        CHUNK = 16384
        if T <= CHUNK:
            return fn(x, sim, sums, cnt, sq, k_live)
        assign = torch.empty(H, T, dtype=torch.long, device=x.device)
        for c0 in range(0, T, CHUNK):
            seg = x[c0:c0 + CHUNK]
            sums, cnt, sq, k_live, a = fn(seg, sim, sums, cnt, sq, k_live)
            assign[:, c0:c0 + seg.shape[0]] = a
        return sums, cnt, sq, k_live, assign

    @staticmethod
    def _greedy_cluster_pass_python(x, sim, sums, cnt, sq, k_live):
        """增量贪心聚类一趟 —— Python 参考实现（E110 原版原样保留；E113b 起
        作为 SGLANG_TLI_GREEDY_KERNEL=0 的回退路径与对拍基准。语义 =
        e64a / E108 probe 的 greedy_cluster_assign：
        token 按序到达，与现有簇心（成员 running mean）余弦相似度最大的活簇
        cos >= sim 则归并，否则新建簇；簇心由 sums/cnt 增量维护（算术均值）。

        x: [T, H, dd]（按序）。sums [H,K,dd] / cnt [H,K] / sq [H,K]（||sums||^2
        增量维护，免每步全量 norm）/ k_live [H]：既有簇状态（全零 = 冷启动；
        非零 = 续跑，与全量重放逐位一致——贪心决策只依赖先验状态）。
        返回 (sums, cnt, sq, k_live, assign[H,T])；容量不足时返回扩容新张量。
        实现注（E108 probe 实测坑原样规避）：归并步的 dot_a 从 dot 直接 gather
        （恒有限）——若用 m·norm·x_n 推导，无活簇首 token 时 m=-inf 经 0 权重
        乘出 NaN 污染 sq；非归并头 index_add 加 0 权重无害（目标槽均为活簇或
        全新零槽）。
        """
        T, H, dd = x.shape
        K = sums.shape[1]
        need = int(k_live.max().item()) + T
        if need > K:   # 扩容（新簇上界 = T）
            pad = need - K
            sums = F.pad(sums, (0, 0, 0, pad))
            cnt = F.pad(cnt, (0, pad))
            sq = F.pad(sq, (0, pad))
            K = sums.shape[1]
        dev = x.device
        EPS = 1e-9
        assign = torch.zeros(H, T, dtype=torch.long, device=dev)
        x_n = x.norm(dim=-1)                                  # [T,H]
        ar_h = torch.arange(H, device=dev)
        live_row = torch.arange(K, device=dev).unsqueeze(0)   # [1,K]
        for i in range(T):
            xi = x[i]                                          # [H,dd]
            xi_n = x_n[i]                                      # [H]
            norm = sq.clamp(min=0).sqrt()                      # [H,K]
            dot = torch.bmm(sums, xi.unsqueeze(-1)).squeeze(-1)  # [H,K]
            cos = dot / (norm * xi_n.unsqueeze(-1) + EPS)
            cos = cos.masked_fill(live_row >= k_live.unsqueeze(-1), float("-inf"))
            a = cos.argmax(-1)                                 # [H]
            m = cos.gather(1, a.unsqueeze(-1)).squeeze(-1)
            upd = m >= sim                                     # [H]
            w = upd.float()
            dot_a = dot.gather(1, a.unsqueeze(-1)).squeeze(-1)  # sums_a·xi（恒有限）
            flat_upd = ar_h * K + a
            sums.view(-1, dd).index_add_(0, flat_upd, xi * w.unsqueeze(-1))
            cnt.view(-1).index_add_(0, flat_upd, w)
            sq.view(-1).index_add_(0, flat_upd, (2 * dot_a + xi_n * xi_n) * w)
            nw = (~upd).float()                                # 新建簇头
            flat_new = ar_h * K + k_live
            sums.view(-1, dd).index_add_(0, flat_new, xi * nw.unsqueeze(-1))
            cnt.view(-1).index_add_(0, flat_new, nw)
            sq.view(-1).index_add_(0, flat_new, xi_n * xi_n * nw)
            assign[:, i] = torch.where(upd, a, k_live)
            k_live = k_live + (~upd).long()
        return sums, cnt, sq, k_live, assign

    def _sim_dims_indices(self, device):
        """E110：sim_greedy 聚类维度（用户设定：nope 维或压缩维）。
        subspace = 与 cluster/kmeans 路径同维（_subspace_indices 全分支；
        注意 'full' 落到 tail 分支——与 far kmeans 构建口径一致）；nope/tail/full
        为显式覆盖（tail 宽度由 cmp_ratio 控制）。"""
        if self.sim_dims == "subspace":
            return self._subspace_indices(device)
        if self.sim_dims == "nope":
            return torch.arange(64, 128, device=device)
        if self.sim_dims == "full":
            return torch.arange(128, device=device)
        delta = 64 // self.cmp_ratio
        return torch.tensor(
            list(range(64 - delta, 64)) + list(range(128 - delta, 128)), device=device
        )

    _km_far_hi_cached = -1

    @staticmethod
    def _gpu_kmeans(x, K, niter, seed=0):
        g = torch.Generator().manual_seed(seed)
        N, d = x.shape
        c = x[torch.randperm(N, generator=g)[:K]].clone()
        for _ in range(niter):
            a = (x @ c.T).argmax(dim=1)
            cnt = torch.bincount(a, minlength=K).float()
            sums = torch.zeros(K, d, device=x.device).index_add_(0, a, x)
            ne = cnt > 0
            c[ne] = sums[ne] / cnt[ne, None]
        return c, a

    # ---------------- L1 打分（A 的子空间 + B 的远端簇分数 + D' 剔除）---------------- #

    def _need_avg_score(self) -> bool:
        """E109a 修复：method=avg 的分数源条件（此前 subspace=full 下全部静默失效）。

        far avg：单池（α=0）与分区（α>0）的 far 池都要用 → 永远需要；
        near avg：near 池仅在分区（α>0 且 β>0）存在 → 仅分区需要。
        """
        return self.far_method == "avg" or (
            self.near_method == "avg" and self.alpha > 0 and self.beta > 0
        )

    def compute_score(self, q, q_ids, index_dict, softmax_scale):
        # 缓存 squeeze 后的 q 供远端簇分数使用
        # 【C3 修复 2026-10-08】缓存时统一乘 softmax_scale——_near_token_score
        # 的簇分会与已缩放的 score_fine 回退段（b_q_full = q_sq[0]*softmax_scale）
        # 进同一 near 池 topk 竞争，未缩放会放大量纲 |1/scale| 倍（D=128 时
        # 11.31×）导致簇段霸榜挤出回退段；far 侧纯簇分 topk 乘正 scale 不变
        # 排序 → far 行为逐位不变（核验报告 research/docs/gpt_audit_verdict_
        # 20261008_round2.md C3 节）。
        if q.shape[0] == 1 and q.shape[1] == 1:
            self._last_q = (
                q.squeeze(0).squeeze(0) * softmax_scale
            ).to(torch.float32)  # [H, D]（已缩放）
        # E109a 修复：full 子空间原来无条件走父类 compute_score（不产
        # score_coarse_avg）→ far/near_method=avg 全部静默退化为 minmax。
        # 现仅在「双池都不用 avg」时才走父类捷径；否则走本类路径
        # （idx_sub=None 时 minmax 分数与父类逐位等价，且计算 k_avg 分数源）。
        # 【E111 D' gate 短路省算（GPT 审计 2026-10-12 第 1 条「包括
        # fallback」）】父类捷径是全量 L1/L2 评分（无 far 省算入口）——
        # skip+e64 组合不得走捷径，改走本类路径（idx_sub=None 时 minmax
        # 分数与父类逐位等价，且本类路径有 dgate_short 省算分支）。
        # 非 skip / 非 e64 组合的捷径行为逐位不变。
        if not self.enable_subspace and not self._need_avg_score() \
                and not (self.skip_far and self.alpha > 0 and self.beta > 0):
            d = super().compute_score(q, q_ids, index_dict, softmax_scale)
            # 075：父类捷径同样携带真实长度（该路径无投影，valid_length ==
            # score_fine 宽度，行为不变；保持 score_dict 接口统一）
            d["valid_length"] = index_dict.get("valid_length")
            return d
        # A：L1 上界只在 d' 子空间维上算（q 同步取子空间，与 k_min/k_max 对齐）
        assert q.shape[0] == 1 and q.shape[1] == 1
        from einops import rearrange as _rearrange
        k_min, k_max = (x.squeeze(0) for x in (index_dict["k_min"], index_dict["k_max"]))
        k_qat = index_dict["k_qat"]
        k_avg = index_dict.get("k_avg")
        if k_avg is not None:
            k_avg = k_avg.squeeze(0)  # [T, Hkv, d]（与 k_min/k_max 同形）
        q_sq = q.squeeze(0)  # [1, H, D]
        H = q_sq.shape[1]
        self.group_size = H // k_min.shape[1]
        coarse_shape = (1, H, k_min.shape[0])
        score_coarse = q_sq.new_full(coarse_shape, float("-inf"))
        # E109a 修复：full（enable_subspace=False）时 idx_sub=None（全维）
        # ——原实现此处无守卫，full 落 _subspace_indices 默认 tail 分支（32 维），
        # 但 k_min/k_max 是 prepare_index 产的全 128 维 → einsum 维度崩。
        # （此前 full 走父类 compute_score 捷径，此路径从未被 full 触发故未暴露）
        idx_sub = self._subspace_indices(q_sq.device) if self.enable_subspace else None
        if self._basis is not None:
            # 投影路径（E64f）：q 子空间投影到 d 维（GQA 组共享 kv 基）
            q_sub = (q_sq[0] * softmax_scale).to(torch.float32)[..., idx_sub]  # [H,32]
            Hkv = self._basis.shape[0]
            G = H // Hkv
            b_q = torch.einsum(
                "hgd,hde->hge", q_sub.reshape(Hkv, G, -1), self._basis
            ).reshape(H, -1)  # [H, r]
        else:
            q_full = (q_sq[0] * softmax_scale).to(torch.float32)
            b_q = q_full if idx_sub is None else q_full[..., idx_sub]  # [H, d']（full=全维）
        # 【E127 inplace 合并】b_k_min/b_k_max repeat 物化下放到各分支
        # 惰性执行（dgate_short / Triton fast / 原版三路径各自物化）——
        # 原无条件物化会使 Triton fast path 的免物化收益归零。
        # 【E111 D' gate 短路省算（用户 2026-10-12 拍板）】skip 层 far 区
        # 粗筛分结构性必弃（compute_mask 将 far 块置 -inf / far 池置空），
        # 全序列 einsum 中 far 段纯白算。e64 分区几何（α>0 且 β>0）下把
        # far 段块 [sink_blocks, near_blks) 从 einsum 排除，score_coarse
        # 该段保持初始 -inf。kt 是 einsum 输出维非收缩维 → 分段对保留段
        # 逐位一致。非 e64（(0,0) 单池等）skip 层仍落 legacy topk（其 far
        # 段分数同样会被置 -inf，理论也可省）——但该域无生产配置，第一版
        # 保守不省，缩小验证面。valid_length 缺失（外部合成 index_dict 的
        # 回放 harness）→ 回退全量（行为不变）。几何公式与 compute_mask
        # B10/F5 修复后口径逐字段一致。
        dgate_short = False
        dgate_near_blks = None
        if self.skip_far and self.alpha > 0 and self.beta > 0:
            _vl = index_dict.get("valid_length")
            if _vl is not None:
                _bs = self.args.tia_block_size
                _sink_tok = self.sink_blocks * _bs
                _swa_tok = self.sliding_window_size
                _mid_len = max(0, int(_vl) - _sink_tok - _swa_tok)
                _near_len_dyn = max(_bs, int(self.alpha * _mid_len))
                _near_blks = max(self.sink_blocks,
                                 ((int(_vl) - _swa_tok) - _near_len_dyn) // _bs)
                if self.sink_blocks < _near_blks:
                    dgate_short = True
                    dgate_near_blks = _near_blks
        if dgate_short:
            b_k_min = repeat(k_min, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
            b_k_max = repeat(k_max, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
            _sb = self.sink_blocks
            _qc0 = b_q.clamp(max=0)
            _qc1 = b_q.clamp(min=0)
            sc_head = (
                einsum(_qc0, b_k_min[:_sb], "h d, kt h d -> h kt")
                + einsum(_qc1, b_k_max[:_sb], "h d, kt h d -> h kt")
            )
            sc_tail = (
                einsum(_qc0, b_k_min[dgate_near_blks:], "h d, kt h d -> h kt")
                + einsum(_qc1, b_k_max[dgate_near_blks:], "h d, kt h d -> h kt")
            )
            score_coarse[0, :, :_sb] = sc_head.to(q_sq.dtype)
            score_coarse[0, :, dgate_near_blks:] = sc_tail.to(q_sq.dtype)
        elif (_HAS_TRITON and self._basis is None and k_min.dim() == 3
                and k_max.dim() == 3 and k_min.is_cuda
                and (k_min.shape[-1] & (k_min.shape[-1] - 1)) == 0
                and k_min.stride(-1) == 1 and k_max.stride(-1) == 1
                and b_q.stride(-1) == 1 and b_q.dtype == torch.float32
                and k_min.shape[-1] == b_q.shape[-1]
                and k_min.shape[1] * self.group_size == b_q.shape[0]
                and self._triton_ok):
            # 【E127 iter8】coarse minmax 段 Triton GQA K 复用（原
            # TLIIndexerFast._fast_score_coarse 并入）：免 b_k_min/b_k_max
            # 两次 [kt,H,d'] fp32 repeat 物化 + clamp 胶水（~10 launches →
            # 1 kernel）；fp32 累加顺序与 cublas 不同 → bf16 舍入后 ULP 级
            # 漂移（实测 ~5e-5 比例元素差 1 ULP）。异形布局/Triton 异常
            # 回退原版表达式（fail-closed）。
            try:
                score_coarse[0] = self._fast_score_coarse(
                    k_min, k_max, b_q, q_sq.dtype)
            except Exception:
                b_k_min = repeat(k_min, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
                b_k_max = repeat(k_max, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
                b_score = (
                    einsum(b_q.clamp(max=0), b_k_min, "h d, kt h d -> h kt")
                    + einsum(b_q.clamp(min=0), b_k_max, "h d, kt h d -> h kt")
                )
                score_coarse[0] = b_score.to(q_sq.dtype)
        else:
            b_k_min = repeat(k_min, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
            b_k_max = repeat(k_max, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
            b_score = (
                einsum(b_q.clamp(max=0), b_k_min, "h d, kt h d -> h kt")
                + einsum(b_q.clamp(min=0), b_k_max, "h d, kt h d -> h kt")
            )
            score_coarse[0] = b_score.to(q_sq.dtype)
        # E89 MoBA 臂：全维 chunk-mean gate 分（gate 用完整 128 维，严格 MoBA 口径）
        score_moba = None
        if self.moba_gate:
            k_avg_full = index_dict.get("k_avg_full")
            b_k_avg_full = repeat(
                k_avg_full.squeeze(0), "t h d -> t (h g) d", g=self.group_size
            ).to(torch.float32)
            b_q_full_gate = (q_sq[0] * softmax_scale).to(torch.float32)
            score_moba = q_sq.new_full(coarse_shape, float("-inf"))
            score_moba[0] = einsum(
                b_q_full_gate, b_k_avg_full, "h d, kt h d -> h kt"
            ).to(q_sq.dtype)
        # E64f：near 粗筛 avg 分数（块均值精确分，不 clamp）
        score_coarse_avg = None
        if k_avg is not None:
            score_coarse_avg = q_sq.new_full(coarse_shape, float("-inf"))
            if dgate_short:
                # E111 省算同款：far 段 avg 分保持 -inf（skip 层 far 池置空
                # 后不可达；分段 einsum 对保留段逐位一致）
                b_k_avg = repeat(k_avg, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
                sc_avg_head = einsum(
                    b_q, b_k_avg[:_sb], "h d, kt h d -> h kt")
                sc_avg_tail = einsum(
                    b_q, b_k_avg[dgate_near_blks:], "h d, kt h d -> h kt")
                score_coarse_avg[0, :, :_sb] = sc_avg_head.to(q_sq.dtype)
                score_coarse_avg[0, :, dgate_near_blks:] = sc_avg_tail.to(q_sq.dtype)
            elif (_HAS_TRITON and self._basis is None and k_avg.dim() == 3
                    and k_avg.is_cuda
                    and (k_avg.shape[-1] & (k_avg.shape[-1] - 1)) == 0
                    and k_avg.stride(-1) == 1
                    and b_q.dtype == torch.float32
                    and k_avg.shape[-1] == b_q.shape[-1]
                    and k_avg.shape[1] * self.group_size == b_q.shape[0]
                    and self._triton_ok):
                # 【E127 iter8】avg 段复用 iter6 _score_fine_kernel 同构
                # （einsum(b_q, repeat(k_avg)) 数学等价，GQA K 复用免物
                # 化）；ULP 口径同 iter6；异形/异常回退原版表达式。
                try:
                    score_coarse_avg[0] = self._fast_score_fine(
                        k_avg, b_q, q_sq.dtype)
                except Exception:
                    b_k_avg = repeat(k_avg, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
                    score_coarse_avg[0] = einsum(b_q, b_k_avg, "h d, kt h d -> h kt").to(q_sq.dtype)
            else:
                b_k_avg = repeat(k_avg, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
                score_coarse_avg[0] = einsum(b_q, b_k_avg, "h d, kt h d -> h kt").to(q_sq.dtype)
        # L2 精筛分数：非投影=全维 k_qat；投影=投影特征 k_qat（E64f 协议）
        # 投影路径必须 fp32：8 维点积幅度 ~O(0.5)，bf16 绝对步长会折叠 top-K 边界
        # 的小 gap 分数成同值 → topk 退化为位置偏置（e2e mass 0.99→0.81 实测）
        fine_shape = (1, H, k_qat.shape[1])
        if self._basis is not None:
            score_fine = q_sq.new_full(fine_shape, float("-inf"), dtype=torch.float32)
        else:
            score_fine = q_sq.new_full(fine_shape, float("-inf"))
        if self._basis is not None:
            b_q_full = b_q
        else:
            b_q_full = (q_sq[0] * softmax_scale).to(torch.float32)
        if dgate_short:
            # E111 省算：far 段 token [sink_tok, near_blks·bs) 的细筛分
            # 保持 -inf（skip 层 far 池置空后不可达）。分段对保留段逐位
            # 一致（token 维是输出维非收缩维）。
            b_k_qat = repeat(k_qat.squeeze(0), "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
            _ftl = _sb * self.args.tia_block_size
            _fth = dgate_near_blks * self.args.tia_block_size
            sf_head = einsum(b_q_full, b_k_qat[:_ftl], "h d, kt h d -> h kt")
            sf_tail = einsum(b_q_full, b_k_qat[_fth:], "h d, kt h d -> h kt")
            if self._basis is not None:
                score_fine[0, :, :_ftl] = sf_head
                score_fine[0, :, _fth:] = sf_tail
            else:
                score_fine[0, :, :_ftl] = sf_head.to(q_sq.dtype)
                score_fine[0, :, _fth:] = sf_tail.to(q_sq.dtype)
        elif (_HAS_TRITON and self._basis is None and k_qat.dim() == 4
                and k_qat.is_cuda
                and k_qat.shape[0] == 1 and k_qat.shape[-1] == 128
                and b_q_full.dtype == torch.float32
                and k_qat.shape[2] * self.group_size == b_q_full.shape[0]
                and self._triton_ok):
            # 【E127 iter6】非投影路径 Triton GQA K 复用（K 读带宽 4× 削减，
            # 免 [T,H,128] fp32 repeat 物化；fp32 累加与原版 einsum 仅累加
            # 顺序不同 → bf16 舍入后 ULP 级漂移，mask 门禁实测逐位不变）。
            # 投影/异形布局/Triton 异常回退原版 repeat+einsum。
            try:
                score_fine[0] = self._fast_score_fine(
                    k_qat.squeeze(0), b_q_full, q_sq.dtype)
            except Exception:
                b_k_qat = repeat(k_qat.squeeze(0), "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
                score_fine[0] = einsum(b_q_full, b_k_qat, "h d, kt h d -> h kt").to(q_sq.dtype)
        elif self._basis is not None:
            b_k_qat = repeat(k_qat.squeeze(0), "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
            score_fine[0] = einsum(b_q_full, b_k_qat, "h d, kt h d -> h kt")
        else:
            b_k_qat = repeat(k_qat.squeeze(0), "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
            score_fine[0] = einsum(b_q_full, b_k_qat, "h d, kt h d -> h kt").to(q_sq.dtype)
        return {
            "score_coarse": score_coarse.unsqueeze(0),
            "score_coarse_avg": None if score_coarse_avg is None else score_coarse_avg.unsqueeze(0),
            "score_moba": None if score_moba is None else score_moba.unsqueeze(0),
            "score_fine": score_fine.unsqueeze(0),
            # 075：真实序列长度随分数透传给 compute_mask（投影路径
            # score_fine 宽 = Tpad ≠ 真实 S；无值时消费端回退宽度口径）
            "valid_length": index_dict.get("valid_length"),
        }

    # ------------------------------------------------ 【E127 iter6/8】Triton
    # fast path（原 TLIIndexerFast._fast_score_fine/_fast_score_coarse 并入）
    def _fast_score_fine(self, kq, b_q_full, out_dtype):
        """score_fine 的 Triton GQA K 复用 kernel（非投影路径专用）。

        数学：score[h, t] = sum_d b_q_full[h, d] * k_qat32[t, h//g, d]，
        与原版 repeat(k_qat,"t h d -> t (h g) d") 后 einsum("h d, kt h d
        -> h kt") 数学等价；GQA 组内 K 寄存器复用（每块 K 只读一次服务
        g 个 q 头），读带宽 [T,H,D] → [T,Hkv,D]（4× 削减），免 fp32
        repeat 物化（S=64K 时 [S,H,128] fp32 = 1.07GB/层常驻退役）。

        数值：kernel 内 bf16→fp32 精确转换 + fp32 乘加累加 + 输出 RNE
        舍入到 out_dtype，与原版 fp32 einsum → .to(bf16) 相比仅累加
        顺序不同（tl.sum D 维 tree-reduce vs cublas gemv 调度）→ fp32
        ULP 级漂移，bf16 舍入后实测差异 ~5e-5 比例元素差 1 ULP；
        mask/topk 选择多步多配置实测逐位不变（门禁实测口径，非数学
        保证）。原版输出为 bf16（q_sq.dtype，非投影路径），fp32 ULP
        漂移绝大部分被 bf16 舍入吸收。

        fail-closed：Triton 编译/执行异常 → 置 _triton_ok=False（本
        实例永久回退原版表达式）并 raise（调用点 except 回退）。
        """
        try:
            T, Hkv, D = kq.shape
            H = b_q_full.shape[0]
            g = H // Hkv
            out = torch.empty((H, T), device=kq.device, dtype=out_dtype)
            grid = (triton.cdiv(T, 64), Hkv)
            _score_fine_kernel[grid](
                kq, b_q_full, out, T,
                kq.stride(0), kq.stride(1), b_q_full.stride(0),
                out.stride(0), out.stride(1),
                Hkv=Hkv, G=g, D=D, BT=64, num_warps=4)
            return out
        except Exception:
            self._triton_ok = False
            raise

    def _fast_score_coarse(self, k_min, k_max, b_q, out_dtype):
        """【iter8】score_coarse 的 Triton GQA K 复用 kernel（非投影路径）。

        数学：score_coarse[h, t] = Σ_d min(b_q[h,d],0)·k_min32[t,h//g,d]
                                       + Σ_d max(b_q[h,d],0)·k_max32[t,h//g,d]，
        与原版 repeat(k_min/k_max) 后 clamp(max=0)/clamp(min=0) 双 einsum
        数学等价（minmax 上界分：负部配 k_min、正部配 k_max）；GQA 组内
        k_min/k_max 块寄存器复用（读带宽 /g），免 [kt,H,d'] fp32 repeat
        物化 ×2 + clamp 胶水（~10 launches → 1 kernel）。

        数值：fp32 累加顺序与 cublas 双 gemv 不同 → bf16 舍入后 ULP 级
        漂移（实测 ~5e-5 比例元素差 1 ULP，与 iter6 fine 同量级）；
        mask/topk 选择为经验验证口径（非数学保证）。

        fail-closed：异常置 _triton_ok=False 并回退原版表达式。"""
        try:
            T, Hkv, D = k_min.shape
            H = b_q.shape[0]
            g = H // Hkv
            out = torch.empty((H, T), device=k_min.device, dtype=out_dtype)
            grid = (triton.cdiv(T, 64), Hkv)
            _score_coarse_kernel[grid](
                k_min, k_max, b_q, out, T,
                k_min.stride(0), k_min.stride(1),
                k_max.stride(0), k_max.stride(1),
                b_q.stride(0),
                out.stride(0), out.stride(1),
                Hkv=Hkv, G=g, D=D, BT=64, num_warps=4)
            return out
        except Exception:
            self._triton_ok = False
            raise

    def _fast_tok_score(self, q, idx_sub, centroids, assign, agg):
        """【E131-S2】簇分数段 q_g+einsum+gather 三件套的 Triton 全融合。

        数学：q_g[h,d'] = AGG_{j<G} q[h·G+j, idx_sub[d']]（AGG=sum 远端 /
        mean 近端，与原版 reshape(...).sum(1)/mean(1) 数学等价）；
        tok_score[h,t] = Σ_d q_g[h,d]·centroids[h, assign[h,t], d]，与
        原版 q 胶水 + einsum("hd,hkd->hk") + gather(1, assign) 数学等价。
        一次 launch 完成（原版 3 个小 launch 的胶水序列两处各付一遍）。
        E131-S2 实测（H20 S=64K cclustersim 稳态）：far 侧 0.107→0.057ms
        （1.89×）/ near 侧 0.062→0.047ms（1.32×）。

        数值口径（同 E127 iter6/8）：q_g 组内累加与 dot 的 fp32 累加顺序
        均与 torch/cublas 不同 → fp32 ULP 级漂移（tok_score 无 bf16 舍入
        吸收，直接进 topk 竞争）→ 并列边界可能翻转，mask 选择为经验
        验证口径（门禁实测 + top-k 重叠率 ≥99.9% 报告精确值，E131-S2
        实测 16 步 mask 逐位 0 差异 + e2e 24 步 12585120 位 100% 一致）。

        fail-closed：Triton 编译/执行异常 → 置 _triton_ok=False 并由
        调用点回退原版表达式（值同原版路径，仅性能回退）。"""
        try:
            Hkv, Kc, dd = centroids.shape
            T = assign.shape[1]
            H = q.shape[0]
            G = H // Hkv
            out = torch.empty((Hkv, T), device=centroids.device, dtype=torch.float32)
            grid = (triton.cdiv(T, 128), Hkv)
            # AGG 除数：sum → 1.0；mean → 1/G（G 为 2 的幂 → 精确二进制
            # 除法，与 torch mean 的 /G 逐位同）
            agg_div = 1.0 if agg == "sum" else 1.0 / G
            _tok_score_kernel[grid](
                q, idx_sub, centroids, assign, out, T, agg_div,
                q.stride(0),
                centroids.stride(0), centroids.stride(1),
                assign.stride(0), assign.stride(1),
                out.stride(0), out.stride(1),
                Hkv=Hkv, G=G, DD=dd, BT=128, num_warps=4)
            return out
        except Exception:
            self._triton_ok = False
            raise

    def _far_token_score(self, q):
        """B：远端 token 的簇分数（E4/E7 语义：按中心分数 token 级展开）。

        q: [H(全部 head), D]。返回 [Hkv, Tfar]（相对 far_lo 的 token 偏移）或 None。
        E110：q 取维改用建簇时记录的 _km_dims（sim_greedy 的 sim_dims 可异于
        _subspace_indices——打分/建簇维度必须同源；kmeans 路径 _km_dims 即
        _subspace_indices 的返回值，行为不变）。
        【E131-S2】Triton 全融合分支（开关 TLI_E131_TOKSCORE_TRITON，默认 1；
        0 或异形布局/编译异常回退本原版表达式，值逐位同原版路径）。
        """
        if self._km_centroids is None:
            return None
        idx_sub = self._km_dims if self._km_dims is not None \
            else self._subspace_indices(q.device)
        H = q.shape[0]
        Hkv = self._km_centroids.shape[0]
        G = H // Hkv
        if (os.environ.get("TLI_E131_TOKSCORE_TRITON", "1")
                not in ("0", "false", "False")
                and _HAS_TRITON
                and getattr(self, "_triton_ok", True)
                and q.is_cuda and q.dtype == torch.float32
                and self._km_centroids.is_cuda
                and self._km_token_assign.is_cuda
                and idx_sub is not None and idx_sub.is_cuda
                and self._km_centroids.dim() == 3
                and self._km_centroids.shape[-1] > 0
                and (self._km_centroids.shape[-1]
                     & (self._km_centroids.shape[-1] - 1)) == 0
                and self._km_centroids.stride(-1) == 1
                and self._km_token_assign.dim() == 2
                and (G & (G - 1)) == 0):
            try:
                return self._fast_tok_score(
                    q, idx_sub, self._km_centroids, self._km_token_assign,
                    agg="sum")
            except Exception:
                pass   # fail-closed：下方原版表达式（值同锚点路径）
        q_sub = q[:, idx_sub].float()                       # [H, d']
        q_g = q_sub.reshape(Hkv, G, -1).sum(1)              # [Hkv, d']（group 求和，与 L1 量纲一致）
        cscore = torch.einsum("hd,hkd->hk", q_g, self._km_centroids)  # [Hkv, K_c]
        tok_score = cscore.gather(1, self._km_token_assign)  # [Hkv, Tfar]
        return tok_score

    def _near_token_score(self, q):
        """E110：near 区 token 的簇分数（与 _far_token_score 同构）。

        q: [H, D]。返回 [Hkv, Tn]（相对 _km_near_lo 的 token 偏移）或 None。
        与 far 侧的唯一差别：group 聚合用 mean——far 侧单独 topk 用 sum 无所谓
        量纲，near 侧簇分数要与「未覆盖段的细筛回退分 sf_g（group-mean 原始
        点积分）」在同一 topk 内竞争，sum 会放大 G 倍导致簇段霸榜，必须 mean。
        【E131-S2】Triton 全融合分支（同 far 侧口径，AGG=mean；开关
        TLI_E131_TOKSCORE_TRITON 默认 1，异形/异常回退原版表达式）。
        """
        if self._km_near_centroids is None:
            return None
        idx_sub = self._km_near_dims
        H = q.shape[0]
        Hkv = self._km_near_centroids.shape[0]
        G = H // Hkv
        if (os.environ.get("TLI_E131_TOKSCORE_TRITON", "1")
                not in ("0", "false", "False")
                and _HAS_TRITON
                and getattr(self, "_triton_ok", True)
                and q.is_cuda and q.dtype == torch.float32
                and self._km_near_centroids.is_cuda
                and self._km_near_assign.is_cuda
                and idx_sub is not None and idx_sub.is_cuda
                and self._km_near_centroids.dim() == 3
                and self._km_near_centroids.shape[-1] > 0
                and (self._km_near_centroids.shape[-1]
                     & (self._km_near_centroids.shape[-1] - 1)) == 0
                and self._km_near_centroids.stride(-1) == 1
                and self._km_near_assign.dim() == 2
                and (G & (G - 1)) == 0):
            try:
                return self._fast_tok_score(
                    q, idx_sub, self._km_near_centroids, self._km_near_assign,
                    agg="mean")
            except Exception:
                pass
        q_sub = q[:, idx_sub].float()            # [H, dd]
        q_g = q_sub.reshape(Hkv, G, -1).mean(1)             # [Hkv, dd]（group 均值，对齐 sf_g 量纲）
        cscore = torch.einsum("hd,hkd->hk", q_g, self._km_near_centroids)  # [Hkv, K_c]
        return cscore.gather(1, self._km_near_assign)       # [Hkv, Tn]

    def _moba_mask(self, score_dict, kt, bs, S_real):
        """E89 MoBA 复现臂：chunk gate top-K 块全展开（training-free，统一 harness）。

        gate = 全维 q·chunk-mean 分数；选 top-(K2/BS) 块 + 当前块强制，
        sink/swa 保送与 PSI 同口径。返回与 score_fine 同宽的 bool mask。
        【075 修复】S_real = 真实序列长度（非投影 == score_fine 宽度，
        行为逐位不变）；投影路径 tok_hi/swa 强制区必须落真实域——
        旧口径 tok_hi=sf 宽=Tpad 使尾块展开 clamp 进 padding、swa 强制区
        落 padded 尾部。
        """
        sm = score_dict.get("score_moba")
        assert sm is not None, "moba 臂需 prepare_index/compute_score 走 moba_gate 路径"
        sm = rearrange(sm, "b qt (h g) kt -> b qt h g kt", g=self.group_size).mean(dim=-2)
        sf = score_dict["score_fine"]
        K2 = min(sf.shape[-1], self.args.tia_level2_topk)
        nb_sel = max(1, K2 // bs)
        sink_tok = self.sink_blocks * bs
        swa_tok = self.sliding_window_size
        # 未来块屏蔽（因果）：gate 只对已见块打分，此处全为历史块，无需额外 mask
        i_blk = torch.topk(sm, min(nb_sel, sm.shape[-1]), dim=-1).indices  # [1,1,Hkv,nb]
        m = torch.zeros(
            sm.shape[:-1] + (sf.shape[-1],), dtype=torch.bool, device=sm.device
        )
        # 块展开：每块 bs 个 token（尾块按序列长度截断）
        # 075：截断上界按真实 S_real（非投影 = sf 宽，逐位不变；投影防选 padding）
        tok_hi = min(S_real, kt * bs)
        tok = (i_blk.unsqueeze(-1) * bs
               + torch.arange(bs, device=sm.device).view(1, 1, 1, bs)
               ).clamp(max=tok_hi - 1).reshape(i_blk.shape + (bs,))  # [1,1,Hkv,nb*bs]
        m.scatter_(-1, tok.reshape(*m.shape[:-1], -1), True)
        # 正交强制区：sink 头部 + swa 尾部（与 PSI/sigma 臂同口径，不占 gate 预算）
        # 075：swa 尾部 = 真实 S 尾（非 padded 尾）；padding 段保持 False
        m[..., :sink_tok] = True
        m[..., max(0, S_real - swa_tok):S_real] = True
        import os as _os
        if _os.environ.get("TLI_DEBUG") and self.layer_idx == 1:
            n_sel = int(m[..., :tok_hi].sum(dim=-1).float().mean().item())
            print(f"[MObadb] nb_sel={nb_sel} sel_total={n_sel}", flush=True)
        return m

    def compute_mask(self, q_ids, score_dict):
        score_coarse = score_dict["score_coarse"]
        score_fine = score_dict["score_fine"]
        # ---- 【075 修复 2026-10-10（GPT 审计 TL-E121-PROJ-PAD-S-075，P1）】----
        # 投影路径（--tli_proj_basis）的 k_qat 从 pad 后的 pad_k 派生 →
        # score_fine 宽度 = Tpad（块对齐 padding 后长度）而非真实 S。
        # 修复三件事（主会话拍板方向，与 GPT 建议一致）：
        #   ① S_real 显式取 prepare_index 携带的 valid_length（= pad 前 k
        #      真实 token 数；非投影路径 == score_fine 宽度，行为逐位不变）；
        #   ② padding token（[S_real, Tpad) 段）在任何选择之前永久置 -inf，
        #      保证 padding 永不入选任何池（topk/σ/强制区展开均免疫）；
        #   ③ SWA 强制区与 near/far 边界全按真实 S 计算。
        # 无 valid_length（外部直接合成 score_dict 驱动 compute_mask 的
        # 既有测试/回放 harness）→ 回退 score_fine 宽度口径，逐位兼容。
        _vl = score_dict.get("valid_length")
        S_width = score_fine.shape[-1]
        S_real = S_width if _vl is None else min(int(_vl), S_width)
        if _vl is not None and S_real < S_width:
            # 原地写 score_dict：p（softmax 源）、σ 分支与 ccluster 分支的
            # sf_raw 均从 score_dict["score_fine"] 现读 → 同一份屏蔽全生效
            score_dict["score_fine"][..., S_real:] = float("-inf")
        # ---- E103：kv-head 共享消融----
        # 共享口径（默认）：组内 mean 聚合到 kv-head 级 [1,1,Hkv,kt]；
        # per_q_head：旁路聚合，保留 per-q-head 分数 [1,1,H,kt] 直接进 topk
        # （topk 沿最后一维，H 维天然独立——32 个 q-head 各选各的块/token）。
        if not self.per_q_head:
            score_coarse = rearrange(
                score_coarse, "b qt (h g) kt -> b qt h g kt", g=self.group_size
            ).mean(dim=-2)  # [1,1,Hkv,kt]

        kt = score_coarse.shape[-1]
        bs = self.args.tia_block_size
        t = kt * bs - 1  # 当前序列长度（pad 后）近似
        # ---- E64 α/β/γ 分区：near 区动态长度（α=0 走老逻辑 near_len=2048）----
        # 区域权威定义（2026-09-29 用户澄清）：sink/swa 为正交强制区（不进创新管线
        # 不占配额）；mid = 除 sink/swa；near = mid 靠 q 的 α 比例；far = mid 其余。
        # 修复：near 池上界切到 swa_lo_blk（旧口径 near 池含 swa 8 块，抢 β 配额 25%）
        e64_partition = self.alpha > 0 and self.beta > 0
        nb_near = 0
        swa_tok = self.sliding_window_size   # e9acd1e 回归修复：α=0 路径 swa_tok 未定义
        # 【B10 修复 2026-10-10（kimi3 清单 §8 / 2026-10-10_0112 复审 Bug1）】
        # near 左界基准分支化：e64 分区臂从 swa 起点（S − swa_tok）往前推
        # near_len_dyn（=α·mid_len），使 near 区实际宽 = α·mid_len——TASK.md L137
        # 权威定义 near_L = α·mid_L、区间 [S−swa−near_L, S−swa)。旧口径从序列
        # 末尾（kt*bs）推 → near 实际宽 = α·mid_len − swa_tok（swa 被扣两次，
        # 短序列低 α 偏差 15-32%；S=4416/swa=192/α=0.5 实测 near POOL 1856
        # 应 2048）。(0,0) 单池（near_len_dyn=swa_tok）与老逻辑分支保持旧式
        # （kt*bs − near_len_dyn）逐位不变——(0,0) 的 far_hi = S − swa_tok
        # 恰为正确单池语义，不受本修复影响。
        # 【F5 修复 2026-10-10（kimi3 清单 2026-10-08）】e64 分区臂 mid/near_base
        # 用真实 S 而非 pad 后 kt*bs——非对齐 S 下 α·mid 偏差 ≤ α·63 token
        # （decode 每步序列长多数不整除 bs；SG 侧 taskmd 本按真实因果长
        # S_r=t+1 计算，本修使 HF/SG 非对齐行口径一致）。对齐 S 行为逐位
        # 不变；(0,0)/老逻辑分支保持 pad 口径（_maybe_build_kmeans 的 far_hi
        # 必须块对齐——E72 decode 缓存依赖，不动）。
        # 【075 修复（2026-10-10）】F5 原取「真实 S = score_fine 宽度」只在
        # 非投影路径成立（k_qat 从真实 k 派生）；投影路径 score_fine 宽 =
        # Tpad，F5 的真实 S 实取 pad 后长度 → mid/near/far 边界右移、SWA
        # 强制区落 padded 尾部（S=6145/bs=64 实测 far_hi 5312 应 5248、
        # 真实 SWA 只保 65/128）。现 S_real 统一取函数顶部的 valid_length
        # 口径（见函数头 075 注释）；非投影路径 valid_length == 宽度，
        # F5 修复行为逐位不变。
        # S_real 已在函数顶部解析（valid_length 优先，回退 score_fine 宽度）
        if e64_partition:
            sink_tok = self.sink_blocks * bs
            mid_len = max(0, S_real - sink_tok - swa_tok)
            near_len_dyn = max(bs, int(self.alpha * mid_len))  # near 只算 mid 部分（不含 swa）
            near_base = S_real - swa_tok   # near 右界 = swa 起点（B10；真实 S 口径 F5）
        elif self.far_select != "4bit" and self.alpha == 0 and self.beta == 0:
            # TASK.md L231 严格口径：cluster 组合 (0,0) 单池点 = 纯 cluster 全 mid
            near_len_dyn = swa_tok
            near_base = kt * bs
        else:
            near_len_dyn = self.near_len
            near_base = kt * bs
        near_blks = max(self.sink_blocks, (near_base - near_len_dyn) // bs)
        far_lo_blk, far_hi_blk = self.sink_blocks, near_blks  # 远端块区间（含 far_hi 前一块）
        # near 池上界 = swa 起点（swa 块完全排除出双池，纯靠 mask 强制）
        swa_lo_blk = max(near_blks, kt - max(1, swa_tok // bs))

        # ---- D'：跳层 → 远端块全部 -inf（保留 sink + 近端）----
        if self.skip_far:
            # 【E127 iter5】far_range 恒为连续区间 [far_lo_blk, far_hi_blk)——
            # tensor 高级索引赋值（index_put_）触发 host 同步，连续区间
            # 切片赋值数学恒等（区间空时两者同为空操作）→ 零同步改写
            score_coarse[..., far_lo_blk:far_hi_blk] = float("-inf")

        # ---- B：L2 分区 topk（far/near 独立预算，防远端被近端高分挤出）----
        # E4c 修正（E4b 有整簇超选 bug）：簇分数 token 级 topk 严格预算下平均捕获
        # 0.45-0.79，相对 minmax 块上界无一致优势；far 区保留 TIA 4bit token 级
        # 精筛（质量 ≈ oracle，L03 0.999），分区仅保证 far 预算不被挤出。
        # cluster 模式（簇分数选 token）留作消融 + negative result。
        # 门控修正（2026-09-29）：E64 α/β/γ 分区路径也须走 L2 分区预算，否则 γ 是死参数
        # （B5/B6 输出 200/200 逐字一致的根因——use_partition 仅挂在 enable_kmeans 下）
        # 【E111 D' gate 短路（用户 2026-10-12 拍板）】skip 层不再一律排除出
        # 分区路径：e64 分区几何（α>0 且 β>0）下 skip 层走「分区路径仅 far
        # 池置空」——far 粗筛/细筛结构性不选（compute_score 已省算），
        # far 池 k2_far=0 → near 池在 [far_tok_hi, swa_lo_tok) 拿满 K2_mid
        # 预算，sink/swa 强制不占预算（下方 L1368-1371 口径）。修掉 B1 修复
        # 时登记的残余口径差（旧版 skip 层 swa 以 p=1.0 占 K2 预算、near
        # 细筛实际 0 选中、far p=0 填充进 mask）。非分区组合（(0,0) 单池 /
        # enable_kmeans-only）skip 层仍落 legacy 尾部 topk（B1 sink 强制
        # 保留）——用户方案几何依赖 α/β 分区，该域无生产配置，行为不变。
        use_partition = (self.enable_kmeans or e64_partition) \
            and not (self.skip_far and not e64_partition)
        far_tok_score = None
        if use_partition and self.far_select in ("cluster", "sim_greedy"):
            q_last = getattr(self, "_last_q", None)
            if q_last is not None and self._km_centroids is not None:
                far_tok_score = self._far_token_score(q_last)  # [Hkv, Tfar]
        # E110：near 侧簇分数（ccluster / ccluster_sim 臂）
        near_tok_score = None
        near_cluster_on = self.near_select in ("cluster", "sim_greedy")
        if use_partition and near_cluster_on:
            q_last = getattr(self, "_last_q", None)
            if q_last is not None:
                near_tok_score = self._near_token_score(q_last)  # [Hkv, Tn]

        score_coarse[..., -1] = float("inf")  # TIA 语义：当前块强制
        k1 = self.args.tia_level1_topk
        # E109a：avg 分数源提取提前到分支外（单池路径也要按 far_method 选分数源）
        score_coarse_avg = score_dict.get("score_coarse_avg")
        if score_coarse_avg is not None and not self.per_q_head:
            score_coarse_avg = rearrange(
                score_coarse_avg, "b qt (h g) kt -> b qt h g kt", g=self.group_size
            ).mean(dim=-2)
        if e64_partition:
            # ---- E64 L1 双池：far 池=minmax 上界分（α 区外），near 池=avg 分（近区）----
            nb_near = max(1, int(round(k1 * self.beta)))
            # TASK.md L158 严格口径：far_budget_page_topk = budget_page_topk
            # − near_budget_page_topk（无保底；topk k=0 已验证安全）
            nb_far = max(0, k1 - nb_near)
            if self.skip_far:
                # D' 兑现省算：far 块全 -inf 时收缩 far 池。
                # 【E127 iter5】far 区间刚被上方整段置 -inf，计数数学恒
                # 0——host 常量替代 GPU 统计 + .item() host 同步（逐位
                # 等价；区间空时原计数亦为 0）
                n_valid_far = 0
                nb_far = min(nb_far, max(n_valid_far, 0))
            # far 池分数源：minmax（默认=块上界）或 avg（块均值，aavg 组合）
            if self.far_method == "avg" and score_coarse_avg is not None:
                sc_far = score_coarse_avg[..., far_lo_blk:far_hi_blk]
            else:
                sc_far = score_coarse[..., far_lo_blk:far_hi_blk]
            i_f = torch.topk(sc_far, min(nb_far, sc_far.shape[-1]), dim=-1).indices + far_lo_blk
            # near 池上界 = swa_lo_blk：swa 块（含当前块）完全排除出双池，
            # 纯靠 mask 强制（p[-swa:]=1）——正交区不占创新管线配额
            # near 池分数源：avg（默认=块均值）或 minmax（块上界，mminmax 组合）
            if self.near_method == "minmax" or score_coarse_avg is None:
                sc_near = score_coarse[..., near_blks:swa_lo_blk]
            else:
                sc_near = score_coarse_avg[..., near_blks:swa_lo_blk]
            sc_near = sc_near.clone()
            if sc_near.shape[-1] > 0:
                i_n = torch.topk(sc_near, min(nb_near, sc_near.shape[-1]), dim=-1).indices + near_blks
            else:
                i_n = i_f[..., :0]  # near 池空（mid 不足一块）→ 不选
            # 严格口径（2026-09-29 用户最终澄清）：sink 块完全不进 L1 池——
            # 正交强制区不进粗筛管线、不占 K1 配额。sink token 由 L2 最终
            # mask 直接置位（必选），mass 覆盖不受影响（旧 cat 版 sink 走管线
            # 抢配额；此前「漏 sink 即崩」由 mask 强制兜住）
            indices = torch.cat([i_f, i_n], dim=-1)
            topk_mask = torch.zeros_like(score_coarse, dtype=torch.bool).scatter_(
                -1, indices, torch.ones_like(indices, dtype=torch.bool)
            )
            import os as _os
            # TLI_DEBUG 显式布尔解析（字符串 "0"/"" 均为关——原字符串真值
            # 会把 TLI_DEBUG=0 也判开）；far-empty（i_f 空）时 min/max 用
            # -1 哨兵（与 i_n 同款保护，TL-DEBUG-FAR-EMPTY-006）
            _dbg = _os.environ.get("TLI_DEBUG", "").strip().lower() not in ("", "0", "false")
            if _dbg and self.layer_idx == 1:
                i_n_min = i_n.min().item() if i_n.numel() else -1
                i_n_max = i_n.max().item() if i_n.numel() else -1
                i_f_min = i_f.min().item() if i_f.numel() else -1
                i_f_max = i_f.max().item() if i_f.numel() else -1
                print(f"[L1dbg] kt={kt} near_blks={near_blks} swa_lo_blk={swa_lo_blk} "
                      f"nb_far={nb_far} nb_near={nb_near} "
                      f"i_f_blk_min={i_f_min} i_f_blk_max={i_f_max} "
                      f"i_n_blk_min={i_n_min} i_n_blk_max={i_n_max}", flush=True)
        else:
            # E109a 修复：单池（α=0 或 β=0）L1 分数源按 far_method 选择
            # （原实现固定 score_coarse=minmax → aavg/mavg(0,0) 单池实际跑成纯 minmax）
            sc_pool = score_coarse
            if self.far_method == "avg" and score_coarse_avg is not None:
                sc_pool = score_coarse_avg.clone()
                sc_pool[..., -1] = float("inf")  # TIA 语义：当前块强制（avg 分无此置位）
                if self.skip_far:  # D' skip 语义与 score_coarse 同步（far 区 -inf）
                    # 【E127 iter5】同上：连续区间切片赋值零同步等价改写
                    sc_pool[..., far_lo_blk:far_hi_blk] = float("-inf")
            if self.skip_far:
                # D' 真正兑现省算：跳层时 far 块已 -inf，topk 只取有效块数
                # （否则 topk 会用 -inf far 块填满 K1，far token 照样进 L2，白算）。
                # 【E127 iter5】sc_pool 除 far 区（minmax/avg 两口径均在
                # 上方整段置 -inf）外全有限（分数矩阵被全量覆写过）→
                # 有效块数 = kt − far 区块数，host 可算免同步（逐位等价）
                n_valid = kt - max(0, far_hi_blk - far_lo_blk)
                k1 = min(k1, max(n_valid, 1))
            values, indices = torch.topk(
                sc_pool, min(sc_pool.shape[-1], k1), dim=-1
            )
            topk_mask = torch.zeros_like(score_coarse, dtype=torch.bool).scatter_(
                -1, indices, torch.ones_like(values, dtype=torch.bool)
            )
        if self.per_q_head:
            # E103：mask 已是 per-q-head [1,1,H,kt]，只做块→token 展开（不按 G 扩）
            topk_mask = repeat(
                topk_mask, "b qt h kt -> b qt h (kt bs)", bs=bs
            )
        else:
            topk_mask = repeat(
                topk_mask, "b qt h kt -> b qt (h g) (kt bs)", g=self.group_size, bs=bs
            )

        if self.enable_async:
            if self.prev_mask is not None:
                topk_mask, self.prev_mask = self.prev_mask, topk_mask
                if topk_mask.shape[-1] < score_fine.shape[-1]:
                    pad_len = score_fine.shape[-1] - topk_mask.shape[-1]
                    topk_mask = F.pad(topk_mask, (0, pad_len), value=True)
            else:
                self.prev_mask = topk_mask

        topk_mask = topk_mask[..., : score_fine.shape[-1]]
        score_fine = torch.where(topk_mask, score_fine, float("-inf"))
        p = F.softmax(score_fine, dim=-1).nan_to_num(nan=0)
        if not self.per_q_head:
            # E103：共享口径组内 mean；per_q_head 保留 [1,1,H,T] 直接 topk
            p = rearrange(p, "b qt (h g) kt -> b qt h g kt", g=self.group_size).mean(dim=-2)
        # 【075 修复】SWA 强制区按真实 S：p[..., -swa:] 在投影路径落 padded
        # 尾部（消费端裁剪后真实序列尾只保 swa−(Tpad−S) 个 token，S=6145
        # 实测 65/128）。改为 [S_real−swa, S_real)；非投影路径 S_real==宽度
        # → 与旧 p[..., -swa:] 逐位等价（S_real<swa 时两者同为全宽度置位）。
        p[..., max(0, S_real - swa_tok):S_real] = 1.0
        # 【B4 修复（kimi3 清单 F4，2026-10-08）】σ 死参数 fail-closed：
        # sigma_select≠none 但分区选择路径不可用（use_partition=False：
        # enable_kmeans 与 α/β 分区均未开，或 skip_far 跳层）或与
        # moba_gate 组合（moba 分支提前 return）时，σ 分支静默不可达 →
        # 实验跑的不是声明的方法（静默退化为默认两级 topk）。置于 moba
        # 分支之前以覆盖全部三种组合。生产默认 sigma_select="none" 不触发，
        # E87 σ 臂（use_partition=True 且非 moba）行为不变。
        # 【E111 补充（2026-10-12）】skip+e64 现在走分区路径（use_partition
        # =True），但 σ×skip 组合从未实测、无数据背书——fail-closed 纪律
        # 不冒称，skip_far 条件独立保留（B4 ② 用例保持红证语义）。
        if self.sigma_select != "none" and (
                not use_partition or self.skip_far or self.moba_gate):
            raise ValueError(
                f"tli_sigma_select={self.sigma_select!r} 需要分区选择路径"
                f"（enable_kmeans 开启或 α>0 且 β>0）且不与 moba 组合；"
                f"当前 use_partition={use_partition}（enable_kmeans="
                f"{self.enable_kmeans}, alpha={self.alpha}, beta={self.beta},"
                f" skip_far={self.skip_far}）、moba_gate={self.moba_gate}"
                f" —— σ 会静默退化为默认两级 topk，拒绝执行。请开启分区"
                f"或改 tli_sigma_select=none。"
            )
        # ---- E89：MoBA 复现臂（统一 harness 公平口径）----
        # training-free chunk gate：全维 chunk-mean 分数 top-(K2/BS) 块全展开。
        # 无两级、无量化、无分区；sink/swa 保送与 PSI 同口径（预算 K2=1024
        # → 16 块 + sink 2 块 + swa 16 块，选中 token 数与 PSI 严格对齐）。
        if self.moba_gate:
            return self._moba_mask(score_dict, kt, bs, S_real)
        # ---- E87：top-σ 选择（用户 2026-10-02 定义）----
        # sigma 侧不做两级：该区全部 token 的细筛原始分（GQA group-mean 口径与 p
        # 一致）≥ sink token 分数 max − σ 即选中；预算可变（σ 为质量-预算旋钮）。
        # far=far 侧 top-σ + near 侧两级 / near=反向混合 / mid=整 mid 不分区 top-σ
        if use_partition and self.sigma_select != "none":
            K2_sig = min(p.shape[-1], self.args.tia_level2_topk)
            far_tok_lo = far_lo_blk * bs
            far_tok_hi = min(near_blks * bs, p.shape[-1])
            sink_tok = far_tok_lo
            # 075：swa 起点按真实 S（p 宽度在投影路径是 Tpad）
            swa_lo_tok = max(0, S_real - self.sliding_window_size)
            sf_raw = score_dict["score_fine"][..., : p.shape[-1]].to(torch.float32)
            sf_g = rearrange(sf_raw, "b qt (h g) kt -> b qt h g kt",
                             g=self.group_size).mean(dim=-2)
            sink_max = sf_g[..., :sink_tok].max(dim=-1, keepdim=True).values
            thr = sink_max - self.sigma
            m = torch.zeros_like(p, dtype=torch.bool)
            if self.sigma_select in ("far", "mid"):
                m[..., far_tok_lo:far_tok_hi] = (
                    sf_g[..., far_tok_lo:far_tok_hi] >= thr)
            else:
                far_p = p[..., far_tok_lo:far_tok_hi]
                k2_far = min(max(64, K2_sig), far_tok_hi - far_tok_lo)
                i_f = torch.topk(far_p, k2_far, dim=-1).indices + far_tok_lo
                m.scatter_(-1, i_f, True)
            if self.sigma_select in ("near", "mid"):
                if swa_lo_tok > far_tok_hi:
                    m[..., far_tok_hi:swa_lo_tok] = (
                        sf_g[..., far_tok_hi:swa_lo_tok] >= thr)
            else:
                near_p = p[..., far_tok_hi:swa_lo_tok] if swa_lo_tok > far_tok_hi \
                    else p[..., :0]
                k2_near = min(K2_sig, near_p.shape[-1])
                i_n = torch.topk(near_p, k2_near, dim=-1).indices + far_tok_hi
                m.scatter_(-1, i_n, True)
            # 正交强制区直接置位（必选不占预算）：sink 头部 + swa 尾部
            # 075：swa 尾部 = 真实 S 尾（非 padded 尾）；padding 段保持 False
            m[..., :sink_tok] = True
            m[..., swa_lo_tok:S_real] = True
            import os as _os
            if _os.environ.get("TLI_DEBUG") and self.layer_idx == 1:
                n_sig = int(m[..., far_tok_lo:swa_lo_tok].sum(dim=-1).float().mean().item())
                print(f"[SIGdbg] mode={self.sigma_select} sigma={self.sigma} "
                      f"mid_selected={n_sig} sink_max={float(sink_max.mean()):.3f}",
                      flush=True)
            return m
        # ---- B：L2 分区 topk（C = C_near + C_far 独立预算）----
        if use_partition:
            K2 = min(p.shape[-1], self.args.tia_level2_topk)
            far_tok_lo = far_lo_blk * bs
            # far 细筛池上界 = far/near 块边界（near_len_dyn 修复后只含 α·mid，
            # 不能再用 kt*bs-near_len_dyn——那会把 near 区吞进 far 池）
            # 【静态簇尾巴覆盖修复（GPT 审计 TL cluster tail coverage，2026-10-11，
            #   主 AI 核验 confirmed）】触发五元组（α=β=0 + far_select≠4bit +
            #   enable_kmeans + skip_far=false + 簇缓存已初始化）下，生产者
            #   _maybe_build_kmeans 的 far 上界按真实 S 块对齐向下（far_hi =
            #   max(sink+1, (S−swa)//bs)·bs ≤ S−swa），而 near_blks 在消费分支
            #   （L916-923）从 padding 过界的 kt*bs 推 → far_tok_hi 被撑大
            #   （最多 bs−1 token）→ 守卫 swa_lo_tok > far_tok_hi 恒 FALSE →
            #   near 补齐分支被跳过，而簇 topk 候选只到缓存覆盖 far_hi 为止
            #   → 残留尾巴 [far_hi, S−swa)（宽 = (S−swa) mod bs）结构性不可达。
            #   S=4097/bs=64/sink=swa=128 实测：far_hi=3968、far_tok_hi=4032、
            #   swa_lo_tok=3969 → token 3968 无论分数多高不可选。
            # 修复（单一事实源）：far 池上界不再两侧各算各的公式，直接读生产者
            #   写进簇缓存条目的 far_hi（_km_far_hi_cached，_build_far_kmeans /
            #   _update_far_greedy 全路径在建簇时维护 = 缓存真实覆盖右缘），
            #   簇覆盖 [far_lo, far_hi) 与消费上界严格一致；S%bs==0 时
            #   far_hi == S−swa == 旧 near_blks·bs → 行为逐位不变。
            #   far_tok_score is not None 蕴含 far_select∈{cluster,sim_greedy}
            #   且簇缓存已初始化（触发五元组后两条）+ skip_far=false（本
            #   use_partition 块已保证）。其余路径（e64 分区 / far_select=4bit /
            #   无簇分）保持旧口径逐位不变。
            if self.alpha == 0 and self.beta == 0 and far_tok_score is not None:
                far_tok_hi = min(self._km_far_hi_cached, p.shape[-1])
            else:
                far_tok_hi = min(near_blks * bs, p.shape[-1])
            # 严格口径（2026-09-29 用户最终澄清）：sink/swa 保护 token 必选且
            # 自动从预算排除——mid 预算 K2_mid = K2 − sink_tok − swa_tok；
            # 固定区 token 不进任何 topk 池，最终 mask 直接置位
            # （如 K2=2048、sink=128、swa=128 → mid_token = 1792 全给创新管线）
            sink_tok = far_tok_lo
            # 075：swa 起点/强制区宽度按真实 S——投影路径 p 宽 = Tpad，
            # 旧口径 swa 强制区落 padded 尾且 K2_mid 把 padding 段当 swa 扣除
            swa_lo_tok = max(0, S_real - self.sliding_window_size)
            K2_mid = max(0, K2 - sink_tok - (S_real - swa_lo_tok))
            # ---- E122：γ=off 自由竞争（用户 2026-10-10 指令，cavg 探索臂）----
            # 取消 near/far 配额分割：全部 mid 候选在统一预算 K2_mid 内单池
            # topk 竞争。语义边界（全部保持不变）：
            #   * sink/swa 保护带：正交强制区必选不占预算（下方直接置位）；
            #   * L1 的 β near 块预算：γ 只作用于 L2 配额，L1 双池照旧；
            #   * far 池空 guard（C-1）/near 区宽截断：单池天然覆盖
            #     （far 段宽 w_far = max(0, min(Tfar, 区宽))，near 段 = 区切片）；
            #   * 下方 nt_near/far_budget/k2_far/k2_near 配额逻辑整体旁路。
            # 【量纲混合风险（GPT 7c3190921 核验 2026-10-10，验收门未闭合）】
            # far_tok_score 路径（cavg/ccluster）下 far 侧簇分数
            # （_far_token_score 为 group-sum 点积原始量纲，无界）与 near 侧
            # 细筛 softmax 概率 p（[0,1]）混池竞争量纲不一致——far 点积
            # 系统性霸榜/挤出 near 风险高。far_method=avg 只改 L1 分数源，
            # 不归一化 L2 簇分，「配 cavg 即量纲一致」的说法无实现依据
            # （原注释此句已删，GPT 1930 补记指正）。默认路径（far 也用
            # 细筛分 p）两侧同源无此问题。归一化变体（far 段 row-wise
            # softmax/minmax 后再混池）列为 E123 判决后 matched 对照条件项。
            # ccluster 臂注意：γ=off 下 near 段固定用细筛分 p（本探索臂口径），
            # near 簇分数（near_tok_score）不参与自由竞争。
            if self.gamma is None:
                if far_tok_score is not None:
                    # cavg/ccluster 路径：far 区簇分数 + near 区细筛分拼接，
                    # 单次 topk 后索引回映射。far 段截到 far 区上界
                    # （far_tok_hi）：簇覆盖区（_km_far_lo 起 Tfar 宽）与
                    # near 段 [far_tok_hi, swa_lo_tok) 不重叠不漏——簇覆盖
                    # 越界块（far_hi_blk 用 max(sink+1,·) 可能比 near_blks 多
                    # 一块）的 token 由 near 段细筛分代表（质量 ≥ 簇代表分），
                    # 不重复占 K2_mid 预算。
                    # 短序列退化（far_tok_hi > swa_lo_tok，near 区倒挂）时 far 段
                    # 上界收到 swa_lo_tok：swa 强制区 token 不进竞争不占预算
                    w_far = max(0, min(far_tok_score.shape[-1],
                                       min(far_tok_hi, swa_lo_tok) - far_tok_lo))
                    fs = far_tok_score[..., :w_far]
                    if self.alpha == 0 and self.beta == 0:
                        # 【尾巴覆盖修复】(0,0)+非4bit 触发组合：far_tok_hi 已
                        # 改为生产者簇缓存右缘（上方单一事实源修复）→ near_seg
                        # 恰为簇缓存结构性覆盖不到的残留尾巴 [far_tok_hi,
                        # swa_lo_tok)（宽 < bs）。尾巴 token 用原始细筛分参与
                        # 自由竞争：p 是 softmax 概率（≤1）与无界簇分混池
                        # （上方量纲风险注），高分尾巴也竞争不过簇分 → 仍不可
                        # 选。改用 group-sum 细筛分（与 _far_token_score 的
                        # group-sum 簇分同量纲，ccluster near 池 C3 修复同款
                        # 口径理由），量纲一致公平竞争。e64 分区臂（α/β>0）的
                        # near 段保持 p 口径逐位不变。
                        sf_raw_goff = score_dict["score_fine"][..., : p.shape[-1]].to(torch.float32)
                        sf_gsum = rearrange(
                            sf_raw_goff, "b qt (h g) kt -> b qt h g kt",
                            g=self.group_size).sum(dim=-2)
                        near_seg = sf_gsum[..., far_tok_hi:swa_lo_tok] \
                            if swa_lo_tok > far_tok_hi else p[..., :0]
                    else:
                        near_seg = p[..., far_tok_hi:swa_lo_tok] \
                            if swa_lo_tok > far_tok_hi else p[..., :0]
                    cat_sc = torch.cat(
                        [fs.unsqueeze(0).unsqueeze(0), near_seg], dim=-1)
                    k2_free = min(K2_mid, cat_sc.shape[-1])
                    idx_free = torch.topk(cat_sc, k2_free, dim=-1).indices
                    # 索引回映射：拼接向量内 < w_far 是 far 段（簇分数相对
                    # _km_far_lo 偏移），否则 near 段（far_tok_hi 偏移）
                    i_tok = torch.where(
                        idx_free < w_far,
                        idx_free + self._km_far_lo,
                        idx_free - w_far + far_tok_hi,
                    )
                else:
                    # 默认路径：far 与 near 同用细筛分 p，整个 mid 一体竞争
                    mid_p = p[..., far_tok_lo:swa_lo_tok]
                    k2_free = min(K2_mid, mid_p.shape[-1])
                    i_tok = torch.topk(mid_p, k2_free, dim=-1).indices \
                        + far_tok_lo
                m_off = torch.zeros_like(p, dtype=torch.bool)
                m_off.scatter_(-1, i_tok, True)
                # 正交强制区直接置位（必选不占预算）：sink 头部 + swa 尾部
                # 075：swa 尾部 = 真实 S 尾；padding 段保持 False
                m_off[..., :sink_tok] = True
                m_off[..., swa_lo_tok:S_real] = True
                return m_off
            nt_near = 0
            # 【C-1 修复 2026-10-08（pool_starvation_audit §5-1）】far token 池空
            # （near_blks==sink_blocks：α=1 / 短序列 mid≤1 块 / α·mid 舍入<bs）时
            # 旧版把 near 池选择 + sink/swa 强制 + return 整体包在
            # `if far_tok_hi > far_tok_lo:` 内 → 静默落单池兜底：sink 不保证、
            # mid 超预算（896>768）、短序列全注意力。修复：guard 只保留 far
            # 选择段；near 池选择与强制区无条件执行。far 空时初始化空 i_f
            # （形状对齐 scatter，且下方 `i_n = i_f[..., :0]` 依赖该变量）。
            k2_far = 0
            i_f = torch.zeros_like(p[..., :0], dtype=torch.long)
            # 【E111 D' gate 短路】skip 层 far 池结构性置空：far 段粗筛分
            # 已 -inf（compute_score 省算或上方 L945 置位）→ L1 双池 far 池
            # 空载 → 细筛 p far 段全 0，旧版 topk 会把 p=0 的 far token 当
            # 填充选中占预算（mask 混入无效 far 候选）。直接跳过整个 far
            # 选择体 → k2_far=0 → k2_near=K2_mid 全额让渡 near 池
            # （「near 最终多取几个 topk」，用户方案语义）。
            if not self.skip_far and far_tok_hi > far_tok_lo:
                if far_tok_score is not None:
                    # 消融：远端按簇分数 token 级 topk
                    Tfar = far_tok_score.shape[-1]
                    if e64_partition:
                        # TASK.md L79/L172 严格口径：γ 对所有 method 组合生效
                        # （cavg/ccluster 的 far 簇分路径同样按 γ 切
                        # near_token/far_token 预算；原「cavg far_tokens=512
                        # γ 死参数」回归口径随 E109a 污染重跑一并废弃）
                        nt_near = min(int(nb_near * bs * self.gamma), K2_mid)
                        # TASK.md L172 严格口径（与 E109a-γ 同步）：无 64 保底
                        far_budget = max(0, K2_mid - nt_near)
                    elif self.alpha == 0 and self.beta == 0:
                        # TASK.md L231 严格口径：cluster 组合 (0,0) 单池点
                        # far（簇分）拿全部 mid 预算（near 区已空）。
                        # 【尾巴覆盖修复】far_tok_hi 已改为生产者簇缓存右缘
                        # （上方单一事实源修复），守卫 swa_lo_tok > far_tok_hi
                        # 在有残留尾巴时为 TRUE，但旧口径 far_budget=K2_mid
                        # 全给簇 topk → k2_near = K2_mid−k2_far = 0，near
                        # 补齐分支即使进得去也无预算（尾巴只挂名不覆盖）。
                        # 修复：先从 far 预算扣除尾巴宽度 gap_w = swa_lo_tok −
                        # far_tok_hi（< bs，簇缓存块对齐的结构性盲区），由
                        # 下方 near 补齐分支用细筛分给尾巴 token 正常打分
                        # 参与选择；mid 总预算 K2_mid 逐位不变（预算身份：
                        # K2 1024 = sink 128 + swa 128 + mid K2_mid 保持）。
                        # S%bs==0 时 gap_w=0 → far_budget 与旧口径逐位相同。
                        far_budget = max(0, K2_mid - max(0, swa_lo_tok - far_tok_hi))
                    else:
                        far_budget = self.far_tokens
                    k2_far = min(far_budget, Tfar, K2_mid)
                    i_f = (torch.topk(far_tok_score, k2_far, dim=-1).indices
                           + self._km_far_lo).unsqueeze(0).unsqueeze(0)
                else:
                    # E64 γ：near 细筛折扣 nt_near = nb_near·bs·γ，far 拿剩余预算
                    # TASK.md L172 严格口径：far_budget_token = budget_token −
                    # near_budget_token（去掉旧 64 保底——γ 高值下 far=0 即
                    # near 让渡满额，topk k=0 已验证安全）
                    if e64_partition:
                        nt_near = min(int(nb_near * bs * self.gamma), K2_mid)
                        far_budget = max(0, K2_mid - nt_near)
                    else:
                        far_budget = self.far_tokens
                    # 默认：远端与近端同用 4bit 精筛分数，仅在独立池内 topk
                    k2_far = min(far_budget, far_tok_hi - far_tok_lo, K2_mid)
                    far_p = p[..., far_tok_lo:far_tok_hi]
                    i_f = torch.topk(far_p, k2_far, dim=-1).indices + far_tok_lo
                    import os as _os
                    if _os.environ.get("TLI_DEBUG") and self.layer_idx == 1:
                        nfin = int((far_p > float("-inf")).sum(dim=-1).max().item())
                        print(f"[L2dbg] K2={K2} K2_mid={K2_mid} sink_tok={sink_tok} "
                              f"swa_tok={p.shape[-1] - swa_lo_tok} nt_near={nt_near} "
                              f"far_budget={far_budget} far_lo={far_tok_lo} "
                              f"far_hi={far_tok_hi} k2_far={k2_far} "
                              f"far_p_finite={nfin} pshape={list(p.shape)}", flush=True)
            # near 池只取 mid-near 区 [far_tok_hi, swa_lo_tok)——
            # sink/swa 不进 topk 竞争（旧版 near 池=全序列减 far 区，
            # swa p=1 满分 + sink 残留都会进池占预算）
            # 【C-1 修复 2026-10-08】本段从 `if far_tok_hi > far_tok_lo:` 内移出：
            # far 池空时 near 池照常选择、sink/swa 照常强制、K2_mid 预算照常生效
            # （far 空 ⇒ k2_far=0 ⇒ k2_near=K2_mid 全给 near 池，池内截断 min(k2_near,
            # near 区宽)——饥饿语义：剩余预算不跨池回补 far）。
            k2_near = max(0, K2_mid - k2_far)
            if swa_lo_tok > far_tok_hi:
                near_p = p[..., far_tok_hi:swa_lo_tok]
                if near_tok_score is not None:
                    # E110 ccluster：near 池换「簇分数（簇覆盖段）+ 细筛原始分
                    # （未覆盖段）」。两源均为 group-mean 缩放点积分（量纲一致：
                    # 【C3 修复 2026-10-08】簇分经 _last_q（已乘 softmax_scale）
                    # 计算，sf_g 来自 score_fine（b_q_full 同乘 scale）；
                    # _near_token_score 用 mean 聚合对齐 GQA 维）；且都不过 L1 块池
                    # 门控——簇代表分本身就是粗筛（TASK.md cluster 语义），
                    # 与 far 簇分路径口径一致。未覆盖段 = 块对齐右缘尾巴 +
                    # 建簇/消费边界漂移，用细筛分回退（细筛分质量 ≥ 簇代表分，
                    # 语义无损偏保守）
                    sf_raw = score_dict["score_fine"][..., : p.shape[-1]].to(torch.float32)
                    sf_g = rearrange(
                        sf_raw, "b qt (h g) kt -> b qt h g kt", g=self.group_size
                    ).mean(dim=-2)
                    near_sc = sf_g[..., far_tok_hi:swa_lo_tok].clone()
                    cs_lo = max(self._km_near_lo, far_tok_hi)
                    cs_hi = min(self._km_near_hi, swa_lo_tok)
                    if cs_hi > cs_lo:
                        seg = near_tok_score[..., cs_lo - self._km_near_lo:
                                              cs_hi - self._km_near_lo]
                        near_sc[..., cs_lo - far_tok_hi:cs_hi - far_tok_hi] = \
                            seg.unsqueeze(0).unsqueeze(0)
                    near_p = near_sc
                i_n = torch.topk(near_p, min(k2_near, near_p.shape[-1]), dim=-1).indices + far_tok_hi
            else:
                i_n = i_f[..., :0]
            topk_mask = torch.zeros_like(p, dtype=torch.bool)
            topk_mask.scatter_(-1, i_f, True)
            topk_mask.scatter_(-1, i_n, True)
            # 正交强制区直接置位（必选不占预算）：sink 头部 + swa 尾部
            # 075：swa 尾部 = 真实 S 尾；padding 段保持 False
            topk_mask[..., :sink_tok] = True
            topk_mask[..., swa_lo_tok:S_real] = True
            return topk_mask
        # 【075 修复】legacy 全序列 topk 只在真实 token 域 [0, S_real) 内选：
        # 投影路径 p 宽 = Tpad，旧 topk k=min(Tpad, K2) 在预算 ≥ 候选数时会把
        # p=0 的 padding 位选进 mask（挤占真实 token 预算 + 虚增 select_tokens
        # 统计）。非投影路径 S_real==p 宽 → 逐位不变。
        p_real = p[..., :S_real]
        values, indices = torch.topk(p_real, min(p_real.shape[-1], self.args.tia_level2_topk), dim=-1)
        topk_mask = (
            torch.zeros_like(p, dtype=torch.bool)
            .scatter_(-1, indices, torch.ones_like(values, dtype=torch.bool))
        )
        # 【B1 修复（kimi3 清单 F1，2026-10-08）】skip 层（skip_far=True）经
        # use_partition=False 落本 legacy 全序列 topk：e64/ccluster 分区臂的
        # L1 双池按设计排除 sink 块（sink 不进粗筛管线、不占 K1 配额），
        # sink token 的细筛分被 L1 门控成 -inf → p=0 → 本 topk 永不选中
        # sink（attention sink 永久丢失；S=4352/mavg 冠军配置实测 sink
        # 选中 0/128 token，对照非 skip 层 128/128）。swa 已由上方
        # p[..., -swa:]=1.0 兜住。此处对 skip 层补 sink 头部正交强制
        # （与分区路径 topk_mask[..., :sink_tok]=True 同款，不占 K2 预算）。
        # 仅 skip_far 触发——普通 (0,0) 单池尾部路径 sink 走 L1 池竞争
        # （历史走廊语义），保持逐位不变。
        # 【E111 D' gate 短路兑现（用户 2026-10-12 拍板）】原「已知残余
        # 口径差（记录不修）：skip 层 swa 仍以 p=1.0 占 K2 预算」已在
        # e64 分区组合兑现修复——use_partition 条件放开 skip 层 + far 池
        # 置空（见上方两处 E111 注释），skip+e64 走分区路径、强制区不占
        # 预算、near 拿满 K2_mid。本 legacy 掉落现在仅剩非分区组合
        # （(0,0) 单池 / enable_kmeans-only，无生产配置），B1 sink 强制
        # 保留为该域防线。
        if self.skip_far:
            sink_tok = min(self.sink_blocks * bs, topk_mask.shape[-1])
            topk_mask[..., :sink_tok] = True
        return topk_mask

    def clear(self):
        super().clear()
        self._km_centroids = None
        self._km_token_assign = None
        self._km_blk_ids = None
        self._km_far_hi_cached = -1
        self._km_dims = None
        # E110：near 侧簇缓存 + far 侧 greedy 增量状态跨请求必须重置
        # （clear() 在每次 prefill 前调用；far greedy 增量状态残留会把上一请求
        # 的簇心带入新请求的增量指派——贪心时序语义跨请求不成立）
        self._km_near_centroids = None
        self._km_near_assign = None
        self._km_near_lo = self._km_near_hi = -1
        self._km_near_key = None
        self._km_near_dims = None
        self._km_greedy_sums = None
        self._km_greedy_cnt = None
        self._km_greedy_sq = None
        self._km_greedy_klive = None
        # E85f：跨请求必须重置——clear() 在每次 prefill 前调用，随后
        # observe_prefill_q 用本请求的 q 统计重选 pair（否则残留上一请求）
        self._pair_idx = None
        # 【C3 修复 2026-10-08】_last_q 同样跨请求必须重置——否则任何
        # 「prefill 期调用 compute_mask 且簇已构建」的路径（TLI_DEBUG 重放类）
        # 会用上一请求 decode 的 stale q 打 near 簇分
        self._last_q = None
        # 【B5 修复（kimi3 清单 F7，2026-10-08）】_basis 跨请求重置：
        # 惰性提取于 prepare_index（投影基层切片 [Hkv, 32, r]），clear()
        # 不重置则 proj_basis × static_pair 同开时上一请求提取的切片跨
        # 请求残留。_basis_all（离线校准常量）不重置——_basis 由其在下次
        # prepare_index 重新派生，语义自愈（两开关当前均默认关，纯防御）。
        self._basis = None
        # 【E127 inplace 合并】增量缓存跨请求失效——clear() 在每次 prefill
        # 前调用，残留会把上一请求的 k_min/k_max/k_avg/k_qat 前缀带入新
        # 请求（k append-only 契约跨请求不成立），新请求首步必须全量重建。
        self._inc = None

    def get_block_size(self):
        return 1


# ======================================================================
# 【E127 iter6/iter8】Triton fast path kernels（原 tli_indexer_fast.py
# 并入；_HAS_TRITON=False 时整体不定义，调用点守卫短路走原版表达式）。
# ======================================================================
if _HAS_TRITON:

    @triton.jit
    def _score_fine_kernel(
        KQ, Q, OUT,   # KQ [T,Hkv,D] bf16；Q [H,D] fp32；OUT [H,T] out_dtype
        T, stride_kt, stride_kh, stride_qh, stride_oh, stride_ot,
        Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
    ):
        """GQA K 复用 score_fine：每 program 处理 (token 块 × 1 个 kv 头)，
        K 块 [BT, D] 载入寄存器后循环 g 个 q 头逐头点积，K 读带宽 /g。"""
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_t = pid_t * BT + tl.arange(0, BT)
        offs_d = tl.arange(0, D)
        mask_t = offs_t < T
        k = tl.load(KQ + offs_t[:, None] * stride_kt + pid_h * stride_kh + offs_d[None, :],
                    mask=mask_t[:, None], other=0.0).to(tl.float32)
        for j in tl.static_range(G):
            h = pid_h * G + j
            q = tl.load(Q + h * stride_qh + offs_d)      # [D] fp32（已乘 scale）
            s = tl.sum(k * q[None, :], axis=1)            # [BT] fp32
            tl.store(OUT + h * stride_oh + offs_t * stride_ot,
                     s.to(OUT.dtype.element_ty), mask=mask_t)

    @triton.jit
    def _score_coarse_kernel(
        KMIN, KMAX, Q, OUT,   # KMIN/KMAX [T,Hkv,D] bf16；Q [H,D] fp32；OUT [H,T]
        T, stride_kt, stride_kh, stride_kt2, stride_kh2,
        stride_qh, stride_oh, stride_ot,
        Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
    ):
        """【iter8】GQA K 复用 coarse minmax 上界分（min/max 两份 K 独立
        stride）：s[h,t] = Σ_d min(q,0)·k_min + Σ_d max(q,0)·k_max（负部配
        k_min、正部配 k_max，与原版 clamp 双 einsum 数学等价）。K 块
        [BT,D] 寄存器复用服务 g 个 q 头（读带宽 /g，免 repeat 物化）。"""
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_t = pid_t * BT + tl.arange(0, BT)
        offs_d = tl.arange(0, D)
        mask_t = offs_t < T
        kmin = tl.load(KMIN + offs_t[:, None] * stride_kt + pid_h * stride_kh
                       + offs_d[None, :],
                       mask=mask_t[:, None], other=0.0).to(tl.float32)   # [BT, D]
        kmax = tl.load(KMAX + offs_t[:, None] * stride_kt2 + pid_h * stride_kh2
                       + offs_d[None, :],
                       mask=mask_t[:, None], other=0.0).to(tl.float32)
        for j in tl.static_range(G):
            h = pid_h * G + j
            q = tl.load(Q + h * stride_qh + offs_d)      # [D] fp32（已缩放）
            q_low = tl.minimum(q, 0.0)                   # clamp(max=0)：负部 → k_min
            q_high = tl.maximum(q, 0.0)                  # clamp(min=0)：正部 → k_max
            s = tl.sum(kmin * q_low[None, :], axis=1) + tl.sum(kmax * q_high[None, :], axis=1)
            tl.store(OUT + h * stride_oh + offs_t * stride_ot,
                     s.to(OUT.dtype.element_ty), mask=mask_t)

    # 【E131-S2】簇分数 gather-dot 全融合 kernel（q_g 胶水 + einsum +
    # gather 三件套 → 单 launch；E127 GQA K 复用同款设计）。数学 =
    # _far/_near_token_score 原版表达式（AGG=sum 远端 / mean=AGG_DIV·sum
    # 近端，G 为 2 的幂 → 1/G 精确二进制除法）；fp32 累加顺序与 cublas
    # 不同 → ULP 级漂移口径同 iter6/8（mask 门禁实测，E131-S2 实测
    # 16 步逐位 0 差异 + e2e 24 步 12585120 位逐位一致）。
    @triton.jit
    def _tok_score_kernel(
        Q, IDXSUB, CENT, ASSIGN, OUT,
        # Q [H,D] fp32（已乘 scale）；IDXSUB [DD] int64；CENT [Hkv,Kc,DD] fp32；
        # ASSIGN [Hkv,T] int64；OUT [Hkv,T] fp32
        T, AGG_DIV,
        stride_qh, stride_cenh, stride_cenk, stride_ah, stride_at,
        stride_oh, stride_ot,
        Hkv: tl.constexpr, G: tl.constexpr, DD: tl.constexpr, BT: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        h = tl.program_id(1)
        offs_t = pid_t * BT + tl.arange(0, BT)
        mask_t = offs_t < T
        # ---- q_g：组内 G 行 q 取 idx_sub 维聚合（sum / mean=AGG_DIV·sum）----
        offs_d = tl.arange(0, DD)
        offs_j = tl.arange(0, G)
        sub = tl.load(IDXSUB + offs_d)                       # [DD]
        q_rows = tl.load(Q + (h * G + offs_j)[:, None] * stride_qh
                        + sub[None, :])                      # [G, DD] fp32
        qg = tl.sum(q_rows, axis=0) * AGG_DIV                # [DD]
        # ---- gather-dot：簇号值域 [0,Kc) < 2^31 → int32 索引（int64 地址
        # 计算慢，randn 单例口径实测 0.079→0.036ms）----
        a = tl.load(ASSIGN + h * stride_ah + offs_t * stride_at,
                    mask=mask_t, other=0).to(tl.int32)
        c = tl.load(CENT + h * stride_cenh + a[:, None] * stride_cenk
                    + offs_d[None, :], mask=mask_t[:, None], other=0.0)
        s = tl.sum(c * qg[None, :], axis=1)
        tl.store(OUT + h * stride_oh + offs_t * stride_ot, s, mask=mask_t)
