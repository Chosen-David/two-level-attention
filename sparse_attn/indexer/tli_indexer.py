"""TLI (Two-Level Indexer): tail-subspace L1 + far/near L2 partition + layer pruning.

  A  tail-32 subspace block min-max/avg coarse scoring
  B  far 4-bit token-level fine scoring + cluster alternatives + sink/swa + nb/nt/nl geometry
  D' static / dynamic far-skip layer pruning

Extends TIAIndexer, reusing its L2 4-bit quantized token-level fine scoring semantics.
"""
import json
import math
import operator
import os

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

from .tia_indexer import TIAIndexer
from ..info import DEFAULT_LAYER_SKIP_MASK_PATH

# Graph state (shared across 36 per-layer indexers); fallback to full rebuild if unavailable.
try:
    from ..ops.cuda_graph_state import graph_state as _graph_state_fn
except Exception:
    _graph_state_fn = None

# Triton fast path (graceful fallback when unavailable)
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
    _TL_HAS_SORT = hasattr(tl, "sort")
    _TL_HAS_GATHER = hasattr(tl, "gather")
except Exception:
    triton = None; tl = None
    _HAS_TRITON = False; _TL_HAS_SORT = False; _TL_HAS_GATHER = False

DEFAULT_MASK = DEFAULT_LAYER_SKIP_MASK_PATH
_GREEDY_TRITON_FN = None  # Triton lazy import for sim_greedy kernel


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


_GRAPH_GEO_CACHE = {}  # step-level geometry cache (shared by 36 per-layer indexers)


def _graph_geo_lookup(valid_len, geo_key):
    try:
        # DeviceDecodeInputs 的 Triton 写入不推进 PyTorch version counter，
        # 额外消费其逐步调用的显式修订号，避免 eager 跨步复用旧 S_real。
        version = (valid_len._version,
                   getattr(valid_len, '_tli_device_revision', 0))
    except RuntimeError:  # inference_mode tensors have no mutation counter
        return None
    ent = _GRAPH_GEO_CACHE.get(valid_len.data_ptr())
    if (ent is not None
            and ent[1] == version
            and ent[2] == geo_key):
        return ent[3]
    return None


def _graph_geo_store(valid_len, geo_key, geo):
    try:
        version = (valid_len._version,
                   getattr(valid_len, '_tli_device_revision', 0))
    except RuntimeError:
        return geo  # Never reuse geometry without a reliable mutation version.
    if len(_GRAPH_GEO_CACHE) > 8:   # 上限清理（B>1 行缓存 + 防无界增长）
        _GRAPH_GEO_CACHE.clear()
    _GRAPH_GEO_CACHE[valid_len.data_ptr()] = (
        valid_len, version, geo_key, geo)
    return geo


class TLIIndexer(TIAIndexer):
    """A+B+D' 全开为 method='tli'；参数可单独关闭做消融。"""

    def __init__(self, args, snapshot=None) -> None:
        # snapshot = 生产入口注入的冻结 TreatmentSnapshot
        # snapshot=None 保留旧路径读取行为（既有非 benchmark 入口不破），
        # 但生产入口（benchmark pred.py / pred_ruler.py）必须注入。
        super().__init__(args)
        self.layer_idx = None            # 由 register_patch 注入
        # ---- A：子空间维度选择（2026-09-29 用户指令：参数化 rope/nope/full，默认 full）----
        #   full = 全 128 维（默认）；rope = 前 64 旋转维；nope = 后 64 非旋转维；
        #   tail = 旧口径低频尾维 32（E71/B7/C0 主表口径，向后兼容保留）
        self.subspace = getattr(args, "tli_subspace", "full")
        # mla_rope: DeepSeek-V2 MLA 布局（rope 64d 在尾段）
        if self.subspace not in ("full", "rope", "nope", "tail", "random", "highfreq", "mla_rope"):
            raise ValueError(f"tli_subspace 非法取值: {self.subspace}")
        self.enable_subspace = getattr(args, "tli_enable_subspace", True)
        if self.subspace == "full":
            self.enable_subspace = False      # 全维（用户默认）
        elif self.subspace in ("rope", "nope", "mla_rope"):
            self.enable_subspace = True       # 显式子空间段
        # L2 细筛子空间：same=与 L1 一致 / full=全 128 维量化
        self.l2_subspace = getattr(args, "tli_l2_subspace", "same")
        if self.l2_subspace not in ("same", "full"):
            raise ValueError(f"tli_l2_subspace 非法取值: {self.l2_subspace}")
        # 模型注册入口可覆写为 GLM 的 256；未注入时保持 Qwen/Llama 128、
        # DeepSeek mla_rope 192 的既有语义。
        self.head_dim = 192 if self.subspace == "mla_rope" else 128
        # ---- B：远端 kmeans（far_select='cluster' 时才构建，默认 4bit 分区）----
        self.enable_kmeans = getattr(args, "tli_enable_kmeans", True)
        self.far_select = getattr(args, "tli_far_select", "4bit")
        self.far_clusters = getattr(args, "tli_far_clusters", 256)
        self.far_niter = getattr(args, "tli_far_niter", 10)
        self.far_blocks = getattr(args, "tli_far_blocks", 16)   # 远端块预算 K1_far
        self.far_tokens = getattr(args, "tli_far_tokens", 512)  # 远端 token 预算 K2_far
        # 【E151c oracle 逐层 far 配额终审臂】env TLI_E151C_ORACLE_QUOTA 指向
        # 36 元 JSON 数组（e151_layer_demand_v1_compact.json oracle far_sel，
        # min 233/mean 320/max 397，36/36 层超 K1=128 平铺供给 128）时，
        # compute_mask e64 分支的 far_budget 按层查表覆盖为 oracle 需求，
        # near 池自动吃 K2_mid 剩余（768−≈320≈448，与 oracle near 需求 416
        # 同量级）。生效范围仅 prefill 稀疏选择段——eager_prefill 在调用
        # prepare_mask 前后置/清 _e151c_on 旗标；decode 路径旗标恒 False
        # 不受扰（与 E138 锚单变量对拍）。未设 env 时 _e151c_quota=None，
        # 全部路径逐位不变（compute_mask 短路在 None 上）。
        self._e151c_quota = None
        self._e151c_on = False
        _e151c_path = os.environ.get("TLI_E151C_ORACLE_QUOTA")
        if _e151c_path:
            with open(_e151c_path) as _f:
                quotas = json.load(_f)
            if not isinstance(quotas, list) or not quotas:
                raise ValueError("E151c quota must be a nonempty JSON array")
            parsed = []
            for value in quotas:
                if type(value) not in (int, float):
                    raise ValueError("E151c quotas must be finite nonnegative numbers")
                try:
                    number = float(value)
                except OverflowError as exc:
                    raise ValueError("E151c quotas must be finite nonnegative numbers") from exc
                if not math.isfinite(number) or number < 0:
                    raise ValueError("E151c quotas must be finite nonnegative numbers")
                # Preserve the existing nearest-even rounding of oracle means.
                parsed.append(int(round(number)))
            self._e151c_quota = parsed
        self.near_len = 2048                                      # 近端窗（D'/B 共用）
        self.far_method = getattr(args, "tli_far_method", "minmax")
        self.near_method = getattr(args, "tli_near_method", "avg")
        # far_select/near_select: 4bit / cluster / sim_greedy
        self.near_select = getattr(args, "tli_near_select", "4bit")
        if self.near_select not in ("4bit", "cluster", "sim_greedy"):
            raise ValueError(f"tli_near_select 非法取值: {self.near_select}")
        self.sim = getattr(args, "tli_sim", 0.9)
        self.sim_dims = getattr(args, "tli_sim_dims", "subspace")
        if self.sim_dims not in ("subspace", "nope", "tail", "full"):
            raise ValueError(f"tli_sim_dims 非法取值: {self.sim_dims}")
        # top-σ 选择
        self.sigma_select = getattr(args, "tli_sigma_select", "none")
        self.moba_gate = getattr(args, "tli_moba", False)
        self.sigma = getattr(args, "tli_sigma", 8.0)
        self.sink_blocks = 2                                      # 历史兼容锚点；实际由 sink_tokens 推导
        self.sink_tokens = int(getattr(args, "tli_sink_tokens", 128))
        if self.sink_tokens < 0:
            raise ValueError("tli_sink_tokens 必须 >= 0")
        if self.sliding_window_size < 1:
            raise ValueError("tli_swa_size 必须 >= 1")
        # swa 必须块对齐（sink 可子块对齐，StreamingLLM sink=4 兼容）
        _bs = self.args.tia_block_size
        if self.sliding_window_size % _bs != 0:
            raise ValueError(
                f"tli_swa_size={self.sliding_window_size} 必须是 "
                f"tia_block_size={_bs} 的整数倍（非对齐 swa 让边界块 mid "
                "token 结构性不可选）。sink 允许子块值（如 4）")
        self.sinkswa_only = bool(getattr(args, "tli_sinkswa_only", False))  # 归因臂：sink+swa 强制，mid 全丢
        # D' gate 动态标定：prefill 尾行观测 far 质量 < eps 的层 skip far
        self.dyn_dgate = bool(getattr(args, "tli_dyn_dgate", False))
        self.dyn_dgate_eps = float(getattr(args, "tli_dyn_dgate_eps", 0.05))
        self.dyn_dgate_rows = int(getattr(args, "tli_dyn_dgate_rows", 256))
        if not math.isfinite(self.dyn_dgate_eps) \
                or not (0.0 <= self.dyn_dgate_eps <= 1.0):
            raise ValueError(
                f"tli_dyn_dgate_eps={self.dyn_dgate_eps!r} 非法：必须为有限值"
                "且 0<=eps<=1（far 区概率质量损失上界阈值）")
        if self.dyn_dgate_rows < 1:
            raise ValueError(
                f"tli_dyn_dgate_rows={self.dyn_dgate_rows} 非法：必须 >= 1"
                "（标定用 prefill q 尾行数）")
        # decode-frozen 生命周期：prefill 期末暂存，decode 首步冻结进 skip_far
        self._dyn_skip_pending = False
        self._dyn_frozen = False
        if self.dyn_dgate and self.sinkswa_only:
            raise ValueError("--tli_dyn_dgate 与 --tli_sinkswa_only 互斥")
        if self.sinkswa_only:
            if self.sigma_select != "none":
                raise ValueError("--tli_sinkswa_only 与 --tli_sigma_select 互斥（归因臂不做 σ 选择）")
            if self.moba_gate:
                raise ValueError("--tli_sinkswa_only 与 --tli_moba 互斥（归因臂不做块 gate）")
            if getattr(args, "tli_per_q_head", False):
                raise ValueError("--tli_sinkswa_only 与 --tli_per_q_head 互斥（归因臂用共享 kv-head mask）")
        if self.near_select != "4bit":
            if self.sigma_select != "none":
                raise ValueError(
                    "--tli_near_select 与 --tli_sigma_select 互斥"
                    "（σ 路径绕过两级选择）")
            if self.moba_gate:
                raise ValueError("--tli_near_select 与 --tli_moba 互斥")
        # per-q-head：旁路 GQA 组内聚合，逐 q-head 独立选择（与 cluster/moba/σ 互斥）
        self.per_q_head = getattr(args, "tli_per_q_head", False)
        if self.per_q_head:
            if self.moba_gate:
                raise ValueError(
                    "--tli_per_q_head 与 --tli_moba 互斥"
                    "（gate 是 kv-head 级语义）")
            if self.far_select != "4bit":
                raise ValueError(
                    "--tli_per_q_head 与 --tli_far_select cluster/sim_greedy "
                    "互斥（簇分数是 kv-head 级语义）")
            if self.near_select != "4bit":
                raise ValueError(
                    "--tli_per_q_head 与 --tli_near_select cluster/sim_greedy "
                    "互斥（簇分数是 kv-head 级语义）")
            if self.sigma_select != "none":
                raise ValueError(
                    "--tli_per_q_head 与 --tli_sigma_select 互斥")
        self.enable_layer_skip = getattr(args, "tli_enable_layer_skip", True)
        # 动态 D' gate 与静态层掩码互斥（静态会覆盖动态冻结的 skip_far）
        if self.dyn_dgate and self.enable_layer_skip:
            raise ValueError(
                "--tli_dyn_dgate 与 --tli_enable_layer_skip true 互斥："
                "静态层掩码会覆盖动态冻结的 skip_far")
        self.skip_far = False
        self._skip_ids: set[int] | None = None
        if snapshot is not None:
            if self.enable_layer_skip:
                if snapshot.skip_ids is None:
                    raise SystemExit(
                        "[GATE-FAIL] snapshot.skip_ids=None 但 "
                        "tli_enable_layer_skip=True")
                self._skip_ids = set(snapshot.skip_ids)
        else:
            mask_path = getattr(args, "tli_layer_skip_path", None) or DEFAULT_MASK
            if self.enable_layer_skip:
                try:
                    with open(mask_path) as f:
                        self._skip_ids = set(json.load(f)["skip"])
                except (OSError, KeyError, ValueError):
                    print(f"[TLI] 警告: 层掩码 {mask_path} 不可读, D' 关闭")
                    self._skip_ids = None
        # 聚类缓存（prefill 一次；decode 期 far 区不变）
        self._km_near_centroids = None; self._km_near_assign = None
        self._km_near_lo = -1; self._km_near_hi = -1
        self._km_near_key = None; self._km_near_dims = None
        self._km_greedy_sums = None      # [Hkv, Kcap, d']  簇成员和（簇心 = sums/cnt）
        self._km_greedy_cnt = None       # [Hkv, Kcap]
        self._km_greedy_sq = None        # [Hkv, Kcap]  ||sums||^2 增量维护（免每步全量 norm）
        self._km_greedy_klive = None     # [Hkv] 活簇数
        # 增量缓存 + Triton fast path 健康标志（异常后 permanent fail-closed 回退）
        self._stat_ok = True
        self._quant_ok = True
        self._qat_idx_cache = None
        # α/β/γ legacy 比例口径（与 nb/nt/nl 显式几何互斥）
        _a_raw = getattr(args, "tli_alpha", None)
        _b_raw = getattr(args, "tli_beta", None)
        self.alpha = 0.0 if _a_raw is None else float(_a_raw)
        self.beta = 0.0 if _b_raw is None else float(_b_raw)
        # 比例同时控制分区开关与区宽/页预算；NaN 会静默关闭分区，
        # beta>1 可让 near 页配额超过 K1。构造期拒绝，不截断或改算法。
        for name, value in (("tli_alpha", self.alpha), ("tli_beta", self.beta)):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须是 [0, 1] 内的有限比例，收到 {value!r}")
        # γ 三态：数值=配额折扣；None=legacy 默认；"off"=自由竞争
        _g_raw = getattr(args, "tli_gamma", None)
        if isinstance(_g_raw, str) and _g_raw.strip().lower() == "off":
            self.gamma = None
        else:
            self.gamma = 1.0 if _g_raw is None else float(_g_raw)
        _abg_explicit = (_a_raw is not None or _b_raw is not None
                         or _g_raw is not None)
        # nb/nt/nl 固定配额几何（far 侧 = K1/K2 减法推出）
        _nt_raw = getattr(args, "tli_near_tokens", None)
        _nt_off = isinstance(_nt_raw, str) and str(_nt_raw).strip().lower() == "off"
        def _quota_integer(name, value):
            # Direct Python callers bypass argparse's integer conversion.
            # int(-0.5) == 0 would otherwise bypass the nonnegative guards.
            if value is None:
                return None
            if not isinstance(value, bool):
                try:
                    # Preserve existing integer-string inputs and integer-like
                    # objects, but never truncate floating point quota values.
                    return int(value) if isinstance(value, str) else operator.index(value)
                except (TypeError, ValueError, OverflowError):
                    pass
            raise ValueError(f"{name} 须为整数计数，收到 {value!r}")

        self.nt_near_pages = _quota_integer(
            "tli_near_pages", getattr(args, "tli_near_pages", None))
        self.nt_near_tokens = None if _nt_off else _quota_integer("tli_near_tokens", _nt_raw)
        self.nt_near_len = _quota_integer(
            "tli_near_len", getattr(args, "tli_near_len", None))
        _geo = (self.nt_near_pages is not None or _nt_raw is not None
                or self.nt_near_len is not None)
        if _geo:
            if _abg_explicit:
                raise ValueError(
                    "tli_near_pages/tokens/len 与 tli_alpha/beta/gamma 不可同传"
                )
            if self.nt_near_pages is None:
                raise ValueError(
                    "固定配额几何须显式传 tli_near_pages / tli_near_tokens / tli_near_len")
            if self.nt_near_tokens is None and not _nt_off:
                raise ValueError(
                    "tli_near_tokens 须传非负整数或 'off'（自由竞争哨兵）——"
                    "只传 tli_near_pages 而缺 nt 属歧义配置")
            if self.nt_near_len is None:
                raise ValueError(
                    "固定配额几何须显式传 tli_near_len（E155d：几何派单下 "
                    "α/β/γ 禁传，near 区宽无比例规则回退）")
            # E151c replaces far_budget only during prefill, overriding nt.
            # Do not silently combine that per-layer experiment with a fixed
            # nb/nt/nl contract (even if one layer happens to match its quota).
            if self._e151c_quota is not None:
                raise ValueError(
                    "fixed geometry (tli_near_pages/tokens/len) cannot be combined "
                    "with TLI_E151C_ORACLE_QUOTA (E151c prefill quota override)")
            # These branches return before fixed L2 quotas are consumed.
            # nt=off is fixed-budget free competition, not variable top-sigma.
            if self.sinkswa_only or self.moba_gate or self.sigma_select != "none":
                raise ValueError(
                    "fixed geometry (tli_near_pages/tokens/len) requires ordinary "
                    "two-level top-k; no sinkswa_only, MoBA or sigma selection")
            if _nt_off:
                # nt=off 归一：γ → 自由竞争（消费端与 E122 γ-off 同一分支，
                # 逐位等价）；nb/nl 显式覆盖照常生效。
                self.gamma = None
            # 激活分区（e64_partition = α>0 且 β>0；几何显式时 α/β/γ 对
            # 主路径全部失活——nb/nt/nl 三处消费点均显式优先，L522/L995/
            # L1411 的 α·mid_len 消费点同已改为 nt_near_len 显式优先）。
            self.alpha = 1.0
            self.beta = 1.0
            # 构造期几何约束（用户 2026-10-19 三维网格：nb·bs ≤ nl；
            # nt ≤ nb·bs；far = K2 − sink − swa − nt ≥ 0）。
            _bs_c = int(getattr(args, "tia_block_size", 64))
            _k2_c = int(getattr(args, "tia_level2_topk", 1024))
            _nb_c = int(self.nt_near_pages)
            _nl_c = int(self.nt_near_len)
            if _nb_c < 0:
                raise ValueError(f"tli_near_pages 须非负，收到 {_nb_c}")
            # L1 两池共享 K1；far 截到 0 并不能限制过大的 near 配额。
            # nt=off 只取消 L2 分区，仍须遵守同一 L1 页预算。
            if _nb_c > self.args.tia_level1_topk:
                raise ValueError(
                    f"tli_near_pages={_nb_c} 不可超过 "
                    f"tia_level1_topk={self.args.tia_level1_topk}（L1 总页预算）")
            if _nb_c * _bs_c > _nl_c:
                raise ValueError(
                    f"网格约束破坏：nb·bs ≤ nl（nb={_nb_c}·bs={_bs_c}="
                    f"{_nb_c * _bs_c} > nl={_nl_c}）——near 页配额超出 "
                    "near 区宽度能供给的页数")
            # nt=off 仅取消 near/far 的 L2 分额；保护区仍计入 K2。
            # K2_mid 截到 0 不能消除最终 mask 强制保留的 sink/SWA。
            if _k2_c < self.sink_tokens + self.sliding_window_size:
                raise ValueError(
                    f"K2={_k2_c} 不可小于 sink+swa="
                    f"{self.sink_tokens + self.sliding_window_size}（保护区总预算）")
            if not _nt_off:
                _nt_c = int(self.nt_near_tokens)
                if _nt_c < 0:
                    raise ValueError(f"tli_near_tokens 须非负，收到 {_nt_c}")
                if _nt_c > _nb_c * _bs_c:
                    raise ValueError(
                        f"网格约束破坏：nt ≤ nb·bs（nt={_nt_c} > nb={_nb_c}·"
                        f"bs={_bs_c}={_nb_c * _bs_c}）——near token 配额超出 "
                        "near 页几何容量")
                _far_c = (_k2_c - int(self.sink_tokens)
                          - int(self.sliding_window_size) - _nt_c)
                if _far_c < 0:
                    raise ValueError(
                        f"网格约束破坏：far = K2−sink−swa−nt = {_far_c} < 0"
                        f"（K2={_k2_c}, sink={self.sink_tokens}, "
                        f"swa={self.sliding_window_size}, nt={_nt_c}）")
        # 边界页保送（不占 K1 配额）+ L2 细筛去量化
        self.boundary_bypass = getattr(args, "tli_boundary_bypass", True)
        self.l2_quant = getattr(args, "tli_l2_quant", False)
        # 精筛候选化：仅对 L1 候选 + 强制区打分（消除全宽 einsum）
        self._fine_gather = (
            os.environ.get("TLI_FINE_GATHER", "1").strip() != "0")
        self._tmask_ok = True
        # 【E127 inplace 合并】Triton fast path + 尾段统计/量化融合
        # kernel 健康门（异常置 False 后本实例永久回退原版表达式）。
        self._triton_ok = True
        self._stat_ok = True
        self._quant_ok = True
        # E151c is implemented only in the E64 split-budget top-k branches.
        # Other selection modes return before those branches or use a different
        # budget contract; accepting the table there would silently ignore it.
        if self._e151c_quota is not None and (
            not (self.alpha > 0 and self.beta > 0)
            or self.gamma is None
            or self.sinkswa_only
            or self.moba_gate
            or self.sigma_select != "none"
        ):
            raise ValueError(
                "E151c oracle quota requires alpha>0, beta>0, numeric gamma "
                "and ordinary two-level top-k (no sinkswa_only, MoBA or sigma selection)"
            )
        self._basis_all = None
        bp_path = getattr(args, "tli_proj_basis", None)
        if bp_path:
            if self.subspace != "tail":
                raise ValueError(
                    f"tli_proj_basis 定义在 tail-32 子空间上，与 tli_subspace={self.subspace} "
                    "不兼容（投影基的输入是 32 维特征）；请用 --tli_subspace tail"
                )
            # A1（TL-E134-PROJ-L2FULL-137）：投影 L1 + 全维 L2 组合未实现，
            # fail-closed。投影路径下 L2 细筛仍以 rank-r 投影特征执行
            # （k_qat 由投影后特征量化），与 --tli_l2_subspace full 声明的
            # 「L2 全 128 维量化」不符——此前静默接受 = CLI 声明与实际
            # 执行行为背离，构造期拒绝。
            if self.l2_subspace == "full":
                raise ValueError(
                    "--tli_proj_basis 与 --tli_l2_subspace full 组合未实现："
                    "投影路径的 L2 细筛以 rank-r 投影特征执行，非 CLI 声明的"
                    "全 128 维量化，拒绝构造（需「投影 L1 + 全维 L2」时请先"
                    "实现该组合再放开）"
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

        # ---- far-skip KV eviction codex 接入 ----
        self.far_skip_backend = getattr(args, "tli_far_skip_backend", "legacy")
        self.far_skip_cache = getattr(args, "tli_far_skip_cache", "retain")
        if self.far_skip_backend != "legacy" or self.far_skip_cache != "retain":
            from ..ops.far_skip_runtime import configure_far_skip
            configure_far_skip(self)

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

        # ---- far-skip compact 配置（必须在 nt_near_len/nt_near_pages 之后）----
        self.far_skip_backend = getattr(args, "tli_far_skip_backend", "legacy")
        self.far_skip_cache = getattr(args, "tli_far_skip_cache", "retain")
        if self.far_skip_backend != "legacy" or self.far_skip_cache != "retain":
            from ..ops.far_skip_runtime import configure_far_skip
            configure_far_skip(self)

    # ---------------- E135：sink/swa 参数化的单一事实源 ---------------- #

    @property
    def _sink_pool_lo_blk(self):
        """E135：L1/L2/簇池左界（块对齐）——floor(sink_tokens/bs)。

        默认 sink_tokens=128 → 128//64 = 2 ≡ 历史 sink_blocks，全部派生值
        逐位不变。子块 sink（如 4 tok < bs，E143 起生产允许）时左界为
        0——sink 所在块参与池竞争，与强制区并集，无 token 结构性丢失
        （ceil 口径会在 [sink_tokens, ceil·bs) 产生既不强制也不可选的
        死区，弃用）。"""
        return max(0, self.sink_tokens // self.args.tia_block_size)

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
        if (os.environ.get("TLI_DEBUG", "").strip().lower() not in ("", "0", "false")
                and self.layer_idx is not None):
            print(f"[E85f] L{self.layer_idx} static pairs (freq j): "
                  f"{sorted(top_pairs.tolist())}", flush=True)

    # ---------------- E137：D' gate prefill 动态标定 ---------------- #

    def calibrate_dyn_dgate(self, q, k):
        """E137 动态 D' gate 标定（qwen3_attn_patch prefill 分支调用）。

        q [B,H,S,D] post-RoPE / k [B,Hkv,S,D]（均 patch 侧 transpose 后布局，
        E137 冒烟实证：误按 [B,S,Hkv,D] 多转置一次 → Hkv 读成 D → G=0 崩）。
        取 q 尾 dyn_dgate_rows 行（默认 256，与 observe_prefill_q E85f 同口径），
        per-q-head 全维 softmax（k GQA 展开到 H 头——组内 mean 会改变 softmax
        分布，E137 口径对齐判决 e137_calib_alignment 已证其失真），
        far 区 = 生产 far 区 [sink_tokens, sink+(1-α)·mid_len)（mid_len =
        S-sink-swa；该区在 skip_far 时不被 near 选择/sink/swa 任一覆盖，
        其概率质量即真实质量损失上界的 head-mean 近似），
        r_l = rows×heads 全均值。r_l < eps → 该层 decode 期 skip_far=True
        （_resolve_skip 消费链零新造，E111 dgate_short 省算五消费点全复用）。

        口径对齐判决（e137_calib_alignment_all.json，16 样本全量）：
        E6 静态口径「13/36 层可跳」系 sum-over-G 分数聚合 artifact（组内
        4 head far 分数互抵消把聚合分布压零，per-head 口径同层高一个量级，
        E129 静态 gate e2e -4.47 与此自洽）；per-head 生产 far 区口径下
        τ=0.05 割 22/576 层（4%）、τ=0.10 割 104/576（18%）。eps 默认 0.05
        （保守），e2e 双臂 0.05/0.10 判决。head-max 保守口径全量零头寸
        ——head 间 far 依赖不均是硬约束，mean 口径阈值即质量-速度旋钮。

        有效性背书：E60 已离线验证 prefill 尾行 far mass 与 decode 尾行
        far mass 层间 corr 0.966~0.991——far 依赖性是层属性，prefill 标定
        对 decode 有效。成本 ~[rows,H,S,D] GEMM/层（64K/256 行 <30ms 级），
        prefill 一次付清。
        """
        if not self.dyn_dgate:
            return
        S = q.shape[2]
        sink_tok = self.sink_tokens
        swa_tok = self.sliding_window_size
        mid_len = S - swa_tok - sink_tok
        # 生产 far 区右界（E135 analyze 同式）：far = [sink, sink+(1-α)·mid)。
        # E155d：near 区宽显式（nt_near_len）时覆盖 α 规则（与 compute_mask
        # L2000 同源）——几何派单下 α 已填充 1.0，不覆盖会静默把 far 区
        # 标定成空（far_hi=sink）。
        if self.nt_near_len is not None:
            _nl_c = max(int(self.nt_near_len), 0)
            far_hi = sink_tok + max(0, mid_len - _nl_c)
        else:
            far_hi = sink_tok + int((1.0 - self.alpha) * mid_len)
        far_lo = sink_tok
        if far_hi <= far_lo:          # 序列太短：far 区空，跳过标定不 skip
            return
        rows = min(self.dyn_dgate_rows, S)
        qe = q.detach()[:, :, -rows:, :].float()          # [B,H,rows,D]
        ke = k.detach()[:, :, -S:, :].float()            # [B,Hkv,S,D]（已是 patch 转置后布局）
        B, H, _, D = qe.shape
        Hkv = ke.shape[1]
        G = H // Hkv
        scale = D ** -0.5
        # per-q-head 概率（E137 口径）：k GQA 展开到 H 头，不做组内 mean
        ke_e = ke.repeat_interleave(G, dim=1)             # [B,H,S,D]
        s = torch.einsum("bhrd,bhsd->bhrs", qe, ke_e) * scale
        del ke_e
        # 因果 mask：每个尾行 r 可见 k ∈ [0, S-rows+r]
        pos_q = torch.arange(S - rows, S, device=q.device).view(1, 1, -1, 1)
        pos_k = torch.arange(S, device=q.device).view(1, 1, 1, -1)
        s = s.masked_fill(pos_k > pos_q, float("-inf"))
        p = torch.softmax(s, dim=-1)
        r_l = float(p[..., far_lo:far_hi].sum(-1).mean())
        del s, p
        # A6（TL-E137-PREFILL-GATE-145）：只写 pending，不直接落 skip_far。
        # 决策由 decode 首步 freeze_dyn_dgate 冻结生效——prefill 期
        # skip_far 恒 False：dense prefill（默认路径）不消费 skip_far，
        # 行为逐位不变；TLI_SPARSE_PREFILL=1 时稀疏 prefill chunk 不再
        # 消费「完整 prompt 标定」的决策（未来后缀影响较早 chunk 路由的
        # 非因果缺陷修复）。对现产线（decode 才消费）等价。
        self._dyn_skip_pending = bool(r_l < self.dyn_dgate_eps)
        if (os.environ.get("TLI_DEBUG", "").strip().lower() not in ("", "0", "false")
                and self.layer_idx is not None):
            print(f"[E137] L{self.layer_idx} r_l={r_l:.6f} eps={self.dyn_dgate_eps} "
                  f"pending_skip_far={self._dyn_skip_pending}", flush=True)

    def freeze_dyn_dgate(self):
        """A6（TL-E137-PREFILL-GATE-145）：decode 首步一次性冻结标定决策。

        qwen3_attn_patch decode 分支（prepare_mask 前）调用：首次 decode 步
        把 prefill 期挂起的 _dyn_skip_pending 写入 skip_far 并置 _dyn_frozen
        哨兵（后续 decode 步幂等 no-op）。clear() 复位 pending/frozen 两状态
        （新请求 prefill 重新标定）。非 dyn_dgate 实例恒 no-op。
        """
        if self._dyn_frozen or not self.dyn_dgate:
            return
        self.skip_far = bool(self._dyn_skip_pending)
        self._dyn_frozen = True
        if (os.environ.get("TLI_DEBUG", "").strip().lower() not in ("", "0", "false")
                and self.layer_idx is not None):
            print(f"[E137] L{self.layer_idx} decode 首步冻结 "
                  f"skip_far={self.skip_far}", flush=True)


    # ---------------- D' ---------------- #

    def _resolve_skip(self):
        if self._skip_ids is not None and self.layer_idx is not None:
            self.skip_far = self.layer_idx in self._skip_ids

    # ---------------- A：子空间 L1 ---------------- #

    def _tail_indices(self, device):
        """L1/L2 共用的低频配对维；索引对应 post-RoPE 布局。"""
        if self.head_dim == 256:
            # GLM 的 apply_rotary_pos_emb_interleave 输出 cat(real, imag)，
            # RoPE64 位于 192:256，每半区 32 维；低频16对分别在两半区尾部。
            return torch.cat((torch.arange(208, 224, device=device),
                              torch.arange(240, 256, device=device)))
        delta = 64 // self.cmp_ratio
        return torch.tensor(
            list(range(64 - delta, 64)) + list(range(128 - delta, 128)),
            device=device)

    def _subspace_indices(self, device):
        # 子空间选择（用户指令参数化）：rope/nope=模型对应的旋转/非旋转维 /
        # tail=低频尾维（旧口径，cmp_ratio 控制宽度）/ full 不进此函数
        # E85f：静态 pair 已定（prefill q 统计）→ 用 per-layer 完整旋转对替换 tail32
        if self.static_pair and self._pair_idx is not None:
            return self._pair_idx.to(device)
        # 【E133b 原生化】实例级缓存：本函数在 tail 模式下每 decode step 被
        # prepare_index/compute_score 各调一次，原每次 CPU list 构造 +
        # pageable H2D（~20-40µs/步 × 2-3 次）——是 tail 臂实测 +5.7% 稳态
        # 回归的主因。子空间配置是实例常量 → 按 device 缓存零构造零拷贝。
        # clear() 不失效（与 _qat_idx_cache 同口径：配置级常量非请求态）。
        cache_key = (self.head_dim, device)
        cache = getattr(self, "_idx_sub_cache", None)
        if cache is not None and cache[0] == cache_key:
            return cache[1]
        if self.subspace == "rope":
            out = torch.arange(0, 64, device=device)
        elif self.subspace == "nope":
            # GLM-4.7 的 256 维布局为 [NoPE192 | RoPE64]。只取 RoPE
            # 低频 tail 会丢掉全部 NoPE 内容特征，检索选择严重退化；
            # 使用 NoPE 前 64 维作为 L1/L2 共用内容子空间。D=128 的
            # [RoPE64 | NoPE64] 布局保持原映射不变。
            out = (torch.arange(0, 64, device=device) if self.head_dim == 256
                   else torch.arange(64, 128, device=device))
        elif self.subspace == "mla_rope":
            # 【E142a】DeepSeek-V2 MLA：k = [nope 128 | rope 64] 192 维，
            # rope 分量在尾段（与 Qwen3 rotate_half 两半布局不同）
            out = torch.arange(128, 192, device=device)
        elif self.subspace == "random":
            # E90 消融臂（对齐 E3 mass 口径）：固定 seed 随机 32 维
            g = torch.Generator().manual_seed(20261002)
            out = torch.randperm(128, generator=g)[:32].sort().values.to(device)
        elif self.subspace == "highfreq":
            # E90 消融臂：头 16+头 16（RoPE 旋转对前元素，最高频对）
            delta = 64 // self.cmp_ratio
            out = torch.tensor(
                list(range(0, delta)) + list(range(64, 64 + delta)), device=device)
        else:
            out = self._tail_indices(device)
        self._idx_sub_cache = (cache_key, out)
        return out

    # 声明支持外部 valid_length 覆盖（base.prepare_mask 据此决定是否传
    # kwarg；其他 indexer 不支持时不传 → None 分支逐位不变）。
    supports_valid_length = True

    def prepare_index(self, k, cu_seqlens_k, valid_length=None):
        """【E127 inplace 合并（用户 2026-10-12 指令）】decode 热路径增量缓存
        入口（原 TLIIndexerFast.prepare_index 语义原样并入，env 门退役——
        fast 成为唯一路径）：k append-only 且前置条件满足时尾段增量重算，
        否则全量。逐位一致性为数学保证（块内独立归约 + quant 逐 token 独立）。
        契约（fail-closed）：
          - k append-only（HF DynamicCache decode 语义）；S_now ≤ S_prev
            或任一前置不符 → 全量重建缓存。
          - clear() 使 _inc 失效 → 新请求首步全量。
          - 投影基/moba/异形 D≠128 一律全量（值同原版路径，逐位不变）。
        【阶段 2a 固定宽度】valid_length：固定全宽 KV 下 k.shape[1]=S_cap
        （容量）≠ 真实有效长。外部显式传入（batch decode 每行 L_b）时覆盖
        返回 dict 的 "valid_length"（_prepare_index_base/_inc 内部设 k.shape[1]，
        该 key 只被 compute_score/_gather_fine_cand 消费 → 入口覆盖语义安全）。
        valid_length=None（可变宽/单序列旧入口）行为逐位不变。
        """
        # This implementation owns exactly one request. Validate before the
        # incremental dispatch: that path indexes k[0] and would otherwise
        # silently discard every additional request once a B=1 cache exists.
        # Use explicit exceptions because Python -O removes assertions.
        if k.shape[0] != 1:
            raise ValueError(
                f"TLIIndexer prepare_index requires batch=1, got batch={k.shape[0]}")
        expected_dim = getattr(
            self, "head_dim", 192 if self.subspace == "mla_rope" else 128)
        if k.shape[-1] != expected_dim:
            raise ValueError(
                f"tli_subspace={self.subspace} 需要的 k 末维 = {expected_dim}，"
                f"实际 {k.shape[-1]}")
        if expected_dim == 256 and (
                self.subspace != "nope" or self.l2_subspace != "same"
                or self.cmp_ratio != 4 or self.l2_quant
                or self.static_pair or self._basis is not None
                or self._basis_all is not None):
            raise ValueError(
                "D=256 TLI 仅支持 nope + l2_subspace=same + cmp_ratio=4 "
                "+ l2_quant=False，且禁用 static_pair/投影基；"
                "RoPE-only tail32 已证实丢失 NoPE 检索内容，拒绝执行")
        # 【E152v4】graph 增量优先：cuda_graph_state 模式（decode 整步
        # capture/replay）下走 pos_buf 驱动的容量宽持久 buffer 增量更新
        # （_prepare_index_graph；每步仅重算 pos 所在块统计 + k_qat[pos]
        # 行，替代 patch 旧口径下每步每层的 _prepare_index_base 容量宽
        # 全量重建 ~15ms/步 index 税）。不兼容配置返回 None → 回退下方
        # 原路径，行为逐位不变（fail-closed）。_ginc 生命周期独立于 _inc
        # （clear() 同时失效两者 = 请求边界全量重建；graph 步进不再每步
        # clear —— 见 patch _tl_decode_batched_graph）。
        if (_graph_state_fn is not None
                and os.environ.get("TLI_GRAPH_INC", "1") != "0"):
            _gs = _graph_state_fn()
            if _gs.enabled:
                _gd = self._prepare_index_graph(k, cu_seqlens_k)
                if _gd is not None:
                    return _gd
        inc = getattr(self, "_inc", None)
        S_now = k.shape[1]
        if (
            inc is not None
            and S_now > inc["S"]
            # Reusing old buffers across metadata changes can broadcast a
            # single KV head or cast the tail into the previous dtype/device.
            # Shape/device/dtype are host metadata: no device synchronization.
            and inc.get("input_signature") == (
                k.device, k.dtype, k.shape[2], k.shape[3], self.args.tia_block_size)
            and self._basis is None
            and self._basis_all is None
            and not self.moba_gate
            and k.shape[-1] in (128, 256)
            and inc["sub_key"] == self._sub_key()
            and (not self._need_avg_score() or inc["k_avg"] is not None)
        ):
            d = self._prepare_index_inc(k, cu_seqlens_k, inc)
        else:
            d = self._prepare_index_full(k, cu_seqlens_k,
                                         valid_length=valid_length)
        if valid_length is not None:
            d["valid_length"] = valid_length
        return d

    def _sub_key(self):
        """子空间一致性 key：跨步 idx_sub 不变才可增量（static_pair 的
        _pair_idx 在 prefill 后冻结；rope/nope/tail 为常量分支）。"""
        if not self.enable_subspace:
            return ("off", 0, self.head_dim)
        if self.static_pair and self._pair_idx is not None:
            return ("pair", int(self._pair_idx.numel()), self.head_dim)
        return (self.subspace, self.cmp_ratio, self.head_dim)

    def _prepare_index_full(self, k, cu_seqlens_k, valid_length=None):
        d = self._prepare_index_base(k, cu_seqlens_k, valid_length=valid_length)
        # 建缓存（k_qat 仅非投影路径可作增量 buffer——投影路径 k_qat 为
        # 全维量化特征，散布语义不同；_basis 非 None 时 can_inc 已挡，
        # 此处再防御一次）
        k_qat_buf = None
        if self._basis is None and not self.moba_gate and k.shape[-1] in (128, 256):
            k_qat_buf = d["k_qat"].squeeze(0).clone() if d["k_qat"].dim() == 4 \
                else None
        # 【E133c-A1】k_min/k_max/k_avg 预分配块统计 buffer（与 k_qat_buf
        # 同模式）：inc 路径尾段直写 [fb, T_blk)，消除每步 3 次 cat 的
        # 全量 [1,T,Hkv,d'] 重分配 + 前缀复制。cat 产物 ≡ buffer 直写
        # （前缀值归纳不变、尾段同源重算）→ 逐位一致。k_qat_buf None ⇒
        # inc 路径不可达（prepare_index 前置门），统计 buffer 同门创建。
        k_min_buf = k_max_buf = k_avg_buf = None
        if k_qat_buf is not None:
            k_min_buf = d["k_min"].clone()
            k_max_buf = d["k_max"].clone()
            k_avg_buf = d["k_avg"].clone() if d.get("k_avg") is not None else None
        self._inc = {
            "S": k.shape[1],
            "input_signature": (
                k.device, k.dtype, k.shape[2], k.shape[3], self.args.tia_block_size),
            "k_min": d["k_min"],
            "k_max": d["k_max"],
            "k_avg": d.get("k_avg"),
            "k_min_buf": k_min_buf,
            "k_max_buf": k_max_buf,
            "k_avg_buf": k_avg_buf,
            "stat_blk_cap": k_min_buf.shape[1] if k_min_buf is not None else 0,
            "k_qat_buf": k_qat_buf,
            "k_qat_cap": k.shape[1] if k_qat_buf is not None else 0,
            "sub_key": self._sub_key(),
        }
        return d

    def _prepare_index_graph(self, k, cu_seqlens_k):
        """【E152v4】graph 专用增量 prepare（pos_buf 驱动，capture 安全）。

        背景：graph 模式下 patch 每步每层 clear()+prepare_index →
        _prepare_index_base 对整条 S_cap 容量全量重建（index_put_/
        nonzero/amin/amax/mean/gather 全量 kernel，with_stack 归因
        ~15ms/步 36 层 index 税）。decode 稳态每步只写 1 个新 token
        （pos_buf 指示列），前缀 token 内容跨步不变（静态 cache 追加写）
        + pad 区恒零 → 容量宽块统计/k_qat 可持久化，每步只增量更新 pos
        所在块 + k_qat[pos] 行（~7 个固定形状小 kernel/层）。

        等价性（与每步全量重建逐位一致）：
          - 首调（warmup，capture 外）全量建 buffer：_prepare_index_base
            产物原样接管（valid_length=None 容量宽口径同款）；
          - 增量步：块统计块独立归约（E127 归纳）→ 单块重算 ≡ 全量对应
            块；k_qat 逐 token 独立（切片/quant 均行独立）→ 单行直写 ≡
            全量对应行；末块不满时全量路径 pad 零，这里 clamp+where(+0)
            复现（where 显式给 +0，规避乘零的 -0 陷阱）；
          - 均为纯固定形状张量 op（index_select/reduce/index_copy_），
            无 host 读回 → capture/replay 合法。

        生命周期：clear() 失效 _ginc（prefill 前每层 clear = 请求边界
        全量重建）；跨 repeat 同存储 refills 时 e152v4 协议保证逐位相同
        prompt（input_ids sha 落盘）→ 前缀恒有效。生产 pred_ruler 每
        请求先 prefill（clear）→ 首步 decode 全量重建，安全。

        前置门（host 侧，仅 capture 外评估；不满足返回 None → 调用方
        回退全量路径，fail-closed）：graph_state.enabled 且 pos_buf 就
        绪、B=1 容量宽、非投影/非 moba/D∈{128,256}、非 kmeans 消费臂。
        """
        if _graph_state_fn is None:
            return None
        gs = _graph_state_fn()
        if not gs.enabled or gs.pos_buf is None:
            return None
        if (self._basis is not None or self._basis_all is not None
                or self.moba_gate or k.dim() != 4 or k.shape[0] != 1
                or not k.is_cuda or k.shape[-1] not in (128, 256)):
            return None
        if (self.enable_kmeans
                and (self.far_select != "4bit" or self.near_select != "4bit")):
            return None
        bs = self.args.tia_block_size
        if (bs & (bs - 1)) != 0:
            return None
        S_cap = k.shape[1]
        kt_cap = (S_cap + bs - 1) // bs
        need_avg = self._need_avg_score()
        dev = k.device
        _g = getattr(self, "_ginc", None)
        stale = _g is not None and _g.get("stale", False)
        if (_g is None
                or stale
                or _g["S_cap"] != S_cap
                or _g["kt_cap"] != kt_cap
                or _g["bs"] != bs
                or _g["data_ptr"] != k.data_ptr()
                or _g["dtype"] != k.dtype
                or _g["sub_key"] != self._sub_key()
                or _g["need_avg"] != need_avg):
            # 首调/缓存失效：全量建 buffer。必须发生在 capture 外
            #（warmup 必先跑）；capture 内触发 = 状态异常 → 拒绝增量。
            if torch.cuda.is_current_stream_capturing():
                return None
            self._resolve_skip()
            d = self._prepare_index_base(k, cu_seqlens_k)
            k_qat = d["k_qat"]
            # l2_subspace=full 且 l2_quant=False 时 base 返回 k 本体
            #（视图）——持久 buffer 会被下方 index_copy_ 原地改写 →
            # 污染静态 cache，必须物化拷贝
            if k_qat is k:
                k_qat = k_qat.clone()
            if (k_qat.dim() != 4 or k_qat.shape[1] != S_cap
                    or k_qat.data_ptr() == k.data_ptr()):
                return None
            if need_avg and d.get("k_avg") is None:
                return None
            # 【E152v4-修复·根因】同形状 stale 旧 buffer 存在（图 replay
            # 仍 pin 其内存）→ in-place 全量刷新代替重新分配：旧图 kernel
            # 读写同一地址，跨请求复用图安全；若改为新分配，旧 buffer 被
            # allocator 回收复用 → replay 静默读写被覆写内存。
            if (stale
                    and _g["S_cap"] == S_cap
                    and _g["kt_cap"] == kt_cap
                    and _g["bs"] == bs
                    and _g["data_ptr"] == k.data_ptr()
                    and _g["dtype"] == k.dtype
                    and _g["sub_key"] == self._sub_key()
                    and _g["need_avg"] == need_avg
                    and k_qat.shape == _g["k_qat"].shape
                    and d["k_min"].shape == _g["k_min"].shape):
                _g["k_min"].copy_(d["k_min"])
                _g["k_max"].copy_(d["k_max"])
                if need_avg:
                    _g["k_avg"].copy_(d["k_avg"])
                _g["k_qat"].copy_(k_qat)
                _g.pop("stale", None)
                return {
                    "k_min": _g["k_min"],
                    "k_max": _g["k_max"],
                    "k_avg": _g["k_avg"],
                    "k_avg_full": None,
                    "k_qat": _g["k_qat"],
                    "cu_seqlens_k_coarse": _g["cu_coarse"],
                    "cu_seqlens_k_fine": cu_seqlens_k,
                    "valid_length": S_cap,
                }
            self._ginc = {
                "S_cap": S_cap, "kt_cap": kt_cap, "bs": bs,
                "data_ptr": k.data_ptr(), "dtype": k.dtype,
                "sub_key": self._sub_key(), "need_avg": need_avg,
                "k_min": d["k_min"], "k_max": d["k_max"],
                "k_avg": d.get("k_avg"),
                "k_qat": k_qat,
                "cu_coarse": d["cu_seqlens_k_coarse"],
                "ar_bs": torch.arange(bs, device=dev),
                "zero_k": k.new_zeros(()),
            }
            return d
        # ---- 增量步（固定形状纯张量 op，capture/replay 安全）----
        pos = gs.pos_buf                       # [1] long（本步写列）
        if pos.dtype != torch.long or pos.device != dev:
            return None
        # 【E166 iter3d】摘要维护链融合：原 ~15 op/层（fb//、idx_blk 乘加、
        # inb 比较、clamp、index_select、where 零垫、子空间 gather、
        # amin/amax/mean 三归约、3×index_copy_、row index_select、row
        # 子采样、k_qat index_copy_）→ 单 Triton kernel（块 tile 直载 +
        # 序无关 min/max + fp32 累加 mean + 行直写）。值域=拷贝+序无关
        # 归约+常量垫零；mean 归约序逐位一致性由单测+e2e bitexact 裁决。
        idx_sub = (self._subspace_indices(dev)
                   if self.enable_subspace else None)
        _m3d = False
        if (_HAS_TRITON and self._triton_ok
                and idx_sub is not None
                and self.l2_subspace != "full"
                and not self.l2_quant
                and k.dim() == 4 and k.shape[0] == 1 and k.is_cuda
                and k.dtype in (torch.bfloat16, torch.float16)
                and k.stride(-1) == 1
                and (idx_sub.numel() & (idx_sub.numel() - 1)) == 0
                and _g["k_qat"].shape[-1] == idx_sub.numel()
                and _g["k_min"][0].is_contiguous()
                and _g["k_max"][0].is_contiguous()
                and (not need_avg
                     or (_g["k_avg"] is not None
                         and _g["k_avg"][0].is_contiguous()))
                # k/k_qat 为 head-major 置换视图（k[...,idx_sub] 对非连续
                # k 返回非连续结果）——kernel 按实际 stride 寻址，布局无关，
                # 只需末维 stride=1
                and _g["k_qat"][0].stride(2) == 1
                and k[0].stride(2) == 1):
            try:
                _m3d = _iter3_maint(
                    k[0], pos, idx_sub,
                    _g["k_min"][0], _g["k_max"][0],
                    _g["k_avg"][0] if need_avg else None,
                    _g["k_qat"][0], S_cap, bs, need_avg)
            except Exception:
                _m3d = False
        if _m3d:
            return {
                "k_min": _g["k_min"],
                "k_max": _g["k_max"],
                "k_avg": _g["k_avg"],
                "k_avg_full": None,
                "k_qat": _g["k_qat"],
                "cu_seqlens_k_coarse": _g["cu_coarse"],
                "cu_seqlens_k_fine": cu_seqlens_k,
                "valid_length": S_cap,
            }
        fb = pos // bs                         # [1] 新 token 所在块
        idx_blk = fb * bs + _g["ar_bs"]        # [bs] 该块 token 下标
        inb = idx_blk < S_cap                  # 末块越界门（pad 零语义）
        blk = k[0].index_select(0, idx_blk.clamp(max=S_cap - 1))
        blk = torch.where(inb.view(bs, 1, 1), blk, _g["zero_k"])
        blk_sub = blk[..., idx_sub] if idx_sub is not None else blk
        # 归约形态对齐全量表达式（[1,1,bs,Hkv,d] mean(dim=2)/amin(dim=2)，
        # 同元素序同 stride → 逐位一致）。blk_sub 恒 3D [bs,Hkv,d']（末维
        # gather/直用均不升维）
        t5 = blk_sub.view(1, 1, bs, blk_sub.shape[1], blk_sub.shape[2])
        _g["k_min"][0].index_copy_(0, fb, t5.amin(dim=2)[0, 0].unsqueeze(0))
        _g["k_max"][0].index_copy_(0, fb, t5.amax(dim=2)[0, 0].unsqueeze(0))
        if need_avg:
            _g["k_avg"][0].index_copy_(0, fb, t5.mean(dim=2)[0, 0].unsqueeze(0))
        # k_qat 行：逐 token 独立 → 单行 gather + 直写
        row = k[0].index_select(0, pos)         # [1,Hkv,D]
        if self.l2_subspace == "full":
            seg = self.min_max_per_token_quant(row) if self.l2_quant else row
        else:
            indices = idx_sub if idx_sub is not None else self._qat_indices(dev)
            seg = row[..., indices]
            if self.l2_quant:
                seg = self.min_max_per_token_quant(seg)
        _g["k_qat"][0].index_copy_(0, pos, seg)
        return {
            "k_min": _g["k_min"],
            "k_max": _g["k_max"],
            "k_avg": _g["k_avg"],
            "k_avg_full": None,
            "k_qat": _g["k_qat"],
            "cu_seqlens_k_coarse": _g["cu_coarse"],
            "cu_seqlens_k_fine": cu_seqlens_k,
            "valid_length": S_cap,
        }

    def _prepare_index_inc(self, k, cu_seqlens_k, inc):
        """【E127 iter4】增量缓存：复用块 [0, fb) + 尾段重算（pad 语义
        与全量版一致）。b_k_qat 物化缓存 + pad_cu 向量化 + quant indices
        缓存（原每步 H2D 构造）均为逐位等价替换。"""
        self._resolve_skip()   # 与原版头部同构（幂等；影响 _maybe_build_kmeans 条件）
        bs = self.args.tia_block_size
        S_prev, S_now = inc["S"], k.shape[1]
        # Append-only input: complete prefix pages cannot change. At an exact
        # page boundary start at the new page, not the previous complete page.
        fb = S_prev // bs              # partial previous page, or first new page
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
        # ---- 【E133c-A1/A4】尾段块统计：预分配 buffer 直写 + 融合 kernel ----
        # 尾段块数与 pad：与全量版一致 = ceil(S_now/bs)*bs 覆盖下的尾块集
        # （lo 为块边界，前缀块已对齐——尾段 pad 量 = 全量 pad 量）
        pad_total = (bs - S_now % bs) % bs
        n_tok = S_now - lo                              # 尾段实际 token 数
        n_blk = (n_tok + pad_total) // bs                # 尾段块数（含 pad 块）
        T_blk = fb + n_blk
        need_avg = self._need_avg_score()
        k_min_buf, k_max_buf, k_avg_buf = (
            inc["k_min_buf"], inc["k_max_buf"], inc["k_avg_buf"])
        if k_min_buf is not None and T_blk > inc["stat_blk_cap"]:
            # 扩容（翻倍摊销；只搬前缀 [0, fb)——尾段本步全量重写）
            new_cap = max(2 * inc["stat_blk_cap"], T_blk)
            # Only [:T_blk] is exposed: prefix copy plus tail writes initialize
            # every live row. Spare capacity is never read, so do not zero it.
            _nb = k_min_buf.new_empty((1, new_cap) + k_min_buf.shape[2:])
            _nb[:, :fb] = k_min_buf[:, :fb]
            k_min_buf = _nb
            _nb = k_max_buf.new_empty((1, new_cap) + k_max_buf.shape[2:])
            _nb[:, :fb] = k_max_buf[:, :fb]
            k_max_buf = _nb
            if k_avg_buf is not None:
                _nb = k_avg_buf.new_empty((1, new_cap) + k_avg_buf.shape[2:])
                _nb[:, :fb] = k_avg_buf[:, :fb]
                k_avg_buf = _nb
            inc["stat_blk_cap"] = new_cap
        stat_fused = False
        if (k_min_buf is not None and _HAS_TRITON and self._triton_ok
                and self._stat_ok and k.dim() == 4 and k.shape[0] == 1
                and k.is_cuda and k.stride(-1) == 1
                and (bs & (bs - 1)) == 0
                and (idx_sub is None or (idx_sub.numel() & (idx_sub.numel() - 1)) == 0)):
            # 【E133c-A4】尾段统计融合：tail_pad 构造（new_zeros+copy）+
            # gather 物化 + amin/amax/mean + 3 次直写 ≈ 9 launch → 1
            # kernel（pad 语义用 mask other=0 复现）。数值实证（50 轮
            # 随机数据全过）：mean 升序 fp32 顺序累加 /64 → RNE bf16 与
            # PyTorch 逐位一致；min/max 为选择运算无舍入。异常由 wrapper
            # 置 _stat_ok=False 后回退下方 PyTorch 路径（fail-closed）。
            try:
                self._fast_stat_tail(k, idx_sub, lo, n_tok, n_blk, fb,
                                     k_min_buf, k_max_buf,
                                     k_avg_buf if need_avg else None)
                stat_fused = True
            except Exception:
                pass
        if stat_fused:
            k_min = k_min_buf[:, :T_blk]
            k_max = k_max_buf[:, :T_blk]
            k_avg = k_avg_buf[:, :T_blk] if need_avg else None
        elif k_min_buf is not None:
            # 【E133c-A1】buffer 直写版（原 cat → 直写 [fb, T_blk)，消除
            # 3 次全量 [1,T,Hkv,d'] 重分配+前缀复制；值逐位同 cat 产物）
            tail_k = k[:, lo:S_now]                      # [1, n, Hkv, D]
            tail_pad = tail_k.new_zeros((1, n_tok + pad_total) + tail_k.shape[2:])
            tail_pad[:, :n_tok] = tail_k
            t_coarse = rearrange(
                tail_pad[..., idx_sub] if idx_sub is not None else tail_pad,
                "b (t bs) h d -> b t bs h d", bs=bs,
            )
            k_min_buf[:, fb:T_blk] = t_coarse.amin(dim=2)
            k_max_buf[:, fb:T_blk] = t_coarse.amax(dim=2)
            if need_avg:
                k_avg_buf[:, fb:T_blk] = t_coarse.mean(dim=2)
            k_min = k_min_buf[:, :T_blk]
            k_max = k_max_buf[:, :T_blk]
            k_avg = k_avg_buf[:, :T_blk] if need_avg else None
        else:
            # 防御保底（k_qat_buf None ⇒ inc 路径不可达，理论不触发）：原 cat 语义
            tail_k = k[:, lo:S_now]                      # [1, n, Hkv, D]
            tail_pad = tail_k.new_zeros((1, n_tok + pad_total) + tail_k.shape[2:])
            tail_pad[:, :n_tok] = tail_k
            t_coarse = rearrange(
                tail_pad[..., idx_sub] if idx_sub is not None else tail_pad,
                "b (t bs) h d -> b t bs h d", bs=bs,
            )
            k_min = torch.cat([inc["k_min"][:, :fb], t_coarse.amin(dim=2)], dim=1)
            k_max = torch.cat([inc["k_max"][:, :fb], t_coarse.amax(dim=2)], dim=1)
            k_avg = None
            if need_avg:
                k_avg = torch.cat([inc["k_avg"][:, :fb], t_coarse.mean(dim=2)], dim=1)

        # ---- k_qat 增量：预分配 buffer 步进写（E133b 原生 [cap,Hkv,delta]
        # 紧凑布局直写——量化覆盖全部 delta 维，与全量版紧凑 k_qat 同构；
        # E134 l2_subspace=full 时 buffer [cap,Hkv,128] 全维量化整段直写）----
        indices = idx_sub if idx_sub is not None else self._qat_indices(k.device)
        buf, cap = inc["k_qat_buf"], inc["k_qat_cap"]
        if S_now > cap:   # 扩容（翻倍摊销）
            new_cap = max(2 * cap, S_now)
            # [:S_prev] and [lo:S_now] cover all exposed tokens (lo <= S_prev).
            # Future tokens are initialized before their views are returned.
            new_buf = buf.new_empty((new_cap,) + buf.shape[1:])
            new_buf[:S_prev] = buf[:S_prev]
            inc["k_qat_buf"], inc["k_qat_cap"] = new_buf, new_cap
            buf = new_buf
        if self.l2_subspace == "full":
            # 【E155b 去量化（新口径默认）】l2_quant=False 时细筛直接用原始
            # 切片（假量化无压缩收益，仅数值罚 + launch 开销），true 回旧口径
            seg = k[0, lo:S_now]
            if self.l2_quant:
                seg = self.min_max_per_token_quant(seg)     # 全 128 维量化直写
            buf[lo:S_now] = seg
        else:
            # 【E133c-A5】量化链融合：gather 物化 + minmax 4bit 量化（~10 个
            # elementwise/reduction 小 launch）+ buf 直写 ≈ 12 launch → 1
            # kernel 直写 buf[lo:S_now]。逐算子仿真（每算子 fp32 计算 + RNE
            # bf16，PyTorch CUDA opmath=fp32 口径实测逐位一致）；torch.round
            # 的 round-half-to-even 用 libdevice.rint 复现。异常由 wrapper 置
            # _quant_ok=False 后回退原版（fail-closed）。
            # 【E155b】l2_quant=False（新口径默认）时整段旁路——gather 直拷
            # 进 buf（量化 kernel 链不发射）。
            quant_fused = False
            if (self.l2_quant
                    and _HAS_TRITON and self._triton_ok and self._quant_ok
                    and k.dim() == 4 and k.shape[0] == 1 and k.is_cuda
                    and k.stride(-1) == 1 and indices.dtype == torch.int64
                    and (indices.numel() & (indices.numel() - 1)) == 0):
                try:
                    self._fast_qat_tail(k, indices, lo, n_tok, buf)
                    quant_fused = True
                except Exception:
                    pass
            if not quant_fused:
                seg = k[0, lo:S_now, :, indices]
                if self.l2_quant:
                    seg = self.min_max_per_token_quant(seg)
                buf[lo:S_now] = seg
        k_qat = buf[:S_now].unsqueeze(0)                     # [1, S, Hkv, delta] 视图（l2=full 时 128 宽）

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

        # 缓存滚动（视图归位；统计/k_qat buffer 原地复用——扩容时归位新 buf）
        inc["S"] = S_now
        inc["k_min"], inc["k_max"], inc["k_avg"] = k_min, k_max, k_avg
        inc["k_min_buf"], inc["k_max_buf"], inc["k_avg_buf"] = (
            k_min_buf, k_max_buf, k_avg_buf)
        return index_dict

    def _qat_indices(self, device):
        """【iter5】非子空间 quant indices 缓存（原每步 list→tensor 的
        pageable H2D 构造；cmp_ratio 为实例级常量 → 命中后零构造零拷贝）。
        GPU 端 arange 构造（纯 kernel 零 H2D，int64 与原 torch.tensor(list)
        同 dtype，值逐位同）。"""
        key = (self.head_dim, self.cmp_ratio, device)
        cache = getattr(self, "_qat_idx_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        idx = self._tail_indices(device)
        self._qat_idx_cache = (key, idx)
        return idx

    def _prepare_index_base(self, k, cu_seqlens_k, valid_length=None):
        from .utils import prepare_pad_mask, pad_tensor
        self._resolve_skip()
        idx_sub = self._subspace_indices(k.device) if self.enable_subspace else None
        # 【阶段 2a 固定宽度】块统计有效区：固定全宽 KV（k.shape[1]=S_cap
        # 容量 > valid_length 有效长）下 pad 列进入块统计 → k_min/k_max/k_avg
        # 与变长路径（无 pad 列）不一致 → score_coarse 漂移 → L1 选择/mask
        # 漂移（G5 aavg_e64 normal 反例）。非投影分支的块统计从有效区切片算
        # （宽 ceil(vl/bs)），下游 kt=score_coarse 宽随之有效化 → 近池/
        # swa_lo_blk 边界与变长一致（G6 bypass 断言反例同源消解）。k_qat
        # 保持全宽（非候选化路径 score_fine 需全宽 S_cap；pad 列由
        # compute_mask 显式置 -inf）。投影分支（_basis 非 None，消融臂）
        # 保持全宽旧行为逐位不变。
        k_stat = k
        cu_stat = cu_seqlens_k
        if (valid_length is not None and valid_length < k.shape[1]
                and self._basis is None):
            k_stat = k[:, :valid_length]
            cu_stat = torch.tensor([0, valid_length],
                                   dtype=cu_seqlens_k.dtype,
                                   device=cu_seqlens_k.device)
        # 【阶段 2c-5】CUDA Graph capture 纯张量路径：prepare_pad_mask 的
        # 逐元素迭代与 pad_tensor 的 bool 掩码散射都需要 host 侧元素个数
        # 同步（capture 期非法）。单行（k_stat.shape[0]==1，本函数断言）且
        # pad_mask 恒为前缀 True 布局 → 静态 total + 切片写入，值逐位相同。
        if k.is_cuda and torch.cuda.is_current_stream_capturing():
            _pbs = self.args.tia_block_size
            _total = (k_stat.shape[1] + _pbs - 1) // _pbs * _pbs
            pad_mask, pad_cu_seqlens_k = prepare_pad_mask(_pbs, cu_stat,
                                                          total=_total)
            pad_k = k_stat.new_zeros((k_stat.shape[0], _total)
                                     + tuple(k_stat.shape[2:]))
            pad_k[:, :k_stat.shape[1]] = k_stat
        else:
            pad_mask, pad_cu_seqlens_k = prepare_pad_mask(
                self.args.tia_block_size, cu_stat)
            pad_k = pad_tensor(k_stat, pad_mask)
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
            # 【E176】全量块统计融合（投影路径）：kp 投影后特征，
            # amin+amax+mean 一次读数据。
            _proj_need_avg = self._need_avg_score()
            _proj_use_fused = (_HAS_TRITON and self._triton_ok
                               and self._stat_ok and kp.is_cuda
                               and kp.stride(-1) == 1
                               and (self.args.tia_block_size & (self.args.tia_block_size - 1)) == 0)
            if _proj_use_fused:
                try:
                    k_min, k_max, k_avg_proj = self._fast_stat_full(
                        kp, None, _proj_need_avg)
                except Exception:
                    if not self._stat_ok:
                        _proj_use_fused = False
                    else:
                        raise
            if not _proj_use_fused:
                k_coarse = rearrange(kp,
                    "b (t bs) h d -> b t bs h d", bs=self.args.tia_block_size)
                k_min = k_coarse.amin(dim=2)
                k_max = k_coarse.amax(dim=2)
                k_avg_proj = k_coarse.mean(dim=2) if _proj_need_avg else None
            cu_seqlens_k_coarse = pad_cu_seqlens_k // self.args.tia_block_size
            # L2 4bit 精筛：投影特征逐 token 量化（任意 last-dim 通用）
            # 【E155b 去量化】l2_quant=False 直接用原始投影特征（同 dtype 视图语义）
            k_qat = self.min_max_per_token_quant(kp) if self.l2_quant else kp
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
            if _proj_need_avg:
                # near/far 粗筛 avg 分数（E64f：near=avg / far=minmax；far=avg 单池也需）
                # 【E176】k_avg_proj 已由 _fast_stat_full 或 fallback 产出
                index_dict["k_avg"] = k_avg_proj
            # ---- B：远端/近端聚类（cluster/sim_greedy 消融模式构建；默认 4bit 分区无需聚类）----
            if (self.enable_kmeans and not self.skip_far
                    and (self.far_select != "4bit" or self.near_select != "4bit")):
                self._maybe_build_kmeans(k, index_dict)
            return index_dict
        # 【E176】全量块统计融合：amin + amax + mean 一次读数据
        # （替代 rearrange + amin + amax + mean 四次读）。Triton 异常
        # 时 _fast_stat_full 置 _stat_ok=False 并 raise，except 走原版链。
        need_avg = self._need_avg_score()
        _use_fused = (_HAS_TRITON and self._triton_ok and self._stat_ok
                      and k_for_coarse.is_cuda
                      and (self.args.tia_block_size & (self.args.tia_block_size - 1)) == 0
                      and k_for_coarse.stride(-1) == 1)
        if _use_fused:
            try:
                # 注意：k_for_coarse 在此处已是子空间切片后的 tensor
                # （L1169: pad_k[..., idx_sub] if idx_sub is not None），
                # 不能再传 idx_sub 给 kernel 做二次 gather（越界）。
                # _fast_stat_full 内部会根据 stride(-1) 与 shape(-1) 判断是否需 sub。
                k_min, k_max, k_avg = self._fast_stat_full(
                    k_for_coarse, None, need_avg)
            except Exception:
                if not self._stat_ok:
                    _use_fused = False  # 永久回退原版链
                else:
                    raise
        if not _use_fused:
            k_coarse = rearrange(k_for_coarse,
                "b (t bs) h d -> b t bs h d", bs=self.args.tia_block_size)
            k_min = k_coarse.amin(dim=2)
            k_max = k_coarse.amax(dim=2)
            k_avg = k_coarse.mean(dim=2) if need_avg else None
        # E89 MoBA 臂：全维 chunk-mean（gate 分数用完整 128 维 pad_k，严格 MoBA 口径）
        k_avg_full = None
        if self.moba_gate:
            kc_full = rearrange(
                pad_k, "b (t bs) h d -> b t bs h d", bs=self.args.tia_block_size
            )
            k_avg_full = kc_full.mean(dim=2)
        cu_seqlens_k_coarse = pad_cu_seqlens_k // self.args.tia_block_size

        # 【E142a】维度门：Qwen3 GQA 128 维 / DeepSeek MLA mla_rope 臂 192 维
        # （nope 128 + rope 64；选择子空间 mla_rope 只取尾 64 维）。
        # 其他模式（full/tail/rope/nope/random/highfreq）保持 128 硬门不变。
        # E85f 口径修正：L2 细筛 k_qat 与 L1 粗筛用同一子空间（idx_sub）——
        # E85e 重放口径是粗筛+细筛同特征；tail 臂 idx_sub 与旧硬编码 tail32
        # 逐位相同（cmp_ratio=2），仅 rope/nope 探索臂与静态 pair 臂行为改变
        # 【E133b 原生化】k_qat 直接产出 [1,S,Hkv,delta] 紧凑布局（原为
        # zeros_like(k) 128 维零填充 dense + 尾段 scatter——96 个零维参与
        # fine 打分读带宽 4× 浪费；零维乘积恒为 0，紧凑布局数学等价，仅
        # fp32 累加顺序变化 → bf16 舍入后 ULP 级漂移，E127 iter6 同口径）。
        # full 模式 L2 同样用 tail-delta 尾段（cmp_ratio 口径，idx_sub=None
        # 时 _qat_indices 缓存路径），full 锚点 fine 段同样受益。
        # 【E134】l2_subspace=full 时 L2 细筛全 128 维量化（L1 仍由 idx_sub
        # 决定）——sub+full 臂消费端，score_fine 宽度 = 全维（128 宽 dense，
        # 无紧凑化问题）。
        if self.l2_subspace == "full":
            # 【E155b 去量化】l2_quant=False 直接用原始 k（无压缩收益的假量化旁路）
            k_qat = self.min_max_per_token_quant(k) if self.l2_quant else k   # [1, S, Hkv, 128]
        else:
            indices = idx_sub if idx_sub is not None else self._qat_indices(k.device)
            # 【E155b 去量化】同上——原始子空间切片直用（gather 拷贝保持紧凑布局）
            k_qat = (self.min_max_per_token_quant(k[..., indices]) if self.l2_quant
                     else k[..., indices])
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
        # far_lo 对齐 sink 池左界（E135：floor(sink_tokens/bs) 块对齐，默认 2 块逐位不变）
        far_lo = self._sink_pool_lo_blk * self.args.tia_block_size
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
            sink_tok = self.sink_tokens          # E135：强制区宽单一事实源
            mid_len = max(0, S - sink_tok - swa_tok)
            if self.nt_near_len is not None:
                # E155d：near 区宽显式整数（α·mid_len 规则被覆盖，与
                # compute_mask 同源）——几何派单下 α 已填充 1.0，不覆盖
                # 会把簇覆盖范围错扩到全 mid。
                near_len_dyn = max(bs, int(self.nt_near_len))
            else:
                near_len_dyn = max(bs, int(self.alpha * mid_len))
            near_base = S - swa_tok
        elif self.far_select != "4bit" and self.alpha == 0 and self.beta == 0:
            # TASK.md L231 严格口径：cluster 组合 (0,0) 单池点 = 纯 cluster 全 mid
            # （near 区空；swa 仍正交强制，计入 K2 预算）
            near_len_dyn = swa_tok
            near_base = S
        else:
            near_len_dyn = self.near_len
            near_base = S
        # E135：sink 池左界块数 +1 保底（默认 floor(128/64)+1 = 3 ≡ 历史 sink_blocks+1）
        far_hi_blk = max(self._sink_pool_lo_blk + 1, (near_base - near_len_dyn) // bs)
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

        【E131-S2（原 fast.py override 并入）】默认先走 v2 SPLIT 协作 kernel
        （greedy_triton.greedy_pass_triton_v2：grid=(H,SPLIT) 同头多 program
        均分活簇扫描 + 两阶段软件 barrier，n_regs 死锁防御钳制 SPLIT）。E131
        阶段 1 实测 v1 grid=(H,) 仅 H=8 program、78 SM 闲置 90%——v2 far
        冷建 8077→958ms（8.4×），大 k_live 续跑 9.5×（H20 randn 最坏口径）。
        对拍 exp/bench/e131s2_v2_validate.py V1-V5 全过（assign/k_live 逐位
        + 状态 allclose）。开关 TLI_E131_GREEDY_V2=0 回退 v1；v2 编译/执行
        异常 fail-closed 回退本调度器（值同锚点路径，仅性能回退）。
        【e113b T3/T5 存量失败根因修复 2026-10-20】kill-switch
        SGLANG_TLI_GREEDY_KERNEL=0 必须**先于 v2/v1 kernel 分支判定**——
        原 v2 分支判于前直接 return，kill-switch 被绕过（env=0 仍走
        Triton），违反本 docstring 与 greedy_triton.py 头注的「0 = 强制
        Python 循环」契约。生产零扰动：不设 env 时路由不变（仍 v2）。
        """
        if (os.environ.get("SGLANG_TLI_GREEDY_KERNEL", "1") in ("0", "false", "False")):
            return TLIIndexer._greedy_cluster_pass_python(x, sim, sums, cnt, sq, k_live)
        if (os.environ.get("TLI_E131_GREEDY_V2", "1") not in ("0", "false", "False")
                and x.is_cuda
                and x.dim() == 3
                and x.shape[-1] > 0
                and (x.shape[-1] & (x.shape[-1] - 1)) == 0):
            try:
                from .greedy_triton import greedy_pass_triton_v2
                # 复用锚点的 chunk=16384 分段驱动（跨 launch 串行，语义 =
                # 全量重放，E113 T3 口径；fn 形参即逐段实现）
                return TLIIndexer._greedy_cluster_pass_triton(
                    x, sim, sums, cnt, sq, k_live, greedy_pass_triton_v2)
            except Exception:
                pass   # fail-closed：v2 不可用 → v1/Python 锚点调度器（值同）
        if x.is_cuda:
            fn = _load_greedy_triton()
            T, H, dd = x.shape
            if fn is not None and T > 0 and dd > 0 and (dd & (dd - 1)) == 0:
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

    def _gather_fine_cand(self, score_coarse, score_coarse_avg, k_qat,
                          b_q_full, valid_length, sc_mean=None,
                          sc_avg_mean=None):
        """【E156 单源】L1 候选 token 集 fine 打分 helper（per-head）。

        compute_score 主路径与父类捷径共用同一实现（杜绝两份口径漂移）：
        用 _select_l1_blocks 选 L1 块 → 每 kv-head 独立候选 token 集（本
        head L1 块 ∪ E155b bypass ∪ sink ∪ swa）→ 对角 GQA gather k →
        批量 einsum。候选集与 compute_mask 的 _select_l1_blocks 单源。

        【E156b 修复 2026-10-20】初版用「所有 head 选中块并集」——多 head
        attention 下并集覆盖近全序列（32K 实测 27904/32768），einsum 无省
        算且 nonzero/unique/sort 开销反超（+7×）。改 per-head 行矩阵：每个
        kv-head 只对「本 head 的 K1=128 块 + 强制区」打分（~8.5K/head vs
        全宽 32K → fine 打分 ~4× 下降）。返回 q-head 空间 [H,R_max] 带
        每 head 实长 cand_len（短序列/尾块截断时 pad，重建设 scatter 用
        cand_len 门控防 sentinel 污染）。

        返回 (score_fine_cand [1,H,R_max], cand_idx [Hkv,R_max],
        cand_len [Hkv], l1_sel dict)；任一异常 / 形状不符 / 门控不适用 →
        (None,None,None,None) fail-closed（调用方回退原全宽路径，值逐位
        不变）。l1_sel = 内部 _select_l1_blocks 完整产出 dict（【AKO
        iter-6】compute_mask fused 臂复用 L1 选择用；None 伴随 fail 路径）。
        【AKO iter-20】cand_idx/cand_len 以 **kv-head 空间**返回（行在
        GQA 组内恒同——cand_gqa_shared 契约），score_dict 附 cand_kv_space
        标记；消费端 kernel 用 h//g 折叠行地址，冷路径按需展开回 q-head。
        E156 门控内联：
        σ / ccluster（near/far cluster 分）实验臂直接读全宽原始 score_fine
        可超出候选集 → 这些臂拒绝候选化。
        async 消费上一轮 L1 mask，当前候选不保证覆盖它，必须回退全宽评分。
        """
        try:
            if not (self._fine_gather and self._basis is None
                    and not self.moba_gate and not self.per_q_head
                    and not self.enable_async):
                return None, None, None, None
            _raw = (self.sigma_select != "none"
                    or self.far_select in ("cluster", "sim_greedy")
                    or self.near_select in ("cluster", "sim_greedy"))
            if _raw:
                return None, None, None, None
            _S_r = (int(valid_length)
                    if valid_length is not None else k_qat.shape[1])
            # 【AKO iter-21】粗筛 kernel 同趟产出的 GQA 组均值透传（None =
            # 回退臂，_select_l1_blocks 内部 torch mean 逐位不变）
            _sel = self._select_l1_blocks(
                score_coarse, score_coarse_avg, _S_r,
                sc_mean=sc_mean, sc_avg_mean=sc_avg_mean)
            _bs_c = _sel["bs"]
            _tmb = _sel["topk_mask_block"]          # [1,1,Hkv,kt]
            _dev = _tmb.device
            _Hkv = _tmb.shape[2]
            _swa_lo = max(0, _S_r - _sel["swa_tok"])
            # 每 kv-head 独立候选 token 集（sink/swa 全 head 共享；块与强制区
            # 重叠产生重复索引无害——打分重复物化、重建 scatter 幂等同值）。
            # 【E156c 向量化构建】per-head Python 循环 = 8×~20 个小 CUDA
            # launch ≈ 1ms host 开销（32K 实测 build_cand=1.04ms，占候选
            # compute_score 55%），且 Phase 1 batch 化后该开销再乘 B——必须
            # 向量化。语义逐位等价：
            #   L1 段：torch.nonzero 行优先天然按 head 分组、块升序（与循环
            #   `torch.where(_tmb[0,0,h])[0]` 逐位同序）；截断 < S_r 全局等价
            #   于逐 head 截断（只有尾块可能超界）。
            #   共享段（bypass+sink+swa）：每 head 相同，scatter 到各自行的
            #   [cnt_l1[h], cnt_l1[h]+M) 列区间（= 循环 concat 顺序）。
            # 【AKO iter-16】免同步候选构建（消 4 处/步 device 同步 + E134
            # graph capture 阻断项）：旧 torch.nonzero(_tmb4)（输出形状依赖
            # count → D2H 同步）+ 2 处 bool 索引过滤（同步）+
            # (_l1_cnt+_M).max().item()（同步）+ bincount/cumsum/col 推导/
            # L1 scatter 链 → 全 host 已知形状张量算术。序恒等：i_all 与
            # topk_mask True 集恒等（见 _select_l1_blocks i_all 注），per-head
            # sort 升序展开 = nonzero 行主序（head 升序、head 内块升序、
            # token 升序）。截断 token（tok≥S_r）所在块恒排行尾 → 有效候选
            # = 行前缀，cnt = Σ_b clamp(S_r-b·bs,0,bs) 与旧 bincount(keep)
            # 逐位同值；R_max = nb·bs + M host 常量（S_r%bs==0 时与旧
            # max().item() 严格相等；有部分块时仅多 ≤bs-1 列 -inf pad，
            # cand_len 门控下 mask 逐位不变）。探针：128 步对拍 0 失配
            # （e64/非 e64/sink 变体 ×32）。
            _i_all = _sel.get("i_all")
            if _i_all is None:
                return None, None, None, None   # 理论不可达（两分支均产出）
            _bp = _sel["bypass_blks"]
            # 【AKO iter-22】共享段 host 端点/宽度直算（零宽段剔除，段序 =
            # 旧 cat 程序序：bypass 序 + sink + swa）：fused 路径 4×(lo,w)
            # 标量直进 K2（免 arange×4 + cat + bp as_tensor H2D）；
            # fallback 路径按同段表现建 arange（值逐位同旧链）。
            # 宽度全 host 可知：bypass 段 = min((b+1)·bs, S_r) − b·bs
            # （iter-16 host 宽度直构口径）；sink = min(sink_tok, S_r)；
            # swa = S_r − max(0, S_r−swa_tok)。
            _seg_los, _seg_ws = [], []
            for _b in _bp:
                _lo, _hi = _b * _bs_c, min(_b * _bs_c + _bs_c, _S_r)
                if _hi > _lo:
                    _seg_los.append(_lo)
                    _seg_ws.append(_hi - _lo)
            _sink_w = min(_sel["sink_tok"], _S_r)
            if _sink_w > 0:
                _seg_los.append(0)
                _seg_ws.append(_sink_w)
            if _S_r > _swa_lo:
                _seg_los.append(_swa_lo)
                _seg_ws.append(_S_r - _swa_lo)
            _M = int(sum(_seg_ws))
            _nb = _i_all.shape[-1] + len(_bp)
            # 【AKO iter-17】融合构建 K1（per-head 排序+cnt）+ K2（行装配），
            # 2 launch 替代 torch 链 ~16 launch（cat+sort+arange 展开+
            # clamp.sum+zeros+直填+index_put_）；纯整数域与 torch 链逐位
            # 恒等（哨兵 pad 排序 ≡ torch.sort 升序，互异块号无 tie）。
            # 门禁：末维连续 / nb ≥ 2（tl.sort 单元素编译崩，iter-9 坑）/
            # nb_pow2 ≤ 256（iter-9 同界）；异常回退 torch 链值同。
            _use_fused = (self._triton_ok and _HAS_TRITON
                          and _i_all.stride(-1) == 1
                          and _i_all.shape[-1] + len(_bp) >= 2)
            if _use_fused:
                try:
                    _cand, _l1_cnt = self._fast_cand_build(
                        _i_all, _bp, _seg_los, _seg_ws, _S_r, _bs_c, _nb)
                except Exception:
                    _use_fused = False
            if not _use_fused:
                _shared = [torch.arange(_lo, _lo + _w, device=_dev)
                           for _lo, _w in zip(_seg_los, _seg_ws)]
                _shared_t = torch.cat(_shared) if _shared else None
                if _bp:
                    _bt = torch.as_tensor(_bp, dtype=torch.long, device=_dev)
                    _i_all = torch.cat(
                        [_i_all, _bt.view(1, 1, 1, -1).expand(
                            *_i_all.shape[:-1], len(_bp))], dim=-1)
                _nb2 = _i_all.shape[-1]
                _blk = torch.sort(_i_all, dim=-1).values.reshape(_Hkv, _nb2)
                _l1_tok = (_blk * _bs_c).unsqueeze(-1) \
                    + torch.arange(_bs_c, device=_dev).view(1, 1, _bs_c)
                _l1_tok = _l1_tok.reshape(_Hkv, _nb2 * _bs_c)   # 行 token 升序
                _l1_cnt = torch.clamp(
                    _S_r - _blk * _bs_c, min=0, max=_bs_c
                ).sum(dim=-1).to(torch.long)                    # [Hkv]
                _R_max = _nb2 * _bs_c + _M
                if _R_max == 0:
                    return None, None, None, None
                _cand = torch.zeros((_Hkv, _R_max),
                                    dtype=torch.long, device=_dev)
                _cand[:, :_nb2 * _bs_c] = _l1_tok  # L1 段直填（升序前缀）
                if _M:
                    _sh_h = torch.arange(_Hkv, device=_dev
                                         ).view(-1, 1).expand(_Hkv, _M)
                    _sh_col = (_l1_cnt.view(-1, 1).expand(_Hkv, _M)
                               + torch.arange(_M, device=_dev
                                              ).view(1, -1).expand(_Hkv, _M))
                    _cand[_sh_h.reshape(-1), _sh_col.reshape(-1)] = (
                        _shared_t.expand(_Hkv, _M).reshape(-1))
            _R_max = _nb * _bs_c + _M
            if _R_max == 0:
                return None, None, None, None
            _cand_len = _l1_cnt + _M                      # [Hkv]
            # 对角 GQA gather：行 h 只用 kv-head h 的 k 特征（免 [Hkv,R,Hkv,d]
            # 全展开物化——首版 70MB 中间量，对角直取省 8×）。
            # 【E156c】Triton 融合候选 kernel（_fast_score_fine_cand）替代
            # gather+repeat+einsum 链：单 launch 内 cand gather + GQA K 寄存器
            # 复用 + fp32 累加，免 fp32 repeat 物化。门控对齐 _fast_score_fine
            # （非投影/紧凑 4D/连续末维/fp32 q/同维同头数）；测试臂强制
            # _triton_ok=False 走下方原版 einsum 表达式（两臂同构隔离）。
            _use_triton = (self._triton_ok and _HAS_TRITON
                           and k_qat.is_cuda and k_qat.dim() == 4
                           and k_qat.shape[0] == 1
                           and (k_qat.shape[-1] & (k_qat.shape[-1] - 1)) == 0
                           and k_qat.stride(-1) == 1
                           and b_q_full.stride(-1) == 1
                           and b_q_full.dtype == torch.float32
                           and k_qat.shape[-1] == b_q_full.shape[-1]
                           and k_qat.shape[2] * self.group_size
                           == b_q_full.shape[0])
            if _use_triton:
                try:
                    _sfc = self._fast_score_fine_cand(
                        k_qat[0], b_q_full, _cand, _cand_len, _R_max,
                        k_qat.dtype)
                except Exception:
                    _use_triton = False
            if not _use_triton:
                _kv_id = torch.arange(_Hkv, device=_dev).view(-1, 1)   # [Hkv,1]
                _kqc_diag = k_qat[0][_cand, _kv_id, :]                 # [Hkv,R,d']
                _bkqc = repeat(_kqc_diag, "kv r d -> (kv g) r d",
                               g=self.group_size).to(torch.float32)   # [H,R,d']
                _sfc = einsum(b_q_full, _bkqc, "h d, h r d -> h r")   # [H,R]
                _sfc = _sfc.to(k_qat.dtype)
            # 【AKO iter-17】Triton 路径 -inf 门控已在 _score_fine_cand_kernel
            # 内完成（L4207 offs_r < cl，cl = _cand_len 组内共享同界）→ 外层
            # arange/repeat/where/new_full 尾链纯冗余（对 -inf 再 where 逐位
            # 同值），免 6+ launch；einsum 回退路径保持原 where 链。
            if _use_triton:
                _score_fine_cand = _sfc.unsqueeze(0)
            else:
                _r_mask = (torch.arange(_R_max, device=_dev).view(1, _R_max)
                           < _cand_len.view(_Hkv, 1))                 # [Hkv,R]
                _r_mask_h = repeat(_r_mask, "kv r -> (kv g) r",
                                   g=self.group_size)                 # [H,R]
                _score_fine_cand = k_qat.new_full(
                    (1, b_q_full.shape[0], _R_max), float("-inf"))
                _score_fine_cand[0] = torch.where(
                    _r_mask_h, _sfc, torch.full_like(_sfc, float("-inf")))
            # 【AKO iter-20】候选索引保持 kv-head 空间（组内行恒同 =
            # cand_gqa_shared 契约），免 repeat expand+clone 物化
            # （(8,4,8576) int64 clone+copy ~2.2MB/步）；消费端
            # _ako_rebuild_scatter_kernel 用 h//g 折叠，冷路径按需展开。
            return _score_fine_cand, _cand, _cand_len, _sel
        except Exception:
            # fail-closed：候选化任何异常回退全宽原分支（值同原路径）。
            # 返回值独立于调用方的 score_fine（不污染全宽原张量）。
            return None, None, None, None

    def compute_score(self, q, q_ids, index_dict, softmax_scale):
        if q.shape[0] != 1 or q.shape[1] != 1:
            raise ValueError(
                "TLIIndexer compute_score requires batch=1 and one query token; "
                f"got query shape prefix={tuple(q.shape[:2])}")
        # 缓存 squeeze 后的 q 供远端簇分数使用
        # 【C3 修复 2026-10-08】缓存时统一乘 softmax_scale——_near_token_score
        # 的簇分会与已缩放的 score_fine 回退段（b_q_full = q_sq[0]*softmax_scale）
        # 进同一 near 池 topk 竞争，未缩放会放大量纲 |1/scale| 倍（D=128 时
        # 11.31×）导致簇段霸榜挤出回退段；far 侧纯簇分 topk 乘正 scale 不变
        # 排序 → far 行为逐位不变（核验报告 research/docs/gpt_audit_verdict_
        # 20261008_round2.md C3 节）。
        # 【E133c-A3】_last_q 仅被簇分数分支消费（compute_mask far/near
        # far_select/near_select ∈ {cluster, sim_greedy} 两处 getattr）；
        # mavg 等臂每步 2 个 launch 白付 → 按消费方条件化（未命中时保持
        # 旧值 None 语义，消费端 getattr 回退口径不变）。
        if q.shape[0] == 1 and q.shape[1] == 1 and (
            self.far_select in ("cluster", "sim_greedy")
            or self.near_select in ("cluster", "sim_greedy")
        ):
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
            # 【E133b 适配修复】E133b 后 k_qat 为紧凑布局 [1,cap,Hkv,64//r]，
            # 父类 TIA compute_score 捷径按全维 dense 消费 → einsum 形状崩
            # （基线 cccf756 同炸，E133b 既有缺陷非 E133c 引入）。喂父类前
            # 临时展开零填充 dense 副本——零列贡献精确 +0.0，分数逐位不变；
            # 浅拷贝 dict 不污染调用方。
            _kq = index_dict.get("k_qat")
            if _kq is not None and _kq.dim() == 4 and _kq.shape[-1] != q.shape[-1]:
                _ind = self._qat_indices(_kq.device)
                _dense = _kq.new_zeros(_kq.shape[0], _kq.shape[1],
                                       _kq.shape[2], q.shape[-1])
                _dense[..., _ind] = _kq
                index_dict = dict(index_dict)
                index_dict["k_qat"] = _dense
            d = super().compute_score(q, q_ids, index_dict, softmax_scale)
            # 075：父类捷径同样携带真实长度（该路径无投影，valid_length ==
            # score_fine 宽度，行为不变；保持 score_dict 接口统一）
            d["valid_length"] = index_dict.get("valid_length")
            # 【E156】父类捷径同样候选化（单源 _gather_fine_cand）：父类
            # 返回全宽 score_fine [1,1,H,T] 与 b_q=(q*scale).to(f32)（全维，
            # 对应 k_qat 展开 dense 后 128 宽）。候选位 fine 打分逐 token 独立
            # → 候选 gather einsum 与全宽 einsum 逐位一致（N 维不同仅累加
            # 调度）。k_qat 此处为 dense 展开副本（L1487-1494），consumed
            # 维度与 b_q 匹配。
            # 【R4 修复 2026-10-20】async 两步 topk 同列排除：候选化使
            # score_fine 收成候选宽 + cand_idx，compute_mask scatter 回全宽
            # 后 score_fine 只在本步 L1 候选块位置有 finite 值（其余 -inf）。
            # async step2 swap 后 topk_mask = step1 块，where 后 step1 块
            # 位置因非 step2 候选 → -inf → L2 topk 有效 token 不足 K2 → 用
            # p=0 填充 → 选出 step2 自己的候选块（async 候选替换语义被
            # 破坏，test_e121_fix_079 R4 红）。enable_async 默认关，开后须
            # 全宽 score_fine 保证 step2 在 step1 候选位置有分。
            if self._fine_gather and not self.enable_async:
                _cd, _ci, _cl, _l4 = self._gather_fine_cand(
                    d["score_coarse"], None, index_dict["k_qat"],
                    (q[0, 0] * softmax_scale).to(torch.float32),
                    index_dict.get("valid_length"))
                if _cd is not None and _ci is not None:
                    d["score_fine"] = _cd.unsqueeze(0)
                    d["cand_idx"] = _ci
                    d["cand_len"] = _cl
                    d["cand_width"] = index_dict["k_qat"].shape[1]
                    # 【AKO iter-3】同主路径：组内行共享契约标记（见下）
                    d["cand_gqa_shared"] = True
                    # 【AKO iter-20】同主路径：kv 空间候选标记
                    d["cand_kv_space"] = True
                    # 【AKO iter-6】同主路径：L1 选择产出透传（见主路径注释）
                    d["l1_sel"] = _l4
            return d
        # A：L1 上界只在 d' 子空间维上算（q 同步取子空间，与 k_min/k_max 对齐）
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
                _sink_tok = self.sink_tokens          # E135：强制区宽单一事实源
                _swa_tok = self.sliding_window_size
                _mid_len = max(0, int(_vl) - _sink_tok - _swa_tok)
                if self.nt_near_len is not None:
                    # E155d：near 区宽显式整数（α·mid_len 规则被覆盖，与
                    # compute_mask 同源）
                    _near_len_dyn = max(_bs, int(self.nt_near_len))
                else:
                    _near_len_dyn = max(_bs, int(self.alpha * _mid_len))
                _near_blks = max(self._sink_pool_lo_blk,
                                 ((int(_vl) - _swa_tok) - _near_len_dyn) // _bs)
                if self._sink_pool_lo_blk < _near_blks:
                    dgate_short = True
                    dgate_near_blks = _near_blks
        # 【AKO iter-21】候选化臂资格 + pow2 守卫提前判定（原 L2015 处
        # _raw_fine_read 条件原样上移复用，纯代码移动零语义变化）：
        # 粗筛 Triton 分支据此决定是否同趟写 GQA 组均值（均值缓冲只在
        # 候选化路径被 _select_l1_blocks 消费；臂不适用时零分配零浪费）。
        _raw_fine_read = (self.sigma_select != "none"
                          or self.far_select in ("cluster", "sim_greedy")
                          or self.near_select in ("cluster", "sim_greedy"))
        _cand_eligible = (self._fine_gather and self._basis is None
                          and not self.moba_gate and not self.per_q_head
                          and not _raw_fine_read and not self.enable_async)
        # 【AKO iter-19】非 dgate_short 路径三个粗筛分支（dgate/Triton/
        # einsum）均整片覆写 [0] 切片 → new_full(-inf) 填充是死代码，
        # new_empty + Triton out= 直写消 fill+copy 各 2 launch；dgate_short
        # 分段写保留 -inf 语义不变。
        if dgate_short:
            score_coarse = q_sq.new_full(coarse_shape, float("-inf"))
        else:
            score_coarse = q_sq.new_empty(coarse_shape)
        # 【AKO iter-21】GQA 组均值缓冲（kv 空间 [1,Hkv,T]）：kernel 同趟
        # 直写，_select_l1_blocks 免 torch mean reduce launch（×2 ~11us）。
        # 仅候选化臂 + pow2 G 分配；dgate_short / einsum / 投影等回退臂
        # 恒 None → _select_l1_blocks 回退 torch mean 逐位不变。
        _mean_out = _cand_eligible and (self.group_size & (self.group_size - 1)) == 0
        sc_mean = None
        sc_avg_mean = None
        if dgate_short:
            b_k_min = repeat(k_min, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
            b_k_max = repeat(k_max, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
            _sb = self._sink_pool_lo_blk             # E135：池左界块数单一事实源
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
                # 【AKO iter-19】kernel out= 直写 [0] 切片（免中间物化+copy）
                if _mean_out:
                    sc_mean = q_sq.new_empty((1, k_min.shape[1], k_min.shape[0]))
                    self._fast_score_coarse(
                        k_min, k_max, b_q, q_sq.dtype, out=score_coarse[0],
                        mean_out=sc_mean[0])
                else:
                    self._fast_score_coarse(
                        k_min, k_max, b_q, q_sq.dtype, out=score_coarse[0])
            except Exception:
                sc_mean = None
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
            # 【AKO iter-19】同 score_coarse：非 dgate_short 全量覆写 → 免 -inf 填充
            if dgate_short:
                score_coarse_avg = q_sq.new_full(coarse_shape, float("-inf"))
            else:
                score_coarse_avg = q_sq.new_empty(coarse_shape)
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
                    and k_avg.stride(-1) == 1 and b_q.stride(-1) == 1
                    and b_q.dtype == torch.float32
                    and k_avg.shape[-1] == b_q.shape[-1]
                    and k_avg.shape[1] * self.group_size == b_q.shape[0]
                    and self._triton_ok):
                # 【E127 iter8】avg 段复用 iter6 _score_fine_kernel 同构
                # （einsum(b_q, repeat(k_avg)) 数学等价，GQA K 复用免物
                # 化）；ULP 口径同 iter6；异形/异常回退原版表达式。
                # 【E132b TL-E127-FEATURE-STRIDE-129】补 b_q 末维 stride
                # 检查（kernel 以 base+offs_d 硬编码末维 stride=1 访问 Q；
                # coarse 门禁同款检查已有 k 侧，此处补齐 q 侧）。
                try:
                    # 【AKO iter-19】kernel out= 直写 [0] 切片
                    if _mean_out:
                        sc_avg_mean = q_sq.new_empty(
                            (1, k_avg.shape[1], k_avg.shape[0]))
                        self._fast_score_fine(
                            k_avg, b_q, q_sq.dtype, out=score_coarse_avg[0],
                            mean_out=sc_avg_mean[0])
                    else:
                        self._fast_score_fine(
                            k_avg, b_q, q_sq.dtype, out=score_coarse_avg[0])
                except Exception:
                    sc_avg_mean = None
                    b_k_avg = repeat(k_avg, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
                    score_coarse_avg[0] = einsum(b_q, b_k_avg, "h d, kt h d -> h kt").to(q_sq.dtype)
            else:
                b_k_avg = repeat(k_avg, "t h d -> t (h g) d", g=self.group_size).to(torch.float32)
                score_coarse_avg[0] = einsum(b_q, b_k_avg, "h d, kt h d -> h kt").to(q_sq.dtype)
        # L2 精筛分数：非投影=全维 k_qat；投影=投影特征 k_qat（E64f 协议）
        # 投影路径必须 fp32：8 维点积幅度 ~O(0.5)，bf16 绝对步长会折叠 top-K 边界
        # 的小 gap 分数成同值 → topk 退化为位置偏置（e2e mass 0.99→0.81 实测）
        # 【iter-18】全宽 score_fine 分配下沉到 cand_idx is None 分支——
        # 候选化成功时（生产默认）原分配即被 _score_fine_cand 整体替换，
        # 4.2MB new_full(-inf) 是纯死代码；失败回退路径在分支内按需分配。
        if self._basis is not None:
            b_q_full = b_q
        else:
            # 【E133b 原生】k_qat 紧凑 [.,.,delta] → q 同步 gather 到量化维
            # （被丢弃维乘积恒为 0，数学等价；fp32 累加顺序变化 → bf16 舍入
            # 后 ULP 级漂移，E127 iter6 同口径）。indices 规则与 prepare 端
            # k_qat 量化完全同源（idx_sub 优先，full 落 _qat_indices）。
            # 【E133c-A2】tail 模式 _qi == idx_sub → b_q_full ≡ b_q（原
            # mul+to+gather 3 launch 纯冗余，复用 L931 产物）；full 模式
            # 共享 q_full 仅 gather 一次（省 mul+to 2 launch）。值逐位同
            # 原表达式（q_full 与重算的 (q_sq[0]*scale).to(f32) 确定性同值）。
            # 【E134】l2_subspace=full 时 k_qat 为 128 宽全维 → q 直接用
            # 全维 q_full（量化覆盖全部维度，无 gather 必要）。
            if self.l2_subspace == "full":
                b_q_full = q_full
            elif idx_sub is not None:
                b_q_full = b_q
            else:
                b_q_full = q_full[..., self._qat_indices(q_sq.device)]
        # ---- 【E156 精筛候选化】只对 L1 候选 token + 强制区打分 ----
        # 消除全宽 [1,H,T] einsum 物化（T=全序列 token）：fine 打分逐 token
        # 独立（q·k_qat），softmax 归约集合 = 候选∪强制区 ≡ 现状 where 屏蔽
        # 后的非 -inf 集合（未选位置 -inf 的 exp=0 精确不贡献）→ 逐位等价。
        # 候选集 = L1 选中块 ∪ E155b bypass ∪ sink ∪ swa（与 compute_mask 共用
        # _select_l1_blocks 单源）。env TLI_FINE_GATHER=0 关 / 投影 / per_q_head /
        # moba / 任何异常 fail-closed 回退原全宽路径。
        # 【E156 门控】σ / ccluster（near/far cluster 分）实验臂直接读
        # score_dict["score_fine"] 全宽原始分（未 L1 屏蔽），可选中 token 集合
        # 可超出 L1 候选集——候选化把这些位置置 -inf 会语义漂移 → 这些臂
        # fail-closed 走原全宽路径（行为逐位不变）。生产默认（σ=none、
        # far/near=4bit）非此列。
        # 【AKO iter-21】_raw_fine_read / 臂资格已上移至粗筛分支前（复用同值）
        cand_idx = None
        cand_len = None
        _score_fine_cand = None
        _l1_sel = None
        if _cand_eligible:
            # 单源 _gather_fine_cand（与父类捷径共用）：内部 fail-closed，
            # 任一异常/形状不符返回 (None,None,None,None) → 走原全宽路径
            # （值逐位不变）。
            _score_fine_cand, cand_idx, cand_len, _l1_sel = \
                self._gather_fine_cand(
                score_coarse, score_coarse_avg, k_qat, b_q_full,
                index_dict.get("valid_length"),
                sc_mean=sc_mean, sc_avg_mean=sc_avg_mean)
        if cand_idx is not None and _score_fine_cand is not None:
            score_fine = _score_fine_cand
        if cand_idx is None:
            # 【iter-18】全宽 -inf 预填充在此按需分配（候选化成功时不执行）。
            fine_shape = (1, H, k_qat.shape[1])
            if self._basis is not None:
                score_fine = q_sq.new_full(fine_shape, float("-inf"), dtype=torch.float32)
            else:
                score_fine = q_sq.new_full(fine_shape, float("-inf"))
            # ---- 原全宽 L2 精筛打分（dgate_short / Triton / 投影 / einsum）----
            # 注意：整个 if/elif 链（dgate_short/Triton/投影/einsum）必须整体
            # 嵌套在 `if cand_idx is None:` 内——候选化成功时（cand_idx 非 None）
            # 本链完全跳过（score_fine 已是候选尺寸），否则 elif 链会被错误地
            # 接到 cand_idx 判定上导致形状错位（2026-10-20 实测 shape 1277 vs
            # 4093 即此因）。
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
                    and k_qat.shape[0] == 1
                    and (k_qat.shape[-1] & (k_qat.shape[-1] - 1)) == 0
                    and k_qat.stride(-1) == 1 and b_q_full.stride(-1) == 1
                    and b_q_full.dtype == torch.float32
                    and k_qat.shape[-1] == b_q_full.shape[-1]
                    and k_qat.shape[2] * self.group_size == b_q_full.shape[0]
                    and self._triton_ok):
                # 【E127 iter6】非投影路径 Triton GQA K 复用（K 读带宽 4× 削减，
                # 免 [T,H,128] fp32 repeat 物化；fp32 累加与原版 einsum 仅累加
                # 顺序不同 → bf16 舍入后 ULP 级漂移，mask 门禁实测逐位不变）。
                # 投影/异形布局/Triton 异常回退原版 repeat+einsum。
                # 【E132b TL-E127-FEATURE-STRIDE-129】补 k_qat/b_q_full 末维
                # stride 检查：_score_fine_kernel 以 base+offs_d 硬编码末维
                # stride=1 访问特征，合法非连续布局（如 permute/转置 dense 视
                # 图）会静默读错物理地址——与 coarse 门禁（k_min/k_max/b_q 三
                # 检查）对齐；非连续输入回退原版 einsum 表达式（fail-closed，
                # 值同）。
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
            # 【E156 精筛候选化】per-head 候选 token 绝对位置 [H,R_max]
            # （None = 全宽路径）。compute_mask 检测到非 None 时把稀疏
            # score_fine scatter 回全宽（cand_len 门控防 pad sentinel 污染）。
            "cand_idx": cand_idx,
            # 【E156b】per-head 候选实长 [H]（短序列/尾块截断时 < R_max；
            # 重建设 scatter 只写实长前缀）。
            "cand_len": cand_len,
            # 【E156】候选化时的全宽（= k_qat 原宽，scatter 重建目标宽；
            # 非投影路径 == valid_length == 原 score_fine 宽）。None = 全宽路径。
            "cand_width": (None if cand_idx is None else k_qat.shape[1]),
            # 【AKO iter-3】候选行 GQA 组内共享契约标记：_gather_fine_cand
            # 经 repeat(kv g) 展开到 q-head 空间，组内行恒同——compute_mask
            # 的 compact softmax 路径依赖该不变量做 kv-head view 折叠（mean
            # 与 scatter 索引行折叠）。合成回放输入缺此键 → fused 门拒绝 →
            # E156 全宽重建路径，逐位兼容。
            "cand_gqa_shared": cand_idx is not None,
            # 【AKO iter-20】候选索引 kv-head 空间标记（组内行恒同）：
            # fused kernel 用 h//g 折叠；缺标记（合成回放）= q-head 空间
            # 旧口径，消费端按旧行为处理。
            "cand_kv_space": cand_idx is not None,
            # 【AKO iter-6】L1 选择复用：_gather_fine_cand 内 _select_l1_blocks
            # 的完整产出 dict（topk_mask_block/几何/i_f/i_n/l1_s_real）——
            # compute_mask fused 臂在 kt/bs/S_real/几何逐项一致时直取，
            # 消 2×topk+cat+scatter+bypass GPU 写重跑（同源 score_coarse、
            # 同 GQA mean 聚合、逐字同实现 → bit-identical）。None = 全宽
            # 路径/fail-closed/合成回放 → compute_mask 回退内联 L1 逐位不变。
            "l1_sel": _l1_sel,
        }

    # ------------------------------------------------ 【E127 iter6/8】Triton
    # fast path（原 TLIIndexerFast._fast_score_fine/_fast_score_coarse 并入）
    # 【E133c-A6】打分 kernel tile 参数按 D 查表：D=128 维持 (64,4) 原值
    # 保证 full 臂零回归；D=32 由 exp/bench/e133c_tile_scan.py 实测扫描定参
    # （2xh20-reg0519 H20，真实 mavg+tail32 链路捕获输入，cudaEvent 稳态中位
    # warmup50+rep200）：coarse 最优 (128,8)=0.0374ms vs (64,4)=0.0378（-1%）；
    # fine 最优 (16,1)=0.0394 vs (64,4)=0.0412（-4.4%，D=32 下 warps>1 会把
    # 32 维 tile 切碎致劣化）。D 缺省回退 (64,4)。扫描数据 tile_scan.json。
    _TILE_FINE = {32: (16, 1), 64: (64, 4), 128: (64, 4)}
    _TILE_COARSE = {32: (128, 8), 64: (64, 4), 128: (64, 4)}

    def _fast_score_fine(self, kq, b_q_full, out_dtype, out=None, mean_out=None):
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
            # 【AKO iter-19】out= 直写调用方切片（[H,T] 连续），免中间物化
            # + 整片 copy_；缺省自建保持既有调用点（compute_mask 内联段等）。
            if out is None:
                out = torch.empty((H, T), device=kq.device, dtype=out_dtype)
            bt, nw = self._TILE_FINE.get(D, (64, 4))
            grid = (triton.cdiv(T, bt), Hkv)
            # 【AKO iter-21】可选 GQA 组均值同趟写出（消 _select_l1_blocks
            # torch mean reduce launch）；非 pow2 G fail-closed raise（调用
            # 方 except 回退 einsum 臂 + torch mean，值逐位不变）。
            if mean_out is not None:
                if (g & (g - 1)) != 0 or not mean_out.is_contiguous():
                    raise ValueError(
                        f"mean_out 守卫失败：g={g} 非 2 幂或 MOUT 非连续")
                mout, smh, smt = mean_out, mean_out.stride(0), mean_out.stride(1)
            else:
                mout, smh, smt = out, out.stride(0), out.stride(1)
            _score_fine_kernel[grid](
                kq, b_q_full, out, mout, T,
                kq.stride(0), kq.stride(1), b_q_full.stride(0),
                out.stride(0), out.stride(1),
                smh, smt,
                Hkv=Hkv, G=g, D=D, BT=bt,
                HAS_MEAN=mean_out is not None, num_warps=nw)
            return out
        except Exception:
            self._triton_ok = False
            raise

    def _fast_score_fine_cand(self, kq, b_q_full, cand_idx, cand_len, R,
                              out_dtype):
        """【E156c】候选 fine 打分 Triton 融合 kernel（非投影路径专用）。

        数学：score[h, r] = sum_d b_q_full[h, d] * kq[cand_idx[kv, r], kv, d]
        （kv = h // g），与 _gather_fine_cand 的 einsum 路径数学等价。单
        launch 内完成 cand gather + GQA K 寄存器复用（读带宽 /g）+ fp32 累加
        + -inf 门控，替代 gather + repeat fp32 物化 + einsum + where 链。

        数值：与 _fast_score_fine 同口径（fp32 累加顺序 vs einsum 不同 →
        bf16 舍入后 ULP 级漂移，mask 门禁实测逐位不变）；测试臂强制
        _triton_ok=False 走 einsum 表达式，本 kernel 仅生产路径触发。

        fail-closed：Triton 编译/执行异常 → 置 _triton_ok=False 并 raise
        （调用点 except 回退 einsum 表达式，值同）。"""
        try:
            T_, Hkv, D = kq.shape
            H = b_q_full.shape[0]
            g = H // Hkv
            out = torch.empty((H, R), device=kq.device, dtype=out_dtype)
            bt, nw = self._TILE_FINE.get(D, (64, 4))
            grid = (triton.cdiv(R, bt), Hkv)
            _score_fine_cand_kernel[grid](
                kq, b_q_full, cand_idx, cand_len, out, R,
                kq.stride(0), kq.stride(1), b_q_full.stride(0),
                cand_idx.stride(0), cand_idx.stride(1),
                out.stride(0), out.stride(1),
                Hkv=Hkv, G=g, D=D, BT=bt, num_warps=nw)
            return out
        except Exception:
            self._triton_ok = False
            raise

    def _fast_cand_build(self, i_all, bypass_blks, seg_los, seg_ws,
                         s_r, bs, nb):
        """【AKO iter-17】候选构建融合 wrapper（K1 排序+cnt / K2 行装配，
        2 launch 替代 iter-16 torch 链 ~16 launch：cat+sort+arange 展开+
        clamp.sum+zeros+直填+index_put_）。

        【AKO iter-22】共享段下沉：bypass 块号标量化（b0/b1，≤2 门禁）+
        共享段 4×(lo,w) 标量直出（消 bp as_tensor H2D + arange×4 +
        cat）——段序 = 旧 torch _shared cat 程序序（bypass 序 + sink +
        swa），段内 token = lo + (k−段前缀宽) ≡ arange(lo,lo+w) 连接，
        纯整数域逐位恒等。

        数学：与 iter-16 torch 链逐位恒等（纯整数域）——K1 哨兵 pad 排序
        ≡ torch.sort 升序（互异块号无 tie）；cnt 同 clamp 行和（零宽
        bypass 块贡献恒 0，与旧链 clamp(s_r−b·bs,0,bs)=0 同值）；K2 列三态
        与「直填 L1 + index_put shared + zeros pad」程序序同值。探针：
        4 臂变体逐位对拍 + bench bitexact 双门。

        fail-closed：异常 raise（调用点 except 回退 torch 链，值同）。"""
        Hkv = i_all.shape[2]
        n_l1 = i_all.shape[-1]
        n_bp = len(bypass_blks)
        if n_bp > 2:
            raise ValueError(f"cand_build n_bp={n_bp} > 2 标量守卫")
        if len(seg_los) > 4 or len(seg_ws) > 4:
            raise ValueError("cand_build 共享段 >4 守卫")
        _M = int(sum(seg_ws))
        dev = i_all.device
        nb_pow2 = triton.next_power_of_2(max(nb, 2))
        if nb_pow2 > 256:
            raise ValueError(f"cand_build nb_pow2={nb_pow2} > 256 守卫")
        scr = torch.empty((Hkv, nb_pow2), dtype=torch.long, device=dev)
        cnt = torch.empty((Hkv,), dtype=torch.long, device=dev)
        b0 = int(bypass_blks[0]) if n_bp > 0 else 0
        b1 = int(bypass_blks[1]) if n_bp > 1 else 0
        _ako_cand_sort_kernel[(Hkv,)](
            i_all, scr, cnt, n_l1, n_bp, b0, b1, s_r, bs,
            i_all.stride(2), scr.stride(0), NB_POW2=nb_pow2)
        _R_max = nb * bs + _M
        cand = torch.empty((Hkv, _R_max), dtype=torch.long, device=dev)
        los = list(seg_los) + [0] * (4 - len(seg_los))
        ws = list(seg_ws) + [0] * (4 - len(seg_ws))
        _ako_cand_build_kernel[(Hkv, triton.cdiv(_R_max, 512))](
            scr, cnt, cand, nb, bs, _M, _R_max,
            los[0], ws[0], los[1], ws[1], los[2], ws[2], los[3], ws[3],
            scr.stride(0), cand.stride(0), BLOCK=512)
        return cand, cnt

    def _fast_topk_mask(self, indices, bypass_blks, kt):
        """【AKO iter-19】topk_mask 装配融合 wrapper（1 launch 替代
        zeros_like+scatter_+ones_like+bypass setitem ×2 ≈ 5 launch）。

        数学：与 torch 装配程序序逐位恒等（纯 bool/整数域）——全 False
        基座 + topk 索引/bypass 块位置置 True；_ako_topk_mask_kernel 注
        （barrier 写序 / 索引互异无位置竞争）。

        fail-closed：K>128 / bypass>2 / 异常 raise（调用点 _build_topk_mask
        回退 torch 装配链，值同；不动 _triton_ok——独立 kernel 独立门）。
        """
        heads = indices.shape[-2]
        K = indices.shape[-1]
        if K > 128 or len(bypass_blks) > 2:
            raise ValueError(f"topk_mask K={K}/bypass={len(bypass_blks)} 守卫")
        b0 = bypass_blks[0] if len(bypass_blks) > 0 else -1
        b1 = bypass_blks[1] if len(bypass_blks) > 1 else -1
        out = torch.empty((heads, kt), dtype=torch.bool,
                          device=indices.device)
        _ako_topk_mask_kernel[(heads, triton.cdiv(kt, 512))](
            indices, out, K, kt, b0, b1,
            indices.stride(-2), indices.stride(-1), out.stride(0),
            BLOCK=512, num_warps=4)
        return out.view(*indices.shape[:-2], heads, kt)

    def _build_topk_mask(self, indices, bypass_blks, like):
        """topk_mask 装配单源（e64/非 e64 两分支共用）：
        Triton 融合优先（_tmask_ok 独立永久门），超界/异常回退 torch
        zeros_like+scatter_+ones_like+setitem 链（值同）。like = 与
        topk_mask 同形的分数张量（取 heads/kt/设备）。"""
        if getattr(self, "_tmask_ok", True) and self._triton_ok \
                and _HAS_TRITON:
            try:
                return self._fast_topk_mask(indices, bypass_blks,
                                            like.shape[-1])
            except Exception:
                self._tmask_ok = False
        topk_mask = torch.zeros_like(like, dtype=torch.bool).scatter_(
            -1, indices, torch.ones_like(indices, dtype=torch.bool))
        for _b in bypass_blks:
            topk_mask[..., _b] = True
        return topk_mask

    def _fast_score_coarse(self, k_min, k_max, b_q, out_dtype, out=None,
                           mean_out=None):
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
            # 【AKO iter-19】out= 直写调用方切片（[H,T] 连续），同 _fast_score_fine
            if out is None:
                out = torch.empty((H, T), device=k_min.device, dtype=out_dtype)
            bt, nw = self._TILE_COARSE.get(D, (64, 4))
            grid = (triton.cdiv(T, bt), Hkv)
            # 【AKO iter-21】可选 GQA 组均值同趟写出（守卫同 _fast_score_fine）
            if mean_out is not None:
                if (g & (g - 1)) != 0 or not mean_out.is_contiguous():
                    raise ValueError(
                        f"mean_out 守卫失败：g={g} 非 2 幂或 MOUT 非连续")
                mout, smh, smt = mean_out, mean_out.stride(0), mean_out.stride(1)
            else:
                mout, smh, smt = out, out.stride(0), out.stride(1)
            _score_coarse_kernel[grid](
                k_min, k_max, b_q, out, mout, T,
                k_min.stride(0), k_min.stride(1),
                k_max.stride(0), k_max.stride(1),
                b_q.stride(0),
                out.stride(0), out.stride(1),
                smh, smt,
                Hkv=Hkv, G=g, D=D, BT=bt,
                HAS_MEAN=mean_out is not None, num_warps=nw)
            return out
        except Exception:
            self._triton_ok = False
            raise

    def _fast_stat_tail(self, k, idx_sub, lo, n_tok, n_blk, fb,
                        k_min_buf, k_max_buf, k_avg_buf):
        """【E133c-A4】尾段块统计融合 wrapper（1 launch 替代 tail_pad 构造
        + gather + amin/amax/mean + 3 直写 ≈ 9 launch）。

        数学：对尾段每块（BS token × 1 kv 头）按升序逐行 fp32 顺序累加
        min/max/sum；mean = sum/BS（BS=64 精确除）。pad 行 mask other=0
        复现全量版 tail_pad 零填充语义。数值实证（50 轮随机 bf16 全过，
        /tmp/e133c_numeric_probe.py 口径）：mean 升序 fp32 顺序累加 → RNE
        bf16 与 PyTorch mean(dim) 逐位一致；min/max 为选择运算无舍入。

        fail-closed：异常置 _stat_ok=False（本实例永久回退 PyTorch 路径）
        并 raise（调用点 except 走 buffer 直写/cat 回退，值同原版）。"""
        try:
            Hkv = k.shape[2]
            if idx_sub is not None:
                DD, sub_t = idx_sub.numel(), idx_sub
            else:
                # full 模式：全维直读（HAS_SUB=False，IDXSUB 占位不读）
                DD, sub_t = k.shape[3], k_min_buf
            _coarse_stat_kernel[(n_blk, Hkv)](
                k, sub_t, k_min_buf, k_max_buf,
                k_avg_buf if k_avg_buf is not None else k_min_buf,
                n_tok, lo, fb,
                k.stride(1), k.stride(2),
                k_min_buf.stride(1), k_min_buf.stride(2),
                Hkv=Hkv, DD=DD, BS=self.args.tia_block_size,
                HAS_SUB=idx_sub is not None,
                HAS_AVG=k_avg_buf is not None,
                num_warps=max(1, DD // 32))
        except Exception:
            self._stat_ok = False
            raise

    def _fast_stat_full(self, k_for_coarse, idx_sub, need_avg):
        """【E176】全量块统计融合 wrapper —— 替代 rearrange + amin +
        amax + mean（4 次读同一块数据）。1 个 Triton kernel 读一次
        [BS, DD]，同时算 min/max/mean，fp32 顺序累加与 PyTorch
        mean(dim=2) 逐位一致（累加序同 _coarse_stat_kernel）。

        参数：
          k_for_coarse: [1, S_pad, Hkv, DD] bf16（已 pad 到 P*BS）
          idx_sub: [DD] int64 或 None（None = full 模式全维直读）
          need_avg: bool（是否需要 k_avg 输出）

        返回 (k_min, k_max, k_avg_or_None)，形状 [1, P, Hkv, DD]。
        fail-closed：异常置 _stat_ok=False 并 raise（调用方 except 回退
        PyTorch rearrange+amin+amax+mean 链，值不变）。"""
        try:
            Hkv = k_for_coarse.shape[2]
            S_pad = k_for_coarse.shape[1]
            P = S_pad // self.args.tia_block_size
            if idx_sub is not None:
                DD = idx_sub.numel()
                sub_t = idx_sub
            else:
                DD = k_for_coarse.shape[3]
                sub_t = torch.empty(0, dtype=torch.int64,
                                     device=k_for_coarse.device)
            dev = k_for_coarse.device
            k_min = torch.empty((1, P, Hkv, DD), dtype=k_for_coarse.dtype,
                                device=dev)
            k_max = torch.empty_like(k_min)
            k_avg = torch.empty_like(k_min) if need_avg else k_min
            _stat_full_kernel[(P, Hkv)](
                k_for_coarse.squeeze(0), sub_t,
                k_min.squeeze(0), k_max.squeeze(0), k_avg.squeeze(0),
                S_pad, P, Hkv,
                k_for_coarse.squeeze(0).stride(0),
                k_for_coarse.squeeze(0).stride(1),
                k_min.squeeze(0).stride(0),
                k_min.squeeze(0).stride(1),
                BS=self.args.tia_block_size,
                DD=DD,
                HAS_SUB=idx_sub is not None,
                HAS_AVG=need_avg,
                num_warps=max(1, DD // 32))
            k_avg_out = k_avg if need_avg else None
            return k_min, k_max, k_avg_out
        except Exception:
            self._stat_ok = False
            raise

    def _fast_qat_tail(self, k, indices, lo, n_tok, buf):
        """【E133c-A5】量化链融合 wrapper（1 launch 替代 gather 物化 +
        minmax 4bit 量化 ~10 个小 launch + buf 直写 ≈ 12 launch）。

        数学：与 TIAIndexer.min_max_per_token_quant 逐位一致——逐算子
        仿真（每算子 fp32 计算 + 输出 RNE bf16，PyTorch CUDA elementwise
        opmath=fp32 口径）；torch.round 的 round-half-to-even 用
        libdevice.rint（fp32 RN 模式默认 half-to-even）复现。实证 50 轮
        随机 bf16 全过（/tmp/e133c_numeric_probe.py）。

        fail-closed：异常置 _quant_ok=False 并 raise（调用点 except 回退
        PyTorch 原版量化，值同）。"""
        try:
            Hkv, DD = buf.shape[1], indices.numel()
            BT = 64
            _qat_quant_kernel[(triton.cdiv(n_tok, BT), Hkv)](
                k, indices, buf, n_tok, lo,
                k.stride(1), k.stride(2), buf.stride(0), buf.stride(1),
                Hkv=Hkv, DD=DD, BT=BT, num_warps=max(1, DD // 32))
        except Exception:
            self._quant_ok = False
            raise

    def _fast_tok_score(self, q, idx_sub, centroids, assign, agg):
        """【E131-S2（原 fast.py override 并入）】簇分数段 q_g+einsum+gather
        三件套的 Triton 全融合。

        数学：q_g[h,d'] = AGG_{j<G} q[h·G+j, idx_sub[d']]（AGG=sum 远端 /
        mean 近端，与原版 reshape(...).sum(1)/mean(1) 数学等价）；
        tok_score[h,t] = Σ_d q_g[h,d]·centroids[h, assign[h,t], d]，与
        原版 q 胶水 + einsum("hd,hkd->hk") + gather(1, assign) 数学等价。
        一次 launch 完成（原版 3 个小 launch 的胶水序列两处各付一遍）。
        E131-S2 实测（H20 S=64K cclustersim 稳态）：far 侧 0.135→~0.05ms。

        数值口径（同 E127 iter6/8）：q_g 组内累加与 dot 的 fp32 累加顺序
        均与 torch/cublas 不同 → fp32 ULP 级漂移（tok_score 无 bf16 舍入
        吸收，直接进 topk 竞争）→ 并列边界可能翻转，mask 选择为经验
        验证口径（门禁实测 + top-k 重叠率 ≥99.9% 报告精确值；E131-S2
        实测 16 步 mask 逐位 0 差异 + e2e 24 步 12585120/12585120）。

        fail-closed：Triton 编译/执行异常 → 调用点回退原版表达式（值同
        原版路径，仅性能回退）。"""
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

        【E131-S2（原 fast.py override 并入）】默认先走 Triton 全融合
        （_fast_tok_score，开关 TLI_E131_TOKSCORE_TRITON，fail-closed
        回退原版表达式）；far 侧 0.107→0.057ms（1.89×）。
        """
        if self._km_centroids is None:
            return None
        idx_sub = self._km_dims if self._km_dims is not None \
            else self._subspace_indices(q.device)
        H = q.shape[0]
        Hkv = self._km_centroids.shape[0]
        G = H // Hkv
        if (_HAS_TRITON
                and os.environ.get("TLI_E131_TOKSCORE_TRITON", "1")
                not in ("0", "false", "False")
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
                pass   # fail-closed：下方原版表达式（值同原版路径）
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

        【E131-S2（原 fast.py override 并入）】Triton 全融合同构（AGG=mean
        对齐 sf_g 量纲）；near 侧 0.062→0.047ms（1.32×）。
        """
        if self._km_near_centroids is None:
            return None
        idx_sub = self._km_near_dims
        H = q.shape[0]
        Hkv = self._km_near_centroids.shape[0]
        G = H // Hkv
        if (_HAS_TRITON
                and os.environ.get("TLI_E131_TOKSCORE_TRITON", "1")
                not in ("0", "false", "False")
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
        if sm is None:
            raise ValueError(
                "moba 臂缺少 score_moba；需 prepare_index/compute_score "
                "走 moba_gate 路径")
        sm = rearrange(sm, "b qt (h g) kt -> b qt h g kt", g=self.group_size).mean(dim=-2)
        sf = score_dict["score_fine"]
        K2 = min(sf.shape[-1], self.args.tia_level2_topk)
        nb_sel = max(1, K2 // bs)
        sink_tok = self.sink_tokens              # E135：强制区宽单一事实源
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
        if (_os.environ.get("TLI_DEBUG", "").strip().lower() not in ("", "0", "false")
                and self.layer_idx == 1):
            n_sel = int(m[..., :tok_hi].sum(dim=-1).float().mean().item())
            print(f"[MObadb] nb_sel={nb_sel} sel_total={n_sel}", flush=True)
        return m

    def compute_mask(self, q_ids, score_dict):
        # 【E178】fail-closed：TLI_L2_BYPASS 旁路仅在 compute_mask_graph
        # 实现。eager 路径混跑该 env 会静默跑成细筛口径（实验臂数字
        # 口径错配）→ 显式拒绝执行。
        if os.environ.get("TLI_L2_BYPASS", "").strip() == "1":
            raise NotImplementedError(
                "TLI_L2_BYPASS=1 仅支持 graph 路径（compute_mask_graph）；"
                "eager compute_mask 未实现细筛旁路，fail-closed 拒绝执行")
        oracle_far_budget = None
        if self._e151c_quota is not None and self._e151c_on:
            # layer_idx is injected after construction. Never silently use
            # layer zero or Python's negative indexing for a missing identity.
            layer = self.layer_idx
            if type(layer) is not int or not 0 <= layer < len(self._e151c_quota):
                raise ValueError(
                    f"E151c layer_idx must be an integer in [0, {len(self._e151c_quota)}), "
                    f"got {layer!r}"
                )
            oracle_far_budget = self._e151c_quota[layer]
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
        # 【E156 精筛候选化】compute_score 返回稀疏 score_fine [1,1,H,R_cand]
        # + cand_idx（候选 token 绝对位置，升序）时，先 scatter 回全宽
        # [1,1,H,T] 再走原消费逻辑。未选位置保持 -inf（exp(-inf)=0 精确不
        # 贡献 softmax/max/mean 归约 → 与现状 where 屏蔽后的归约集合逐位
        # 等价）。σ/ccluster 臂因全宽原始分语义已由 compute_score 侧门控
        # fail-closed 走全宽路径，本分支只被生产默认路径触发。
        cand_idx = score_dict.get("cand_idx")
        _cw = score_dict.get("cand_width") if cand_idx is not None else None
        _cand_len = score_dict.get("cand_len")
        _vl = score_dict.get("valid_length")
        # 【AKO iter-2】fused 重建门：cand_width/cand_len/valid_length 齐备
        # 且均 host int（无 .item()/nonzero 同步需求）→ 跳过全宽 scatter
        # 重建，S_width/S_real 全 host 口径，全宽化推迟到 L1 topk_mask
        # 产出后的单次静态形状哨兵 scatter（见下方 where 站点）完成。
        # cand_idx 存在 ⇒ compute_score 侧生产门（L1947-1949）已排除
        # σ/ccluster/moba/async/per_q_head 全部全宽 sf_raw 消费臂；本门
        # 再硬挡 compute_mask 内的 shape 消费者（sinkswa_only）与合成
        # 回放 harness（cand_width/cand_len/valid_length 缺失或非 host
        # int）→ 回退原重建路径，逐位不变。
        # 【AKO iter-3】新增 cand_gqa_shared 契约检查：compact softmax 的
        # kv-head view 折叠依赖候选行组内共享不变量（仅 _gather_fine_cand
        # 单源生产恒成立）——契约缺位（合成回放）一律回退 E156 重建。
        _fused_rebuild = (
            cand_idx is not None
            and isinstance(_cw, int) and not isinstance(_cw, bool)
            and _cand_len is not None
            and isinstance(_vl, int) and not isinstance(_vl, bool)
            and score_dict.get("cand_gqa_shared") is True
            and not self.sinkswa_only)
        if cand_idx is not None and not _fused_rebuild:
            # 【AKO iter-20】kv 空间候选（生产标记）→ q-head 按需展开
            # （本分支消费者按 q-head 行语义 scatter；冷路径可付拷贝）。
            if score_dict.get("cand_kv_space"):
                cand_idx = repeat(
                    cand_idx, "kv r -> (kv g) r", g=self.group_size)
                if _cand_len is not None:
                    _cand_len = repeat(
                        _cand_len, "kv -> (kv g)", g=self.group_size)
            S_width = int(_cw) if _cw is not None else \
                (int(cand_idx.max().item()) + 1)
            # 【E156c】取消 `score_fine.shape[-1] != S_width` 形状守卫：
            # 候选矩阵列是「候选秩」而非 token 索引（per-head 候选按
            # L1∪bypass∪sink∪swa 拼接），列序与 token 序只在候选集恰好
            # 全覆盖且有序时重合——S=kt·bs 整块边界（候选 R_max 恰等于
            # S_width，如 4096）时形状相等但列序 ≠ token 序，旧守卫跳过
            # 重建会让下游把候选秩当 token 索引 → mask 错位。cand_idx
            # 存在即无条件 scatter 重建（块∪强制区重复索引 scatter 幂等
            # 同值；R_max<S_width 与 R_max==S_width 一视同仁）。
            # 【E156b】per-head 重建：cand_idx [H,R_max]（q-head 空间），
            # cand_len [H] 门控——只写实长前缀（pad sentinel 0 不污染
            # token 0 的真实 sink 分；块∪强制区重复索引 scatter 幂等同值）。
            sf_full = torch.full(
                (score_fine.shape[0], score_fine.shape[1],
                 score_fine.shape[2], S_width),
                float("-inf"), dtype=score_fine.dtype,
                device=score_fine.device)
            _H_h = score_fine.shape[2]
            _R_max = cand_idx.shape[1]
            _hidx = torch.arange(_H_h, device=score_fine.device
                                 ).view(-1, 1).expand(_H_h, _R_max)
            if _cand_len is None:
                # 防御：无 cand_len（外部合成 score_dict 回放 harness）
                # → 假定无 padding（逐位兼容旧 1D 语义）。
                _cand_len = torch.full(
                    (_H_h,), _R_max, dtype=torch.long,
                    device=score_fine.device)
            _rmask = (torch.arange(_R_max, device=score_fine.device
                                   ).view(1, _R_max)
                      < _cand_len.view(-1, 1))
            _ih = _hidx[_rmask]
            _it = cand_idx[_rmask]
            sf_full[0, 0, _ih, _it] = score_fine[0, 0][_rmask]
            score_dict["score_fine"] = sf_full
            score_fine = sf_full
        if _fused_rebuild:
            # score_fine 保持稀疏 [1,1,H,R_max]；075 pad 段语义由 fused
            # scatter 的 `cand_idx < S_real` 门承载（越界候选不写入 →
            # 目标位保持 -inf），无需整段写 -inf。
            S_width = int(_cw)
            S_real = min(int(_vl), S_width)
        else:
            S_width = score_fine.shape[-1]
            S_real = S_width if _vl is None else min(int(_vl), S_width)
            if _vl is not None and S_real < S_width:
                # 原地写 score_dict：p（softmax 源）、σ 分支与 ccluster 分支的
                # sf_raw 均从 score_dict["score_fine"] 现读 → 同一份屏蔽全生效
                score_dict["score_fine"][..., S_real:] = float("-inf")
        # ---- E135a：sinkswa_only 归因臂（用户 2026-10-13 批准）----
        # mask 只含正交强制区（sink 头 sink_tokens + swa 尾 swa_tok），
        # mid 全丢、L1/L2 两级索引分数全旁路（prepare_index/compute_score
        # 照算——本臂测质量归因不测速度，分数不进任何 mask 路径即语义等价
        # 于「纯 sink+swa」基线）。D' 层同样只给强制区（归因臂逐层统一，
        # far 池被旁路本身无意义）。075：[S_real, S_width) padding 段保持
        # False；形状与共享口径最终 mask 一致（[b,qt,Hkv,T]，消费端
        # eager_decoding 按 GQA 组展开同款）。
        if self.sinkswa_only:
            shape = list(score_fine.shape)          # [b, qt, H, T]
            shape[2] = shape[2] // self.group_size   # → Hkv（互斥守卫保证非 per_q_head）
            m = torch.zeros(shape, dtype=torch.bool,
                            device=score_fine.device)
            m[..., :min(self.sink_tokens, S_real)] = True
            m[..., max(0, S_real - self.sliding_window_size):S_real] = True
            return m
        # ---- E103：kv-head 共享消融----
        # 共享口径（默认）：组内 mean 聚合到 kv-head 级 [1,1,Hkv,kt]；
        # per_q_head：旁路聚合，保留 per-q-head 分数 [1,1,H,kt] 直接进 topk
        # （topk 沿最后一维，H 维天然独立——32 个 q-head 各选各的块/token）。
        # 【AKO iter-12】L1 复用外门前移 + score_coarse GQA mean 稳态免跑：
        # iter-6 复用门判决的补完——复用命中臂（生产同调用稳态）下游对
        # mean 结果零消费（i_f/i_n/topk_mask 全取自 l1_sel dict；sc_far/
        # sc_near/sc_pool/zeros_like 全在 not _l1_reuse 臂）→ mean
        # （reduce_kernel<512,1> ~3.4us/步）与其后的 skip_far 区间写均死
        # 代码。外门（与原 L2634 处逐字同式；kt 取 mean 前后同值、bs/e64
        # 表达式同源）任一不符（合成回放/几何漂移/非 fused 臂/per_q_head
        # 生产门天然缺席）→ mean 照跑逐位不变。内门（e64 池端点）失败的
        # 补偿路径在下方 e64 分支 `if not _l1_reuse:` 前 fail-closed 补跑
        # （保真复刻旧序：mean → skip_far 区间写；[-1]=inf 与 avg mean 在
        # 外门通过时点旧路径本就跳过，同序不复刻，见 L2643 注释）。
        _l1_rs = score_dict.get("l1_sel") if _fused_rebuild else None
        _l1_reuse = (
            isinstance(_l1_rs, dict)
            and _l1_rs.get("kt") == score_coarse.shape[-1]
            and _l1_rs.get("bs") == self.args.tia_block_size
            and _l1_rs.get("l1_s_real") == S_real
            and _l1_rs.get("e64_partition") == (self.alpha > 0 and self.beta > 0)
            and "topk_mask_block" in _l1_rs
        )
        _sc_mean_skipped = False
        if not self.per_q_head:
            if _l1_reuse:
                _sc_mean_skipped = True
            else:
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
            sink_tok = self.sink_tokens          # E135：强制区宽单一事实源
            mid_len = max(0, S_real - sink_tok - swa_tok)
            if self.nt_near_len is not None:
                # E155 固定几何：near 区宽显式整数（α·mid_len 规则被覆盖）
                near_len_dyn = max(bs, int(self.nt_near_len))
            else:
                near_len_dyn = max(bs, int(self.alpha * mid_len))  # near 只算 mid 部分（不含 swa）
            near_base = S_real - swa_tok   # near 右界 = swa 起点（B10；真实 S 口径 F5）
        elif self.far_select != "4bit" and self.alpha == 0 and self.beta == 0:
            # TASK.md L231 严格口径：cluster 组合 (0,0) 单池点 = 纯 cluster 全 mid
            near_len_dyn = swa_tok
            near_base = kt * bs
        else:
            near_len_dyn = self.near_len
            near_base = kt * bs
        # E135：池左界 = floor(sink_tokens/bs)（默认 2 ≡ sink_blocks，逐位不变）；
        # near_blks 保底同步用池左界（子块 sink 时 near 池可含 sink 所在块）
        near_blks = max(self._sink_pool_lo_blk, (near_base - near_len_dyn) // bs)
        far_lo_blk, far_hi_blk = self._sink_pool_lo_blk, near_blks  # 远端块区间（含 far_hi 前一块）
        # near 池上界 = swa 起点（swa 块完全排除出双池，纯靠 mask 强制）
        swa_lo_blk = max(near_blks, kt - max(1, swa_tok // bs))

        # ---- D'：跳层 → 远端块全部 -inf（保留 sink + 近端）----
        # 【AKO iter-12】复用命中臂（_sc_mean_skipped）下本写目标为死代码
        # （mean 结果无消费者）且免跑后 dict 原张量绝不被触碰（旧路径写
        # 进 mean 新张量、同样不触 dict）→ 两路 dict 张量均零变异。
        if self.skip_far and not _sc_mean_skipped:
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
        # 预算，sink/swa 强制计入 K2 总预算（不占 K2_mid 选择名额，下方 L1368-1371 口径）。修掉 B1 修复
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

        # 【AKO iter-6】L1 选择复用门：_gather_fine_cand 内 _select_l1_blocks
        # 已在 compute_score 侧用同源 score_coarse（同张量 lineage、同 GQA
        # mean 聚合、逐字同实现）完成同一 L1 选择 → 下方 e64/非 e64 两分支
        # 的 2×topk + cat + zeros+scatter + clone + bypass GPU 写全链免重跑，
        # 直接复用 bit-identical 的 topk_mask/i_f/i_n。硬门：fused 臂（生产
        # 单源契约已在 compute_score 侧排除 σ/ccluster/moba/async/per_q_head）
        # + kt/bs/S_real/e64_partition 逐项一致 + 键齐备；任一不符 → 下方
        # 原内联 L1 段逐位不变（合成回放/几何漂移 fail-closed）。
        # 【AKO iter-12】外门已前移至 GQA mean 前（见 E103 消融段上方）——
        # 此处直接复用前置处算得的 _l1_rs/_l1_reuse（kt/bs/e64 前后取值
        # 恒等：mean 不改末维宽，bs/e64 表达式同源），几何单源不二算。
        if not _l1_reuse:
            # TIA 语义：当前块强制（复用臂生产侧 _select_l1_blocks 已在
            # 其聚合副本上完成同置位；本张量复用臂下无后续消费）
            score_coarse[..., -1] = float("inf")
        k1 = self.args.tia_level1_topk
        # E109a：avg 分数源提取提前到分支外（单池路径也要按 far_method 选分数源）
        score_coarse_avg = score_dict.get("score_coarse_avg")
        if not _l1_reuse and score_coarse_avg is not None \
                and not self.per_q_head:
            score_coarse_avg = rearrange(
                score_coarse_avg, "b qt (h g) kt -> b qt h g kt", g=self.group_size
            ).mean(dim=-2)
        # 【AKO iter-1 候选域压缩前置】e64 分支内填充（L1 块级索引 + 池端点）；
        # 非 e64 臂保持 None/() → _compact_topk 拒绝 → 旧全宽 topk 逐位不变。
        _l1_far_blk = _l1_near_blk = None
        _far_pool_hi_tok = _near_pool_hi_tok = 0
        bypass_blks = ()
        if e64_partition:
            # ---- E64 L1 双池：far 池=minmax 上界分（α 区外），near 池=avg 分（近区）----
            if self.nt_near_pages is not None:
                # E155 固定配额：nb_near 显式整数（β 语义被覆盖）
                nb_near = max(0, int(self.nt_near_pages))
            else:
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
            # 【E155b 边界页保送（新口径默认开）】sink 跨界页 + swa 跨界页
            # 移出双池、直通 L2 细筛候选。旧口径下 sink 页靠「minmax 上界被
            # sink 离群值撑大」隐性赢得 L1（暗通道：页统计混保护/竞争 token，
            # 挤占其他 far 页配额）；swa 跨界页同理混 swa 保护 token。新口径
            # 显式保送：双池页统计纯竞争 token、保送页不占 K1 配额（nb_far/
            # nb_near 照旧从收缩后的池内选）、保送页 token 仍在 L2 按配额
            # 竞争（sink/swa 强制 token 语义不变，L2 分区边界天然覆盖保送页
            # ——sink 页在 far 段 [far_tok_lo, far_tok_hi)、swa 页在 near 段）。
            # 跨界判定（token 域单一事实源）：sink_tok % bs != 0 → sink 跨界页
            # = far_lo_blk（floor 口径下恰为 far 池第一块；sink 整页对齐时本就
            # 被池左界排除，无跨界页）；swa_lo_tok % bs != 0 → swa 跨界页 =
            # swa_lo_tok // bs（mid 侧最后一块；swa 整块对齐时被池右界排除）。
            # boundary_bypass=false 逐位回旧口径（对拍/消融臂）。
            bypass_blks = ()
            far_pool_lo, far_pool_hi = far_lo_blk, far_hi_blk
            near_pool_lo, near_pool_hi = near_blks, swa_lo_blk
            if self.boundary_bypass:
                _bp = []
                if sink_tok % bs != 0:
                    _bp.append(far_lo_blk)
                _swa_lo_tok = max(0, S_real - swa_tok)
                if _swa_lo_tok % bs != 0:
                    _cb = _swa_lo_tok // bs
                    if _cb >= far_lo_blk:
                        _bp.append(_cb)
                bypass_blks = _bp
                for _b in _bp:
                    # 双池区间端点收缩（跨界页恒位于区间端：sink 页=far 左端、
                    # swa 页=near 右缘或退化短序列时 far 右缘），池内块不受影响。
                    # GPT 审计 2026-10-19（端点隐性假设显式化）：保送页必在
                    # 池端点——当前几何三种形态（near 非空/空/短序列退化）已
                    # 逐一验证恒成立；未来几何变化若破坏该假设，非端点分支
                    # 会静默多收缩一整段池，fail-closed 断言把契约变成硬错误。
                    if far_pool_lo <= _b < far_pool_hi:
                        if _b == far_pool_lo:
                            far_pool_lo = _b + 1
                        elif _b == far_pool_hi - 1:
                            far_pool_hi = _b
                        else:
                            raise ValueError(
                                f"boundary bypass 页 {_b} 非 far 池端点 "
                                f"[{far_pool_lo},{far_pool_hi})——池收缩几何"
                                "假设被破坏（当前几何下保送页恒位于端点）")
                    if near_pool_lo <= _b < near_pool_hi:
                        if _b == near_pool_hi - 1:
                            near_pool_hi = _b
                        elif _b == near_pool_lo:
                            near_pool_lo = _b + 1
                        else:
                            raise ValueError(
                                f"boundary bypass 页 {_b} 非 near 池端点 "
                                f"[{near_pool_lo},{near_pool_hi})——池收缩几何"
                                "假设被破坏（当前几何下保送页恒位于端点）")
            # 【AKO iter-6】几何一致性硬门：dict 几何与本步 host 几何逐项
            # 对齐才复用（生产同调用内恒真；任何漂移 fail-closed 落回下方
            # 原 GPU 选择段，逐位不变）。
            if _l1_reuse:
                if (_l1_rs["nb_near"] == nb_near
                        and _l1_rs["nb_far"] == nb_far
                        and tuple(_l1_rs["bypass_blks"]) == tuple(bypass_blks)
                        and _l1_rs["far_pool_lo"] == far_pool_lo
                        and _l1_rs["far_pool_hi"] == far_pool_hi
                        and _l1_rs["near_pool_lo"] == near_pool_lo
                        and _l1_rs["near_pool_hi"] == near_pool_hi):
                    i_f = _l1_rs["i_f"]
                    i_n = _l1_rs["i_n"]
                    topk_mask = _l1_rs["topk_mask_block"]
                else:
                    _l1_reuse = False
            if not _l1_reuse:
                if _sc_mean_skipped:
                    # 【AKO iter-12】外门过而内门败（生产不可达——同调用内
                    # 几何逐项恒一致，仅未来几何漂移可入）：补跑被免的 mean +
                    # skip_far 区间写，张量状态与旧路径该点逐位一致（mean 新
                    # 张量，dict 原张量两路均零变异）。
                    score_coarse = rearrange(
                        score_coarse, "b qt (h g) kt -> b qt h g kt",
                        g=self.group_size
                    ).mean(dim=-2)  # [1,1,Hkv,kt]
                    if self.skip_far:
                        score_coarse[..., far_lo_blk:far_hi_blk] = float("-inf")
                    _sc_mean_skipped = False
                if self.far_method == "avg" and score_coarse_avg is not None:
                    sc_far = score_coarse_avg[..., far_pool_lo:far_pool_hi]
                else:
                    sc_far = score_coarse[..., far_pool_lo:far_pool_hi]
                i_f = torch.topk(sc_far, min(nb_far, sc_far.shape[-1]), dim=-1).indices + far_pool_lo
                # near 池上界 = swa_lo_blk：swa 块（含当前块）完全排除出双池，
                # 纯靠 mask 强制（p[-swa:]=1）——正交区不占创新管线配额
                # near 池分数源：avg（默认=块均值）或 minmax（块上界，mminmax 组合）
                if self.near_method == "minmax" or score_coarse_avg is None:
                    sc_near = score_coarse[..., near_pool_lo:near_pool_hi]
                else:
                    sc_near = score_coarse_avg[..., near_pool_lo:near_pool_hi]
                sc_near = sc_near.clone()
                if sc_near.shape[-1] > 0:
                    i_n = torch.topk(sc_near, min(nb_near, sc_near.shape[-1]), dim=-1).indices + near_pool_lo
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
                # E155b：保送页 OR 进 L1 候选集（全 head 广播；indices 不含保送
                # 页 → 不占 K1 配额；其 token 在 L2 细筛按 far/near 配额竞争）
                for _b in bypass_blks:
                    topk_mask[..., _b] = True
            # 【AKO iter-1 捕获】L2 复用 i_f/i_n 名字前保存块级索引（含
            # far_pool_lo/near_pool_lo 偏移后的绝对块号）与池收缩后端点。
            # 池端点 host 不变量：far_pool_hi·bs ≤ far 段上界、
            # near_pool_hi·bs ≤ swa_lo_tok（保送页收缩/对齐时成立；
            # 退化几何由 _compact_topk 的 pool_hi_tok>seg_hi 守卫回退）。
            _l1_far_blk, _l1_near_blk = i_f, i_n
            _far_pool_hi_tok = far_pool_hi * bs
            _near_pool_hi_tok = near_pool_hi * bs
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
                      f"nb_far={nb_far} nb_near={nb_near} bypass={list(bypass_blks)} "
                      f"i_f_blk_min={i_f_min} i_f_blk_max={i_f_max} "
                      f"i_n_blk_min={i_n_min} i_n_blk_max={i_n_max}", flush=True)
        else:
            # E109a 修复：单池（α=0 或 β=0）L1 分数源按 far_method 选择
            # （原实现固定 score_coarse=minmax → aavg/mavg(0,0) 单池实际跑成纯 minmax）
            # 【AKO iter-6】复用臂直取生产侧 topk_mask_block（几何一致性外门
            # 已核 kt/bs/S_real/e64/near_blks/far_lo/far_hi/swa_lo；skip 臂
            # k1 收缩在生产侧同 shrink 算过）。
            if _l1_reuse:
                topk_mask = _l1_rs["topk_mask_block"]
            else:
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
        # 【AKO iter-3】块级 L1 掩码快照（token repeat 物化前）——fused
        # compact 路径用「候选块 = cand_idx//bs 的块掩码查表」替代
        # [1,1,H,kt*bs] token 级 bool repeat（token ∈ 选中块 ⟺ 块掩码
        # 选中，含保送页 OR，逐位同集）。
        _topk_blk = topk_mask
        # 【AKO iter-3】fused compact 模式跳过 [1,1,H,kt*bs] token 级 bool
        # repeat 物化与 async prev_mask 交换（fused 生产门已排除 async 臂；
        # token 掩码在本模式的唯一消费者——p 哨兵 scatter 的 _keep——已改由
        # 上方块掩码快照查表得到，见下方 fused 分支）。非 fused 臂逐位不变。
        if not _fused_rebuild:
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

        _sf_w = score_fine.shape[-1]
        if _fused_rebuild:
            # 【AKO iter-5】keep 链 + 哨兵 scatter 融合为单 Triton kernel
            # （_ako_rebuild_scatter_kernel）：L1 块与完整保护区取并集；
            # 保护区必须进入 GQA softmax 分母。融合 arange+compare+and+div+clamp+
            # index_select+gather+where+scatter_ 九个小 kernel 的逐元素
            # 链单趟化。纯整数/布尔/拷贝语义、无浮点归约参与 → 逐位
            # 等价（重复索引同值幂等；idx<S_real 门无条件施加——
            # S_real≥S_width 时恒真，与旧条件分支逐位同）。
            # 【AKO iter-8】无效槽路由 = 牺牲位 DUMMY[h]（Triton 路径）
            # / 尾列哨兵 W 切片（回退路径），可见域两路逐位同。
            # 非连续输入回退 iter-4 torch 链。
            _R = cand_idx.shape[-1]
            _H = score_fine.shape[2]
            _sf2 = score_fine[0, 0]
            _blk2 = _topk_blk[0, 0]
            if (_HAS_TRITON and score_dict.get("cand_kv_space")
                    and cand_idx.is_contiguous()
                    and _cand_len.is_contiguous() and _blk2.is_contiguous()):
                # 【AKO iter-20】kernel CAND/CLEN 行地址按 h//g 折叠——仅
                # kv 空间候选（生产 cand_kv_space 标记）可用；缺标记
                # （合成回放 q-head 口径）回退 torch 链逐位不变。
                # 【AKO iter-8】哨兵列退役：[W] 连续重建目标（keep 位同值
                # 同址、未访位 -inf 预填充同值 → 重建结果逐位同旧）+ 微型
                # 牺牲位缓冲吸收无效槽写入（内容永不读）；score_fine 不再
                # 是 [W+1] 的行距切片 → softmax 免 strided 触发的 4MB
                # contiguous clone（aten::contiguous→clone→copy_ 微测
                # ~22us/步，共居口径 direct_copy ~8.8us）。
                sf_full = torch.full(
                    (score_fine.shape[0], score_fine.shape[1],
                     score_fine.shape[2], S_width),
                    float("-inf"), dtype=score_fine.dtype,
                    device=score_fine.device)
                _dummy = torch.empty(
                    (_H, 1), dtype=score_fine.dtype,
                    device=score_fine.device)
                _n = _H * _R
                _ako_rebuild_scatter_kernel[(
                    triton.cdiv(_n, 1024),)](
                    cand_idx, _sf2, _blk2, _cand_len, sf_full[0, 0],
                    _dummy,
                    _R, _n, S_width, S_real, bs, kt, self.group_size,
                    self.sink_tokens, swa_tok,
                    _sf2.stride(0), _sf2.stride(1),
                    cand_idx.stride(0), cand_idx.stride(1),
                    _blk2.stride(0), _blk2.stride(1),
                    BLOCK=1024, num_warps=4)
                score_fine = sf_full
            else:
                sf_full = torch.full(
                    (score_fine.shape[0], score_fine.shape[1],
                     score_fine.shape[2], S_width + 1),
                    float("-inf"), dtype=score_fine.dtype,
                    device=score_fine.device)
                # iter-4 torch 链回退（非连续/无 Triton）。
                # 【AKO iter-20】kv 空间候选冷路径按需展开回 q-head 空间
                # （expand 后 reshape 触发拷贝，冷路径可付）。
                if score_dict.get("cand_kv_space"):
                    cand_idx = repeat(
                        cand_idx, "kv r -> (kv g) r", g=self.group_size)
                    _cand_len = repeat(
                        _cand_len, "kv -> (kv g)", g=self.group_size)
                _hkv = getattr(self, "_ako_hkv_row", None)
                if _hkv is None or _hkv.device != cand_idx.device \
                        or _hkv.numel() != _H:
                    _hkv = torch.arange(
                        _H, device=cand_idx.device) // self.group_size
                    self._ako_hkv_row = _hkv
                _bid = torch.div(cand_idx, bs, rounding_mode="floor").clamp(
                    max=kt - 1)
                _keep = _blk2.index_select(
                    0, _hkv).gather(-1, _bid)
                _rmask2 = (torch.arange(_R, device=cand_idx.device
                                        ).view(1, _R) < _cand_len.view(-1, 1))
                _protected = ((cand_idx < self.sink_tokens)
                              | (cand_idx >= max(0, S_real - swa_tok)))
                _keep = (_keep | _protected) & _rmask2
                if S_real < S_width:
                    _keep = _keep & (cand_idx < S_real)
                _safe = torch.where(
                    _keep, cand_idx, S_width)      # 无效槽 → 尾列哨兵
                sf_full[0, 0].scatter_(1, _safe, _sf2)
                score_fine = sf_full[..., :S_width]
        else:
            if topk_mask.shape[-1] > _sf_w:
                topk_mask = topk_mask[..., :_sf_w]
            elif topk_mask.shape[-1] < _sf_w:
                # 【阶段 2a 固定宽度】块统计有效区化后 kt_eff=ceil(S_real/bs) →
                # topk_mask 宽 kt_eff*bs 可小于全宽 score_fine（S_cap）——pad
                # False 补齐（pad 列 score_fine 已被上方显式置 -inf，where 后
                # 仍 -inf，softmax 不归约，逐位等价变长路径）。
                topk_mask = F.pad(topk_mask,
                                  (0, _sf_w - topk_mask.shape[-1]), value=False)
            # The normalization domain includes every protected token, even
            # when its page was not selected by L1 (including full-width fallback).
            _tokens = torch.arange(_sf_w, device=score_fine.device)
            _protected = ((_tokens < self.sink_tokens)
                          | (_tokens >= max(0, S_real - swa_tok))) & (_tokens < S_real)
            score_fine = torch.where(topk_mask | _protected, score_fine, float("-inf"))
        _G_ = self.group_size
        _sm = F.softmax(score_fine, dim=-1)
        if (not self.per_q_head and _HAS_TRITON and _sm.is_contiguous()
                and _sm.dtype == torch.bfloat16
                and isinstance(_G_, int) and _G_ > 0
                and (_G_ & (_G_ - 1)) == 0
                and _sm.shape[2] % _G_ == 0):
            # 【AKO iter-11】softmax 输出 nan_to_num + GQA 组 mean 单趟
            # 融合（_ako_gqa_mean_kernel）：strided reduce + nan_to_num 两
            # kernel → 1。等价性论证见 kernel docstring；守卫不齐（per_q_head
            # / G 非 2 幂 / 非连续）→ 回退 torch 链逐位不变。
            _Wm = _sm.shape[-1]
            _Rm = _sm.shape[2] // _G_
            p = torch.empty(
                (_sm.shape[0], _sm.shape[1], _Rm, _Wm),
                dtype=_sm.dtype, device=_sm.device)
            # 【AKO iter-13】swa 强制区 [max(0,S_real−swa_tok), S_real)
            # 直写 1.0 折叠进 mean kernel——消独立 setitem fill kernel
            # （FillFunctor ~1.8us/步）+ torch 2.8 标量 setitem 固有
            # lift_fresh→fill_→item CPU 分发链（~10us host，共居下被 GPU
            # 积压掩盖但计入 launch 开销）。等价性见 kernel docstring；
            # S_real/swa_tok 非 host int → 传 (0,0)（forced 恒空）+ 下方
            # 旧 fill 行照跑 fail-closed 逐位不变。
            if isinstance(S_real, int) and isinstance(swa_tok, int):
                _swa_lo_t = S_real - swa_tok
                if _swa_lo_t < 0:
                    _swa_lo_t = 0
                _swa_s_t = S_real
                _swa_filled = True
            else:
                _swa_lo_t = 0
                _swa_s_t = 0
                _swa_filled = False
            _ako_gqa_mean_kernel[(_Rm, triton.cdiv(_Wm, 1024))](
                _sm[0, 0], p[0, 0], _Wm,
                _swa_lo_t, _swa_s_t,
                G=_G_, BLOCK=1024, num_warps=4)
        else:
            p = _sm.nan_to_num(nan=0)
            if not self.per_q_head:
                # E103：共享口径组内 mean；per_q_head 保留 [1,1,H,T] 直接 topk
                p = rearrange(p, "b qt (h g) kt -> b qt h g kt", g=self.group_size).mean(dim=-2)
            _swa_filled = False
        # 【075 修复】SWA 强制区按真实 S：p[..., -swa:] 在投影路径落 padded
        # 尾部（消费端裁剪后真实序列尾只保 swa−(Tpad−S) 个 token，S=6145
        # 实测 65/128）。改为 [S_real−swa, S_real)；非投影路径 S_real==宽度
        # → 与旧 p[..., -swa:] 逐位等价（S_real<swa 时两者同为全宽度置位）。
        # 【AKO iter-13】host int 几何 + fused mean 臂已由 kernel 直写覆盖
        # （_swa_filled）→ 本 fill 免跑；其余臂照旧。
        if not _swa_filled:
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
            sink_tok = self.sink_tokens   # E135：σ 锚 = 强制区宽（默认 == far_tok_lo 逐位不变）
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
            # 正交强制区直接置位（必选，计入 K2 预算）：sink 头部 + swa 尾部
            # 075：swa 尾部 = 真实 S 尾（非 padded 尾）；padding 段保持 False
            m[..., :sink_tok] = True
            m[..., swa_lo_tok:S_real] = True
            import os as _os
            if (_os.environ.get("TLI_DEBUG", "").strip().lower() not in ("", "0", "false")
                    and self.layer_idx == 1):
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
            sink_tok = self.sink_tokens   # E135：强制区宽 = token 级（子块 sink 时
            # far_tok_lo=floor(sink/bs)·bs < sink_tokens，绑池左界会让头
            # sink<bs 段强制区丢失——sink=16 实测 head64=0 即此 bug）。
            # mid topk 域仍从 far_tok_lo 起（floor 口径：sink 所在块参与
            # 竞争），[0, sink_tokens) 可能被 topk 重复选中，仅浪费名额
            # 不破坏并集语义。
            # 075：swa 起点/强制区宽度按真实 S——投影路径 p 宽 = Tpad，
            # 旧口径 swa 强制区落 padded 尾且 K2_mid 把 padding 段当 swa 扣除
            swa_lo_tok = max(0, S_real - self.sliding_window_size)
            K2_mid = max(0, K2 - sink_tok - (S_real - swa_lo_tok))
            # ---- E122：γ=off 自由竞争（用户 2026-10-10 指令，cavg 探索臂）----
            # 取消 near/far 配额分割：全部 mid 候选在统一预算 K2_mid 内单池
            # topk 竞争。语义边界（全部保持不变）：
            #   * sink/swa 保护带：正交强制区必选，计入 K2 预算（下方直接置位）；
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
                    # 上界收到 swa_lo_tok：swa 强制区 token 不进竞争，计入 K2 预算
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
                # 正交强制区直接置位（必选，计入 K2 预算）：sink 头部 + swa 尾部
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
            # 【AKO iter-10】装配融合守卫：几何标量全 host int + p 连续 +
            # Triton 可用 → far/near 走 _raw 出口（cand_idx+sel），装配尾部
            # 单趟化（_ako_l2_mask_kernel）。任一守卫失败 → _raw=False →
            # far/near/装配全链与旧口径逐位相同（fail-closed）。
            _asm_fuse = (_HAS_TRITON and p.is_contiguous()
                         and isinstance(sink_tok, int)
                         and isinstance(swa_lo_tok, int)
                         and isinstance(S_real, int))
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
                        if self.nt_near_tokens is not None:
                            # E155 固定配额：nt_near 显式整数（γ 语义被覆盖）
                            nt_near = min(int(self.nt_near_tokens), K2_mid)
                        else:
                            nt_near = min(int(nb_near * bs * self.gamma), K2_mid)
                        # TASK.md L172 严格口径（与 E109a-γ 同步）：无 64 保底
                        far_budget = max(0, K2_mid - nt_near)
                        if oracle_far_budget is not None:
                            # E151c oracle 逐层配额终审：far 覆盖为该层 oracle 需求
                            far_budget = min(oracle_far_budget, K2_mid)
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
                        if self.nt_near_tokens is not None:
                            # E155 固定配额：nt_near 显式整数（γ 语义被覆盖）
                            nt_near = min(int(self.nt_near_tokens), K2_mid)
                        else:
                            nt_near = min(int(nb_near * bs * self.gamma), K2_mid)
                        far_budget = max(0, K2_mid - nt_near)
                        if oracle_far_budget is not None:
                            # E151c oracle 逐层配额终审（minmax/avg 臂实际走本分支）
                            far_budget = min(oracle_far_budget, K2_mid)
                    else:
                        far_budget = self.far_tokens
                    # 默认：远端与近端同用 4bit 精筛分数，仅在独立池内 topk
                    k2_far = min(far_budget, far_tok_hi - far_tok_lo, K2_mid)
                    # 【AKO iter-1 候选域压缩】far 段 p>0 support = L1 far 池
                    # 选中块 ∪ far 侧保送页 token（其余位 softmax(-inf)=0，
                    # k2_far ≤ 选中块 token 数时永不出线）→ 段全宽 topk 换
                    # 候选域 topk，选择集合等价、宽度 48960 → ~nb_far·bs。
                    i_f = self._compact_topk(
                        p, far_tok_lo, far_tok_hi, k2_far,
                        _l1_far_blk, _far_pool_hi_tok, bypass_blks, bs,
                        _raw=_asm_fuse)
                    if i_f is None:
                        far_p = p[..., far_tok_lo:far_tok_hi]
                        i_f = torch.topk(far_p, k2_far, dim=-1).indices \
                            + far_tok_lo
                    import os as _os
                    if (_os.environ.get("TLI_DEBUG", "").strip().lower() not in ("", "0", "false")
                            and self.layer_idx == 1):
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
                    # 【AKO iter-1 候选域压缩】near 段 p>0 support = L1 near
                    # 池选中块 ∪ near 侧保送页（swa 跨界页 < swa_lo_tok 部分）
                    # → 段全宽 topk 换候选域 topk（同 far 侧注释）。
                    k_near_eff = min(k2_near, swa_lo_tok - far_tok_hi)
                    i_n = self._compact_topk(
                        p, far_tok_hi, swa_lo_tok, k_near_eff,
                        _l1_near_blk, _near_pool_hi_tok, bypass_blks, bs,
                        _raw=_asm_fuse)
                    if i_n is None:
                        near_p = p[..., far_tok_hi:swa_lo_tok]
                        i_n = torch.topk(near_p, min(k2_near, near_p.shape[-1]),
                                         dim=-1).indices + far_tok_hi
            else:
                # 【AKO iter-10】far 为 _raw 形态（cand_f, sel_f）时空近端
                # 同走 raw：(cand_f, sel_f[..., :0])——回退路径 gather 出
                # [..., :0] 与旧 i_f[..., :0] 同形同 dtype 同空值；融合路径
                # kn=0 近端零置位（探针 trial1 判据）。
                if isinstance(i_f, tuple):
                    i_n = (i_f[0], i_f[1][..., :0])
                else:
                    i_n = i_f[..., :0]
            # 【AKO iter-10】L2 mask 装配尾部单趟化（_ako_l2_mask_kernel）：
            # back-gather（2× cand_idx.gather）+ zeros 填充 + 2× scatter_ +
            # sink/swa 强制区 → 1 launch。纯整数/幂等 bool 置位语义 → 逐位
            # 恒等。far/near 任一非 raw 形态（compact 守卫失败/全宽回退/
            # 簇分臂）→ 手动 gather 补齐后走旧 torch 链逐位不变。
            if isinstance(i_f, tuple) and isinstance(i_n, tuple):
                cand_f, sel_f = i_f
                cand_n, sel_n = i_n
                kf = int(sel_f.shape[-1])
                kn = int(sel_n.shape[-1])
                W = p.shape[-1]
                R = p.numel() // W if W else 0
                topk_mask = torch.empty(
                    p.shape, dtype=torch.bool, device=p.device)
                _ako_l2_mask_kernel[(R, triton.cdiv(W, 4096))](
                    cand_f, sel_f, cand_n, sel_n, topk_mask,
                    kf, kn, int(cand_f.shape[-1]), int(cand_n.shape[-1]),
                    W, sink_tok, swa_lo_tok, S_real,
                    KF_POW2=max(1, triton.next_power_of_2(kf)),
                    KN_POW2=max(1, triton.next_power_of_2(kn)),
                    BLOCK=4096, num_warps=4)
                return topk_mask
            if isinstance(i_f, tuple):
                i_f = i_f[0].gather(-1, i_f[1])
            if isinstance(i_n, tuple):
                i_n = i_n[0].gather(-1, i_n[1])
            topk_mask = torch.zeros_like(p, dtype=torch.bool)
            topk_mask.scatter_(-1, i_f, True)
            topk_mask.scatter_(-1, i_n, True)
            # 正交强制区直接置位（必选，计入 K2 预算）：sink 头部 + swa 尾部
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
            sink_tok = min(self.sink_tokens, topk_mask.shape[-1])  # E135：强制区宽单一事实源
            topk_mask[..., :sink_tok] = True
        return topk_mask

    # ---------------- 【AKO iter-1】L2 段 topk 候选域压缩 ----------------
    # p = softmax(where(L1 块掩码 ∪ 保送页, score_fine, -inf)) → p>0 的
    # support 恰为「L1 选中块 + 保送页」的 token；段内其余位恒 p=0（严格
    # 大于 0 的正候选数 = 选中块·bs + 保送页相交 token）。k ≤ 选中块 token
    # 数时 p=0 位两侧都永不出线 → 段全宽 topk 与候选域 topk 选择集合逐位
    # 等价，输入宽度从 O(S) 降到 O(K1·bs)。等价性前提不满足（k 超正候选
    # 下界 / 池端点越段 / 非 e64 单池臂 / 退化几何）时返回 None，调用方
    # 回退旧全宽 topk——各消融臂逐位不变。
    def _compact_topk(self, p, seg_lo, seg_hi, k, l1_blk, pool_hi_tok,
                      bypass_blks, bs, _raw=False):
        if (l1_blk is None or k <= 0 or seg_hi <= seg_lo
                or seg_hi > p.shape[-1] or pool_hi_tok > seg_hi
                or k > l1_blk.shape[-1] * bs):
            return None
        dev = p.device
        offs = getattr(self, "_ct_offs", None)
        if offs is None or offs.device != dev or offs.numel() != bs:
            offs = torch.arange(bs, device=dev, dtype=torch.int64)
            self._ct_offs = offs
        # 块升序展开成 token 索引（升序保 tie-break 的原 token 顺序；
        # L1 块 token 全部真实有限分 → p>0，构成正候选下界）
        lead = l1_blk.shape[:-1]
        nb = l1_blk.shape[-1]
        # 保送页与段相交 token（宿主侧段枚举；保送块已移出双池 → 与
        # L1 块无重复索引；bypass_blks ≤ 2 块（sink 跨界页 + swa 跨界页）
        # → 相交段 ≤ 2，段序 = 旧 cat 序逐位复刻）
        segs = []
        for b in bypass_blks:
            lo_b = max(b * bs, seg_lo)
            hi_b = min((b + 1) * bs, seg_hi)
            if hi_b > lo_b:
                segs.append((lo_b, hi_b))
        n_byp = sum(hi - lo for lo, hi in segs)
        R_cand = nb * bs + n_byp
        # 【AKO iter-15】排序+展开融合（_ako_cand_expand_sort_kernel）：
        # iter-9 l1_sort + iter-7 cand_expand 两 kernel → 1 launch。每列
        # 切片 program 冗余 load 未排序块号行 → 寄存器 pad 哨兵 tl.sort
        # 升序 → tl.gather 动态取 BLK[j]（clamp nb-1 防 bypass lane 越界）
        # → 展开/bypass/p.gather。互异块号升序唯一 → 与独立 kernel 写回
        # sorted blk 后展开逐位恒等。守卫不齐（无 tl.sort/tl.gather /
        # nb=1 退化（tl.sort 单元素编译 bug）/ nb>256 / 非连续 / >2 bypass
        # 段 / lead 与 p 不匹配）→ 回退 iter-9+iter-7 两 kernel 路径或
        # torch 链逐位不变。探针 5 形状 ×3 重复对拍全 PASS。
        if (_HAS_TRITON and _TL_HAS_SORT and _TL_HAS_GATHER
                and l1_blk.is_contiguous()
                and 2 <= nb and 1 << (nb - 1).bit_length() <= 256
                and len(segs) <= 2 and p.is_contiguous()
                and tuple(p.shape[:-1]) == tuple(lead)):
            S_w = p.shape[-1]
            R_l = p.numel() // S_w
            ba = segs[0] if len(segs) > 0 else (0, 0)
            bb = segs[1] if len(segs) > 1 else (0, 0)
            cand_idx = torch.empty(
                (R_l, R_cand), dtype=torch.int64, device=dev)
            cand_p = torch.empty(
                (R_l, R_cand), dtype=p.dtype, device=dev)
            _ako_cand_expand_sort_kernel[(R_l, triton.cdiv(R_cand, 256))](
                l1_blk, p, cand_idx, cand_p,
                R_cand, nb, S_w,
                ba[0], ba[1], bb[0], bb[1],
                S_w,
                BS=bs, NB_POW2=1 << (nb - 1).bit_length(),
                BLOCK=256, num_warps=4)
            cand_idx = cand_idx.view(*lead, R_cand)
            cand_p = cand_p.view(*lead, R_cand)
        else:
            # 【AKO iter-9】升序排序内联 Triton 化（_ako_l1_sort_kernel）：
            # L1 块号来自 topk indices 互异 → 升序唯一 → 与 torch.sort 逐位
            # 恒等，消 2×(radixSortKVInPlace<long>+bitonic+DtoD 首拷)
            # ~19us/步；nb>256 / 非连续 / 无 tl.sort 回退 torch.sort 逐位不变。
            if (_HAS_TRITON and _TL_HAS_SORT and l1_blk.is_contiguous()
                    and 0 < nb and 1 << (nb - 1).bit_length() <= 256):
                _nbp = 1 << (nb - 1).bit_length()
                _Rl = l1_blk.numel() // nb
                blk = torch.empty((_Rl, nb), dtype=torch.int64, device=dev)
                _ako_l1_sort_kernel[(_Rl,)](
                    l1_blk, blk, nb, nb, nb,
                    NB_POW2=_nbp, num_warps=max(1, min(8, _nbp // 32)))
                blk = blk.view(*lead, nb)
            else:
                blk = torch.sort(l1_blk, dim=-1).values
            # 【AKO iter-7】块展开 × bypass 拼接 × p.gather 单趟 Triton 融合
            # （_ako_cand_expand_kernel）：mul+add+arange+cat+gather 五 kernel
            # 链 → 1 launch，输出布局逐位恒等（升序展开 + 旧 cat 段序）→
            # cand_p 逐位一致 → topk tie-break 逐位不变。守卫不齐（无
            # Triton / >2 bypass 段 / 非连续 / lead 与 p 不匹配）→ 回退旧
            # torch 链逐位不变。
            if (_HAS_TRITON and len(segs) <= 2 and p.is_contiguous()
                    and blk.is_contiguous()
                    and tuple(p.shape[:-1]) == tuple(lead)):
                S_w = p.shape[-1]
                R_l = p.numel() // S_w
                ba = segs[0] if len(segs) > 0 else (0, 0)
                bb = segs[1] if len(segs) > 1 else (0, 0)
                cand_idx = torch.empty(
                    (R_l, R_cand), dtype=torch.int64, device=dev)
                cand_p = torch.empty(
                    (R_l, R_cand), dtype=p.dtype, device=dev)
                _ako_cand_expand_kernel[(R_l, triton.cdiv(R_cand, 256))](
                    blk, p, cand_idx, cand_p,
                    R_cand, nb, S_w,
                    ba[0], ba[1], bb[0], bb[1],
                    nb, S_w, R_cand, R_cand,
                    BS=bs, BLOCK=256, num_warps=4)
                cand_idx = cand_idx.view(*lead, R_cand)
                cand_p = cand_p.view(*lead, R_cand)
            else:
                cand_idx = (blk * bs).unsqueeze(-1).add(offs).reshape(*lead, -1)
                for lo, hi in segs:
                    part = torch.arange(
                        lo, hi, device=dev, dtype=torch.int64)
                    cand_idx = torch.cat(
                        [cand_idx,
                         part.view(*((1,) * len(lead)), -1).expand(*lead, -1)],
                        dim=-1)
                cand_p = p.gather(-1, cand_idx)
        sel = torch.topk(cand_p, k, dim=-1).indices
        # 【AKO iter-10】_raw=True：装配融合守卫齐备时免尾部 back-gather
        # （(cand_idx, sel) 直交 _ako_l2_mask_kernel 内联 gather）；默认
        # False 与旧口径逐位相同。
        if _raw:
            return cand_idx, sel
        return cand_idx.gather(-1, sel)

    # ---------------- 【阶段 2c-2】graph 专用全张量 compute_mask ----------------
    # CUDA Graph decode 专用路径：与 eager compute_mask（Python 分支版）逐位
    # 一致（有限步验收口径），但零 host 同步 / 零 Python int 分支 / 全程固定
    # 形状——capture 后 replay 可用 copy_ 注入 q/valid_len 重复执行。
    #
    # 核心简化（decode 语义保证，下面各点详细论证）：
    #   * decode 期间 StaticCache 前缀 [0, S_cap) 不覆盖（append-only）→
    #     index_dict 容量宽一次烘焙（capture 前 prepare_index 全量），replay
    #     期间为常量张量；尾块（含新 token 的块）统计过时但恒在 swa 强制区
    #     （swa_lo_blk ≤ kt_eff−max(1,swa//bs)）→ 不进 L1 池、不进候选集、
    #     不消费 → 过时值零影响。
    #   * 当前块强制（eager `score_coarse[..., -1]=inf`）：强制块 = 有效区最后
    #     块 = 新 token 所在块 ∈ swa 区 → 不进 L1 池 → inf 不影响任何选择 →
    #     graph 省略，与 eager 选择逐位一致。
    #   * 池区间（far∪near = [sink_pool_lo, swa_lo_blk)）内块全部在前缀真实
    #     KV 区 → 烘焙统计正确。
    # 动态几何（near_blks/swa_lo_blk/swa_lo_tok/K2_mid/bypass/pad 界）全部从
    # valid_len [1] long tensor 派生，形状固定（张量比较/区间掩码/固定 k
    # topk + rank 截断），零 Python 分支。
    #
    # 生产主配置强制（其余 fail-closed raise，调用方回 eager 逐行路径）：
    #   e64 固定几何（alpha=beta=1.0 由 E155d 归一）+ gamma=None（nt=off 自由
    #   竞争）+ boundary_bypass + 非投影 + 非 per_q_head + 非 cluster 簇分 +
    #   非 moba/sigma/async + skip_far=False。

    def _build_graph_geometry(self, valid_len, S_cap, kt_cap, _cst, *, compact_only=False):
        """【E165 iter2a】步级几何单次构建（compute_mask_graph 消费）。

        内容 = 原 compute_mask_graph 段 1/2（动态几何/池掩码）+ 段 5 的
        bypass 槽构建 + 段 7/8 的 mid_mask/swa_force/k2_mid 逐字抽取。
        这些量是 valid_len 内容与配置常量的纯函数，与层无关 → 36 层共享
        （模块级 _GRAPH_GEO_CACHE）。逐位一致性：kernel 序列与值逐字
        未动，仅执行次数从 36 次/步降为 1 次/步（约 50 个小 kernel/层，
        E165 census：graph 8422 launches/步、均值 2.92us，launch 数受限）。
        """
        dev = valid_len.device
        bs = self.args.tia_block_size
        sink_tok = self.sink_tokens
        swa_tok = self.sliding_window_size
        nl = max(bs, int(self.nt_near_len))
        K2 = self.args.tia_level2_topk
        sink_pool_lo = self._sink_pool_lo_blk
        ar_blk = _cst["ar_blk"]
        ar_tok = _cst["ar_tok"]
        ar_bs = _cst["ar_bs"]
        zero_l = _cst["zero_l"]
        # ---- 原 段1. 动态几何（全部从 valid_len tensor 派生）----
        S_real = valid_len[0].to(torch.long)                 # 0-dim
        swa_lo_tok = torch.clamp(S_real - swa_tok, min=0)
        kt_eff = (S_real + bs - 1) // bs
        # near_blks = max(sink_pool_lo, (S_real − swa_tok − nl) // bs)（B10/F5）
        near_blks = torch.clamp((S_real - swa_tok - nl) // bs, min=sink_pool_lo)
        # swa_lo_blk = max(near_blks, kt_eff − max(1, swa_tok//bs))
        swa_lo_blk = torch.maximum(
            near_blks, kt_eff - max(1, swa_tok // bs))
        swa_byp_blk = swa_lo_tok // bs
        swa_byp_cond = (swa_lo_tok % bs != 0) & (
            swa_byp_blk >= sink_pool_lo)                    # 0-dim bool
        # Short prompts can put sink and SWA boundaries in the same page.
        # Eager uses a set of bypass pages; graph slots must not double-count
        # that page in softmax/GQA aggregation either.
        if sink_tok % bs != 0:
            swa_byp_cond = swa_byp_cond & (swa_byp_blk != sink_pool_lo)
        # ---- 原 段2. 池掩码（tensor 区间，固定形状 kt_cap）----
        far_lo_blk = sink_pool_lo
        far_pool_mask = (ar_blk >= far_lo_blk) & (ar_blk < near_blks)
        near_pool_mask = (ar_blk >= near_blks) & (ar_blk < swa_lo_blk)
        if sink_tok % bs != 0:
            far_pool_mask = far_pool_mask & (ar_blk != far_lo_blk)
        byp_excl = (ar_blk == swa_byp_blk.view(-1)) & swa_byp_cond.view(-1)
        near_pool_mask = near_pool_mask & ~byp_excl
        if sink_tok % bs != 0:
            sink_in_near = (ar_blk == far_lo_blk) & (
                ar_blk >= near_blks) & (ar_blk < swa_lo_blk)
            near_pool_mask = near_pool_mask & ~sink_in_near
        # ---- 原 段7/8 层无关量 ----
        far_tok_lo = far_lo_blk * bs
        if compact_only:
            mid_mask = mid_mask_inv = swa_force = None
        else:
            mid_mask = (ar_tok >= far_tok_lo) & (ar_tok < swa_lo_tok.view(-1))
            mid_mask_inv = ~mid_mask
            swa_force = (ar_tok >= swa_lo_tok.view(-1)) & (ar_tok < S_real.view(-1))
        k2_mid = torch.clamp(S_real.clamp(max=K2) - sink_tok - (S_real - swa_lo_tok), min=0)
        # ---- 原 段5 bypass 槽（slot0 静态 / slot1 = swa 跨界页）----
        slot0 = torch.where(_cst["sink_cross"], _cst["far_lo"], zero_l)
        slot1 = swa_byp_blk
        byp_blk2 = torch.stack([slot0, slot1.view(-1)[0]]).view(1, 1, 1, 2, 1)
        byp_tok = (byp_blk2 * bs + ar_bs.view(1, 1, 1, 1, -1))  # [1,1,1,2,bs]
        byp_keep = byp_tok < S_real.view(1, 1, 1, 1, 1)
        byp_keep = byp_keep & torch.stack([
            _cst["sink_cross"], swa_byp_cond]).view(1, 1, 1, 2, 1)
        return {
            "S_real": S_real, "swa_lo_tok": swa_lo_tok,
            "near_blks": near_blks, "swa_lo_blk": swa_lo_blk,
            "swa_byp_blk": swa_byp_blk, "swa_byp_cond": swa_byp_cond,
            "far_pool_mask": far_pool_mask, "near_pool_mask": near_pool_mask,
            "mid_mask": mid_mask, "mid_mask_inv": mid_mask_inv,
            "swa_force": swa_force, "k2_mid": k2_mid,
            "far_tok_lo": far_tok_lo,
            "byp_tok": byp_tok, "byp_keep": byp_keep,
        }

    def compute_selection_graph(self, q, index_dict, softmax_scale, valid_len):
        """与 compute_mask_graph 相同的选择策略，直出紧凑 TokenSelection。

        返回 TokenSelection（indices [1,Hkv,K2_static] int32 + counts
        [1,Hkv] int32），替代 full-width bool mask。operator pipeline
        → partial_attention 直接消费紧凑表示，免全宽 scatter/mask 开销。
        语义等效：to_mask(S_cap) == compute_mask_graph(...)（单元测试逐位对拍）。
        """
        return self.compute_mask_graph(q, index_dict, softmax_scale, valid_len,
                                       return_selection=True)

    def compute_candidate_inputs_graph(self, q, index_dict, softmax_scale, valid_len):
        """Produce one request's exact L1/bypass candidates, stopping before L2.

        This does not implement batch indexing. Later consumers may batch fine
        scoring/normalization/ranking after validating compatible feature storage
        and geometry. Existing selector denominator candidates/order are retained.
        """
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != 1:
            raise ValueError('candidate producer requires one decode request')
        if valid_len.shape != (1,) or valid_len.dtype != torch.int64:
            raise ValueError('candidate producer requires int64 valid_len[1]')
        return self.compute_mask_graph(q, index_dict, softmax_scale, valid_len,
                                       return_selection=True, return_candidates=True)

    def compute_mask_graph(self, q, index_dict, softmax_scale, valid_len, *,
                           return_selection=False, return_candidates=False):
        """graph 专用全张量 compute_mask（生产主配置，逐位对齐 eager）。

        输入：q [1,1,H,D]（replay 注入）、index_dict（容量宽：k_min/k_max/
        k_avg/k_qat 全宽块统计，capture 前烘焙为常量）、softmax_scale、
        valid_len [1] long（replay 注入，decode 当前行有效长）。
        return_selection：True → 返回 TokenSelection 紧凑格式（Phase 1 新路径）；
                          False → 返回 [1,1,Hkv,S_cap] bool mask（默认，兼容旧消费端）。
        return_candidates：内部候选生产接口；True 时在 fine score 前返回 CandidateInputs。
        输出：默认 mask，或显式请求的 TokenSelection/CandidateInputs。
        """
        # ---- 0. 生产主配置守卫（fail-closed 回 eager 逐行路径）----
        if not (self.alpha > 0 and self.beta > 0 and self.gamma is None
                and self.nt_near_pages is not None
                and self.nt_near_len is not None
                and not self.per_q_head and self._basis is None
                and self.far_select not in ("cluster", "sim_greedy")
                and self.near_select not in ("cluster", "sim_greedy")
                and not self.moba_gate and self.sigma_select == "none"
                and not self.enable_async):
            raise NotImplementedError(
                "compute_mask_graph 仅支持生产主配置（e64 固定几何 + γ=None "
                "自由竞争 + boundary_bypass + 非投影 + 非 per_q_head + 非簇分/"
                "moba/sigma/async）；其余配置调用方应回退 eager "
                "逐行路径（行为不变）")
        dev = q.device
        bs = self.args.tia_block_size
        sink_tok = self.sink_tokens
        swa_tok = self.sliding_window_size
        K1 = self.args.tia_level1_topk
        K2 = self.args.tia_level2_topk
        # 【E178】L2 细筛旁路实验臂 env 门（默认关，仅 graph 路径支持）
        _l2_bypass = os.environ.get("TLI_L2_BYPASS", "").strip() == "1"
        if return_candidates and _l2_bypass:
            raise NotImplementedError('candidate producer excludes L2 bypass')
        nb_near = max(0, int(self.nt_near_pages))
        nb_far = max(0, K1 - nb_near)
        nl = max(bs, int(self.nt_near_len))
        sink_pool_lo = self._sink_pool_lo_blk
        H = q.shape[2]
        # 【2c-2 修复】graph 专用路径自洽：Hkv/group_size 从容量宽 index_dict
        # 派生（等价 compute_score 设的 self.group_size = H // k_min.shape[1]）。
        # replay 时 compute_score 不先跑、实例状态不可依赖（原 self.group_size
        # 在 eager 路径因 compute_score 先跑而掩盖，graph 直调必 None 崩）。
        k_min = index_dict["k_min"].squeeze(0)   # [kt_cap, Hkv, d]（前置供 Hkv 派生）
        Hkv = k_min.shape[1]
        group_size = H // Hkv
        # 容量宽特征（squeeze(0) 后）[kt_cap, Hkv, d] / [S_cap, Hkv, d]
        k_max = index_dict["k_max"].squeeze(0)
        k_qat = index_dict["k_qat"]            # [1, S_cap, Hkv, d]
        k_avg = index_dict.get("k_avg")
        if k_avg is not None:
            k_avg = k_avg.squeeze(0)
        kt_cap = k_min.shape[0]
        S_cap = k_qat.shape[1]
        from ..ops.sparse_policy import live_normalization_enabled, stats_policy
        live_normalization = live_normalization_enabled()
        self._graph_normalization_width_route = None
        # A later request may choose another route on the same indexer.
        # Never report stats inherited from a previous sparse capture.
        self._graph_normalization_stats_route = None
        from ..ops.sparse_policy import selection_fallback
        sparse_reason = selection_fallback(
            q.dtype, dev, S_cap, group_size, Hkv, l2_bypass=_l2_bypass,
            protected_end=sink_pool_lo * bs, topk=K2)
        _sparse_selection = sparse_reason is None
        # Retain the coarse receipt for existing consumers; detail distinguishes
        # unsupported budgets from dtype/geometry and invalid policy settings.
        self._graph_sparse_selection_fallback_detail = sparse_reason
        self._graph_sparse_selection_fallback = (
            None if _sparse_selection else
            'disabled' if sparse_reason in ('disabled', 'selection_policy')
            else 'unsupported_static_geometry')
        # A short cache may contain fewer pages than the configured L1 quota.
        # Clip static slot counts exactly as eager clips each pool's topk.
        nb_far, nb_near = min(nb_far, kt_cap), min(nb_near, kt_cap)
        # The decode-frozen gate removes all far slots, as eager L1 does.
        if self.skip_far:
            nb_far = 0
        K1 = nb_far + nb_near
        # 静态常量（capture 安全）：torch.tensor(..., device=) 是 host→device
        # 拷贝，CUDA Graph capture 期禁止（cudaMemcpyAsync from pageable）→
        # 实例级缓存（首调 = warmup，在 capture 之外构建；key 含形状/dtype
        # 与几何常量，配置级常量）。arange 走设备 kernel，本身可 capture，
        # 一并缓存省 replay 轮次的重复 launch。
        _cst_key = (dev, q.dtype, kt_cap, S_cap, bs, sink_pool_lo, sink_tok, _sparse_selection)
        # Graphs retain raw pointers to constants allocated before capture.
        # Keep each geometry/mode alive for this indexer's lifetime: switching
        # the active key must not invalidate an already captured graph.
        if not hasattr(self, "_graph_const_entries"):
            self._graph_const_entries = {}
        _cst = self._graph_const_entries.get(_cst_key)
        if _cst is None:
            _cst = (_cst_key, {
                "ar_blk": torch.arange(kt_cap, device=dev),
                "ar_tok": None if _sparse_selection else torch.arange(S_cap, device=dev),
                "ar_bs": torch.arange(bs, device=dev),
                "inf_t": torch.tensor(float("-inf"), dtype=q.dtype, device=dev),
                "zero_l": torch.tensor(0, dtype=torch.long, device=dev),
                "sink_cross": torch.tensor(sink_tok % bs != 0, device=dev),
                "far_lo": torch.tensor(sink_pool_lo, dtype=torch.long,
                                       device=dev),
            })
            self._graph_const_entries[_cst_key] = _cst
        self._graph_const_cache = _cst
        ar_blk = _cst[1]["ar_blk"]
        ar_tok = _cst[1]["ar_tok"]
        ar_bs = _cst[1]["ar_bs"]
        inf_t = _cst[1]["inf_t"]
        zero_l = _cst[1]["zero_l"]
        # ---- 1+2. 动态几何 + 池掩码【E165 iter2a：步级共享缓存】----
        # 几何/池掩码/bypass 槽/mid_mask/swa_force/k2_mid 只依赖 valid_len
        # 与配置常量，与层无关（原实现每层重算 ~50 个小 kernel）。缓存键
        # = (data_ptr, _version, 几何键)：warmup/eager 步 inject 的 vl.copy_()
        # 使版本递增 → 每步首层重建；capture 单步内版本恒定 → 36 层命中
        # （几何 kernel 只记录一次，replay 期从 live vl 现值重算 → 跨步
        # 正确）。条目 pin valid_len 视图（强引用）防 data_ptr 复用误命中。
        _geo_key = (S_cap, kt_cap, bs, sink_pool_lo, sink_tok,
                    swa_tok, nl, K2, str(dev), _sparse_selection)
        # A capture-scoped identity prevents reuse of warmup or another graph's
        # geometry. Within the same capture, matching pointer/version/config can
        # share the first layer's device computation. Uninstrumented captures
        # keep rebuilding; mutation-version-less tensors also never reuse.
        from ..ops.cuda_graph_state import geometry_cache_key
        capturing = q.is_cuda and torch.cuda.is_current_stream_capturing()
        cache_key = geometry_cache_key(_geo_key, capturing=capturing,
            stream=torch.cuda.current_stream(q.device).cuda_stream if capturing else None)
        geo = None if cache_key is None else _graph_geo_lookup(valid_len, cache_key)
        if geo is None:
            geo = _graph_geo_store(
                valid_len, cache_key,
                self._build_graph_geometry(valid_len, S_cap, kt_cap, _cst[1],
                                           compact_only=_sparse_selection))
        S_real = geo["S_real"]
        swa_lo_tok = geo["swa_lo_tok"]
        near_blks = geo["near_blks"]
        swa_lo_blk = geo["swa_lo_blk"]
        swa_byp_blk = geo["swa_byp_blk"]
        swa_byp_cond = geo["swa_byp_cond"]
        far_lo_blk = sink_pool_lo
        far_pool_mask = geo["far_pool_mask"]
        near_pool_mask = geo["near_pool_mask"]
        mid_mask = geo["mid_mask"]
        mid_mask_inv = geo["mid_mask_inv"]
        swa_force = geo["swa_force"]
        k2_mid = geo["k2_mid"]
        far_tok_lo = geo["far_tok_lo"]
        # ---- 3. coarse 分（Triton，与 eager 同 kernel 逐位一致）----
        # 【2c-2 修复】Q 侧子空间 gather 与 eager 同源（compute_score L1742-1743
        # b_q = q_full[..., idx_sub]）：生产主配置 tail/l2_subspace=same 下
        # k_qat 为子空间 32 维切片，coarse/fine 打分 kernel 期望 Q [H,32]。
        # 原用全维 b_q [H,128] 传给 _score_fine_cand_kernel → kernel 只读 q
        # 前 32 维 [0:32]（非 tail 子空间 [48:64]∪[112:128]）→ fine 分错误 →
        # p 值偏移（G8 对拍 ~25% 差异根因）。gather 后与 eager 逐位一致。
        q_full32 = (q[0, 0] * softmax_scale).to(torch.float32)   # [H, D]
        idx_sub = (self._subspace_indices(q.device)
                   if self.enable_subspace else None)
        # coarse 用 b_q（与 k_min/k_max 同宽：idx_sub=None 时 = 全维，对应
        # prepare 侧 k_for_coarse 未切片）
        b_q = q_full32 if idx_sub is None else q_full32[..., idx_sub]  # [H, d']
        # 【2c-2 修复】fine 侧 q 的宽度由 k_qat 决定，与 coarse 可能不同：
        # 镜像 eager compute_score L1896-1901（l2_subspace=full → 全维；
        # idx_sub 非 None → b_q；否则 _qat_indices 尾段 32 维）。原实现
        # fine kernel 复用 coarse 的 b_q，full 模式（idx_sub=None）下传入
        # 128 维 → kernel 只读前 32 维（非 tail 子空间）→ fine 分/p 全错
        # （G8 对拍 diff ~7700 根因）。
        if self.l2_subspace == "full":
            b_q_fine = q_full32
        elif idx_sub is not None:
            b_q_fine = b_q
        else:
            b_q_fine = q_full32[..., self._qat_indices(q.device)]
        # 【iter3f-2】out= 直写消 fill+copy：fast_score 已支持 out= 参数，
        # 预分配 empty（不 fill -inf），kernel 直写 [H,kt_cap] 连续视图 →
        # 消除 2×(fill torch.full + setitem copy_) = 4 kernel/层，同值换写入路径。
        score_coarse = torch.empty((1, H, kt_cap), dtype=q.dtype, device=dev)
        _sc_buf = score_coarse[0]  # [H,kt_cap] 连续视图
        try:
            self._fast_score_coarse(
                k_min, k_max, b_q, q.dtype, out=_sc_buf)
        except Exception:
            # 与 eager 回退同路径（einsum minmax，fail-closed 逐位）
            b_k_min = repeat(k_min, "t h d -> t (h g) d", g=group_size).to(torch.float32)
            b_k_max = repeat(k_max, "t h d -> t (h g) d", g=group_size).to(torch.float32)
            _sc_buf.copy_(
                einsum(b_q.clamp(max=0), b_k_min, "h d, kt h d -> h kt")
                + einsum(b_q.clamp(min=0), b_k_max, "h d, kt h d -> h kt")
            ).to(q.dtype)
        score_coarse_avg = None
        if k_avg is not None:
            score_coarse_avg = torch.empty((1, H, kt_cap), dtype=q.dtype, device=dev)
            _sca_buf = score_coarse_avg[0]
            try:
                self._fast_score_fine(
                    k_avg, b_q, q.dtype, out=_sca_buf)
            except Exception:
                b_k_avg = repeat(k_avg, "t h d -> t (h g) d", g=group_size).to(torch.float32)
                _sca_buf.copy_(einsum(
                    b_q, b_k_avg, "h d, kt h d -> h kt").to(q.dtype))
        # GQA 聚合到 Hkv（共享口径，与 eager rearrange+mean 同数学）
        sc = score_coarse.view(1, Hkv, group_size, kt_cap).mean(dim=2)     # [1,Hkv,kt_cap]
        sc = sc.unsqueeze(1)
        if score_coarse_avg is not None:
            sc_avg = score_coarse_avg.view(1, Hkv, group_size, kt_cap).mean(dim=2)
            sc_avg = sc_avg.unsqueeze(1)
        else:
            sc_avg = None
        # ---- 4. L1 双池 topk（固定 k；池外 where -inf，keep 门控剔除垃圾）----
        far_src = sc_avg if (self.far_method == "avg" and sc_avg is not None) else sc
        near_src = sc if (self.near_method == "minmax" or sc_avg is None) else sc_avg
        fp_mask = far_pool_mask.view(1, 1, 1, -1).expand(1, 1, Hkv, kt_cap)
        np_mask = near_pool_mask.view(1, 1, 1, -1).expand(1, 1, Hkv, kt_cap)
        # 【E166 iter3d】L1 双池 scored 融合：4 kernel/层 → 1（值域纯
        # 拷贝+常量 -inf → topk 输入逐位同）；门控不齐回退原链。
        _l1s = None
        if (_HAS_TRITON
                and far_src.is_contiguous() and near_src.is_contiguous()
                and far_pool_mask.is_contiguous()
                and near_pool_mask.is_contiguous()
                and far_src.dtype in (torch.bfloat16, torch.float16,
                                      torch.float32)
                and far_src.shape[-1] == kt_cap):
            try:
                _l1s = _iter3_l1_scored(far_src, near_src,
                                        far_pool_mask, near_pool_mask,
                                        kt_cap, Hkv)
            except Exception:
                _l1s = None
        if _l1s is not None:
            far_scored, near_scored = _l1s
        else:
            far_scored = torch.where(fp_mask, far_src,
                                     torch.full_like(far_src, float("-inf")))
            near_scored = torch.where(np_mask, near_src,
                                      torch.full_like(near_src, float("-inf")))
        i_f = torch.topk(far_scored, nb_far, dim=-1).indices   # [1,1,Hkv,nb_far]
        i_n = torch.topk(near_scored, nb_near, dim=-1).indices  # [1,1,Hkv,nb_near]
        # ---- 5. 候选集构建（L1 块 ∪ bypass 页，固定 R_cap = K1*bs + 2*bs）----
        # 【E166 iter3d】候选装配融合：cat×2 + l1_tok 乘加 + l1_keep lt&and
        # + cat×2（~8 kernel/层）→ 单核直出 [Hkv,R_cap]（纯整数/布尔域 →
        # bitexact by construction；槽布局逐位复刻）。门控不齐回退原链。
        R_cap = K1 * bs + 2 * bs
        _c3d = None
        if (_HAS_TRITON
                and i_f.is_contiguous() and i_n.is_contiguous()
                and far_pool_mask.is_contiguous()
                and near_pool_mask.is_contiguous()
                and K1 >= 1 and K1 <= 2048 and (bs & (bs - 1)) == 0
                and geo["byp_tok"].is_contiguous()
                and geo["byp_keep"].is_contiguous()
                and i_f.dtype == torch.long
                and S_real.dtype == torch.long
                and S_real.numel() == 1):
            try:
                _c3d = _iter3_l1_cand(
                    i_f, i_n, far_pool_mask, near_pool_mask,
                    geo["byp_tok"], geo["byp_keep"], S_real,
                    K1, nb_far, bs, R_cap, Hkv, dev)
            except Exception:
                _c3d = None
        if _c3d is not None:
            cand_idx2, keep2 = _c3d
        else:
            far_ok = fp_mask.gather(-1, i_f)                    # 池内判定（剔除垃圾）
            near_ok = np_mask.gather(-1, i_n)
            blks = torch.cat([i_f, i_n], dim=-1)               # [1,1,Hkv,K1]
            in_pool = torch.cat([far_ok, near_ok], dim=-1)     # [1,1,Hkv,K1]
            l1_tok = (blks.unsqueeze(-1) * bs
                      + ar_bs.view(1, 1, 1, 1, -1))            # [1,1,Hkv,K1,bs]
            l1_keep = (l1_tok < S_real.view(1, 1, 1, 1, 1)) & in_pool.unsqueeze(-1)
            # bypass 槽：slot0 = sink 跨界（静态，cond 门控）、slot1 = swa 跨界
            # （动态）。【E165 iter2a】槽构建与层无关 → 步级 geo 缓存直取，
            # 层内仅剩 Hkv 维 expand（视图零 kernel）。
            byp_tok = geo["byp_tok"].expand(1, 1, Hkv, 2, bs)
            byp_keep = geo["byp_keep"].expand(1, 1, Hkv, 2, bs)
            cand_idx = torch.cat([
                l1_tok.reshape(1, 1, Hkv, K1 * bs),
                byp_tok.reshape(1, 1, Hkv, 2 * bs)], dim=-1)    # [1,1,Hkv,R_cap]
            keep = torch.cat([
                l1_keep.reshape(1, 1, Hkv, K1 * bs),
                byp_keep.reshape(1, 1, Hkv, 2 * bs)], dim=-1)   # bool
            cand_idx2 = cand_idx[0, 0]                         # [Hkv,R_cap]
            keep2 = keep[0, 0]                                 # [Hkv,R_cap]
        # 【iter3f-3】哨兵重定向由 _iter3_compact_kernel 内联
        # （keep=False 槽写 S_CAP_M1）；fallback argsort 分支需在 _iter3_compact
        # 守卫失败时独立执行（见下方 except/else 分支）。
        # ---- 【E178】L2 细筛旁路实验臂（TLI_L2_BYPASS=1，默认关）----
        # 用户 2026-10-27 指令：K1 选中块 + bypass 页的全部候选 token
        # 直接进主注意力（跳过细筛打分/紧凑 softmax/L2 topk/候选压缩
        # 全链），对比 K1=128+K2 细筛的质量与性能。语义 = 把
        # 「tok_top=cand_idx2、k2_mid=K2_static=R_cap（rank_keep 全
        # True）」代入下方旧 torch 装配链：垃圾槽已重定向 S_cap-1 哨兵
        # （落 swa 区）→ & mid_mask 排除；bypass 跨界页中落 sink/swa
        # 强制区的 token 由强制区覆盖；pad 清零/metrics 与主链同式。
        # 纯整数/布尔域 → bitexact by construction。候选 token 量 =
        # R_cap（K1=64 → 4224）+ sink + swa，远超 K2=1024 预算——
        # attention/gather 成本随选中量放大的代价由实验实测裁决。
        if _l2_bypass:
            # Inactive page slots can contain arbitrary/out-of-range IDs.
            # Sanitize before scatter, not after filtering the resulting mask.
            cand_idx2 = torch.where(keep2, cand_idx2, S_cap)
            sel = torch.zeros((1, 1, Hkv, S_cap + 1), dtype=torch.bool, device=dev)
            sel.scatter_(-1, cand_idx2.view(1, 1, Hkv, R_cap), True)
            # A dedicated extra slot also works when SWA is disabled: the
            # last real token must never act as an inactive-slot sentinel.
            sel = sel[..., :S_cap]
            sel = sel & mid_mask.view(1, 1, 1, -1)
            sel[..., :sink_tok] = True
            sel = sel | swa_force.view(1, 1, 1, -1)
            sel = sel & (ar_tok < S_real).view(1, 1, 1, -1)
            # 【Phase 1】紧凑 Selection 路径：return_selection=True 时
            # 不走全宽 mask，直接返回 compact TokenSelection。
            # 用 getattr(locals(),...) 保护 AST exec 测试上下文（无此变量）。
            _do_selection = locals().get('return_selection', False)
            if _do_selection:
                from ..ops.token_selection import graph_middle_selection
                safe = torch.where(keep2, cand_idx2, S_cap)
                return graph_middle_selection(
                    safe, R_cap, mid_mask, S_real, S_cap, sink_tok, swa_tok)
            if not (q.is_cuda and torch.cuda.is_current_stream_capturing()):
                self.metrics.add_select_result(valid_len - 1, sel, 1)
            return sel
        _domain_pack = None
        if _HAS_TRITON and q.is_cuda:
            from .candidate_domain_triton import complete_and_pack
            _domain_pack = complete_and_pack(
                cand_idx2, keep2, S_real, sink_tok, swa_tok, S_cap)
        self._graph_candidate_domain_route = "fused" if _domain_pack is not None else "torch"
        if _domain_pack is not None:
            cand_idx2, keep2, cand_len = _domain_pack
            R_cap = cand_idx2.shape[-1]
        else:
            # Complete the same normalization domain as eager. Move protected
            # tokens out of L1/bypass slots into a single disjoint sink/SWA segment;
            # short requests and overlapping protected intervals must count once.
            from .candidate_domain import complete_protected_candidates
            cand_idx2, keep2 = complete_protected_candidates(
                cand_idx2, keep2, S_real, sink_tok, swa_tok, S_cap)
            R_cap = cand_idx2.shape[-1]
            # 【2c-2 修复】候选紧凑化重排：_score_fine_cand_kernel 的 cand_len
            # 门控语义 = 「前 cand_len 个位置有效」（offs_r < CLEN[kv]）。原布局
            # cand_idx = [far 96 槽(垃圾), near 32 槽(有效), byp 2 槽(有效)]——
            # 有效候选落在第 K1_far*bs 之后的槽位，kernel 只保留前 cand_len
            # （= 有效数）位置 → 保留下来的全是 far 垃圾，near/byp 有效候选
            # 全被 -inf（→ sf_full mid 区全 -inf → p mid 区 0，G8 对拍全崩）。
            # 用 stable argsort(~keep) 把有效候选（keep=True）稳定提到前
            # cand_len 个位置、垃圾后置（scatter 哨兵排除），R_cap 固定形状
            # 与 eager 归约集合（L1∪bypass）逐位一致。
            # 【E166 iter3b】候选压缩链融合：argsort(~keep2) stable + 2×gather
            # + sum + （下游 mid_slot gather）≈5 kernel/层（R_cap=8448 恒定
            # 长度无关 ≈1.3ms/步）换单 Triton 扫描压缩 kernel。逐位等价：
            # stable argsort(~keep) = keep=True 按原序前置 + False 按原序后置
            # ——扫描压缩的 keep 目的位 = 前缀计数、garbage 目的位 = total+序数
            # （含 garbage 区内容保序复刻：旧 gather 把原始 idx/False/mid 值
            # 置换到尾部，下游 topk 边缘语义依赖该内容）；cand_len = total。
            # 纯整数/布尔域无浮点 → bitexact by construction；形态守卫不齐
            # → 旧链 fail-closed 逐位不变。
            _mid_slot3 = None
            if (_HAS_TRITON and cand_idx2.is_contiguous()
                    and keep2.is_contiguous()
                    and (mid_mask is None or mid_mask.is_contiguous())):
                try:
                    cand_idx2, keep2, _mid_slot3, cand_len = _iter3_compact(
                        cand_idx2, keep2, mid_mask, R_cap, Hkv, S_cap, dev)
                except Exception:
                    _mid_slot3 = None
                    cand_idx2 = torch.where(keep2, cand_idx2, S_cap - 1)  # fallback 哨兵
                    _srt2 = torch.argsort(~keep2, dim=-1, stable=True)
                    cand_idx2 = cand_idx2.gather(-1, _srt2)
                    keep2 = keep2.gather(-1, _srt2)
                    cand_len = keep2.sum(-1).to(torch.long)
            else:
                cand_idx2 = torch.where(keep2, cand_idx2, S_cap - 1)  # fallback 哨兵
                _srt2 = torch.argsort(~keep2, dim=-1, stable=True)
                cand_idx2 = cand_idx2.gather(-1, _srt2)
                keep2 = keep2.gather(-1, _srt2)
                cand_len = keep2.sum(-1).to(torch.long)
        if return_candidates:
            from ..ops.candidate_inputs import candidate_inputs
            lower = torch.full_like(S_real, far_tok_lo)
            upper = torch.maximum(lower, swa_lo_tok)
            return candidate_inputs(
                k_qat, b_q_fine, cand_idx2, keep2, cand_len, S_real,
                lower, upper, k2_mid, capacity=S_cap, group_size=group_size,
                output_dtype=q.dtype, rank_width=min(S_cap, K2),
                sink=sink_tok, swa=swa_tok)
        # ---- 6. fine 打分（Triton 候选 kernel，与 eager 同 kernel 逐位）----
        try:
            if self._triton_ok and _HAS_TRITON and k_qat.dim() == 4 \
                    and k_qat.is_cuda:
                sfc = self._fast_score_fine_cand(
                    k_qat[0], b_q_fine, cand_idx2, cand_len, R_cap, q.dtype)
            else:
                raise RuntimeError("triton 候选 kernel 不可用")
        except Exception:
            _kv_id = torch.arange(Hkv, device=dev).view(-1, 1)
            _kqc = k_qat[0][cand_idx2, _kv_id, :]              # [Hkv,R,d]
            _bkqc = repeat(_kqc, "kv r d -> (kv g) r d", g=group_size).to(torch.float32)
            sfc = einsum(b_q_fine, _bkqc, "h d, h r d -> h r").to(q.dtype)
        if _sparse_selection:
            from ..ops.sparse_normalization import candidate_probabilities
            from ..ops.candidate_ranking import rank_candidates
            probability, unique = candidate_probabilities(
                sfc, cand_idx2, keep2, S_cap, group_size, return_unique=True,
                valid_length=valid_len if live_normalization else None)
            self._graph_normalization_width_route = "live_length" if live_normalization else "capacity"
            lower = torch.full_like(S_real, far_tok_lo)
            upper = torch.maximum(lower, swa_lo_tok)
            selection = rank_candidates(cand_idx2, probability, unique, lower, upper,
                                        k2_mid, capacity=S_cap, width=1024, sink=sink_tok,
                                        deduplicated=True)
            self._graph_normalization_route = "sparse_candidate_domain"
            self._graph_normalization_stats_route = (
                "live_length_owner" if live_normalization else
                "capacity_owner" if stats_policy() == "owner"
                else "sorted_candidates")
            self._graph_ranking_route = "candidate_domain_native_1024"
            self._graph_ranking_kth_route = None
            self._graph_ranking_gather_route = None
            self._graph_ranking_fallback = None
            if return_selection:
                return selection
            # Legacy attention consumes a bool mask. Keep probability and
            # ranking candidate-only, then adapt just the final IDs. Protected
            # tokens are outside TokenSelection and must be restored exactly.
            token = torch.arange(S_cap, device=dev)
            selected = selection.to_mask(S_cap).unsqueeze(1)
            protected = (token < sink_tok) | (token >= swa_lo_tok)
            selected = (selected | protected.view(1, 1, 1, -1)) & (
                token < S_real).view(1, 1, 1, -1)
            if not (q.is_cuda and torch.cuda.is_current_stream_capturing()):
                self.metrics.add_select_result(valid_len - 1, selected, 1)
            return selected
        # ---- 7. 完整候选散回 + 全宽 softmax + GQA mean（与 eager 同归约域）----
        # Both eager and Graph normalize L1/bypass union complete sink/SWA.
        # Protected IDs were deduplicated above. The optional compact kernel
        # retains token-position reduction order and full-domain Top-K ties.
        _compact_norm = False
        if live_normalization or os.environ.get("TLI_COMPACT_NORMALIZATION", "0") == "1":
            from ..ops.compact_normalization import supported, compact_probabilities
            _compact_norm = supported(sfc, cand_idx2, keep2, S_cap, group_size)
        if live_normalization and not _compact_norm:
            raise NotImplementedError("live-length normalization requires supported BF16 owner geometry")
        self._graph_normalization_route = (
            "compact_fused_mean" if os.environ.get("TLI_COMPACT_FUSED_MEAN", "1") == "1"
            else "compact_separate_mean") if _compact_norm else "full_width_reference"
        if _compact_norm:
            p = compact_probabilities(sfc, cand_idx2, keep2, S_cap, group_size, mid_mask_inv,
                                      valid_length=valid_len if live_normalization else None)
        else:
            keep_h = repeat(keep2, "kv r -> (kv g) r", g=group_size)
            cand_h = repeat(cand_idx2, "kv r -> (kv g) r", g=group_size)
            sf_full = torch.full(
                (H, S_cap + 1), float("-inf"), dtype=sfc.dtype, device=dev)
            safe_idx = torch.where(keep_h, cand_h, S_cap)
            sf_full.scatter_(1, safe_idx, sfc)
            score_full = sf_full[:, :S_cap]
            sm = F.softmax(score_full, dim=-1)
            if (_HAS_TRITON and sm.is_contiguous() and sm.dtype == torch.bfloat16
                    and (group_size & (group_size - 1)) == 0):
                p = torch.empty((Hkv, S_cap), dtype=sm.dtype, device=dev)
                # SWA 由下方动态 mask 置位；这里传空强制区，mid 值与 eager 相同。
                _ako_gqa_mean_kernel[(Hkv, triton.cdiv(S_cap, 1024))](
                    sm, p, S_cap, 0, 0,
                    G=group_size, BLOCK=1024, num_warps=4)
            else:
                p = sm.nan_to_num(nan=0).view(Hkv, group_size, S_cap).mean(dim=1)
        self._graph_normalization_width_route = "live_length" if live_normalization else "capacity"
        # ---- 8. mid 全域 L2 自由竞争 ----
        # 固定 K2_static 保持 capture 形状；k2_mid 在终掩码装配处动态截断。
        # mid 外全为 -inf，前 k2_mid 名与 eager 的 mid 段 top-k 同集。
        mid_scored = p if _compact_norm else p.masked_fill(mid_mask_inv, float("-inf"))
        K2_static = min(S_cap, K2)
        tok_top = None
        self._graph_ranking_route = "full_width_native"
        self._graph_ranking_kth_route = None
        self._graph_ranking_gather_route = None
        self._graph_ranking_fallback = "disabled"
        if os.environ.get("TLI_COMPACT_RANKING", "0") == "1":
            from ..ops.compact_ranking import compact_topk
            tok_top = compact_topk(mid_scored, cand_idx2, keep2, mid_mask, K2_static)
            self._graph_ranking_fallback = "unsupported_static_geometry" if tok_top is None else None
            if tok_top is not None:
                self._graph_ranking_route = "compact_threshold_native_order"
                from ..ops import compact_ranking
                self._graph_ranking_kth_route = compact_ranking._last_kth_route
                self._graph_ranking_gather_route = compact_ranking._last_gather_route
        if tok_top is None:
            tok_top = torch.topk(mid_scored, K2_static, dim=-1).indices
        if return_selection:
            from ..ops.token_selection import graph_middle_selection
            return graph_middle_selection(
                tok_top, k2_mid, mid_mask, S_real, S_cap, sink_tok, swa_tok, unique_top=True)
        # 【E166 iter3a】终掩码装配融合：i_eff where + full_like + zeros/
        # scatter + and + slice fill + or 六段（~6 kernel/层，[Hkv,S_cap]
        # 全宽 3 遍读写 ≈7.5MB/层@128K）换 2 个 Triton kernel（A 基底
        # sink|swa 每位置恰写一次 + B 选中位条件写 KB 级）。
        # 逐位等价论证：旧 sel = (scatter(i_eff) & mid) | sink | swa；
        # 哨兵位（k≥k2_mid）经 & mid 恒 False（S_cap-1 ∉ mid 域）→ B 跳过
        # 同集；k<k2_mid 位 val = mid[tok] 与旧 scatter&and 同值；A 基底
        # 与旧 sink|swa 同集；纯布尔/整数语义无浮点 → bitexact by
        # construction。形态守卫（连续性/triton 可用）不齐 → 旧链
        # fail-closed 逐位不变。
        # 【iter3f-5】valid_mask 预计算（步级常量，ar_tok < S_real），
        # 传入 sel_base_kernel 内联 AND，替代 host 侧 sel & valid_mask
        _valid_mask = (ar_tok < S_real)              # [S_cap] bool，连续
        _asm3 = None
        if (_HAS_TRITON and K2_static > 0
                and tok_top.is_contiguous()
                and mid_mask.is_contiguous() and swa_force.is_contiguous()
                and _valid_mask.is_contiguous()):
            try:
                _asm3 = _iter3_assemble_mask(
                    tok_top, k2_mid, mid_mask, swa_force,
                    int(sink_tok), S_cap, Hkv, K2_static, dev,
                    valid_mask=_valid_mask)
            except Exception:
                _asm3 = None
        if _asm3 is not None:
            sel = _asm3
        else:
            # 【iter3f-1】rank_keep 仅在回退分支使用 → 从热路径移除
            rank_keep = (torch.arange(K2_static, device=dev)
                         .view(1, -1) < k2_mid.view(1, 1))
            # 截断后多余的 topk 结果替换成 S_cap−1 哨兵（恒在 swa 区，mid_mask 排除）
            i_eff = torch.where(rank_keep, tok_top,
                                torch.full_like(tok_top, S_cap - 1))
            sel = torch.zeros((1, 1, Hkv, S_cap), dtype=torch.bool, device=dev)
            sel.scatter_(-1, i_eff.view(1, 1, Hkv, K2_static), True)
            sel = sel & mid_mask.view(1, 1, 1, -1)
            # 正交强制区（必选，计入 K2 预算）：sink 头部 + swa 尾部
            sel[..., :sink_tok] = True
            sel = sel | swa_force.view(1, 1, 1, -1)
            # 回退分支仍需 host 侧 AND（kernel 未内联）
            sel = sel & _valid_mask.view(1, 1, 1, -1)
        if not (q.is_cuda and torch.cuda.is_current_stream_capturing()):
            self.metrics.add_select_result(valid_len - 1, sel, 1)
        return sel

    # ---------------- 【E156 精筛候选化】L1 块选择单源 helper ----------------
    # compute_score（候选化）与 compute_mask（消费端）共用同一 L1 块级选择
    # 实现，杜绝两份口径漂移。逻辑逐字抽取自 compute_mask 原 L1 选择段
    # （e64 双池 / 单池 / skip / E155b bypass / _sink_pool_lo_blk），原地写
    # score_coarse（skip far 置位 + 当前块强制 inf，与原行为逐字一致）。

    def _select_l1_blocks(self, score_coarse, score_coarse_avg, S_real,
                          sc_mean=None, sc_avg_mean=None):
        """L1 块级选择单源实现（compute_score 候选化 / compute_mask 共用）。

        输入 score_coarse [1,1,Hkv,kt]（共享口径）或 [1,1,H,kt]（per_q_head）、
        score_coarse_avg 可选（同前）；S_real = 真实序列长（valid_length 口径）。
        【AKO iter-21】sc_mean/sc_avg_mean = 粗筛 kernel 同趟写出的 GQA 组
        均值 [1,Hkv,kt]（kv 空间，形态守卫不齐一律回退 torch mean 逐位不变
        ——均值与 score_coarse 的对应关系由生产单调用链保证）。
        返回 dict：topk_mask_block（块级选中 bool mask，同输入聚合后形状）、
        bypass_blks（E155b 保送页块列表）、几何上下文（compute_mask L2 分支
        消费：kt/bs/t/e64_partition/sink_tok/swa_tok/mid_len/near_len_dyn/
        near_base/near_blks/far_lo_blk/far_hi_blk/swa_lo_blk/nb_near/nb_far/
        far_pool_lo/far_pool_hi/near_pool_lo/near_pool_hi/use_partition）。
        """
        per_q_head = self.per_q_head
        if score_coarse.ndim == 3:
            # compute_score 路径传入未 unsqueeze 的 [1,H,kt]；统一成 4D
            # [1,1,H,kt]（与 compute_mask 侧 score_dict 口径一致）
            score_coarse = score_coarse.unsqueeze(0)
            if score_coarse_avg is not None and score_coarse_avg.ndim == 3:
                score_coarse_avg = score_coarse_avg.unsqueeze(0)
            if sc_mean is not None and sc_mean.ndim == 3:
                sc_mean = sc_mean.unsqueeze(0)
            if sc_avg_mean is not None and sc_avg_mean.ndim == 3:
                sc_avg_mean = sc_avg_mean.unsqueeze(0)
        _mean_fused = False
        if not per_q_head:
            # 【AKO iter-21】kernel 同趟 GQA 组均值直用（消 torch mean
            # reduce launch）；形态守卫（形状/dtype/连续）不齐 → 回退 torch
            # mean 逐位不变。探针纪律（坑⑦）：返回 dict 附 mean_fused
            # 标记，对拍探针必须断言 fused 路径真生效。
            _kt_m = score_coarse.shape[-1]
            _hkv_m = score_coarse.shape[2] // self.group_size
            if (sc_mean is not None
                    and tuple(sc_mean.shape) == (1, 1, _hkv_m, _kt_m)
                    and sc_mean.dtype == score_coarse.dtype
                    and sc_mean.is_contiguous()):
                score_coarse = sc_mean
                _mean_fused = True
            else:
                score_coarse = rearrange(
                    score_coarse, "b qt (h g) kt -> b qt h g kt", g=self.group_size
                ).mean(dim=-2)  # [1,1,Hkv,kt]
        kt = score_coarse.shape[-1]
        bs = self.args.tia_block_size
        t = kt * bs - 1  # 当前序列长度（pad 后）近似
        # ---- E64 α/β/γ 分区：near 区动态长度（与 compute_mask 原逻辑逐字一致，
        #      含 B10 near 左界基准分支化 / F5 真实 S 口径 / 075 真实域）----
        e64_partition = self.alpha > 0 and self.beta > 0
        swa_tok = self.sliding_window_size
        sink_tok = self.sink_tokens            # E135：强制区宽单一事实源
        mid_len = max(0, S_real - sink_tok - swa_tok)
        if e64_partition:
            if self.nt_near_len is not None:
                near_len_dyn = max(bs, int(self.nt_near_len))
            else:
                near_len_dyn = max(bs, int(self.alpha * mid_len))
            near_base = S_real - swa_tok
        elif self.far_select != "4bit" and self.alpha == 0 and self.beta == 0:
            near_len_dyn = swa_tok
            near_base = kt * bs
        else:
            near_len_dyn = self.near_len
            near_base = kt * bs
        near_blks = max(self._sink_pool_lo_blk, (near_base - near_len_dyn) // bs)
        far_lo_blk, far_hi_blk = self._sink_pool_lo_blk, near_blks
        swa_lo_blk = max(near_blks, kt - max(1, swa_tok // bs))
        # ---- D'：跳层 → 远端块全部 -inf（与 compute_mask 原 L1 前置一致）----
        if self.skip_far:
            score_coarse[..., far_lo_blk:far_hi_blk] = float("-inf")
        use_partition = (self.enable_kmeans or e64_partition) \
            and not (self.skip_far and not e64_partition)
        score_coarse[..., -1] = float("inf")  # TIA 语义：当前块强制
        k1 = self.args.tia_level1_topk
        # E109a：avg 分数源提取提前到分支外（与 compute_mask 原逻辑一致）
        # 【AKO iter-21】kernel 同趟均值直用（守卫同上）
        if score_coarse_avg is not None and not per_q_head:
            _kt_a = score_coarse_avg.shape[-1]
            _hkv_a = score_coarse_avg.shape[2] // self.group_size
            if (sc_avg_mean is not None
                    and tuple(sc_avg_mean.shape) == (1, 1, _hkv_a, _kt_a)
                    and sc_avg_mean.dtype == score_coarse_avg.dtype
                    and sc_avg_mean.is_contiguous()):
                score_coarse_avg = sc_avg_mean
            else:
                score_coarse_avg = rearrange(
                    score_coarse_avg, "b qt (h g) kt -> b qt h g kt", g=self.group_size
                ).mean(dim=-2)
        # ---- L1 双池 / 单池 topk + E155b bypass（逐字抽取自 compute_mask）----
        bypass_blks = ()
        # 【AKO iter-6】L1 块级索引透传初始化：非 e64 臂不产出（消费端
        # compute_mask 复用门/内联段均以 None 表示无块索引，口径一致）。
        i_f = i_n = None
        far_pool_lo, far_pool_hi = far_lo_blk, far_hi_blk
        near_pool_lo, near_pool_hi = near_blks, swa_lo_blk
        nb_near = 0
        nb_far = 0
        if e64_partition:
            if self.nt_near_pages is not None:
                nb_near = max(0, int(self.nt_near_pages))
            else:
                nb_near = max(1, int(round(k1 * self.beta)))
            nb_far = max(0, k1 - nb_near)
            if self.skip_far:
                n_valid_far = 0
                nb_far = min(nb_far, max(n_valid_far, 0))
            if self.boundary_bypass:
                _bp = []
                if sink_tok % bs != 0:
                    _bp.append(far_lo_blk)
                _swa_lo_tok = max(0, S_real - swa_tok)
                if _swa_lo_tok % bs != 0:
                    _cb = _swa_lo_tok // bs
                    if _cb >= far_lo_blk:
                        _bp.append(_cb)
                bypass_blks = _bp
                for _b in _bp:
                    if far_pool_lo <= _b < far_pool_hi:
                        if _b == far_pool_lo:
                            far_pool_lo = _b + 1
                        elif _b == far_pool_hi - 1:
                            far_pool_hi = _b
                        else:
                            raise ValueError(
                                f"boundary bypass 页 {_b} 非 far 池端点 "
                                f"[{far_pool_lo},{far_pool_hi})——池收缩几何"
                                "假设被破坏（当前几何下保送页恒位于端点）")
                    if near_pool_lo <= _b < near_pool_hi:
                        if _b == near_pool_hi - 1:
                            near_pool_hi = _b
                        elif _b == near_pool_lo:
                            near_pool_lo = _b + 1
                        else:
                            raise ValueError(
                                f"boundary bypass 页 {_b} 非 near 池端点 "
                                f"[{near_pool_lo},{near_pool_hi})——池收缩几何"
                                "假设被破坏（当前几何下保送页恒位于端点）")
            if self.far_method == "avg" and score_coarse_avg is not None:
                sc_far = score_coarse_avg[..., far_pool_lo:far_pool_hi]
            else:
                sc_far = score_coarse[..., far_pool_lo:far_pool_hi]
            i_f = torch.topk(
                sc_far, min(nb_far, sc_far.shape[-1]), dim=-1
            ).indices + far_pool_lo
            if self.near_method == "minmax" or score_coarse_avg is None:
                sc_near = score_coarse[..., near_pool_lo:near_pool_hi]
            else:
                sc_near = score_coarse_avg[..., near_pool_lo:near_pool_hi]
            # 【AKO iter-19】sc_near 后续仅 topk 消费（无 in-place 变异），
            # 旧 clone 为逐字抽取时代残留 → 删除（值不变，-1 launch）。
            if sc_near.shape[-1] > 0:
                i_n = torch.topk(
                    sc_near, min(nb_near, sc_near.shape[-1]), dim=-1
                ).indices + near_pool_lo
            else:
                i_n = i_f[..., :0]  # near 池空（mid 不足一块）→ 不选
            indices = torch.cat([i_f, i_n], dim=-1)
            # 【AKO iter-19】topk_mask 装配融合（5 launch → 1）
            topk_mask = self._build_topk_mask(indices, bypass_blks,
                                              score_coarse)
        else:
            sc_pool = score_coarse
            if self.far_method == "avg" and score_coarse_avg is not None:
                sc_pool = score_coarse_avg.clone()
                sc_pool[..., -1] = float("inf")
                if self.skip_far:
                    sc_pool[..., far_lo_blk:far_hi_blk] = float("-inf")
            if self.skip_far:
                n_valid = kt - max(0, far_hi_blk - far_lo_blk)
                k1 = min(k1, max(n_valid, 1))
            values, indices = torch.topk(
                sc_pool, min(sc_pool.shape[-1], k1), dim=-1
            )
            # 【AKO iter-19】topk_mask 装配融合（3 launch → 1）；非 e64 无 bypass
            topk_mask = self._build_topk_mask(indices, (), score_coarse)
        return {
            "topk_mask_block": topk_mask,
            "bypass_blks": bypass_blks,
            "kt": kt, "bs": bs, "t": t, "e64_partition": e64_partition,
            "sink_tok": sink_tok, "swa_tok": swa_tok, "mid_len": mid_len,
            "near_len_dyn": near_len_dyn, "near_base": near_base,
            "near_blks": near_blks, "far_lo_blk": far_lo_blk,
            "far_hi_blk": far_hi_blk, "swa_lo_blk": swa_lo_blk,
            "nb_near": nb_near, "nb_far": nb_far,
            "far_pool_lo": far_pool_lo, "far_pool_hi": far_pool_hi,
            "near_pool_lo": near_pool_lo, "near_pool_hi": near_pool_hi,
            "use_partition": use_partition,
            # 【AKO iter-6】L1 块级索引 + 生产侧 S_real 透传：e64 臂 i_f/i_n
            # 为偏移后绝对块号（与 compute_mask 内联段捕获口径同源）；非
            # e64 臂恒 None。l1_s_real = 本函数收到的 S_real（valid_length
            # 口径）——compute_mask 复用门逐项一致性校验用。
            "i_f": i_f, "i_n": i_n,
            "l1_s_real": S_real,
            # 【AKO iter-16】L1 块索引单键透传：e64 = cat(i_f,i_n)（局部
            # indices 即此值）；非 e64 = 单池 topk indices。与 topk_mask True
            # 集恒等（topk 位置互异；e64 bypass 恒在池端点、E155b 收缩后
            # 与 i_f/i_n 不相交）→ _gather_fine_cand 免 nonzero 同步构建。
            "i_all": indices,
            # 【AKO iter-21】粗筛 GQA 均值 fused 生效标记（探针断言用，
            # 坑⑦防假 PASS）
            "mean_fused": _mean_fused,
        }

    def clear(self):
        exact_batch = getattr(self, '_exact_linear_batch', None)
        if exact_batch is not None:
            exact_batch.invalidate()
        self._far_skip_workspace = None
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
        # E137：动态 D' 标定跨请求重置——clear() 在每次 prefill 前调用，
        # 本请求 prefill 期末由 calibrate_dyn_dgate 用新 q/k 重新标定。
        # A6：pending/frozen 两状态同步复位（skip_far / _dyn_skip_pending /
        # _dyn_frozen 三态全清），新请求 decode 首步重新冻结。
        if self.dyn_dgate:
            self.skip_far = False
            self._dyn_skip_pending = False
            self._dyn_frozen = False
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
        # 【E152v4】graph 持久 buffer 同门失效：clear() 是请求边界标记
        #（prefill 前每层调用）——新请求首步 decode 必须全量重建容量宽
        # 常量 buffer。graph 步进路径（_tl_decode_batched_graph）不再每步
        # clear（见 patch），否则 _ginc 每步失效退化为全量重建。
        # 【E152v4-修复·根因】置 None 会让 dict 连同 k_min/k_max/k_avg/k_qat
        # 最后一组 Python 引用死亡 → allocator 回收 → 已 capture 的图 kernel
        # 仍持有旧指针（cross-repeat 复用图场景），后续大分配（新请求
        # prefill 的 dyn cache）复用并覆写该内存 → replay 静默读写垃圾
        #（e152v4 探针：repeat0 全对、repeat1 从中间层起 mask/KV 全错）。
        # 改为软失效：保留 buffer 引用 + stale 标记；_prepare_index_graph
        # 冷建分支对同形状旧 buffer 做 in-place 全量刷新（图地址稳定）。
        _ginc = getattr(self, "_ginc", None)
        if _ginc is not None:
            _ginc["stale"] = True

    def get_block_size(self):
        return 1


# ======================================================================
# 【E127 iter6/iter8】Triton fast path kernels（原 tli_indexer_fast.py
# 并入；_HAS_TRITON=False 时整体不定义，调用点守卫短路走原版表达式）。
# ======================================================================
if _HAS_TRITON:
    # 【E133c-A5】libdevice.rint：fp32 round-half-to-even（torch.round 的
    # nearbyint 口径），量化融合 kernel 逐位仿真用
    from triton.language.extra import libdevice  # noqa: E402

    @triton.jit
    def _score_fine_kernel(
        KQ, Q, OUT, MOUT,
        # KQ [T,Hkv,D] bf16；Q [H,D] fp32；OUT [H,T] out_dtype；
        # MOUT [Hkv,T] out_dtype（GQA 组均值；HAS_MEAN=False 时传 OUT 占位不写）
        T, stride_kt, stride_kh, stride_qh, stride_oh, stride_ot,
        stride_mh, stride_mt,
        Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
        HAS_MEAN: tl.constexpr,
        BATCHED: tl.constexpr = False, STRIDE_KB: tl.constexpr = 0,
    ):
        """GQA K 复用 score_fine：每 program 处理 (token 块 × 1 个 kv 头)，
        K 块 [BT, D] 载入寄存器后循环 g 个 q 头逐头点积，K 读带宽 /g。
        【AKO iter-21】HAS_MEAN=true 时同趟写 GQA 组均值 [Hkv,T]：均值
        必须按**落盘舍入值**（OUT 存的 out_dtype 位形）累加——消费端 torch
        mean(dim) 读的就是这些位；升序 j=0..G-1 fp32 累加 + /G（pow2 精确除）
        + RNE 收尾，与 PyTorch MeanOps 逐线程顺序归约同序（iter-11 /
        E133c-A4 50 轮探针实证同口径，iter-21 探针按本 kernel 位形复验）。
        NaN 语义 = torch mean 原样传播（不做 nan_to_num——与 softmax 输出
        域的 _ako_gqa_mean_kernel 不同，本 kernel 分数域无 NaN 清洗）。"""
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        kv = pid_h
        if BATCHED:
            pid_t = pid_t.to(tl.int64)
            pid_h = pid_h.to(tl.int64)
            batch = pid_h // Hkv
            kv = pid_h % Hkv
            KQ += batch * STRIDE_KB
        offs_t = pid_t * BT + tl.arange(0, BT)
        offs_d = tl.arange(0, D)
        mask_t = offs_t < T
        if HAS_MEAN:
            macc = tl.zeros((BT,), dtype=tl.float32)
        k = tl.load(KQ + offs_t[:, None] * stride_kt + kv * stride_kh + offs_d[None, :],
                    mask=mask_t[:, None], other=0.0).to(tl.float32)
        for j in tl.static_range(G):
            h = pid_h * G + j
            q = tl.load(Q + h * stride_qh + offs_d)      # [D] fp32（已乘 scale）
            s = tl.sum(k * q[None, :], axis=1)            # [BT] fp32
            s_o = s.to(OUT.dtype.element_ty)
            tl.store(OUT + h * stride_oh + offs_t * stride_ot,
                     s_o, mask=mask_t)
            if HAS_MEAN:
                macc += s_o.to(tl.float32)
        if HAS_MEAN:
            tl.store(MOUT + pid_h * stride_mh + offs_t * stride_mt,
                     (macc / G).to(MOUT.dtype.element_ty), mask=mask_t)

    @triton.jit
    def _score_coarse_kernel(
        KMIN, KMAX, Q, OUT, MOUT,
        # KMIN/KMAX [T,Hkv,D] bf16；Q [H,D] fp32；OUT [H,T]；
        # MOUT [Hkv,T]（GQA 组均值；HAS_MEAN=False 时传 OUT 占位不写）
        T, stride_kt, stride_kh, stride_kt2, stride_kh2,
        stride_qh, stride_oh, stride_ot, stride_mh, stride_mt,
        Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
        HAS_MEAN: tl.constexpr,
        BATCHED: tl.constexpr = False,
        STRIDE_KB: tl.constexpr = 0, STRIDE_KB2: tl.constexpr = 0,
    ):
        """【iter8】GQA K 复用 coarse minmax 上界分（min/max 两份 K 独立
        stride）：s[h,t] = Σ_d min(q,0)·k_min + Σ_d max(q,0)·k_max（负部配
        k_min、正部配 k_max，与原版 clamp 双 einsum 数学等价）。K 块
        [BT,D] 寄存器复用服务 g 个 q 头（读带宽 /g，免 repeat 物化）。
        【AKO iter-21】HAS_MEAN=true 同趟写 GQA 组均值——按落盘舍入值累加
        （等价性口径同 _score_fine_kernel 注释，iter-21 探针复验）。"""
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        kv = pid_h
        if BATCHED:
            # Widen before stride products, including flattened output heads.
            pid_t = pid_t.to(tl.int64)
            pid_h = pid_h.to(tl.int64)
            batch = pid_h // Hkv
            kv = pid_h % Hkv
            KMIN += batch * STRIDE_KB
            KMAX += batch * STRIDE_KB2
        offs_t = pid_t * BT + tl.arange(0, BT)
        offs_d = tl.arange(0, D)
        mask_t = offs_t < T
        if HAS_MEAN:
            macc = tl.zeros((BT,), dtype=tl.float32)
        kmin = tl.load(KMIN + offs_t[:, None] * stride_kt + kv * stride_kh
                       + offs_d[None, :],
                       mask=mask_t[:, None], other=0.0).to(tl.float32)   # [BT, D]
        kmax = tl.load(KMAX + offs_t[:, None] * stride_kt2 + kv * stride_kh2
                       + offs_d[None, :],
                       mask=mask_t[:, None], other=0.0).to(tl.float32)
        for j in tl.static_range(G):
            h = pid_h * G + j
            q = tl.load(Q + h * stride_qh + offs_d)      # [D] fp32（已缩放）
            q_low = tl.minimum(q, 0.0)                   # clamp(max=0)：负部 → k_min
            q_high = tl.maximum(q, 0.0)                  # clamp(min=0)：正部 → k_max
            s = tl.sum(kmin * q_low[None, :], axis=1) + tl.sum(kmax * q_high[None, :], axis=1)
            s_o = s.to(OUT.dtype.element_ty)
            tl.store(OUT + h * stride_oh + offs_t * stride_ot,
                     s_o, mask=mask_t)
            if HAS_MEAN:
                macc += s_o.to(tl.float32)
        if HAS_MEAN:
            tl.store(MOUT + pid_h * stride_mh + offs_t * stride_mt,
                     (macc / G).to(MOUT.dtype.element_ty), mask=mask_t)

    @triton.jit
    def _score_fine_cand_kernel(
        KQ, Q, CAND, CLEN, OUT,
        # KQ [T,Hkv,D] bf16；Q [H,D] fp32（已乘 scale）；CAND [Hkv,R] int64；
        # CLEN [Hkv] int64；OUT [H,R] out_dtype
        R, stride_kt, stride_kh, stride_qh, stride_ch, stride_cr,
        stride_oh, stride_or,
        Hkv: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
        BATCHED: tl.constexpr = False, STRIDE_KB: tl.constexpr = 0, CAP: tl.constexpr = 0,
    ):
        """【E156c】候选 fine 打分融合 kernel：每 program 处理 (候选列块 × 1
        kv 头)，cand_idx gather k 行 [BT,D] 载入寄存器后循环 g 个 q 头逐头
        点积（K 读带宽 /g，同 _score_fine_kernel）；-inf 门控（r ≥ CLEN[kv]
        的 pad 列）在 kernel 内完成，替代 einsum 路径的 where。

        数学：score[h, r] = Σ_d b_q_full[h, d]·kq[CAND[kv, r], kv, d]，与
        einsum(b_q_full, kq[CAND] 展开) 数学等价；fp32 累加顺序与 einsum
        不同 → bf16 舍入后 ULP 级漂移（与 _fast_score_fine 同口径）。
        """
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        if BATCHED:
            pid_h = pid_h.to(tl.int64)
            pid_t = pid_t.to(tl.int64)
        kv = pid_h
        if BATCHED:
            batch = (pid_h // Hkv).to(tl.int64)
            kv = pid_h % Hkv
            KQ += batch * STRIDE_KB
        offs_r = pid_t * BT + tl.arange(0, BT)
        offs_d = tl.arange(0, D)
        mask_r = offs_r < R
        cl = tl.load(CLEN + pid_h)
        read_r = mask_r
        if BATCHED:
            read_r = mask_r & (offs_r < cl)
        cand = tl.load(CAND + pid_h * stride_ch + offs_r * stride_cr,
                       mask=read_r, other=0)                     # [BT] int64
        if BATCHED:
            read_r = read_r & (cand >= 0) & (cand < CAP)
        k = tl.load(KQ + cand[:, None] * stride_kt + kv * stride_kh
                    + offs_d[None, :],
                    mask=read_r[:, None], other=0.0).to(tl.float32)   # [BT, D]
        for j in tl.static_range(G):
            h = pid_h * G + j
            q = tl.load(Q + h * stride_qh + offs_d)              # [D] fp32
            s = tl.sum(k * q[None, :], axis=1)                   # [BT] fp32
            res = tl.where(offs_r < cl, s, float("-inf"))
            tl.store(OUT + h * stride_oh + offs_r * stride_or,
                     res.to(OUT.dtype.element_ty), mask=mask_r)

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
        """【E131-S2（原 fast.py 并入）】tok_score 全融合：q_g =
        AGG_{j<G} q[h·G+j, idx_sub]（sum 或 mean=AGG_DIV·sum）→
        tok_score[h,t] = Σ_d q_g[h,d]·cent[h,a[h,t],d]。fp32 累加顺序
        ULP 口径见 _fast_tok_score。"""
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

    @triton.jit
    def _bf16_rne(x):
        """fp32 → RNE 舍入 bf16 → 回 fp32（PyTorch CUDA bf16 算子输出的
        opmath=fp32 逐算子口径，E133c-A5 量化仿真用）。"""
        return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _coarse_stat_kernel(
        K, IDXSUB, OMIN, OMAX, OAVG,
        # K [S,Hkv,D] bf16（k[0] 视图）；OMIN/OMAX/OAVG [1,cap,Hkv,DD] bf16；
        # IDXSUB [DD] int64（HAS_SUB=False 时占位不读）
        n_tok, lo, fb,
        stride_kt, stride_kh, stride_ot, stride_oh,
        Hkv: tl.constexpr, DD: tl.constexpr, BS: tl.constexpr,
        HAS_SUB: tl.constexpr, HAS_AVG: tl.constexpr,
    ):
        """【E133c-A4】尾段块统计融合：每 program 处理 1 个（尾块 × kv 头），
        按升序 i=0..BS-1 逐行 fp32 顺序累加 min/max/sum（与 PyTorch
        mean(dim) 实测同序，/tmp/e133c_numeric_probe.py 50 轮全过）；
        pad 行 mask other=0 复现全量版 tail_pad 零填充语义。"""
        pid_b = tl.program_id(0)
        h = tl.program_id(1)
        offs_d = tl.arange(0, DD)
        if HAS_SUB:
            sub = tl.load(IDXSUB + offs_d)                    # [DD] int64
        mn = tl.full((DD,), float("inf"), tl.float32)
        mx = tl.full((DD,), float("-inf"), tl.float32)
        s = tl.zeros((DD,), tl.float32)
        for i in tl.static_range(BS):
            msk = (offs_d >= 0) & ((pid_b * BS + i) < n_tok)  # 向量 mask
            if HAS_SUB:
                x = tl.load(K + (lo + pid_b * BS + i) * stride_kt
                            + h * stride_kh + sub,
                            mask=msk, other=0.0
                            ).to(tl.float32)                  # [DD]
            else:
                x = tl.load(K + (lo + pid_b * BS + i) * stride_kt
                            + h * stride_kh + offs_d,
                            mask=msk, other=0.0).to(tl.float32)
            mn = tl.minimum(mn, x)
            mx = tl.maximum(mx, x)
            s = s + x
        tl.store(OMIN + (fb + pid_b) * stride_ot + h * stride_oh + offs_d,
                 mn.to(OMIN.dtype.element_ty))
        tl.store(OMAX + (fb + pid_b) * stride_ot + h * stride_oh + offs_d,
                 mx.to(OMAX.dtype.element_ty))
        if HAS_AVG:
            tl.store(OAVG + (fb + pid_b) * stride_ot + h * stride_oh
                     + offs_d,
                     (s * (1.0 / BS)).to(OAVG.dtype.element_ty))

    @triton.jit
    def _stat_full_kernel(
        K, IDXSUB, OMIN, OMAX, OAVG,
        # K [1,S,D] bf16（k_for_coarse，已 pad 到 P*BS）；
        # OMIN/OMAX/OAVG [1,P,Hkv,DD] bf16; IDXSUB [DD] int64
        # （HAS_SUB=False 时占位不读）
        S, P, Hkv_idx,
        stride_kt, stride_kh,
        stride_ot, stride_oh,
        BS: tl.constexpr, DD: tl.constexpr,
        HAS_SUB: tl.constexpr, HAS_AVG: tl.constexpr,
    ):
        """【E176 全量块统计融合】每 program 处理 1 个（块 × kv 头），
        一次读取内同时算 min/max/mean —— 替代 rearrange + amin + amax
        + mean 四次读数据。只读一次 [BS, DD] 块数据，fp32 顺序累加，
        与 PyTorch mean(dim=2) 逐位一致（累加序同 _coarse_stat_kernel）。"""
        pid_b = tl.program_id(0)
        h = tl.program_id(1)
        offs_d = tl.arange(0, DD)
        if HAS_SUB:
            sub = tl.load(IDXSUB + offs_d)
        mn = tl.full((DD,), float("inf"), tl.float32)
        mx = tl.full((DD,), float("-inf"), tl.float32)
        s = tl.zeros((DD,), tl.float32)
        for i in tl.static_range(BS):
            tok = pid_b * BS + i
            msk = tok < S
            if HAS_SUB:
                x = tl.load(K + tok * stride_kt + h * stride_kh + sub,
                            mask=msk, other=0.0).to(tl.float32)
            else:
                x = tl.load(K + tok * stride_kt + h * stride_kh + offs_d,
                            mask=msk, other=0.0).to(tl.float32)
            mn = tl.minimum(mn, x)
            mx = tl.maximum(mx, x)
            s = s + x
        tl.store(OMIN + pid_b * stride_ot + h * stride_oh + offs_d,
                 mn.to(OMIN.dtype.element_ty))
        tl.store(OMAX + pid_b * stride_ot + h * stride_oh + offs_d,
                 mx.to(OMAX.dtype.element_ty))
        if HAS_AVG:
            tl.store(OAVG + pid_b * stride_ot + h * stride_oh + offs_d,
                     (s * (1.0 / BS)).to(OAVG.dtype.element_ty))

    @triton.jit
    def _qat_quant_kernel(
        K, IDXSUB, OUT,
        # K [S,Hkv,D] bf16（k[0] 视图）；IDXSUB [DD] int64；OUT=buf
        # [cap,Hkv,DD] bf16；直写 OUT[lo:lo+n_tok)
        n_tok, lo,
        stride_kt, stride_kh, stride_ot, stride_oh,
        Hkv: tl.constexpr, DD: tl.constexpr, BT: tl.constexpr,
    ):
        """【E133c-A5】minmax 4bit 量化融合：每 program 处理（token 块 ×
        1 kv 头），读 K 子空间维 → 逐 token min/max → 量化逐算子 bf16-RNE
        仿真（PyTorch CUDA opmath=fp32 口径逐位一致）→ 直写 buf 尾段。
        round-half-to-even = libdevice.rint（fp32 RN 默认模式）。"""
        pid_t = tl.program_id(0)
        h = tl.program_id(1)
        offs_t = pid_t * BT + tl.arange(0, BT)
        mask_t = offs_t < n_tok
        offs_d = tl.arange(0, DD)
        sub = tl.load(IDXSUB + offs_d)                       # [DD] int64
        x = tl.load(K + (lo + offs_t)[:, None] * stride_kt + h * stride_kh
                    + sub[None, :],
                    mask=mask_t[:, None], other=0.0).to(tl.float32)  # [BT,DD]
        mx = tl.max(x, axis=1)                               # [BT] 选择精确
        mn = tl.min(x, axis=1)
        # scales = (mx - mn).clamp(1e-9) / 15 —— 逐算子 bf16-RNE
        t1 = _bf16_rne(mx - mn)
        t2 = _bf16_rne(tl.maximum(t1, 1e-9))
        scales = _bf16_rne(t2 / 15.0)
        zeros = -mn                                           # bf16 取负精确
        # out = clamp(round((x+zeros)/scales), 0, 15)*scales - zeros
        u = _bf16_rne(x + zeros[:, None])
        v = _bf16_rne(u / scales[:, None])
        w = libdevice.rint(v)                                 # RNE-to-even
        w = tl.minimum(tl.maximum(w, 0.0), 15.0)
        p = _bf16_rne(w * scales[:, None])
        out = _bf16_rne(p - zeros[:, None])
        tl.store(OUT + (lo + offs_t)[:, None] * stride_ot + h * stride_oh
                 + offs_d[None, :], out.to(tl.bfloat16),
                 mask=mask_t[:, None])

    @triton.jit
    def _ako_rebuild_scatter_kernel(
        CAND, SF, BLK, CLEN, OUT, DUMMY,
        # CAND [Hkv,R] int64（候选 token 绝对位，kv 空间组内行恒同）；
        # SF [H,R] bf16（候选分，q-head 空间）；
        # BLK [Hkv,kt] bool（L1 块掩码快照）；CLEN [Hkv] int64（行实长）；
        # OUT [H,W] bf16（-inf 预填充连续重建目标，iter-8 哨兵列退役）；
        # DUMMY [H,1] bf16（无效槽牺牲位，内容永不读）
        R, total, W, S_real, bs_c, kt_c, g, sink, swa,
        stride_sfh, stride_sfr, stride_ch, stride_cr,
        stride_blh, stride_blk,
        BLOCK: tl.constexpr,
    ):
        """【AKO iter-5】fused 哨兵重建融合 kernel：单 launch 内完成
        keep 链（r<cand_len & (L1 块或保护区) & token<S_real）+ 无效槽路由 +
        候选分 scatter。纯整数/布尔/拷贝语义，无浮点归约参与，与 iter-4
        的 arange+compare+and+div+clamp+index_select+gather+where+
        scatter_ 九 kernel 链逐位等价（重复索引同值幂等）。
        【AKO iter-8】哨兵列（[W+1] 尾列路由+切片）退役：keep 槽直写
        [H,W] 连续目标（同值同址），无效槽写牺牲位 DUMMY[h]（重复地址
        竞写、内容丢弃，可见域零影响）→ 重建输出逐位同旧且连续，
        softmax 免 strided 切片触发的 contiguous clone。"""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < total
        h = offs // R
        r = offs - h * R
        # 【AKO iter-20】CAND/CLEN 为 kv-head 空间（组内行恒同）→ 行
        # 地址 h//g 折叠；BLK 查表同折叠，三者一致免 q-head 物化。
        kv = h // g
        idx = tl.load(CAND + kv * stride_ch + r * stride_cr, mask=m, other=0)
        cl = tl.load(CLEN + kv, mask=m, other=0)
        keep = (r < cl) & (idx < S_real)
        bid = tl.minimum(idx // bs_c, kt_c - 1)
        bkeep = tl.load(BLK + (h // g) * stride_blh + bid * stride_blk,
                        mask=m, other=0)
        protected = (idx < sink) | (idx >= tl.maximum(0, S_real - swa))
        keep = keep & ((bkeep != 0) | protected)
        val = tl.load(SF + h * stride_sfh + r * stride_sfr,
                      mask=m, other=float("-inf"))
        tl.store(OUT + h * W + idx, val, mask=m & keep)
        tl.store(DUMMY + h, val, mask=m & (~keep))

    @triton.jit
    def _ako_l1_sort_kernel(
        BLK, OUT,
        # BLK [R_l,nb] int64（L1 块号，topk 产出互异整数）；
        # OUT [R_l,nb] int64（升序排序输出）
        nb, stride_b, stride_o,
        NB_POW2: tl.constexpr,
    ):
        """【AKO iter-9】L1 块号行内升序排序：块号互异整数 → 升序唯一
        → tl.sort 与 torch.sort 逐位恒等（互异值排序无 tie，探针 200/200
        验证含哨兵 pad）。nb 非 2 幂时哨兵 1<<30（> 任何真实块号）pad 排
        尾，store 只写前 nb 个实块。每行一个 program，纯整数排序无浮点。"""
        r = tl.program_id(0)
        o = tl.arange(0, NB_POW2)
        v = tl.load(BLK + r * stride_b + o, mask=o < nb, other=1 << 30)
        tl.store(OUT + r * stride_o + o, tl.sort(v), mask=o < nb)

    @triton.jit
    def _ako_cand_sort_kernel(
        IALL, SCR, CNT,
        # IALL [Hkv,n_l1] int64（L1 topk 块号互异）；SCR [Hkv,NB_POW2]
        # int64（升序输出）；CNT [Hkv] int64（每 head 有效 L1 token 数）
        # 【AKO iter-22】bypass 块号 b0/b1 标量化（n_bp ≤ 2 wrapper 门禁），
        # 消 bp as_tensor H2D；-1 哨兵臂 mask 恒假不进 v。
        n_l1, n_bp, b0, b1, s_r, bs,
        stride_ia, stride_sc,
        NB_POW2: tl.constexpr,
    ):
        """【AKO iter-17 K1】候选构建 per-head 排序 + cnt 归约（消
        torch.sort ~26us + bypass cat + clamp.sum 3 处 aten 调用）。
        行内 L1 ∪ bypass 合并升序排序：哨兵 1<<30 pad 尾（互异块号无
        tie → tl.sort 与 torch.sort 逐位恒等，iter-9 同构）；cnt =
        Σ_b clamp(s_r−b·bs, 0, bs)，哨兵项恒 0（s_r−(1<<30)·bs ≪ 0）。
        纯整数域无浮点 → 与 iter-16 torch 链逐位恒等。"""
        h = tl.program_id(0)
        o = tl.arange(0, NB_POW2)
        BIG = 1 << 30
        v1 = tl.load(IALL + h * stride_ia + o, mask=o < n_l1, other=BIG)
        # bypass 段标量直选（iter-22）：o==n_l1 → b0、o==n_l1+1 → b1，
        # n_bp 门（o < n_l1+n_bp）外一律哨兵——与旧 BP 指针 masked load
        # 逐位同值（bypass 块号 < kt ≪ BIG）
        v2 = tl.where(o < n_l1 + n_bp,
                      tl.where(o == n_l1, b0, b1), BIG)
        v = tl.where(o < n_l1, v1, v2)
        vs = tl.sort(v)
        nb = n_l1 + n_bp
        tl.store(SCR + h * stride_sc + o, vs, mask=o < nb)
        contrib = tl.minimum(tl.maximum(s_r - vs * bs, 0), bs)
        tl.store(CNT + h, tl.sum(contrib))

    @triton.jit
    def _ako_cand_build_kernel(
        SCR, CNT, CAND,
        # SCR [Hkv,NB_POW2] int64（K1 升序块号）；CNT [Hkv] int64；
        # CAND [Hkv,R_max] int64（输出候选行）
        # 【AKO iter-22】共享段下沉：SH [M] 指针 → 4 段 (lo,w) 标量
        # （bypass×2 + sink + swa，段序 = 旧 torch cat 程序序），段内
        # token = lo + (k−段前缀宽)，全 host 已知宽度，免 arange×4 +
        # cat + as_tensor H2D。逐位恒等：标量段算术 ≡ arange(lo,lo+w)
        # 连接（纯整数域）。
        nb, bs, M, R_max,
        lo0, w0, lo1, w1, lo2, w2, lo3, w3,
        stride_sc, stride_cd,
        BLOCK: tl.constexpr,
    ):
        """【AKO iter-17 K2】候选行装配单趟化（消 arange(bs) 展开 + 直填
        copy + shared index_put_ + zeros fill ≈ 10 launch）。列语义与
        iter-16 torch 链程序序恒等：
          j < cnt                → L1 token（升序前缀）
          cnt ≤ j < cnt+M        → 共享段 token（覆盖 L1 pad 尾段，同旧
                                  index_put 列区间）
          其余 j < nb·bs         → L1 token（截断块尾值原样保留，旧直填
                                  语义；cand_len 门控下不可见）
          其余                   → 0（旧 new_zeros 语义）
        单列单写无同址序问题；2D grid (Hkv, cdiv(R_max,BLOCK)) 消 8 CTA
        低占用（iter-14 教训）。"""
        h = tl.program_id(0)
        c = tl.program_id(1)
        j = c * BLOCK + tl.arange(0, BLOCK)
        m_j = j < R_max
        cnt = tl.load(CNT + h)
        nbbs = nb * bs
        in_l1 = j < nbbs
        b = j // bs
        blkv = tl.load(SCR + h * stride_sc + b, mask=in_l1, other=0)
        l1t = blkv * bs + (j - b * bs)
        in_sh = (j >= cnt) & (j < cnt + M)
        k = j - cnt
        c0 = w0
        c1 = c0 + w1
        c2 = c1 + w2
        sht = tl.where(k < c0, lo0 + k,
                       tl.where(k < c1, lo1 + (k - c0),
                                tl.where(k < c2, lo2 + (k - c1),
                                         lo3 + (k - c2))))
        val = tl.where(in_sh, sht, tl.where(in_l1, l1t, 0))
        tl.store(CAND + h * stride_cd + j, val, mask=m_j)

    @triton.jit
    def _ako_topk_mask_kernel(
        IND, OUT,
        # IND [.., heads, K] int64（topk 块索引）；OUT [heads, kt] bool
        K, kt, b0, b1,
        stride_ind_h, stride_ind_k, stride_out_h,
        BLOCK: tl.constexpr,
    ):
        """【AKO iter-19】topk_mask 装配单趟化：zeros_like fill + scatter_
        + ones_like fill + bypass setitem ×2 ≈ 5 launch → 1。纯 bool/整数
        域，与 torch 装配程序序恒等：全 False 基座 + 索引/bypass 位置置
        True（topk 索引互异、bypass 与池收缩后索引不相交 → 无位置竞争；
        barrier 保证基座 False 先于 True 可见——_ako_l2_mask_kernel
        iter-14 同款已验证模式）。K ≤ 128（bypass ≤ 2，-1 哨兵）由
        wrapper 门禁，超界回退 torch 装配链。"""
        h = tl.program_id(0)
        c = tl.program_id(1)
        o = c * BLOCK + tl.arange(0, BLOCK)
        tl.store(OUT + h * stride_out_h + o,
                 tl.zeros([BLOCK], dtype=tl.int1), mask=o < kt)
        tl.debug_barrier()
        j = tl.arange(0, 128)
        m_j = j < K
        idx = tl.load(IND + h * stride_ind_h + j * stride_ind_k,
                      mask=m_j, other=-1)
        pos = idx - c * BLOCK
        hit = m_j & (pos >= 0) & (pos < BLOCK)
        # 地址用绝对列 idx（hit 保证 idx 落在本 CTA 列区间 c*BLOCK+pos）
        tl.store(OUT + h * stride_out_h + idx, tl.full([128], 1, tl.int1),
                 mask=hit)
        # bypass 块：条件折叠进 mask（-1 哨兵恒假），免 scalar-if 控制流
        tl.store(OUT + h * stride_out_h + o, tl.full([BLOCK], 1, tl.int1),
                 mask=(b0 >= 0) & (o == b0))
        tl.store(OUT + h * stride_out_h + o, tl.full([BLOCK], 1, tl.int1),
                 mask=(b1 >= 0) & (o == b1))

    @triton.jit
    def _ako_l2_mask_kernel(
        CIF, SELF, CIN, SELN, OUT,
        # CIF/CIN [R,R_cand] int64（候选 token 绝对位，连续）；
        # SELF/SELN [R,kf]/[R,kn] int64（cand_p topk 的行内选择位）；
        # OUT [R,W] bool（全宽 L2 mask）
        kf, kn, scf, scn, W, sink, swa_lo, s_real,
        KF_POW2: tl.constexpr, KN_POW2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """【AKO iter-10】L2 mask 装配尾部单趟化：back-gather（cand_idx
        .gather(sel)×2）+ zeros 填充 + 2× scatter_ + sink/swa 强制区
        5+2 kernel → 1 launch。纯整数/幂等 bool 置位语义（同址重复写
        同值 True）→ 与旧 torch 链逐位恒等；哨兵 t=W 落界外由 store
        mask 截断。探针 4 形状（含 kn=0 / sink=0 / 非 2 幂 W）对拍全 PASS。
        【AKO iter-14】2D grid (R, cdiv(W, BLOCK)) 化——旧 1D grid (R,)
        每行一 program 串行扫 W/BLOCK 个切片，R=8 行时仅 8 CTA 低占用；
        2D 化后每 program：① 写本切片基座（w<sink | swa_lo≤w<s_real →
        True），切片不相交覆盖全 w<W；② tl.debug_barrier()（__syncthreads
        CTA 内全局写序：base 写先于 scatter 写可见）；③ far/near 散置
        scatter 的 store mask 加 (t≥lo_s)&(t<lo_s+BLOCK) 只写落本切片
        的目标 → 每地址恰被一个 program 写 base、同 program barrier 后
        写 scatter True → 与旧单 program「先 base 后 scatter」程序序逐位
        恒等，零跨 CTA 同址写。探针 5 形状 ×3 重复对拍全 PASS。"""
        h = tl.program_id(0)
        c = tl.program_id(1)
        w = c * BLOCK + tl.arange(0, BLOCK)
        v = (w < sink) | ((w >= swa_lo) & (w < s_real))
        tl.store(OUT + h * W + w, v, mask=w < W)
        tl.debug_barrier()
        lo_s = c * BLOCK
        o = tl.arange(0, KF_POW2)
        mk = o < kf
        s = tl.load(SELF + h * kf + o, mask=mk, other=0)
        t = tl.load(CIF + h * scf + s, mask=mk, other=W)
        in_f = mk & (t < W) & (t >= lo_s) & (t < lo_s + BLOCK)
        tl.store(OUT + h * W + t, True, mask=in_f)
        o2 = tl.arange(0, KN_POW2)
        mn = o2 < kn
        s2 = tl.load(SELN + h * kn + o2, mask=mn, other=0)
        t2 = tl.load(CIN + h * scn + s2, mask=mn, other=W)
        in_n = mn & (t2 < W) & (t2 >= lo_s) & (t2 < lo_s + BLOCK)
        tl.store(OUT + h * W + t2, True, mask=in_n)

    @triton.jit
    def _ako_gqa_mean_kernel(
        IN, OUT,
        # IN [R*G, W] bf16 连续（softmax 输出）；OUT [R, W] bf16（GQA 组均值）
        W,
        swa_lo, s_real,
        G: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """【AKO iter-11】softmax 输出 nan_to_num + GQA 组 mean 单趟融合
        （2 kernel + strided reduce ~12us → 1 kernel）。等价性：v!=v → 0
        与 nan_to_num(nan=0) 同（softmax 输出域 [0,1]∪{NaN}，无 ±inf 支路）；
        bf16→fp32 精确，g 维 fp32 顺序累加（g=0..G-1 内存序）与 PyTorch
        MeanOps 逐线程顺序归约同序；G 为 2 的幂 → 1/G 精确（除法无舍入）；
        fp32→bf16 RNE 收尾同。per_q_head / G 非 2 幂 / 非连续回退 torch 链
        逐位不变。
        【AKO iter-13】swa 强制区折叠：swa_lo≤w<s_real 位直写 1.0（旧
        路径为 mean 后独立 setitem fill 覆写 [max(0,S_real-swa_tok),
        S_real)=1.0 → 强制区位恒 1.0 与 fill 值逐位同（bf16(1.0) 精确）；
        padding 段 w≥s_real 两路均为 mean（fill 从未触及）→ 逐位恒等）。
        非 host int 几何由调用方传 (0,0)（forced 恒空）+ 保留旧 fill 行
        fail-closed。"""
        h = tl.program_id(0)
        w = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = w < W
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for g in range(G):
            v = tl.load(IN + (h * G + g) * W + w, mask=m, other=0.0).to(tl.float32)
            v = tl.where(v != v, 0.0, v)
            acc += v
        val = (acc / G).to(tl.bfloat16)
        forced = (w >= swa_lo) & (w < s_real)
        val = tl.where(forced, 1.0, val)
        tl.store(OUT + h * W + w, val, mask=m)

    @triton.jit
    def _ako_cand_expand_kernel(
        BLK, P, IDX, OUT,
        # BLK [R_l,nb] int64（升序排序后的 L1 块号）；P [R_l,S_w]（softmax
        # 概率行）；IDX/OUT [R_l,R_cand]（候选 token 绝对位 / 对应 p 值）
        R_cand, nb, S_w,
        ba_lo, ba_hi, bb_lo, bb_hi,
        stride_b, stride_p, stride_i, stride_o,
        BS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """【AKO iter-7】候选域展开融合 kernel：单 launch 完成块升序展开
        （tok = BLK[j]·BS + o）× bypass 段拼接（≤2 段、宿主序=旧 cat 序、
        段内升序）× p.gather。纯整数/拷贝语义、无浮点参与，与 mul+add+
        arange+cat+gather 五 kernel 链逐位等价（输出布局逐位一致 →
        cand_p 逐位一致 → 下游 topk tie-break 逐位不变）。"""
        r = tl.program_id(0)
        c = tl.program_id(1)
        offs = c * BLOCK + tl.arange(0, BLOCK)
        m = offs < R_cand
        nbb = nb * BS
        is_byp = offs >= nbb
        in_exp = m & ~is_byp
        # ---- 展开区：j = offs // BS, o = offs % BS ----
        j = offs // BS
        o = offs - j * BS
        blk = tl.load(BLK + r * stride_b + j, mask=in_exp, other=0)
        tok = blk * BS + o
        # ---- bypass 区：段 a→b（宿主序），段内升序 ----
        rb = offs - nbb
        len_a = ba_hi - ba_lo
        len_ab = len_a + (bb_hi - bb_lo)
        in_byp = m & is_byp & (rb < len_ab)
        tok_byp = tl.where(rb < len_a, ba_lo + rb,
                           bb_lo + rb - len_a).to(tl.int64)
        tok = tl.where(is_byp, tok_byp, tok)
        # ---- 界内统一 load/store（展开 tok/ bypass tok 均构造性 < S_w）----
        mm = in_exp | in_byp
        v = tl.load(P + r * stride_p + tok, mask=mm, other=0)
        tl.store(IDX + r * stride_i + offs, tok, mask=mm)
        tl.store(OUT + r * stride_o + offs, v, mask=mm)

    @triton.jit
    def _ako_cand_expand_sort_kernel(
        BLK, P, IDX, OUT,
        # BLK [R_l,nb] int64（未排序 L1 块号，互异）；P [R_l,S_w]；
        # IDX/OUT [R_l,R_cand]（候选 token 绝对位 / 对应 p 值）
        R_cand, nb, S_w,
        ba_lo, ba_hi, bb_lo, bb_hi,
        stride_p,
        BS: tl.constexpr, NB_POW2: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """【AKO iter-15】iter-9 `_ako_l1_sort_kernel` + iter-7
        `_ako_cand_expand_kernel` 融合（每池 2 launch → 1）：每列切片
        program 冗余 load 本行未排序块号（mask 补哨兵 1<<30 pad NB_POW2）
        → tl.sort 升序（互异块号升序唯一 → 与独立 kernel 写回的
        sorted blk 逐位恒等）→ tl.gather 寄存器动态取 BLK[j]（j =
        offs//BS 先 clamp nb-1：in_exp lane 构造性 j≤nb-1 不受影响；
        bypass lane j≥nb 的 gather 越界防 → 该 lane blk 值被 tok_byp
        覆写不可见）→ 展开/bypass 拼接/p.gather 与 iter-7 kernel 逐字
        同构。纯整数/拷贝语义零浮点参与。NB_POW2<2 退化（nb=1）不进
        本 kernel（调用方守卫，tl.sort 单元素编译 bug 回避）。冗余排序
        为纯寄存器 bitonic（NB_POW2≤256），列切片间并行吸收。"""
        r = tl.program_id(0)
        c = tl.program_id(1)
        # ---- 冗余排序段：本行升序块号进寄存器 ----
        ob = tl.arange(0, NB_POW2)
        row = tl.load(BLK + r * nb + ob, mask=ob < nb, other=1 << 30)
        row = tl.sort(row)
        # ---- 展开段（iter-7 同构）----
        offs = c * BLOCK + tl.arange(0, BLOCK)
        m = offs < R_cand
        nbb = nb * BS
        is_byp = offs >= nbb
        in_exp = m & ~is_byp
        j = offs // BS
        j = tl.minimum(j, nb - 1)
        o = offs - j * BS
        blk = tl.gather(row, j, 0)
        tok = blk * BS + o
        rb = offs - nbb
        len_a = ba_hi - ba_lo
        len_ab = len_a + (bb_hi - bb_lo)
        in_byp = m & is_byp & (rb < len_ab)
        tok_byp = tl.where(rb < len_a, ba_lo + rb,
                           bb_lo + rb - len_a).to(tl.int64)
        tok = tl.where(is_byp, tok_byp, tok)
        mm = in_exp | in_byp
        v = tl.load(P + r * stride_p + tok, mask=mm, other=0)
        tl.store(IDX + r * R_cand + offs, tok, mask=mm)
        tl.store(OUT + r * R_cand + offs, v, mask=mm)

    # ---------------- 【E166 iter3a】终掩码装配融合（graph 路径专用） ----------------
    @triton.jit
    def _iter3_sel_base_kernel(SWAF, MIDM, K2MID, VALID, OUT,
                               S_cap, sink_tok, Hkv, K2_static, stride_k,
                               BLOCK: tl.constexpr):
        """iter3a kernel A：基底直写。OUT[h,t] = ((t<sink_tok) | SWAF[t]
        | 哨兵位) & VALID[t]——【iter3f-5】valid_mask 内联消 host 侧
        `sel & (ar_tok < S_real)`（省 1 kernel/层 + 1MB/层读写）。
        VALID[t] = (t < S_real)。纯布尔域 bitexact。"""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < S_cap
        swa = tl.load(SWAF + offs, mask=m, other=0)
        base = (offs < sink_tok) | (swa != 0)
        mid = tl.load(MIDM + offs, mask=m, other=0)
        valid = tl.load(VALID + offs, mask=m, other=0)
        is_last = offs == (S_cap - 1)
        for h in range(Hkv):
            kk = tl.load(K2MID + h * stride_k)
            base_h = (base | (is_last & (mid != 0) & (kk < K2_static))) & valid
            tl.store(OUT + h * S_cap + offs, base_h, mask=m)

    @triton.jit
    def _iter3_sel_scatter_kernel(TOKTOP, K2MID, MIDM, OUT,
                                  K2_static, stride_k, stride_h, S_cap,
                                  BK: tl.constexpr):
        """iter3a kernel B：选中位条件写。k<min(k2_mid,K2_static) 且
        mid[tok] 的槽位写 True——与旧链 `scatter(i_eff) & mid_mask` 同集
        （哨兵 k≥k2_mid 位由 kernel A 兜底）。val=False 不写 → 无写
        冲突；重复 tok（理论不可达：候选互异）同值幂等。纯布尔/整数域。"""
        h = tl.program_id(0)
        k0 = tl.program_id(1) * BK
        offs_k = k0 + tl.arange(0, BK)
        km = offs_k < K2_static
        kk = tl.load(K2MID + h * stride_k)
        kkeep = offs_k < kk
        tok = tl.load(TOKTOP + h * stride_h + offs_k, mask=km, other=-1)
        tm = km & kkeep & (tok >= 0) & (tok < S_cap)
        mid = tl.load(MIDM + tok, mask=tm, other=0)
        val = tm & (mid != 0)
        tl.store(OUT + h * S_cap + tok, val, mask=val)



    # ---------------- 【E166 iter3b】候选压缩扫描 kernel ----------------
    @triton.jit
    def _iter3_compact_kernel(CIDX, KEEP, MIDM,
                              CIDX_OUT, KEEP_OUT, MID_OUT, CLEN,
                              R_cap, S_CAP_M1: tl.constexpr,
                              BLOCK: tl.constexpr, WITH_MID: tl.constexpr = True):
        """iter3b：行内稳定压缩。每 program 一行（kv head）：
        pass1 算 keep 总数 total；pass2 keep=True 槽写前缀目的位
        （carry+cumsum-1）、False 槽写 total+序数目的位（尾部保序），
        同步输出 mid 值（MIDM[idx] 随槽位置换）——与
        argsort(~keep,stable)+gather×2+sum+mid gather 逐位一致
        （含 garbage 区内容保序复刻）。纯整数/布尔域。
        【iter3f-3】哨兵重定向内联：keep=False 槽写 S_CAP_M1 而非原值
        （替代 host 侧 torch.where(keep2, cand_idx2, S_cap-1)），
        省 1 kernel/层 + [Hkv,R_cap] int64 读+写。"""
        r = tl.program_id(0)
        base = r * R_cap
        total = tl.zeros((), tl.int32)
        for c0 in range(0, R_cap, BLOCK):
            offs = c0 + tl.arange(0, BLOCK)
            m = offs < R_cap
            k = tl.load(KEEP + base + offs, mask=m, other=0).to(tl.int32)
            total += tl.sum(k, axis=0)
        carry = tl.zeros((), tl.int32)
        gcarry = tl.zeros((), tl.int32)
        for c0 in range(0, R_cap, BLOCK):
            offs = c0 + tl.arange(0, BLOCK)
            m = offs < R_cap
            k = tl.load(KEEP + base + offs, mask=m, other=0).to(tl.int32)
            idx = tl.load(CIDX + base + offs, mask=m, other=0)
            # Redirect inactive IDs BEFORE any indirect read. Preserve the
            # sentinel's MID value as in the stable argsort reference chain.
            idx = tl.where(k != 0, idx, S_CAP_M1)
            if WITH_MID:
                mid = tl.load(MIDM + idx, mask=m, other=0)
            ksum = tl.cumsum(k, axis=0)
            gsum = tl.cumsum(1 - k, axis=0)
            kp = carry + ksum - 1
            gp = total + gcarry + gsum - 1
            pos = tl.where(k != 0, kp, gp)
            w = m
            # 【iter3f-3】哨兵重定向：keep=False → S_cap-1（恒在 swa 区）
            sidx = tl.where(k != 0, idx, S_CAP_M1)
            tl.store(CIDX_OUT + base + pos, sidx, mask=w)
            tl.store(KEEP_OUT + base + pos, (k != 0), mask=w)
            if WITH_MID:
                tl.store(MID_OUT + base + pos, (mid != 0), mask=w)
            carry += tl.sum(k, axis=0)
            gcarry += tl.sum(1 - k, axis=0)
        tl.store(CLEN + r, total.to(tl.int64))

    # ---------------- 【E166 iter3d】摘要维护融合 kernel ----------------
    @triton.jit
    def _iter3_maint_kernel(K, POS, IDXSUB, KMIN, KMAX, KAVG, KQAT,
                            S_cap, sk_t, sk_h, sm_t, sm_h, sq_t, sq_h,
                            NEED_AVG: tl.constexpr, BS: tl.constexpr,
                            DSUB: tl.constexpr, BATCHED: tl.constexpr = False,
                            ALIVE=None, Hkv: tl.constexpr = 0,
                            STRIDE_KB: tl.constexpr = 0,
                            STRIDE_MINB: tl.constexpr = 0,
                            STRIDE_MAXB: tl.constexpr = 0,
                            STRIDE_AVGB: tl.constexpr = 0,
                            STRIDE_QB: tl.constexpr = 0):
        """iter3d：增量步摘要维护单核融合。每 program 一 kv head：
        加载新 token 所在块 [BS,DSUB] 子空间 tile（越界 +0 垫，与原
        where(zero_k) 同），min/max 序无关逐位一致，mean fp32 累加
        （序与 torch reduce 的逐位一致性由单测+e2e bitexact 裁决），
        k_min/k_max/k_avg 行直写（替代三 reduce+三 index_copy_）、
        k_qat 当前行直写（tile 外独立行载）。值域=纯拷贝+序无关归约
        +常量垫零。"""
        if BATCHED:
            pid = tl.program_id(0).to(tl.int64)
            b, h = pid // Hkv, pid % Hkv
            pos = tl.load(POS + b).to(tl.int64)
            active = tl.load(ALIVE + b) & (pos >= 0) & (pos < S_cap)
            # Widen the program ID before all batch/head stride products.
            K = K + b * STRIDE_KB
            KMIN = KMIN + b * STRIDE_MINB
            KMAX = KMAX + b * STRIDE_MAXB
            KAVG = KAVG + b * STRIDE_AVGB
            KQAT = KQAT + b * STRIDE_QB
        else:
            h = tl.program_id(0)
            pos = tl.load(POS).to(tl.int64)
            active = True
        fb = pos // BS
        offs_t = tl.arange(0, BS)
        offs_d = tl.arange(0, DSUB)
        isub = tl.load(IDXSUB + offs_d).to(tl.int64)
        tok = fb * BS + offs_t.to(tl.int64)
        mt = tok < S_cap
        ptr = K + tok[:, None] * sk_t + h * sk_h + isub[None, :]
        v = tl.load(ptr, mask=mt[:, None] & active, other=0.0)
        tl.store(KMIN + fb * sm_t + h * sm_h + offs_d,
                 tl.min(v, axis=0), mask=active)
        tl.store(KMAX + fb * sm_t + h * sm_h + offs_d,
                 tl.max(v, axis=0), mask=active)
        if NEED_AVG:
            acc = tl.sum(v.to(tl.float32), axis=0) / BS
            tl.store(KAVG + fb * sm_t + h * sm_h + offs_d,
                     acc.to(v.dtype), mask=active)
        rp = K + pos * sk_t + h * sk_h + isub
        rv = tl.load(rp, mask=active, other=0.0)
        tl.store(KQAT + pos * sq_t + h * sq_h + offs_d, rv, mask=active)


    # ---------------- 【E166 iter3d】L1 双池 scored 融合 kernel ----------------
    @triton.jit
    def _iter3_l1_scored_kernel(FSRC, NSRC, FPM, NPM, FOUT, NOUT,
                                kt_cap, BLOCK: tl.constexpr):
        """iter3d：L1 双池门控融合。原 far/near scored 各 1 where +
        1 full_like（4 kernel/层，[1,1,Hkv,kt_cap]）→ 单核双输出。
        逐位等价：out = mask ? src : -inf（纯元素拷贝+常量，浮点值
        不动 → 下游 topk 输入逐位同）。"""
        h = tl.program_id(0)
        pid = tl.program_id(1)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < kt_cap
        fm = tl.load(FPM + offs, mask=m, other=0) != 0
        nm = tl.load(NPM + offs, mask=m, other=0) != 0
        base = h * kt_cap
        fv = tl.load(FSRC + base + offs, mask=m, other=0.0)
        nv = tl.load(NSRC + base + offs, mask=m, other=0.0)
        ninf = float("-inf")
        tl.store(FOUT + base + offs, tl.where(fm, fv, ninf), mask=m)
        tl.store(NOUT + base + offs, tl.where(nm, nv, ninf), mask=m)


    # ---------------- 【E166 iter3d】L1 候选装配融合 kernel ----------------
    @triton.jit
    def _iter3_l1_cand_kernel(IF, IN_, FPM, NPM, BYT, BYK, SREAL,
                              CIDX, KEEP,
                              nb_far, nb_near, R_cap,
                              K1: tl.constexpr, K1P: tl.constexpr,
                              BS: tl.constexpr):
        """iter3d：候选集装配单核融合。原链 cat(blks)+cat(in_pool)+
        l1_tok 乘加+l1_keep lt&and+cat(cand)+cat(keep)（~8 kernel/层）
        → 单核直出 cand_idx2/keep2 [Hkv,R_cap]。槽布局复刻：
        l1 区 slot=k*bs+t、byp 区 slot=K1*bs+s*bs+t。纯整数/布尔域 →
        bitexact by construction。【E172q 修复】tl.arange 要求 2 的幂
        → offs_k 取 K1P=next_pow2(K1)，越界 lane（offs_k≥K1）全部
        load/store 加 mask；K1 本身为 2 的幂时 K1P==K1、mask 全真，
        kernel 行为与原版逐位一致。"""
        h = tl.program_id(0)
        offs_k = tl.arange(0, K1P)
        kv_ = offs_k < K1
        offs_t = tl.arange(0, BS)
        isf = offs_k < nb_far
        fb_ = tl.load(IF + h * nb_far + offs_k, mask=isf, other=0)
        nb_ = tl.load(IN_ + h * nb_near + (offs_k - nb_far),
                      mask=(~isf) & kv_, other=0)
        blk = tl.where(isf, fb_, nb_)
        okf = tl.load(FPM + blk, mask=isf, other=0) != 0
        okn = tl.load(NPM + blk, mask=(~isf) & kv_, other=0) != 0
        ok = tl.where(isf, okf, okn)
        sreal = tl.load(SREAL).to(tl.int64)
        tok = blk[:, None] * BS + offs_t.to(tl.int64)[None, :]
        kp = (tok < sreal) & ok[:, None]
        base = h * R_cap
        slot = offs_k.to(tl.int64)[:, None] * BS + offs_t.to(tl.int64)[None, :]
        # 【E180b】pow2 K1（K1P==K1）走原版无 mask store——谓词化 store
        # 实测 +15us/层（Triton 不折叠 constexpr 比较，E180 A/B k64
        # +0.55ms 回归）；非幂次才需越界 lane mask。
        if K1P == K1:
            tl.store(CIDX + base + slot, tok)
            tl.store(KEEP + base + slot, kp)
        else:
            tl.store(CIDX + base + slot, tok, mask=kv_[:, None])
            tl.store(KEEP + base + slot, kp, mask=kv_[:, None])
        offs_b = tl.arange(0, 2 * BS)
        bt = tl.load(BYT + offs_b)
        bk = tl.load(BYK + offs_b) != 0
        tl.store(CIDX + base + K1 * BS + offs_b, bt)
        tl.store(KEEP + base + K1 * BS + offs_b, bk)

def _iter3_maint(k0, pos, idx_sub, k_min, k_max, k_avg, k_qat,
                 S_cap, bs, need_avg):
    """【E166 iter3d】摘要维护 wrapper。k0=[S_cap,Hkv,D_full] 视图；
    k_min/k_max/k_avg=[kt_cap,Hkv,DSUB]、k_qat=[S_cap,Hkv,DSUB]（调用方
    已门控连续+同 DSUB+indices is idx_sub）。原位更新，返回 True。"""
    Hkv = k0.shape[1]
    DSUB = idx_sub.numel()
    _iter3_maint_kernel[(Hkv,)](
        k0, pos, idx_sub, k_min, k_max,
        k_avg if need_avg else k_min,  # 占位指针（NEED_AVG=False 不写）
        k_qat,
        S_cap, k0.stride(0), k0.stride(1),
        k_min.stride(0), k_min.stride(1),
        k_qat.stride(0), k_qat.stride(1),
        NEED_AVG=need_avg, BS=bs, DSUB=DSUB,
        num_warps=4)
    return True


def _iter3_l1_scored(far_src, near_src, fp_mask, np_mask, kt_cap, Hkv):
    """【E166 iter3d】L1 双池 scored wrapper。far_src/near_src
    [1,1,Hkv,kt_cap] 连续（调用方门控），fp/np_mask [kt_cap] bool。
    返回 (far_scored, near_scored)。"""
    f_out = torch.empty_like(far_src)
    n_out = torch.empty_like(near_src)
    BLOCK = 1024
    _iter3_l1_scored_kernel[(Hkv, triton.cdiv(kt_cap, BLOCK))](
        far_src, near_src, fp_mask, np_mask, f_out, n_out,
        kt_cap, BLOCK=BLOCK, num_warps=4)
    return f_out, n_out


def _iter3_l1_cand(i_f, i_n, far_pool_mask, near_pool_mask,
                   byp_tok, byp_keep, S_real,
                   K1, nb_far, bs, R_cap, Hkv, dev):
    """【E166 iter3d】候选装配 wrapper。i_f/i_n [1,1,Hkv,n] 连续，
    byp_tok/byp_keep [1,1,1,2,bs] 连续，S_real 0-dim long。
    返回 (cand_idx2, keep2)。"""
    nb_near = K1 - nb_far
    cidx = torch.empty((Hkv, R_cap), dtype=torch.long, device=dev)
    keep = torch.empty((Hkv, R_cap), dtype=torch.bool, device=dev)
    # 【E172q 修复】K1P = next_pow2(K1)：K1 为 2 的幂时 ==K1（原行为），
    # 非幂次（如 96）填充到 128，越界 lane 由 kernel 内 mask 排除。
    K1P = 1 << (K1 - 1).bit_length()
    _iter3_l1_cand_kernel[(Hkv,)](
        i_f, i_n, far_pool_mask, near_pool_mask, byp_tok, byp_keep,
        S_real, cidx, keep,
        nb_far, nb_near, R_cap,
        K1=K1, K1P=K1P, BS=bs, num_warps=4)
    return cidx, keep

def _iter3_assemble_mask(tok_top, k2_mid, mid_mask, swa_force,
                         sink_tok, S_cap, Hkv, K2_static, dev,
                         valid_mask=None):
    """【E166 iter3a+iter3f-5】compute_mask_graph 尾部装配融合 wrapper。

    输入：tok_top [Hkv,K2_static] int64（topk 候选 token，k≥k2_mid 位
    视为哨兵）、k2_mid（标量或 [Hkv]，每 head 有效槽上限）、
    mid_mask/swa_force [S_cap] bool（共享跨 head）、sink_tok int、
    valid_mask [S_cap] bool（ar_tok < S_real；None → 全 1 占位）。
    输出：sel [1,1,Hkv,S_cap] bool —— 与旧六段 aten 链逐位一致
    （含哨兵位语义，人工几何 fail-closed 兜底）。
    【iter3f-5】valid_mask 传入 sel_base_kernel 内联 AND，省 host
    侧 `sel & (ar_tok < S_real)`（1 kernel/层 + 1MB 读写）。
    """
    sel = torch.empty((1, 1, Hkv, S_cap), dtype=torch.bool, device=dev)
    flat = sel.view(-1)                       # [Hkv*S_cap] 连续
    k2h = k2_mid.reshape(-1)
    if not k2h.is_contiguous():
        k2h = k2h.contiguous()
    stride_k = k2h.stride(0) if k2h.numel() > 1 else 0
    if valid_mask is not None and valid_mask.is_contiguous():
        _valid = valid_mask
    else:
        _valid = torch.ones(S_cap, dtype=torch.bool, device=dev)
    _iter3_sel_base_kernel[(triton.cdiv(S_cap, 4096),)](
        swa_force, mid_mask, k2h, _valid, flat,
        S_cap, sink_tok, Hkv, K2_static, stride_k,
        BLOCK=4096, num_warps=4)
    if K2_static > 0:
        BK = 256
        _iter3_sel_scatter_kernel[(Hkv, triton.cdiv(K2_static, BK))](
            tok_top, k2h, mid_mask, flat,
            K2_static, stride_k, tok_top.stride(0), S_cap,
            BK=BK, num_warps=4)
    return sel


def _iter3_compact(cand_idx2, keep2, mid_mask, R_cap, Hkv, S_cap, dev):
    """【E166 iter3b】候选压缩 wrapper。返回 (cand_idx2c, keep2c,
    mid_slotc, cand_len)——与旧 argsort 链+mid gather 逐位一致。
    【iter3f-3】S_cap 用于哨兵重定向 S_CAP_M1（替代 host 侧 where）。"""
    mode=os.environ.get('TLI_COMPACT_SCAN_TILED','0')
    if mode not in ('0','1'):
        raise ValueError('TLI_COMPACT_SCAN_TILED must be 0 or 1')
    if mode=='1':
        if cand_idx2.shape!=(Hkv,R_cap) or cand_idx2.device!=dev:
            raise ValueError('compact metadata mismatch')
        from ..ops.tiled_compact import compact_tiled
        return compact_tiled(cand_idx2,keep2,mid_mask,S_cap)
    ci_out = torch.empty_like(cand_idx2)
    kp_out = torch.empty_like(keep2)
    mid_out = torch.empty_like(keep2) if mid_mask is not None else None
    clen = torch.empty((Hkv,), dtype=torch.long, device=dev)
    # 【E166e】launch 配置调优：BLOCK=2048/num_warps=8（扫描 16 组，
    # 33.5→24.7us/层；kernel 语义不动，13 例单测保障）
    _iter3_compact_kernel[(Hkv,)](
        cand_idx2, keep2, mid_mask if mid_mask is not None else keep2,
        ci_out, kp_out, mid_out if mid_out is not None else kp_out, clen,
        R_cap, S_CAP_M1=S_cap - 1, BLOCK=2048, WITH_MID=mid_mask is not None, num_warps=8)
    return ci_out, kp_out, mid_out, clen
