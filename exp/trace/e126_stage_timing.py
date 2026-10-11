#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E126：五臂 TLI 选择器分阶段计时 microbench（成本指纹）——优化前基线（E127 对照组）。

== 目的 ==
对 E125 五臂冻结配置（mavg/aavg/mminmax/cavgsim/cclustersim）测出 TLI 选择器
各生产阶段（S0-S8）的 decode 每步成本指纹 + 建簇成本摊销曲线。本套件测的是
**当前 main 基线**（E127 kernel 优化合入前的对照数字）；E127 验收合入后须
复测出优化后数字，两者成对进 E128 综合表。

== 阶段边界（基于 tli_indexer.py，md5 锚点 bf5e26c4d6123875b2786b9ed886da5f，
   插桩前逐 anchor 硬校验，失配即 GATE-FAIL）==
  S0  pad + 子空间准备          prepare_index L277-291
  S1  L1 块统计 + 4bit 量化     L291-349（decode 每步增量）
  S2  建簇/簇维护               _maybe_build_kmeans L368-409（仅 cavg 系；
      内 _update_far_greedy L437 / _build_far_kmeans L412 / _update_near_cluster L484）
  S3  q 侧打分                  compute_score L682-783（minmax / avg / 细筛 GEMV 分开计）
  S4  簇分数 + L1 准备          compute_mask 内 _far_token_score L785 / _near_token_score
                                  L806 调用区（仅 cavg 系）+ L1 topk 入口准备 L957-977
  mask_setup 分区几何/D'/聚合    compute_mask L865-956（辅助桶，不计入 S0-S8 命名阶段）
  S5  L1 topk                   L970 起（双池 L978 / 单池 L1030）
  S6  softmax p                  L1070-1079
  S7  L2 分区预算 topk           L1150-1372
  S8  最终 attention             eager_decoding_attn（sparse_attn/ops/eager_decoding.py，
                                  qwen3_attn_patch.py:140 的生产消费端）

== 插桩方式（严禁修改生产文件）==
运行时 AST 插桩：读 sparse_attn/indexer/tli_indexer.py 源码 → 在上述 anchor 行
前插入 `self._e126_mark('<TAG>')`（torch.cuda.Event record）→ exec 为独立命名空间
模块。生产文件字节零改动（SHA256/md5 全程可验）；anchor 行内容逐条硬校验，
tli_indexer.py 升级（E127 合入）后行号漂移会 fail-closed 而非静默错标。
开关：环境变量 TLI_STAGE_TIMING=1（默认 0 = mark 全 no-op）。
阶段耗时 = 相邻 mark 的 cuda Event elapsed_time（GPU 时间线，含 host-GPU 空转）。

== 测量协议 ==
  数据：合成 microbench（继承 E64c/E113 口径）——structured（base+noise 混合簇
        C/N≈1/8）+ singleton（纯高斯单例 C/N≈1 最坏）两臂；KV dtype=bf16（Qwen3
        生产口径），S ∈ {4K, 8K, 16K, 32K, 64K}。
  decode 模拟：B=1 q_len=1，每步 k 增长 1 token；steps=128（≥100），warm=3
        （每 rep 前 3 步剔除），rep=10；每 rep 新建 indexer（clear 等价——
        簇缓存跨 rep 不复用，rep0 含冷建簇，供摊销曲线）。
  计时：阶段 span 用 cuda Event（elapsed_time, ms）；每步选择器 wall 用
        time.perf_counter_ns() 整数纳秒原始样本（055 口径：不排序、不舍入、
        按执行顺序持久化，median 由已持久化样本派生）。
  摊销曲线（R2c）：rep0 全 128 步逐步记录 S2/S2_far/S2_near span 与累计值——
        验证块对齐缓存命中（历史 sanity 锚点：不缓存时 61s/step 级反常）。
  阶段求和口径（TL-E126-STAGE-DOUBLECOUNT-001 修复后）：selector_total_ms_
        median_est 与 amortization.selector_ms = 互斥阶段和（EXCLUSIVE_STAGE_
        KEYS 白名单）；S2_far/S2_near 为 S2 的子区间细分，只作 breakdown 不
        重复计入求和。修正值语义 = 互斥阶段和，非 wall time。
  正确性门（fail-closed）：mask 无全 False 行、attention 输出 finite、每头选中
        token 数在 [sink+swa, K2+sink+swa+块宽余量] 合理区间；任一失败 →
        非零退出 + failure.json，性能 JSON 不发布。
  身份闭包：git SHA + tli_indexer.py SHA256/md5 + greedy_triton.py SHA256 +
        harness SHA + 五臂冻结配置 + anchor 校验记录 全落 manifest。
  发布：临时文件 + fsync + os.replace 原子写 + .sha256 sidecar + 输出路径
        flock 跨进程互斥（继承 e113 054/055/056 协议精简版）。

== 红绿测试 ==
exp/trace/test_e126_stage_timing.py：T1 anchor+身份（CPU）；T2 计时开/关同输入
逐位一致（GPU）；T3 mark 序列与 span 非负（GPU）；T4 env 关 → 零事件（CPU+GPU）。

用法：
  TLI_STAGE_TIMING=1 CUDA_VISIBLE_DEVICES=0 python3 exp/trace/e126_stage_timing.py \
      --arms mavg,aavg,mminmax,cavgsim,cclustersim
  python3 exp/trace/e126_stage_timing.py --summary   # 汇总五臂 JSON → e126_summary.json
输出：exp/trace/results/e126_stage_timing_{arm}.json（+ .sha256）；
      --summary → exp/trace/results/e126_summary.json
"""
import argparse
import ast
import fcntl
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
import types

sys.dont_write_bytecode = True

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

# ---- 身份锚点（E125/E116b 在飞实验的 tli_indexer.py 冻结 md5；校验失败 = 基线漂移，
#      须人工确认新基线而非静默继续）----
EXPECTED_TLI_MD5 = "bf5e26c4d6123875b2786b9ed886da5f"

# ===================================================================== #
# 五臂冻结配置（E125 派单口径，勿改）。预算固定 K1=128(page=64)/K2=1024/
# cmp_ratio=4；Hkv=8，D=128，H=32（GQA G=4）；子空间 tail32（dd=32）。
# 公共：tli_enable_layer_skip=False（microbench 单层无 D'；生产层掩码文件
# 不进依赖）、tli_sim_dims=subspace（= tail32）、swa=128/sink=2 块（硬编码）。
# ===================================================================== #
COMMON_ARGS = dict(
    tia_block_size=64, tia_level1_topk=128, tia_level2_topk=1024,
    tia_level2_cmp_ratio=4, tia_enable_async_topk=False,
    tli_subspace="tail", tli_enable_subspace=True,
    tli_enable_layer_skip=False, tli_layer_skip_path=None,
    tli_sim_dims="subspace", tli_sim=0.9,
    tli_sigma_select="none", tli_moba=False, tli_per_q_head=False,
    tli_static_pair=False, tli_proj_basis=None,
    tli_far_clusters=256, tli_far_niter=10,
    tli_far_blocks=16, tli_far_tokens=512,
)

ARM_CONFIGS = {
    "mavg":       dict(far_method="minmax", near_method="avg",
                       far_select="4bit", near_select="4bit",
                       enable_kmeans=False, alpha=0.25, beta=0.125, gamma=0.625),
    "aavg":       dict(far_method="avg", near_method="avg",
                       far_select="4bit", near_select="4bit",
                       enable_kmeans=False, alpha=0.125, beta=0.125, gamma=0.375),
    "mminmax":    dict(far_method="minmax", near_method="minmax",
                       far_select="4bit", near_select="4bit",
                       enable_kmeans=False, alpha=0.875, beta=0.875, gamma=0.125),
    "cavgsim":    dict(far_method="minmax", near_method="avg",
                       far_select="sim_greedy", near_select="4bit",
                       enable_kmeans=True, alpha=0.0, beta=0.0, gamma=0.0),
    "cclustersim": dict(far_method="minmax", near_method="avg",
                        far_select="sim_greedy", near_select="sim_greedy",
                        enable_kmeans=True, alpha=0.125, beta=0.25, gamma=0.125),
}

SIZES = (4096, 8192, 16384, 32768, 65536)
N_STEPS = 128          # ≥100
WARM = 3               # 每 rep 前 3 步剔除（冷建簇在 rep0 摊销曲线中保留）
REPS = 10              # ≥10
DATA_KINDS = ("structured", "singleton")
HKV, D_MODEL, H_Q = 8, 128, 32
G_Q = H_Q // HKV        # GQA G=4

# ===================================================================== #
# AST 插桩：anchor 表（行号, TAG, 行内容前缀）——全部对着
# sparse_attn/indexer/tli_indexer.py @ md5 bf5e26c4 逐条人工核实。
# 任一 anchor 行内容失配 → 插桩前 fail-closed（行号漂移 ≠ 静默错标）。
# ===================================================================== #
ANCHORS = [
    (278,  "S0",      "from .utils import prepare_pad_mask, pad_tensor"),   # prepare_index 入口
    (322,  "S1",      "k_coarse = rearrange("),          # S0→S1：块统计开始（非投影路径）
    (363,  "S2_in",   "if (self.enable_kmeans and not self.skip_far"),    # S2 建簇区入口
    (690,  "S3",      "if q.shape[0] == 1 and q.shape[1] == 1:"),         # compute_score 入口
    (753,  "S3_avg",  "score_coarse_avg = None"),        # S3 细分：avg 分数源段
    (761,  "S3_fine", "fine_shape = (1, H, k_qat.shape[1])"),  # S3 细分：细筛 GEMV 段
    (865,  "CM_in",   'score_coarse = score_dict["score_coarse"]'),        # compute_mask 入口
    (957,  "S4",      "far_tok_score = None"),           # 簇分数调用区（cavg 系）
    (970,  "S5",      'score_coarse[..., -1] = float("inf")'),  # L1 topk 段
    (1070, "S6",      'score_fine = torch.where(topk_mask, score_fine, float("-inf"))'),  # softmax p 段（L1069 slice 划入 S5）
    (1150, "S7",      "if use_partition:"),              # L2 分区 topk 段
    (446,  "S2_far",  "if self._km_greedy_sums is not None and self._km_far_hi_cached == far_hi:"),  # _update_far_greedy 入口
    (493,  "S2_near", "S = k.shape[1]"),                 # _update_near_cluster 入口
]
# 函数出口 mark（Return 语句前插）：prepare_index→S2_out / compute_score→S3_out /
# compute_mask→CM_out（Return 节点由 ast 收集，非硬编码行号）
RETURN_MARKS = {"prepare_index": "S2_out", "compute_score": "S3_out", "compute_mask": "CM_out"}

# 阶段 span = 相邻 mark 对。mask_setup 为辅助桶（分区几何/D' 剔除/GQA 聚合），
# 不在 S0-S8 命名里但如实单独记录。
STAGE_SPANS = [
    ("S0",         ("S0", "S1")),
    ("S1",         ("S1", "S2_in")),
    ("S2",         ("S2_in", "S2_out")),
    ("S2_far",     ("S2_far", "S2_near")),   # near 缺席时由消费端回退 (S2_far→S2_out)
    ("S2_near",    ("S2_near", "S2_out")),
    ("S3_coarse",  ("S3", "S3_avg")),
    ("S3_avg",     ("S3_avg", "S3_fine")),
    ("S3_fine",    ("S3_fine", "S3_out")),
    ("mask_setup", ("CM_in", "S4")),
    ("S4",         ("S4", "S5")),
    ("S5",         ("S5", "S6")),
    ("S6",         ("S6", "S7")),
    ("S7",         ("S7", "CM_out")),
]

# ===================================================================== #
# 互斥阶段白名单（TL-E126-STAGE-DOUBLECOUNT-001 修复）。
# S2_far/S2_near 是 S2 的**子区间**细分（S2 = 前导段 + S2_far + S2_near，
# 父子关系非互斥阶段），只作 breakdown 展示；任何「阶段求和」若把父 S2 与
# 子段一并累加即重复计时。故 selector_total / selector_ms 等求和口径一律
# 用 EXCLUSIVE_STAGE_KEYS（= 全部 STAGE_SPANS 名去掉子段键）。
# 语义边界：修正值 = 互斥阶段和（中位数和 / 逐步和），**不是** wall time。
# ===================================================================== #
SUB_STAGE_KEYS = frozenset({"S2_far", "S2_near"})
EXCLUSIVE_STAGE_KEYS = frozenset(name for name, _ in STAGE_SPANS) - SUB_STAGE_KEYS


def exclusive_stage_sum(step_spans):
    """单步 span 字典的互斥阶段求和（排除 S2_far/S2_near 子段；S2 已覆盖其区间）。"""
    return sum(v for k, v in step_spans.items()
               if v is not None and k in EXCLUSIVE_STAGE_KEYS)


def exclusive_stage_median_sum(stage_raw):
    """互斥阶段中位数和（selector_total_ms_median_est 口径；非 wall time）。"""
    return sum(median(v) for k, v in stage_raw.items()
               if v and k in EXCLUSIVE_STAGE_KEYS)

HELPER_BLOCK = '''

# ==== E126 stage-timing instrumentation（e126_stage_timing.py 运行时注入；
#      生产文件未改动；TLI_STAGE_TIMING=0 时 mark 全 no-op）====
_E126_TIMING = os.environ.get("TLI_STAGE_TIMING", "0").strip() == "1"
_E126_EVENTS = []


def _e126_mark(self, tag):
    if _E126_TIMING:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        _E126_EVENTS.append((tag, ev))


TLIIndexer._e126_mark = _e126_mark
'''


class AnchorMismatchError(SystemExit):
    pass


def build_instrumented_source(src, src_path):
    """AST/text 混合插桩：anchor 行前插 mark + Return 节点前插出口 mark。

    只在独立模块副本上执行；src_path 生产文件字节零改动。"""
    lines = src.split("\n")
    for ln, tag, prefix in ANCHORS:
        actual = lines[ln - 1].strip()
        if not actual.startswith(prefix):
            raise AnchorMismatchError(
                f"[GATE-FAIL] E126 anchor 失配 tag={tag} line={ln}: "
                f"expect prefix {prefix!r}, got {actual!r}（tli_indexer.py 已漂移，"
                f"须人工核对阶段边界后更新 ANCHORS）")
    tree = ast.parse(src)
    cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TLIIndexer":
            cls = node
            break
    if cls is None:
        raise AnchorMismatchError("[GATE-FAIL] 未找到 TLIIndexer 类定义")
    inserts = [(ln, tag) for ln, tag, _ in ANCHORS]
    for fname, tag in RETURN_MARKS.items():
        fns = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == fname]
        if len(fns) != 1:
            raise AnchorMismatchError(f"[GATE-FAIL] 类内函数 {fname} 定位数={len(fns)}")
        n_ret = 0
        for node in ast.walk(fns[0]):
            if isinstance(node, ast.Return):
                inserts.append((node.lineno, tag))
                n_ret += 1
        if n_ret == 0:
            raise AnchorMismatchError(f"[GATE-FAIL] {fname} 无 Return 节点")
    inserts.sort(key=lambda x: -x[0])   # 降序插入防行号漂移
    out = list(lines)
    for ln, tag in inserts:
        indent = re.match(r"\s*", out[ln - 1]).group(0)
        out.insert(ln - 1, f"{indent}self._e126_mark({tag!r})")
    return "\n".join(out) + "\n" + HELPER_BLOCK


def load_instrumented(root, pkg_tag="sparse_attn_e126"):
    """以独立命名空间包加载插桩模块；相对导入（.tia_indexer/.utils/..info）解析到
    root 下真实生产文件（未插桩、未改动）。返回 (module, identity_info)。"""
    tli_path = os.path.join(root, "sparse_attn", "indexer", "tli_indexer.py")
    with open(tli_path, encoding="utf-8") as f:
        src = f.read()
    inst_src = build_instrumented_source(src, tli_path)
    # 命名空间包注册（同 e113 load_sparse_attn 手法）
    if pkg_tag not in sys.modules:
        pkg = types.ModuleType(pkg_tag)
        pkg.__path__ = [os.path.join(root, "sparse_attn")]
        sys.modules[pkg_tag] = pkg
    sub_name = pkg_tag + ".indexer"
    if sub_name not in sys.modules:
        sub = types.ModuleType(sub_name)
        sub.__path__ = [os.path.join(root, "sparse_attn", "indexer")]
        sub.__package__ = pkg_tag
        sys.modules[sub_name] = sub
    ops_name = pkg_tag + ".ops"
    if ops_name not in sys.modules:
        ops = types.ModuleType(ops_name)
        ops.__path__ = [os.path.join(root, "sparse_attn", "ops")]
        ops.__package__ = pkg_tag
        sys.modules[ops_name] = ops
    mod_name = sub_name + ".tli_indexer_inst"
    mod = types.ModuleType(mod_name)
    mod.__file__ = tli_path
    mod.__package__ = sub_name
    sys.modules[mod_name] = mod
    exec(compile(inst_src, tli_path + "<e126-inst>", "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------- 工具（继承 e113 精简版）
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_identity(root):
    def run(*cmd):
        try:
            p = subprocess.run(["git", "-C", root, *cmd],
                               capture_output=True, text=True, timeout=60)
            return p.stdout.strip() if p.returncode == 0 else None
        except Exception:
            return None
    sha = run("rev-parse", "HEAD")
    if sha is None:
        return {"git_sha": None, "git_dirty": None,
                "note": f"{root} 非 git 仓库（rsync 部署）；权威身份 = 传入 --git-sha + impl sha256"}
    porc = run("status", "--porcelain", "--", ".")
    return {"git_sha": sha, "git_dirty": bool(porc),
            "dirty_files": porc.splitlines()[:20] if porc else []}


def collect_versions():
    v = {"torch": torch.__version__, "cuda": torch.version.cuda,
         "python": sys.version.split()[0]}
    try:
        import triton
        v["triton"] = triton.__version__
    except Exception:
        v["triton"] = None
    drv = None
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name",
                            "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=30)
        if p.returncode == 0 and p.stdout.strip():
            parts = p.stdout.strip().splitlines()[0].split(", ")
            drv, v["gpu"] = parts[0], parts[1]
    except Exception:
        pass
    v["driver"] = drv
    if "gpu" not in v:
        v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return v


def canonical_bytes(results):
    r2 = {k: v for k, v in results.items() if k != "output_content_sha256"}
    return json.dumps(r2, indent=1, ensure_ascii=False).encode("utf-8")


def output_content_sha256(results):
    return hashlib.sha256(canonical_bytes(results)).hexdigest()


def verify_output_sha256(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    claimed = d.pop("output_content_sha256", None)
    return claimed is not None and claimed == output_content_sha256(d)


def atomic_write_bytes(path, data):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _fail(msg, code=3):
    print("[STATE-FAIL] " + msg, flush=True)
    raise SystemExit(code)


def supersede_file(path):
    if not os.path.exists(path):
        return None
    tag = file_sha256(path)[:8]
    cand = f"{path}.attempt-{tag}.superseded"
    n = 1
    while os.path.exists(cand):
        cand = f"{path}.attempt-{tag}-{n}.superseded"
        n += 1
    os.replace(path, cand)
    return cand


def acquire_attempt_output_lock(out):
    path = os.path.realpath(out) + ".attempt.lock"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def release_attempt_output_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def median(samples):
    s = sorted(samples)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def tensor_md5_check(root):
    p = os.path.join(root, "sparse_attn", "indexer", "tli_indexer.py")
    md5 = file_md5(p)
    return md5


# ---------------------------------------------------------------- mark 消费与阶段 span
def expected_tag_sequence(arm_cfg):
    """每 decode 步的期望 mark 序列（fail-closed 校验）。"""
    seq = ["S0", "S1", "S2_in"]
    if arm_cfg["enable_kmeans"]:
        seq.append("S2_far")
        if arm_cfg["near_select"] in ("cluster", "sim_greedy"):
            seq.append("S2_near")
    seq += ["S2_out", "S3", "S3_avg", "S3_fine", "S3_out",
            "CM_in", "S4", "S5", "S6", "S7", "CM_out"]
    return seq


def drain_stage_spans(mod, expected_seq, tag):
    """消费 mark 事件 → {stage: ms}。序列失配 fail-closed。"""
    evs = list(mod._E126_EVENTS)
    mod._E126_EVENTS.clear()
    seq = [t for t, _ in evs]
    if seq != expected_seq:
        raise RuntimeError(
            f"[GATE-FAIL] mark 序列失配 ({tag})：expect {expected_seq}，got {seq}")
    if not evs:
        return {}
    spans = {}
    for i in range(len(evs) - 1):
        t0, e0 = evs[i]
        t1, e1 = evs[i + 1]
        spans[(t0, t1)] = e0.elapsed_time(e1)
    out = {}
    for name, pair in STAGE_SPANS:
        if pair in spans:
            out[name] = spans[pair]
    # ---- S2 区间分段求和：cluster 臂的 S2_far/S2_near mark 把 S2 区间切成
    #      [S2_in→S2_far]+[S2_far→S2_near]+[S2_near→S2_out]，主桶须为全长
    #      （非 cluster 臂无中间 mark，(S2_in,S2_out) 已直接命中上方分支）----
    if "S2" not in out:
        segs = []
        cur = "S2_in"
        for nxt in ("S2_far", "S2_near", "S2_out"):
            if (cur, nxt) in spans:
                segs.append(spans[(cur, nxt)])
                cur = nxt
        if cur == "S2_out" and segs:
            out["S2"] = sum(segs)
    # S2_far 出口回退：near 缺席时 = S2_far→S2_out
    if "S2_far" not in out and ("S2_far", "S2_out") in spans:
        out["S2_far"] = spans[("S2_far", "S2_out")]
    return out


# ---------------------------------------------------------------- 数据生成（E113 口径移植）
def gen_structured_kv(T, seed, device, n_base=None, noise=0.10):
    """base+noise 混合簇（C/N≈1/8），[1, T, Hkv, D] bf16——E113 gen_structured 移植
    到 KV 形状（D=128 全维；聚类/粗筛在 tail32 上由生产代码自取）。"""
    g = torch.Generator(device=device).manual_seed(seed)
    n_base = n_base or max(T // 8, 1)
    base = torch.randn(n_base, HKV, D_MODEL, generator=g, device=device)
    idx = torch.randint(0, n_base, (T,), generator=g, device=device)
    x = base[idx] + noise * torch.randn(T, HKV, D_MODEL, generator=g, device=device)
    return x.to(torch.bfloat16).unsqueeze(0).contiguous()


def gen_singleton_kv(T, seed, device):
    """纯高斯单例（C/N≈1 最坏 K 扫描量）。"""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(T, HKV, D_MODEL, generator=g, device=device)
    return x.to(torch.bfloat16).unsqueeze(0).contiguous()


# ---------------------------------------------------------------- decode 模拟
def make_namespace(arm_cfg):
    ns = dict(COMMON_ARGS)
    ns.update(dict(
        tli_far_method=arm_cfg["far_method"], tli_near_method=arm_cfg["near_method"],
        tli_far_select=arm_cfg["far_select"], tli_near_select=arm_cfg["near_select"],
        tli_enable_kmeans=arm_cfg["enable_kmeans"],
        tli_alpha=arm_cfg["alpha"], tli_beta=arm_cfg["beta"], tli_gamma=arm_cfg["gamma"],
    ))
    return argparse.Namespace(**ns)


def run_sim(mod, eager_fn, arm_cfg, k_all, v_all, S, n_steps, device,
            q_seed=20261011, timing=True, collect_gate=False):
    """单次 decode 模拟：返回 (wall_ns_per_step, s8_ms_per_step, stage_samples,
    amort_timeline, gate_info, last_outputs)。

    timing=False 用于干净 e2e wall（mark 全 no-op）。
    """
    exp_seq = expected_tag_sequence(arm_cfg)
    args = make_namespace(arm_cfg)
    indexer = mod.TLIIndexer(args)
    mod._E126_EVENTS.clear()
    mod._E126_TIMING = timing
    softmax_scale = D_MODEL ** -0.5
    wall_ns, s8_ms, stage_samples, timeline = [], [], [], []
    gate = {"min_row_any": None, "max_row_any": None, "out_finite": None,
            "sel_tokens_mean": None}
    last = {}
    for step in range(n_steps):
        St = S + step
        k = k_all[:, :St]
        v = v_all[:, :St]
        cu = torch.tensor([0, St], dtype=torch.int32, device=device).cumsum(0)
        q_ids = cu[1:] - cu[:-1] - 1        # [St-1]（生产 prepare_seqlens-1 口径）
        gq = torch.Generator(device=device).manual_seed(q_seed + step)
        q = torch.randn(1, 1, H_Q, D_MODEL, generator=gq, device=device).to(torch.bfloat16)
        ev_a = torch.cuda.Event(enable_timing=True)
        ev_b = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        mask, bs = indexer.prepare_mask(q, q_ids, k, cu)
        t1 = time.perf_counter_ns()
        ev_a.record()
        out = eager_fn(q, k, v, mask, bs, cu, softmax_scale=softmax_scale)
        ev_b.record()
        torch.cuda.synchronize()
        if timing:
            spans = drain_stage_spans(mod, exp_seq, f"step{step}")
        else:
            # 计时关：mark 全 no-op，无事件可消费（零校验零开销）
            assert len(mod._E126_EVENTS) == 0
            spans = {}
        wall_ns.append(t1 - t0)
        s8_ms.append(ev_a.elapsed_time(ev_b))
        stage_samples.append(spans)
        timeline.append(spans)
        if collect_gate and step == n_steps - 1:
            # 正确性门（末步）：mask 无全 False 行 + 输出 finite + 选中数合理
            m0 = mask.squeeze(0)[0]          # [H 或 Hkv, T]（TLI token 级 mask，bs=1）
            row_any = m0.any(dim=-1)
            gate["min_row_any"] = bool(row_any.all())
            gate["max_row_any"] = bool(row_any.any())
            per_head = m0.float().sum(dim=-1)   # 每行（head）选中 token 数
            gate["sel_tokens_min"] = float(per_head.min())
            gate["sel_tokens_max"] = float(per_head.max())
            gate["sel_tokens_mean"] = float(per_head.mean())
            gate["out_finite"] = bool(torch.isfinite(out.float()).all())
            gate["mask_shape"] = list(mask.shape)
            last["mask_md5"] = hashlib.md5(
                mask.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
            last["out_md5"] = hashlib.md5(
                out.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()
    mod._E126_TIMING = os.environ.get("TLI_STAGE_TIMING", "0").strip() == "1"
    return wall_ns, s8_ms, stage_samples, timeline, gate, last


# ---------------------------------------------------------------- 主流程
def bench_arm(arm, mod, eager_fn, identity, args, device):
    arm_cfg = ARM_CONFIGS[arm]
    results_cases = []
    for data_kind in DATA_KINDS:
        for S in args.sizes:
            seed = 7 if data_kind == "structured" else 11
            if data_kind == "structured":
                k_all = gen_structured_kv(S + N_STEPS, seed, device)
            else:
                k_all = gen_singleton_kv(S + N_STEPS, seed, device)
            v_all = k_all.clone()
            # ---- 正确性门（timing on，末步采样）----
            wall0, s8_0, st0, tl0, gate, _ = run_sim(
                mod, eager_fn, arm_cfg, k_all, v_all, S, n_steps=8, device=device,
                timing=True, collect_gate=True)
            gate_errors = []
            if not gate["min_row_any"]:
                gate_errors.append("mask 存在全 False 行（softmax NaN 契约破坏）")
            if not gate["out_finite"]:
                gate_errors.append("attention 输出非 finite")
            expect_min = 128 + 128          # sink+swa 强制下界
            expect_max = 1024 + 128 + 128 + 64 * 4 + 16
            if not (expect_min - 1 <= gate["sel_tokens_min"] <= expect_max):
                gate_errors.append(f"每头选中 token 数下界异常 min={gate['sel_tokens_min']}")
            if not (expect_min - 1 <= gate["sel_tokens_max"] <= expect_max * 4):
                gate_errors.append(f"每头选中 token 数上界异常 max={gate['sel_tokens_max']}")
            if gate_errors:
                fail_doc = {"status": "failed", "arm": arm, "data": data_kind,
                            "S": S, "errors": gate_errors, "gate": gate,
                            "identity": identity,
                            "timestamp": time.strftime("%F %T")}
                fail_path = args.out + ".failure.json"
                atomic_write_bytes(fail_path, json.dumps(
                    fail_doc, indent=1, ensure_ascii=False).encode("utf-8"))
                print("[GATE-FAIL] " + json.dumps(gate_errors, ensure_ascii=False),
                      flush=True)
                raise SystemExit(1)

            # ---- 正式 reps（timing on）----
            stage_raw = {name: [] for name, _ in STAGE_SPANS}
            wall_raw, s8_raw = [], []
            amort = None
            for rep in range(REPS):
                wall_ns, s8_ms, samples, timeline, _, _ = run_sim(
                    mod, eager_fn, arm_cfg, k_all, v_all, S, n_steps=N_STEPS,
                    device=device, timing=True)
                if rep == 0:
                    amort = timeline          # rep0 全 128 步（含冷建簇）做摊销曲线
                for i in range(WARM, N_STEPS):
                    wall_raw.append(wall_ns[i])
                    s8_raw.append(s8_ms[i])
                    for name in stage_raw:
                        v = samples[i].get(name)
                        if v is not None:      # 非 cluster 臂无 S2_far/S2_near mark
                            stage_raw[name].append(v)
            # ---- 干净 e2e wall（timing off，1 rep）----
            wall_clean, s8_clean, _, _, _, _ = run_sim(
                mod, eager_fn, arm_cfg, k_all, v_all, S, n_steps=N_STEPS,
                device=device, timing=False)
            e2e_clean_ns = int(sum(wall_clean) + sum(s8_clean) * 1e6)  # 粗粒度合并口径

            rec = {
                "arm": arm, "data": data_kind, "S": S,
                "steps": N_STEPS, "warm": WARM, "reps": REPS,
                "samples_per_stage": REPS * (N_STEPS - WARM),
                "gate_last_step": gate,
                # ---- 055：原始样本按执行顺序持久化（不排序不舍入）----
                "stage_ms_samples": {k: v for k, v in stage_raw.items()},
                "wall_ns_selector_samples": wall_raw,
                "s8_ms_samples": s8_raw,
                "e2e_128step_wall_ns_timing_off": e2e_clean_ns,
                # ---- 派生量（从已持久化原始样本派生）----
                "stage_ms_median": {k: (median(v) if v else None)
                                    for k, v in stage_raw.items()},
                "wall_ns_selector_median": median(wall_raw),
                "s8_ms_median": median(s8_raw),
                # 互斥阶段中位数和（白名单：S2_far/S2_near 子段不计入，
                # 修复 TL-E126-STAGE-DOUBLECOUNT-001 父子重复求和）
                "selector_total_ms_median_est": exclusive_stage_median_sum(stage_raw),
                # ---- 摊销曲线（rep0 全 128 步）----
                "amortization": {
                    "step_S": [S + i for i in range(N_STEPS)],
                    "s2_ms": [t.get("S2") for t in amort],
                    "s2_far_ms": [t.get("S2_far") for t in amort],
                    "s2_near_ms": [t.get("S2_near") for t in amort],
                    "s2_cum_ms": list(_cumsum(t.get("S2") for t in amort)),
                    # 逐步互斥阶段和（白名单同上；子段重复计数修复）
                    "selector_ms": [exclusive_stage_sum(t) for t in amort],
                },
            }
            print(f"[{arm}/{data_kind}/S={S}] selector_med="
                  f"{rec['wall_ns_selector_median']/1e6:.2f}ms "
                  f"stage_sum={rec['selector_total_ms_median_est']:.2f}ms "
                  f"s8={rec['s8_ms_median']:.3f}ms", flush=True)
            results_cases.append(rec)
            del k_all, v_all
            torch.cuda.empty_cache()
    return results_cases


def _cumsum(it):
    tot = 0.0
    for x in it:
        tot += x or 0.0
        yield tot


def build_meta(identity, args, arm):
    return {
        "probe": ("E126 五臂 TLI 选择器分阶段计时 microbench（成本指纹）——"
                  "优化前基线（E127 对照组，E127 合入后须复测成对进 E128）"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timing": ("阶段 span = 相邻 cuda Event elapsed_time（GPU 时间线，ms 原始"
                   "样本按执行顺序持久化不排序不舍入，median 由持久化样本派生）；"
                   "每步选择器 wall = perf_counter_ns 整数纳秒（055 口径）；"
                   "S8 = eager_decoding_attn 独立 event 对；"
                   "e2e_128step = 计时关 1 rep 粗粒度合并口径（非服务级精度）；"
                   "selector_total/selector_ms = 互斥阶段和（EXCLUSIVE_STAGE_KEYS；"
                   "S2_far/S2_near 为 S2 子段细分仅作 breakdown 不重复计入，"
                   "TL-E126-STAGE-DOUBLECOUNT-001 修复口径）"),
        "protocol": {"sizes": list(args.sizes), "steps": N_STEPS, "warm": WARM,
                     "reps": REPS, "data_kinds": list(DATA_KINDS),
                     "kv_dtype": "bfloat16", "B": 1, "q_len": 1,
                     "Hkv": HKV, "H": H_Q, "D": D_MODEL, "GQA_G": G_Q,
                     "budget": {"K1": 128, "page": 64, "K2": 1024, "cmp_ratio": 4},
                     "subspace": "tail32 (dd=32)",
                     "swa": 128, "sink_blocks": 2,
                     "layer_skip": "off（microbench 单层无 D'）",
                     "greedy_kernel_env": os.environ.get("SGLANG_TLI_GREEDY_KERNEL", "1")},
        "arm_config": ARM_CONFIGS[arm],
        "common_config": COMMON_ARGS,
        "stage_boundaries": {
            "S0": "prepare_index L277-291 pad+子空间",
            "S1": "L291-349 L1 块统计+4bit 量化（decode 每步增量）",
            "S2": "_maybe_build_kmeans L368-409 建簇（仅 cavg 系；far/near 分解见 S2_far/S2_near）",
            "S3": "compute_score L682-783 q 侧打分（coarse_minmax/avg/细筛 GEMV 三段）",
            "S4": "compute_mask 内簇分数调用区 L957-969（仅 cavg 系）+ L1 准备 L970-977",
            "mask_setup": "compute_mask L865-956 分区几何/D'/GQA 聚合（辅助桶）",
            "S5": "L1 topk L970-1069（双池 L978/单池 L1030）",
            "S6": "softmax p L1070-1079",
            "S7": "L2 分区预算 topk L1150-1372",
            "S8": "eager_decoding_attn（ops/eager_decoding.py；独立 event 对）",
            "note": "S4 span 含 L1 准备段（L970 前的 score 源选择）；anchor 全表见 harness ANCHORS",
        },
        "versions": collect_versions(),
        "identity": identity,
        "started": time.strftime("%F %T"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=lambda s: s.split(","),
                    default=["mavg", "aavg", "mminmax", "cavgsim", "cclustersim"])
    ap.add_argument("--sizes", type=lambda s: tuple(int(x) for x in s.split(",")),
                    default=SIZES)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--git-sha", default=None,
                    help="rsync 部署（无 .git）时的权威 git SHA 传入")
    ap.add_argument("--expect-tli-sha256", default=None,
                    help="部署完整性门：tli_indexer.py sha256 期望值，失配即拒跑")
    args = ap.parse_args()
    for arm in args.arms:
        if arm not in ARM_CONFIGS:
            ap.error(f"未知臂 {arm}（可选：{list(ARM_CONFIGS)}）")

    root = os.path.abspath(args.root)
    tli_path = os.path.join(root, "sparse_attn", "indexer", "tli_indexer.py")
    tli_sha256 = file_sha256(tli_path)
    tli_md5 = file_md5(tli_path)
    if args.expect_tli_sha256 and tli_sha256 != args.expect_tli_sha256:
        print(f"[GATE-FAIL] 部署完整性：tli_indexer.py sha256={tli_sha256} != "
              f"期望 {args.expect_tli_sha256}", flush=True)
        raise SystemExit(2)
    if tli_md5 != EXPECTED_TLI_MD5:
        print(f"[WARN] tli_indexer.py md5={tli_md5} != 冻结锚点 {EXPECTED_TLI_MD5}"
              f"（基线已漂移——若为 E127 合入后的复测属预期，请确认 anchor 仍过校验）",
              flush=True)
    gi = git_identity(root)
    if gi["git_sha"] is None and args.git_sha:
        gi = {"git_sha": args.git_sha, "git_dirty": False,
              "note": "rsync 部署（无 .git）；SHA 由派单侧传入（本地 worktree commit）"}
    identity = {
        "root": root,
        "git": gi,
        "tli_indexer": {"path": tli_path, "sha256": tli_sha256, "md5": tli_md5},
        "greedy_triton": {"path": os.path.join(root, "sparse_attn", "indexer",
                                               "greedy_triton.py"),
                          "sha256": file_sha256(os.path.join(
                              root, "sparse_attn", "indexer", "greedy_triton.py"))},
        "harness_sha256": file_sha256(os.path.abspath(__file__)),
        "anchor_revision": "tli_indexer.py @ md5 bf5e26c4d6123875b2786b9ed886da5f",
        "note": ("E126 优化前基线：E127 kernel 优化合入后复测须成对进 E128 综合表。"
                 "插桩 = 运行时 AST 注入独立模块，生产文件零改动"),
    }
    assert torch.cuda.is_available(), "需 GPU"
    torch.set_grad_enabled(False)
    device = "cuda"

    mod = load_instrumented(root)
    import importlib
    eager_mod = importlib.import_module("sparse_attn_e126.ops.eager_decoding")
    eager_fn = eager_mod.eager_decoding_attn

    os.makedirs(args.out_dir, exist_ok=True)
    for arm in args.arms:
        out = os.path.join(args.out_dir, f"e126_stage_timing_{arm}.json")
        attempt_id = (f"{time.strftime('%Y%m%d%H%M%S')}-{os.getpid()}-"
                      f"{hashlib.sha256((tli_sha256 + arm).encode()).hexdigest()[:8]}")
        lock_fd = acquire_attempt_output_lock(out)
        try:
            superseded = []
            for suffix in ("", ".sha256", ".failure.json"):
                r = supersede_file(out + suffix)
                if r:
                    superseded.append(r)
            meta = build_meta(identity, args, arm)
            meta["attempt"] = {"attempt_id": attempt_id,
                               "superseded_files": superseded}
            print(f"[{arm}] attempt={attempt_id} 起跑（sizes={list(args.sizes)}）",
                  flush=True)
            cases = bench_arm(arm, mod, eager_fn, identity, args, device)
            results = {"meta": meta, "cases": cases,
                       "finished": time.strftime("%F %T")}
            results["output_content_sha256"] = output_content_sha256(results)
            final_bytes = json.dumps(results, indent=1, ensure_ascii=False).encode("utf-8")
            atomic_write_bytes(out, final_bytes)
            atomic_write_bytes(out + ".sha256",
                               (hashlib.sha256(final_bytes).hexdigest() + "\n").encode("utf-8"))
            if not verify_output_sha256(out):
                _fail("发布后内容 SHA 自校验失败")
            fail_path = out + ".failure.json"
            if os.path.exists(fail_path):
                os.remove(fail_path)
            print(f"[{arm}] saved -> {out}（attempt={attempt_id}）", flush=True)
        finally:
            release_attempt_output_lock(lock_fd)


def build_summary():
    """汇总五臂 per-arm JSON → e126_summary.json（五臂×阶段汇总表 + 摊销曲线数据点）。"""
    out_dir = os.path.join(HERE, "results")
    summary = {"meta": {
        "probe": "E126 五臂阶段计时汇总（优化前基线；源 = e126_stage_timing_{arm}.json）",
        "generated": time.strftime("%F %T"),
        "cavg_extra_note": ("cavg 系额外成本 = (S2+S4) 相对 mavg 系同尺寸差值，"
                            "见 extra_cluster_cost_ms 字段"),
    }, "arms": {}}
    for arm in ARM_CONFIGS:
        p = os.path.join(out_dir, f"e126_stage_timing_{arm}.json")
        if not os.path.exists(p):
            print(f"[summary] 缺 {p}，跳过 {arm}", flush=True)
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if not verify_output_sha256(p):
            raise SystemExit(f"[GATE-FAIL] {p} 内容 SHA 校验失败")
        rows = []
        for c in d["cases"]:
            stages = c["stage_ms_median"]
            row = {"data": c["data"], "S": c["S"],
                   **{k: (round(v, 6) if v is not None else None)
                      for k, v in stages.items()},
                   "selector_wall_ms": round(c["wall_ns_selector_median"] / 1e6, 6),
                   "s8_ms": round(c["s8_ms_median"], 6),
                   "e2e_128step_ms_timing_off": round(
                       c["e2e_128step_wall_ns_timing_off"] / 1e6, 3)}
            rows.append(row)
        # cavg 系额外成本：S2+S4 vs mavg 同 (data,S)
        mavg_p = os.path.join(out_dir, "e126_stage_timing_mavg.json")
        if arm != "mavg" and os.path.exists(mavg_p):
            with open(mavg_p, encoding="utf-8") as f:
                mavg = json.load(f)
            mmap = {(c["data"], c["S"]):
                    (c["stage_ms_median"].get("S2") or 0)
                    + (c["stage_ms_median"].get("S4") or 0)
                    for c in mavg["cases"]}
            for row in rows:
                key = (row["data"], row["S"])
                if key in mmap:
                    own = (row.get("S2") or 0) + (row.get("S4") or 0)
                    row["extra_cluster_cost_ms"] = round(own - mmap[key], 6)
        summary["arms"][arm] = {
            "git_sha": d["meta"]["identity"]["git"]["git_sha"],
            "tli_sha256": d["meta"]["identity"]["tli_indexer"]["sha256"],
            "rows": rows,
            "amortization": {f'{c["data"]}/S{c["S"]}': c["amortization"]
                             for c in d["cases"]},
        }
    out = os.path.join(out_dir, "e126_summary.json")
    summary["output_content_sha256"] = output_content_sha256(summary)
    final_bytes = json.dumps(summary, indent=1, ensure_ascii=False).encode("utf-8")
    atomic_write_bytes(out, final_bytes)
    atomic_write_bytes(out + ".sha256",
                       (hashlib.sha256(final_bytes).hexdigest() + "\n").encode("utf-8"))
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    if "--summary" in sys.argv:
        build_summary()
    else:
        main()
