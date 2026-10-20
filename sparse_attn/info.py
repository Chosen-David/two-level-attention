import dataclasses
import hashlib
import io
import json
import os
import re

import torch

# ---- 076（TL-E121-OUTPUT-ID，GPT 2026-10-10 2131 审计）：canonical
# treatment manifest——输出身份必须是 treatment 的单射。
# B09 只把 α/β/γ 加进可读名，仍遗漏多个改变算法行为的参数（GPT CPU
# 复现 6 配置仅 2 名/3 组碰撞：far/near method、far/near select、
# subspace 具体值、投影基、per_q_head、sim/σ/MoBA 等），不同
# treatment 同名 → LongBench 同路径 "w" 模式静默互覆、RULER 同逻辑
# 指针被后跑代切换。修复：全部输出相关参数排序键 JSON 序列化 →
# sha256 前 10 hex 作 treatment hash，追加 "_h<hash10>" 尾段；可读段
# 逐位不动（B09 口径），既有测试锚点只补 hash 段。
# 字段缺省用 getattr 兜底，与 TLIIndexer.__init__ 的运行时缺省逐字段
# 同源（sparse_attn/indexer/tli_indexer.py __init__ getattr 序列）——
# 缺省 namespace 与显式默认值 namespace 必须同 hash（同配置稳定）。
# 【081 + kimi3 0316 追加修复 2】tli_enable_kmeans/tli_enable_layer_skip
# 此前刻意不进 manifest（076 residual 声明「可读段 B/D 二值编码已单射」
# ——被 kimi3 2026-10-11 0316 hourly review 反例推翻）：极端长可读前缀
# 触发 truncate_output_name_keep_hash 截断时，B/D 段被可读中段的 "..."
# 占位吃掉 → 仅在这两个开关上不同的两 treatment 截断后文件名相同 +
# manifest 相同 + hash 相同（kimi3 实测 trunc same? True、hash same?
# True）→ 写门放行互覆。两字段入 manifest 后 hash 恒单射，截断只影响
# 可读性不影响身份。默认 True 与 argparse 缺省 / TLIIndexer.__init__
# 运行时缺省三方同源。
#
# ---- 079（TL-E121-OUTPUT-ID-079，GPT 2026-10-11 0332 复审）两个残余 ----
# ① tia_enable_async_topk 进 manifest：TLI 继承 TIA
#   （sparse_attn/indexer/tia_indexer.py:18 self.enable_async =
#   args.tia_enable_async_topk；tli_indexer.py compute_mask async 分支
#   缓存 prev_mask 并在下一步替换）——异步第 2 个 decode 步起的细筛
#   候选集与同步路径可不同，是输出相关参数而非性能提示。getattr 缺省
#   False 与 argparse store_true 缺省/TIAIndexer 运行时读取三方同源。
# ② tli_proj_basis/tli_layer_skip_path 由「词法路径」升级为「内容身份」：
#   只记路径时同路径不同内容 → 同 manifest/hash → LongBench sidecar
#   写门放行覆盖、RULER 复用同逻辑指针（GPT CPU 复现实锤）。现于
#   manifest 构造时解析为 {path, sha256, shape/n_skip}（值非 None 时），
#   文件缺失/读取失败/解析失败一律 [GATE-FAIL] 079 SystemExit fail
#   closed——解析发生在 get_method_name_with_info 内，即输出路径确定
#   之前；不再用 default=str 隐去身份。LongBench sidecar（pred.py 写
#   门）与 method hash 共用本函数产出的同一份 resolved manifest，
#   两个入口零字段子集各自维护。
#
# ---- 081（TL-E121-OUTPUT-SNAPSHOT-081，GPT 2026-10-11 复审 P1）----
# 079 只共用了解析「函数」，没有冻结一次解析后的「值」——同一文件在
# 一次运行内被多次打开：TLIIndexer 逐层各 open 一次（patch.py 对每个
# attention 层建实例），benchmark 入口在生成后重新调 get_method_name_
# with_info / get_treatment_manifest_json 再读一遍 → 生成窗口内文件被
# 替换时 runtime A / 文件名 B / sidecar C 三方混装可达。配套两个身份缺口：
#   ① 默认 D′ 掩码遗漏：arguments.py 缺省 tli_enable_layer_skip=True、
#     tli_layer_skip_path=None，运行时（tli_indexer）展开 tracked
#     DEFAULT_MASK 并读取；但 079 manifest 只在路径显式非 None 时解析
#     → 默认配置 manifest 恒 null 而可读名含 D（身份缺口）；
#   ② 反向不对称：显式给 tli_layer_skip_path 但 enable_layer_skip=False
#     时，manifest 强制解析文件求 hash，而运行时根本不读它——身份按
#     「argv 声明」而非「实际消费」解析。
# 修复：
#   A. resolve_treatment_snapshot(args) 一次解析 → 不可变 TreatmentSnapshot
#      （manifest / manifest_json / skip_ids / basis_tensor），每文件只
#      open 一次、SHA 与内容解析共用同一批 bytes（不再「open 算 SHA 再
#      torch.load 读第二次」）；
#   B. D′ 掩码按有效值语义解析：只在 effective tli_enable_layer_skip=True
#      时进入（缺省 True + path=None → 展开 DEFAULT_MASK 并解析其内容；
#      enable=False 时无论路径是否声明都不解析、manifest 记 null）——
#      同时修复 ①②；
#   C. benchmark 两个入口在模型加载（文件被任何 attention 层消费）之前
#      冻结 snapshot，文件名 / sidecar / receipt 全程消费同一对象；
#      TLIIndexer(args, snapshot=...) 注入已解析对象，不再按层重开文件。
# 默认掩码路径定义上移到本模块（单一事实源；tli_indexer.DEFAULT_MASK
# 现引用本常量）——manifest 侧与运行时侧展开的必须是同一个文件。
DEFAULT_LAYER_SKIP_MASK_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "exp", "trace",
    "results", "tli_layer_skip_mask.json"))
_TREATMENT_FIELD_DEFAULTS = {
    # 既有可读名字段（hash 同样纳入——可读段整体单射的最强保证）
    "tia_block_size": 64,
    "tia_level1_topk": 128,
    "tia_level2_topk": 1024,
    "tia_level2_cmp_ratio": 2,      # argparse 缺省（可读段直接属性访问，兜底不可达）
    "tli_enable_subspace": True,
    "tli_subspace": "full",
    "tli_l2_subspace": "same",      # E134 子空间矩阵第三臂 sub+full（L2 全维量化）
    "tli_alpha": 0.0,
    "tli_beta": 0.0,
    "tli_gamma": 1.0,               # γ 兜底 1.0（B09 既有约定保持；off=None 走 JSON null）
    "tli_sparse_prefill": False,
    # 【E138 消融①】decode dense 回退开关入 manifest（TL-PREFILL-
    # PROVENANCE-001 同族：开关改变输出必须进身份，否则 P on+DD on 与
    # P on 两臂 manifest 逐字节相同）。默认 False；manifest 扩字段使全部
    # treatment hash 移位一轮（E135 加 sink/swa 先例，既有锚点文件名
    # 不重算不受影响）。
    "tli_decode_dense": False,
    # 081 + kimi3 0316 追加修复 2：B/D 开关入 manifest（见文件头注释——
    # 截断可吃掉可读段 B/D 位，身份必须由 manifest/hash 承载）
    "tli_enable_kmeans": True,
    "tli_enable_layer_skip": True,
    # 076 遗漏字段：改变算法行为但不在可读段
    "tli_far_method": "minmax",
    "tli_near_method": "avg",
    "tli_far_select": "4bit",
    "tli_near_select": "4bit",
    "tli_far_clusters": 256,
    "tli_far_niter": 10,
    "tli_far_blocks": 16,
    "tli_far_tokens": 512,
    "tli_sim": 0.9,
    "tli_sim_dims": "subspace",
    "tli_sigma_select": "none",
    "tli_moba": False,
    "tli_sigma": 8.0,
    "tli_per_q_head": False,
    # 079①：TIA 继承的异步开关改变第 2 步起的细筛候选集，必须进身份
    "tia_enable_async_topk": False,
    # 079②：文件型配置——值非 None 时经 _resolve_file_identity 解析为
    # 内容身份 {path, sha256, shape/n_skip}（fail closed），不再只记路径
    "tli_proj_basis": None,
    "tli_static_pair": False,
    "tli_layer_skip_path": None,
    # E135：sink/swa 正交强制区参数化 + sinkswa_only 归因臂——改变算法
    # 行为（强制区宽/池左界/mask 全旁路），必须进 treatment 身份。默认
    # 128/128/False 与历史硬编码行为逐位一致（同 E134 加 tli_l2_subspace
    # 先例，manifest 扩字段使全部 treatment hash 移位一轮；既有锚点文件
    # 名不重算不受影响）。
    "tli_sink_tokens": 128,
    "tli_swa_size": 128,
    "tli_sinkswa_only": False,
    # E137：D' gate 动态标定（默认关零扰动；eps/rows 非默认值才进可读名）
    "tli_dyn_dgate": False,
    "tli_dyn_dgate_eps": 0.05,
    "tli_dyn_dgate_rows": 256,
    # E155/E155b：固定配额几何三参数 + 新口径双开关（GPT 审计 2026-10-19
    # 阻塞级修复：不入 manifest 则 nb/nt/near_L 不同的格子共享同一
    # treatment hash——pred_postfix 目录隔离之外的任何路径（复用格/重跑/
    # 归档）同名 w 模式静默互覆，B09 族事故形态；叠加 CLI 默认翻转，
    # 旧口径重跑还会同名不同义覆盖归档）。默认 None/None/None/True/False
    # （新口径）；manifest 扩字段使全部 treatment hash 移位一轮（E134/E135
    # 先例，既有锚点文件名不重算不受影响）。
    "tli_near_pages": None,
    "tli_near_tokens": None,
    "tli_near_len": None,
    "tli_boundary_bypass": True,
    "tli_l2_quant": False,
}

# 079②：文件型配置字段集合（值非 None → 内容身份解析，见上）
_FILE_IDENTITY_FIELDS = ("tli_proj_basis", "tli_layer_skip_path")

# 【E155e vendor treatment 身份补齐（用户问题 ①「R3 manifest 登记缺口，
# 2026-10-20」】46 个 vendor CLI 参数此前不进任何 treatment 身份：076 注释
# 「非 tli 方法可读段已编码其全部旋钮」对 vendor 方法不成立——
#   * clusterkv：window/fit_iter/head_sel/balance/dist_t/gqa_policy/
#     topk_stat 7 参数完全不进可读名（异配置同名互覆可达）；
#   * asynctls/jssa：channel/basis 文件只编 basename（同 basename 异内容
#     = 同名，142/149 已修 minference/fasa 的同构缺陷，此两家漏网）；
#   * 其余 vendor（quest_vendor/minference/fasa/sparq/moba/hisa/ffd）旋钮
#     已全参数或内容身份编码，但登记进 manifest 使 hash 承载全部身份
#     （可读段降级为防御性冗余，与 tli 同构）。
# 修复 = vendor 参数按 method 条件入 manifest（tli/none/tia/twi 的
# manifest 逐位不变——E155d 在飞网格与全部 tli hash 锚点稳定），
# vendor 方法名统一追加 _h<hash10> 尾段（身份单射结构性成立；既有
# vendor 输出文件名不追溯改动，同 142/149 追加段先例）。quest/tia/twi/
# none 可读段本就全参数编码，维持 076 旧口径不动。
_VENDOR_TREATMENT_FIELDS = {
    "quest_vendor": {"quest_token_budget": 1024, "quest_chunk_size": 16},
    "minference_vendor": {"minference_starting_layer": -1,
                          "minference_config_path": None},
    "sparq_vendor": {"sparq_k": 1024, "sparq_context_length": None},
    "moba_vendor": {"moba_chunk_size": 4096, "moba_topk": None,
                    "moba_budget": 1024},
    "clusterkv_vendor": {
        "clusterkv_token_budget": 1024, "clusterkv_sink": 16,
        "clusterkv_window": 320, "clusterkv_nlist": 400,
        "clusterkv_fit_iter": 20, "clusterkv_head_sel": "truc",
        "clusterkv_balance": False, "clusterkv_dist_t": "cosine",
        "clusterkv_gqa_policy": None, "clusterkv_topk_stat": False},
    "asynctls_vendor": {"asynctls_token_budget": 1024,
                        "asynctls_block_size": 64, "asynctls_top_kb": 128,
                        "asynctls_channel_file": None},
    "hisa_vendor": {"hisa_budget": 1024, "hisa_k_block_size": 128,
                    "hisa_block_topk": 64},
    "jssa_vendor": {"jssa_budget": 1024, "jssa_rank": 32,
                    "jssa_basis_path": None},
    "fasa_vendor": {"fasa_budget": 1024, "fasa_dataset": "qasper",
                    "fasa_threshold_ratio": 0.8, "fasa_min_k": 16,
                    "fasa_max_k": 16, "fasa_mode": "topk"},
}
# vendor 文件型字段：值非 None → 内容身份 {basename, sha256}（079② 同
# 哲学——运行时必读这些文件，缺失 = 运行时必炸，manifest 侧提前 fail
# closed 不留「能算名不能跑」半配置）。fasa_agreement_dir 是目录型：
# 经 resolve_fasa_agreement_path 归一到具体 pkl 再取内容身份（142 的
# 路径唯一口径复用）。
_VENDOR_FILE_IDENTITY_FIELDS = (
    "minference_config_path", "asynctls_channel_file", "jssa_basis_path")


def _vendor_file_identity(field, path):
    """E155e：vendor 文件型配置的内容身份（fail closed）。"""
    rp = os.path.realpath(path)
    try:
        with open(rp, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise SystemExit(
            f"[GATE-FAIL] E155e: {field} 文件不可读（{path!r}）——"
            f"vendor 文件型配置必须绑定实际内容，拒绝运行，fail loudly"
            f"（{type(e).__name__}: {e}）")
    return {"basename": os.path.basename(rp),
            "sha256": hashlib.sha256(raw).hexdigest()}

# treatment hash 尾段格式：_h + 10 hex。锚定尾正则拆分（可读段以
# g 段/_P 结尾、非 tli 名以数值/async/none 结尾，不与该模式误撞）。
_TREATMENT_HASH_HEX_LEN = 10
_TREATMENT_HASH_RE = re.compile(
    r"^(?P<readable>.+)_(?P<hash>h[0-9a-f]{%d})$" % _TREATMENT_HASH_HEX_LEN,
    re.DOTALL)


def _resolve_skip_content(path, is_default=False):
    """081：D′ 掩码文件的「单次 open」内容解析（fail closed）。

    同一批 bytes 同时产出内容身份与解析好的 skip 列表——不再「open 算
    SHA、再 open 解析 JSON」双开（079 的 _resolve_file_identity 即此
    形态，081 起文件只被读一次，快照冻结才有单一事实源）。is_default=
    True 时错误文案附注该文件为缺省 D′ 掩码展开（tli_layer_skip_path
    未声明时的 DEFAULT_MASK）。

    081 身份口径：is_default=True 时内容身份只记 {default, sha256,
    n_skip}，刻意不含 checkout 相关的 realpath——默认掩码是 repo
    tracked 常量（非 argv 声明），其 provenance 就是「仓库这份内容」，
    记 realpath 会让同一份内容在不同 checkout 下得到不同 hash（WIP
    版实测缺陷：同内容同 SHA 在 worktree/主仓两 checkout 漂移为
    h7c17d0b763 / h3bbe917587，锚点随 checkout 必红）。显式声明的
    path（argv provenance）保留 realpath——路径本身是声明的一部分，
    与 079② 口径一致。"""
    if not isinstance(path, str) or not path:
        raise SystemExit(
            f"[GATE-FAIL] 079: tli_layer_skip_path 路径非法（{path!r}）"
            f"——文件型配置必须绑定实际内容，拒绝运行，fail loudly")
    rp = os.path.realpath(path)
    hint = ("（081：此文件为缺省 D′ 掩码 DEFAULT_MASK 的展开——"
            "tli_layer_skip_path 未声明时运行时同样读它）" if is_default
            else "")
    try:
        with open(rp, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise SystemExit(
            f"[GATE-FAIL] 079: tli_layer_skip_path 文件读取失败（{path} → "
            f"{type(e).__name__}: {e}）——文件型配置必须绑定实际内容，"
            f"拒绝运行，fail loudly{hint}")
    if is_default:
        # 081：默认掩码身份只记内容闭包（default 标记 + sha256 +
        # n_skip），不含 checkout 相关 realpath——见 docstring。
        ident = {"default": True, "sha256": hashlib.sha256(raw).hexdigest()}
    else:
        ident = {"path": rp, "sha256": hashlib.sha256(raw).hexdigest()}
    try:
        doc = json.loads(raw.decode("utf-8"))
        ident["n_skip"] = len(doc["skip"])
        skip_list = list(doc["skip"])
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as e:
        raise SystemExit(
            f"[GATE-FAIL] 079: tli_layer_skip_path 解析失败（{path} → "
            f"{type(e).__name__}: {e}）——须为含 \"skip\" 键的 JSON；"
            f"旧口径静默关 D' 属静默退化，生产入口拒绝运行{hint}")
    return ident, skip_list


def _resolve_basis_content(path):
    """081：投影基文件的「单次 open」内容解析（fail closed）。

    同一批 bytes 同时产出内容身份 {path, sha256, shape} 与加载好的
    CPU tensor（torch.load(io.BytesIO(raw)) 等价于 torch.load(路径)
    的既有加载口径——weights_only、map_location=cpu）——SHA 与 tensor
    同源，杜绝「算 SHA 的字节 ≠ 加载进显存的字节」的混装窗口。"""
    if not isinstance(path, str) or not path:
        raise SystemExit(
            f"[GATE-FAIL] 079: tli_proj_basis 路径非法（{path!r}）——"
            f"文件型配置必须绑定实际内容，拒绝运行，fail loudly")
    rp = os.path.realpath(path)
    try:
        with open(rp, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise SystemExit(
            f"[GATE-FAIL] 079: tli_proj_basis 文件读取失败（{path} → "
            f"{type(e).__name__}: {e}）——文件型配置必须绑定实际内容，"
            f"拒绝运行，fail loudly")
    ident = {"path": rp, "sha256": hashlib.sha256(raw).hexdigest()}
    try:
        t = torch.load(io.BytesIO(raw), map_location="cpu",
                       weights_only=True)
        ident["shape"] = [int(x) for x in t.shape]
    except SystemExit:
        raise
    except Exception as e:   # 加载失败/非裸 tensor——运行时同样会失败
        raise SystemExit(
            f"[GATE-FAIL] 079: tli_proj_basis 加载/取 shape 失败"
            f"（{path} → {type(e).__name__}: {e}）——生产路径要求裸"
            f"tensor [n_layers,Hkv,128,r]，TLIIndexer 稍后同样会失败，"
            f"拒绝运行，fail loudly")
    return ident, t


def _resolve_file_identity(field, path):
    """079②：文件型配置的内容身份解析（fail closed；081 兼容包装）。

    返回 {"path": 规范路径, "sha256": 文件字节 sha256, 摘要}：
      - tli_proj_basis：附加 shape（torch.load weights_only 提取——
        与 TLIIndexer.__init__ 同一加载口径；加载失败意味着运行时
        同样会失败，提前 fail closed 不留「能算名不能跑」的半配置）；
      - tli_layer_skip_path：附加 n_skip（解析 JSON 并要求含 "skip"
        键——TLIIndexer.__init__ 对坏文件静默关 D' 的旧语义属「静默
        退化」家族，生产入口不再放行）。

    文件缺失/读取失败/解析失败 → [GATE-FAIL] 079 SystemExit。
    刻意不做 mtime/size 键的解析缓存：同 size 同 mtime_ns 的陈旧缓存
    会让「同路径、内容已变」拿到旧身份（079 正是要修的缺陷形态）；
    身份宁可每次重读（两文件均为 MB 级以内），不允许任何陈旧身份。
    081：内部已改为单次 open 助手（SHA 与内容解析共用同一批 bytes）；
    D′ 掩码的有效值语义（enable=False 不解析、path=None 展开
    DEFAULT_MASK）见 resolve_treatment_snapshot——本包装保持 079 的
    「声明即解析」口径仅供兼容，生产路径不再经此入口。"""
    if field == "tli_proj_basis":
        return _resolve_basis_content(path)[0]
    return _resolve_skip_content(path)[0]


# ================================================================ 081：冻结快照

@dataclasses.dataclass(frozen=True)
class TreatmentSnapshot:
    """081：一次解析后的冻结 treatment 快照（不可变结构）。

    生产入口（benchmark pred.py / pred_ruler.py）在模型加载前调用
    resolve_treatment_snapshot(args) 取得唯一实例，随后：
      - register_patch → 每个 attention 层的 TLIIndexer 注入同一对象
        （层实例仍逐层建，但共享同一 snapshot——不再按层重开文件）；
      - method_name（文件名 _h<hash10> 段）、LongBench sidecar 写门、
        RULER receipt 的 treatment_manifest_json 全部从本对象派生。
    生成窗口内文件被替换时，runtime / 文件名 / sidecar / receipt 仍
    全部等于冻结时刻的内容身份（081 主缺陷的修复点）。

    字段（frozen——属性不可再赋值；manifest 内容约定只读）：
      manifest      resolved manifest dict（与逐字段重新解析逐位一致，
                    含默认 D′ 掩码的 {path, sha256, n_skip} 身份）
      manifest_json canonical JSON 串（sort_keys；hash 输入即本串）
      skip_ids      D′ 生效时解析好的 skip 集合（frozenset）；
                    enable_layer_skip=False → None（不消费掩码）
      basis_tensor  投影基 CPU float tensor；未声明 → None
    """
    manifest: dict
    manifest_json: str
    skip_ids: object     # frozenset[int] | None
    basis_tensor: object  # torch.Tensor | None


def resolve_treatment_snapshot(args):
    """081：把全部文件型 treatment 配置一次解析并冻结为 TreatmentSnapshot。

    有效值语义（D′ 掩码，081 修复身份缺口 ①②）：
      - effective tli_enable_layer_skip=True（缺省 True）且
        tli_layer_skip_path=None → 展开 DEFAULT_LAYER_SKIP_MASK_PATH
        并解析其内容（默认配置的 manifest 从此含掩码内容身份，
        而非恒 null 而可读名含 D）；
      - tli_enable_layer_skip=False → 无论路径是否声明都不解析、
        manifest 记 null（身份按「实际消费」而非「argv 声明」——
        修复 079 的反向不对称）。
    投影基：声明即解析（与 TLIIndexer 运行时同口径），同一批 bytes
    同时求 SHA256 / shape / tensor。文件缺失/损坏一律 [GATE-FAIL]
    079 SystemExit fail closed（沿用 079 错误口径）——不允许静默关 D′
    或「能算名不能跑」的半配置。

    保证：get_method_name_with_info / get_treatment_manifest_json /
    gate_output_treatment_identity（经 manifest_json）三者共用同一
    snapshot 时，输出与逐字段重新解析逐位一致（manifest 构造只有本
    函数一条路径，无第二套字段子集逻辑）。"""
    manifest = {}
    skip_ids = None
    basis_tensor = None
    # ---- D′：有效值语义 ----
    enable_skip = bool(getattr(args, "tli_enable_layer_skip", True))
    if enable_skip:
        mask_path = getattr(args, "tli_layer_skip_path", None)
        is_default = mask_path is None
        if is_default:
            mask_path = DEFAULT_LAYER_SKIP_MASK_PATH
        ident, skip_list = _resolve_skip_content(mask_path,
                                                 is_default=is_default)
        manifest["tli_layer_skip_path"] = ident
        skip_ids = frozenset(skip_list)
    else:
        # 081②：enable=False 时路径声明不进身份（运行时不消费）
        manifest["tli_layer_skip_path"] = None
    # ---- 投影基：声明即解析（单次 open，身份与 tensor 同 bytes）----
    bp = getattr(args, "tli_proj_basis", None)
    if bp is not None:
        ident, basis_tensor = _resolve_basis_content(bp)
        manifest["tli_proj_basis"] = ident
        basis_tensor = basis_tensor.float()
    else:
        manifest["tli_proj_basis"] = None
    # ---- 其余标量字段（getattr 兜底缺省，与 TLIIndexer 运行时同源）----
    for f, d in _TREATMENT_FIELD_DEFAULTS.items():
        if f in _FILE_IDENTITY_FIELDS:
            continue
        v = getattr(args, f, d)
        # 【E155d 参数退役归一】α/β/γ argparse 缺省翻转为 None=未传，
        # manifest 归一回 legacy 数值缺省（0.0/0.0/1.0）；γ CLI "off"
        # 哨兵字符串归一为 None（JSON null，E122 既有口径）——既有
        # treatment hash 逐位不变，未传 γ 的全部在跑/历史链身份稳定。
        if f == "tli_alpha" and v is None:
            v = 0.0
        elif f == "tli_beta" and v is None:
            v = 0.0
        elif f == "tli_gamma":
            if v is None:
                v = 1.0
            elif isinstance(v, str) and v.strip().lower() == "off":
                v = None
        manifest[f] = v
    # ---- E155e：vendor 参数按 method 条件入 manifest ----
    # tli/none/tia/twi/quest 不进本分支 → manifest 逐位不变（E155d 在飞
    # 网格与全部 tli hash 锚点稳定）。vendor 方法登记 method 键 + 该
    # method 的全部旋钮；文件型字段（channel/basis/config）解析为内容
    # 身份，fasa_agreement_dir 经 142 路径口径归一到具体 pkl 内容身份。
    _m = getattr(args, "method", None)
    if _m in _VENDOR_TREATMENT_FIELDS:
        manifest["method"] = _m
        for f, d in _VENDOR_TREATMENT_FIELDS[_m].items():
            v = getattr(args, f, d)
            frozen_pattern = getattr(args, "_minference_pattern_frozen", None)
            if f == "minference_config_path" and frozen_pattern is not None:
                # Runtime and the readable _cfg suffix already use this
                # snapshot. Reopening the source here can label runtime A as B.
                manifest[f] = {
                    "basename": frozen_pattern.identity_basename or os.path.basename(
                        os.path.realpath(frozen_pattern.config_path)),
                    "sha256": frozen_pattern.sha256}
            elif f in _VENDOR_FILE_IDENTITY_FIELDS and v is not None:
                manifest[f] = _vendor_file_identity(f, v)
            else:
                manifest[f] = v
        if _m == "fasa_vendor":
            frozen_agreement = getattr(args, "_fasa_staged_snapshot", None)
            _agr = resolve_fasa_agreement_path(args)
            if frozen_agreement is not None:
                manifest["fasa_agreement"] = {
                    "basename": frozen_agreement.identity_basename or os.path.basename(
                        os.path.realpath(frozen_agreement.src_path)),
                    "sha256": frozen_agreement.sha256}
            elif _agr is not None and os.path.isfile(_agr):
                manifest["fasa_agreement"] = _vendor_file_identity(
                    "fasa_agreement_dir", _agr)
            else:
                # 142 口径：源文件须先物化（qwen3_identify_fre.py 产出），
                # 缺失如实记 null——stage_fasa_agreement 的 GATE-FAIL
                # 在命名侧先拦截，manifest 不重复设防。
                manifest["fasa_agreement"] = None
    manifest_json = json.dumps(manifest, sort_keys=True,
                              ensure_ascii=False, default=str)
    return TreatmentSnapshot(manifest=manifest, manifest_json=manifest_json,
                             skip_ids=skip_ids, basis_tensor=basis_tensor)


def _treatment_manifest(args, snapshot=None):
    """076：canonical treatment manifest——全部输出相关参数的
    规范化字典（getattr 兜底缺省与 TLIIndexer 运行时同源）。

    079②：文件型配置字段（_FILE_IDENTITY_FIELDS）解析为内容身份
    （realpath + 字节 sha256 + shape/n_skip，fail closed）。
    081：D′ 掩码按有效值语义（见 resolve_treatment_snapshot）；
    snapshot 提供时直接消费冻结值（不重开文件），未提供时逐字段
    重新解析（非 benchmark 入口的既有行为，确定性不依赖缓存）。"""
    if snapshot is not None:
        return dict(snapshot.manifest)
    return resolve_treatment_snapshot(args).manifest


def get_treatment_manifest_json(args, snapshot=None):
    """076：canonical treatment manifest 的规范化 JSON 串。

    与 _treatment_hash 同源同字节（hash 输入 = 本串 utf-8 编码）——
    sidecar 记录与 hash 计算共用单一序列化口径，sidecar 内容可独立
    复核 hash（审计可再算）。排序键 + default=str 兜不可序列化值。
    079：文件型配置已解析为内容身份（realpath/sha256/shape/n_skip），
    同配置多次调用逐位稳定（确定性不依赖缓存）；LongBench 写门与
    RULER receipt 均消费本函数，不各自维护字段子集。
    081：snapshot 提供时返回冻结的 manifest_json（与逐字段重新解析
    逐位一致——单一构造路径保证）；生产入口必须传 snapshot，生成
    窗口内的文件替换不再影响本函数输出。"""
    if snapshot is not None:
        return snapshot.manifest_json
    return json.dumps(_treatment_manifest(args), sort_keys=True,
                      ensure_ascii=False, default=str)


def _treatment_hash(args, snapshot=None):
    """076：manifest 排序键 JSON 序列化 → sha256 前 10 hex。
    同一 args 对象/同配置多次调用稳定；default=str 兜不可序列化值。
    081：snapshot 提供时从冻结 manifest_json 求 hash（与文件名/
    sidecar/receipt 同源）。"""
    payload = get_treatment_manifest_json(args, snapshot)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[
        :_TREATMENT_HASH_HEX_LEN]


# 076 fail-closed 写入门的 sidecar 后缀：{out_path}.tli_manifest.json。
# 刻意不以 .jsonl 结尾（不进 LongBench/RULER 任何 *.jsonl glob）且与
# .tli_gen 指针后缀互不混淆。
TREATMENT_MANIFEST_SIDECAR_SUFFIX = ".tli_manifest.json"


def gate_output_treatment_identity(out_path, manifest_json):
    """076（TL-E121-OUTPUT-ID）fail-closed 写入门：目标路径已存在时，
    必须以 sidecar manifest 证明其与本次 treatment 同一（逐字节一致）
    才允许重写——「文件名同 hash」不再单独作为身份证明（10-hex 碰撞/
    手工改名/异配置同名一律字节级复核）。同配置重跑幂等通过
    （SKIP 语义：同 manifest → 允许重写，内容同配置等价）。

    三类拒绝（fail loudly 拒绝运行，不覆盖、不切指针）：
      ① 目标存在但缺 sidecar（provenance 未知：旧协议残留/手工放置）；
      ② sidecar 读取失败（I/O）；
      ③ sidecar manifest ≠ 本次 manifest（hash 碰撞或异配置同路径）。
    目标不存在 → 直接放行（返回 sidecar 路径）。

    调用方约定：先写 sidecar 再写数据——中断在数据写之前只留孤儿
    sidecar，下次运行目标不存在 → 门自然放行，无死锁态；目标已存在
    且 manifest 相同时重写 sidecar 为同字节幂等操作。无锁：LongBench
    生成侧单写者纪律，并发同路径写 declare residual（RULER 侧同路径
    互斥由 056/059 flock 承载，指针协议见 yarn_receipt.py）。
    """
    sidecar = out_path + TREATMENT_MANIFEST_SIDECAR_SUFFIX
    if os.path.lexists(out_path):
        if not os.path.isfile(sidecar):
            raise SystemExit(
                f"[GATE-FAIL] 076: 输出目标 {out_path} 已存在但缺 "
                f"treatment manifest 记录（{sidecar}）——provenance 未知"
                f"（旧协议残留/手工放置），拒绝覆盖（不覆盖、不切指针），"
                f"fail loudly")
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                existing = f.read()
        except OSError as e:
            raise SystemExit(
                f"[GATE-FAIL] 076: {sidecar} 读取失败（{e}）——无法证明"
                f"目标与本次 treatment 同一，拒绝覆盖，fail loudly")
        if existing != manifest_json:
            raise SystemExit(
                f"[GATE-FAIL] 076: 输出目标 {out_path} 已存在且 treatment "
                f"manifest 与本次不同（10-hex hash 碰撞或异配置同路径）"
                f"——拒绝覆盖。existing 前 200 字节={existing[:200]!r}，"
                f"current 前 200 字节={manifest_json[:200]!r}")
    return sidecar


def split_method_name_hash(method_name):
    """076 配套：把 method_name 拆成 (可读段, 身份 hash 段)。

    hash 段形如 "_h<10 hex>"（含前导下划线，可直接拼回）；非 tli 名
    （quest/twia/tia/none——无 hash 段）返回 ("原名", "")。供 LongBench
    pred.py 的超长名截断「先截可读中段、保 hash 尾段」使用。"""
    m = _TREATMENT_HASH_RE.match(method_name)
    if not m:
        return method_name, ""
    return m.group("readable"), "_" + m.group("hash")


def truncate_output_name_keep_hash(dataset_prefix, method_name, t,
                                    limit=230):
    """076 配套：LongBench 输出文件名超长截断——身份 hash 尾段优先保留。

    旧口径 out_fn[:245]+"..." 会从尾部截，method_name 尾部的
    "_h<hash10>" 被截后不同 treatment 可再次同名互覆（076 回归）。
    新口径：可读中段先截（"..." 占位），dataset 前缀 / hash 尾段 /
    "-t" 后缀全保；非 tli 名（无 hash 段）回退旧口径（其可读段已全
    参数编码）。

    【081 + kimi3 0316 追加修复 1】limit 缺省 245 → 230：上限与
    sidecar 后缀链绑定——out_fn 落盘时实际还要拼 ".jsonl"（6 字节），
    且 076 写门会为它创建 sidecar "{out_fn}.tli_manifest.json"（18
    字节），故 245 截断下 sidecar 实际文件名 = 245+6+18 = 269 字节
    > ext4 单文件名 255 上限，open(sidecar,"w") 抛 OSError 36 裸
    traceback，破坏 076 fail-closed 统一口径（kimi3 实测复现）。
    230 = 255 − 6 − 18 − 1 余量。极端长 prefix/t 下宁可略超限也
    不丢身份段（如实声明，不静默截 hash）；调用方传显式 limit 时
    须自行保证 sidecar 链长约束。
    """
    t = str(t)
    readable, hash_seg = split_method_name_hash(method_name)
    if not hash_seg:
        full = f"{dataset_prefix}-{method_name}-{t}"
        return full[:limit] + "..."
    room = limit - (len(dataset_prefix) + 1 + len("...") +
                    len(hash_seg) + 1 + len(t))
    return f"{dataset_prefix}-{readable[:max(room, 0)]}...{hash_seg}-{t}"


# ---- 142（TL-FASA-CALIBRATION-IDENTITY-142，GPT 审计 B 批）----
# 修复前缺陷：--fasa_agreement_dir 在 sparse_attn/arguments.py 注册后
# 从未被任何入口消费（全仓仅定义一处），pred.py / pred_ruler.py 的
# fasa_vendor 分支硬编码 vendor 内 dimension_rankings/ 为源、仓库根
# dimension_rankings/ 为目标，且「目标已存在时不比较源/目标」——
# 不同校准内容（源被替换/重标定）会静默复用同一份 stale 目标并共享
# 同一 treatment 身份。以下三个函数是 FASA agreement 身份的唯一解析
# 口径（LongBench/RULER 两入口与 treatment 名三方同源）：
#
# ---- 142 reopen（GPT 审计 8b56bbd，2026-10-13 2341）----
# B 批修复后仍存在「同一次运行内两次独立读源」窗口：stage 先把源 A 复制
# 到运行时目标（runtime 在模型加载阶段即冻结为目标副本 A），输出名却在
# 模型加载后经 _fasa_agreement_sha8 重新打开可变源求 SHA——窗口内源被
# 替换为 B 时同一次运行形成 runtime=A、name=B 错绑。修复：stage 时
# 一次性读取源 bytes 构造不可变 FasaAgreementSnapshot（完整 sha256 +
# 8 位段 + raw bytes），挂 args._fasa_staged_snapshot；runtime records
# 与输出名共用同一 snapshot（物化目标从快照 bytes 原子写出——临时文件 +
# os.replace；命名侧消费冻结 sha8，不再重新打开源）。
#
# ---- 142 并发残余（GPT advice C+ 批，2026-10-14）----
# C 批修复后单次运行内三方同源，但物化目标仍是仓库根共享固定路径
# dimension_rankings/<model>/<dataset>_agreement.pkl——两个不同内容的
# 源并发首次 stage 同一共享目标时，双方都通过 isfile 不存在检查
# （stale 拒绝分支不触发），A 先 os.replace、B 后 os.replace 覆盖 A →
# A 进程 name/snapshot 绑 A、官方 loader 却读到 B（原子替换 ≠ 内容归属
# 锁定；GPT 双进程 CPU 复现 + 主 AI 复跑红证确认）。修复（GPT 建议
# 「内容寻址/进程私有不可变文件」口径）：物化目标改为内容寻址进程
# 私有目录 dimension_rankings_staged/<sha8>/dimension_rankings/<model>/...
# ——同内容双进程写同一路径但内容逐字节相同、os.replace 幂等无害；
# 不同内容天然隔离到不同 sha8 目录，跨进程错绑结构性不可能。官方
# loader 的 CWD 由 adapter（qwen3_fasa_patch.enable_fasa）切到私有
# staged 根（见该函数），旧仓库根 dimension_rankings/ 共享固定路径
# 不再被写入也不再被读到（历史产物原地保留，不删不动不读）。
#
# ---- 142 sha8 前缀碰撞条件残余（GPT 审计 P2，2026-10-14）----
# C+ 批的 <sha8> 目录段仍有条件残余：两个不同内容若 sha256 前 8 hex
# 碰撞且并发首次 stage，双方都过 isfile 不存在检查（不同内容、不同
# full sha，stale 拒绝分支不触发），A 先 os.replace 后被 B 覆盖 → A
# 的快照 dst_path 指向 B 的内容（原子替换 ≠ 内容归属锁定在碰撞档
# 复活）。修复：目录段升级为完整 SHA256
# dimension_rankings_staged/<full_sha256>/（sha8 只作 treatment 名
# 显示标识）；os.replace 后写后复核（读回 dst 与快照 raw 逐字节比对
# 不一致即 SystemExit fail-closed）；旧 sha8 段 staged 残骸与旧仓库
# 根 dimension_rankings/ 一样不删不动不读。

@dataclasses.dataclass(frozen=True)
class FasaAgreementSnapshot:
    """142 reopen：stage 时刻单次读源冻结的 agreement 内容身份快照。

    一次运行内 runtime（fasa_utils 经 adapter chdir 读内容寻址目标
    副本，目标由本快照 raw bytes 物化而来）与 treatment 名（_agr<sha8>
    段）共用同一对象——stage 之后源文件再被替换也不影响本次运行的
    身份（name 恒等于 runtime 实际消费的内容）。frozen dataclass：
    属性不可再赋值。"""
    src_path: str    # stage 时解析出的源绝对路径（身份/报错口径）
    dst_path: str    # 内容寻址物化目标绝对路径（staged_root 下）
    sha256: str      # 源 bytes 完整 SHA256（stale 目标比对用同一值）
    sha8: str        # 前 8 hex（treatment 名 _agr<sha8> 尾段，仅显示标识——
                     # 目录段用完整 sha256，见 stage_fasa_agreement）
    raw: bytes       # stage 时刻的源 bytes（目标物化与身份共用同一批）
    staged_root: str = ""   # 内容寻址 staged 根（adapter chdir 目标）
    identity_basename: str = ""  # Freeze realpath basename before source replacement.


def resolve_fasa_agreement_path(args, repo_root=None):
    """142：--fasa_agreement_dir 接线——FASA agreement 源文件路径唯一口径。

    - 绝对路径：直接作为 agreement 源目录（自定义校准产物位置）；
    - 相对路径：相对 fasa_vendor 目录解析——默认值 'dimension_rankings'
      即原硬编码源 exp/vendor_baselines/fasa_vendor/dimension_rankings/，
      默认行为逐位向后兼容。
    返回 <源目录>/<model>/<dataset>_agreement.pkl（不校验存在性——
    存在性与 SHA 对比由 stage_fasa_agreement / 名字消费方按各自时序
    fail closed，本函数只负责路径身份单一事实源）。"""
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vendor_dir = os.path.join(
        repo_root, "exp", "vendor_baselines", "fasa_vendor")
    agr_dir = getattr(args, "fasa_agreement_dir", None) or "dimension_rankings"
    src_root = agr_dir if os.path.isabs(agr_dir) \
        else os.path.join(vendor_dir, agr_dir)
    fasa_dataset = getattr(args, "fasa_dataset", "qasper")
    return os.path.join(src_root, args.model,
                        f"{fasa_dataset}_agreement.pkl")


def _fasa_agreement_sha8(args):
    """142 reopen：消费 stage 冻结快照的 sha8——不再重新打开可变源。

    旧实现（B 批）在模型加载后二次读源求 SHA：runtime 已在模型加载阶段
    冻结为目标副本 A，窗口内源被替换为 B 时同一次运行 runtime=A、
    name=B 错绑（GPT 审计 A/B 复现）。现命名侧只消费
    args._fasa_staged_snapshot（stage_fasa_agreement 挂载）——name 恒
    等于 runtime 实际消费的内容。无快照（未经 stage 的路径，如直接调
    命名函数的单元测试）一律 fail-closed SystemExit：不存在「绕开
    staging 单独算名」的合法路径（回退读源即 142 reopen 的错绑窗口
    本身）。"""
    snap = getattr(args, "_fasa_staged_snapshot", None)
    if snap is None:
        raise SystemExit(
            "[GATE-FAIL] 142: fasa_vendor 命名缺少冻结快照——必须先经 "
            "stage_fasa_agreement（LongBench/RULER 两入口在模型加载前"
            "调用，快照随 args 冻结 runtime/名字共用的同一批源 bytes）。"
            "未经 staging 的直接命名不再回退读源（旧回退即 142 reopen "
            "的 A/B 错绑窗口），fail closed")
    return snap.sha8


def stage_fasa_agreement(args, repo_root=None):
    """142 + 142 reopen + 142 并发残余：单次读源冻结快照 + 内容寻址
    原子物化，返回 staged 根绝对路径（adapter chdir 目标）。

    官方 constructe_adaptive_selection 硬编码相对路径
    dimension_rankings/（相对 CWD 拼接 <model>/<dataset>），故源
    （vendor 目录或 --fasa_agreement_dir 自定义位置）须先物化。C+ 批
    起物化目标是内容寻址进程私有目录，sha8 前缀碰撞条件残余修复
    （TL-FASA-CALIBRATION-IDENTITY-142 P2）起目录段升级为完整
    SHA256：
      <repo_root>/dimension_rankings_staged/<full_sha256>/
        dimension_rankings/<model>/<dataset>_agreement.pkl
    （142 并发残余修复：旧共享固定路径 <repo_root>/dimension_rankings/
    下，两进程并发首次 stage 不同内容时后者覆盖前者、前者的官方
    loader 读到后者的内容——原子替换 ≠ 内容归属锁定。内容寻址下
    同内容双进程写同一路径且内容逐字节相同、os.replace 幂等无害，
    不同内容天然隔离，错绑结构性不可能。目录段用完整 SHA256 而
    非 sha8 前缀：sha256 前 8 hex 碰撞时两个不同内容共享同一目录，
    并发首次 stage 双方都过 isfile 检查、后写覆盖先写——sha8 只作
    treatment 名的显示标识，不承担文件隔离保证。）行为：
      - 私有目标已存在时源/目标字节 SHA256 全量对比，不一致
        SystemExit fail closed（内容寻址目录被篡改/损坏的 stale
        目标；比对两侧 = 冻结快照 bytes vs 目标实读 bytes）；
      - 私有目标不存在时物化（临时文件 + os.replace 原子写，内容 =
        冻结快照 bytes——绝不二次读源），写后复核：读回目标与快照
        raw 逐字节比对，不一致 SystemExit fail closed（防任何残余
        覆盖——目录段换名窗口 / 外部并发写 / 磁盘损坏均在射程内）；
      - 每次调用单次读源 bytes 构造 FasaAgreementSnapshot 挂
        args._fasa_staged_snapshot——runtime records 与输出名共用
        同一快照（stale 拒绝时不更新快照，args 保持上一次成功
        stage 的身份）；
      - 旧仓库根 dimension_rankings/（共享固定路径）与旧 sha8 段
        staged 残骸（dimension_rankings_staged/<sha8>/）：不删不动
        不读（历史产物原地保留，新代码只认 <full_sha256>/ 段）——
        官方 loader 的 CWD 由 adapter 切到返回的 staged 根，旧路径
        不再可达。"""
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_agr = resolve_fasa_agreement_path(args, repo_root)
    fasa_dataset = getattr(args, "fasa_dataset", "qasper")
    if not os.path.isfile(src_agr):
        raise SystemExit(
            f"[GATE-FAIL] 142: FASA 校准产物缺失: {src_agr}"
            f"（--fasa_agreement_dir 解析所得；先跑 qwen3_identify_fre.py"
            f" + convert_agreement.py），fail closed")
    # 142 reopen：单次读源——stale 比对 / 目标物化 / 名字身份全部消费
    # 同一批 bytes（快照），stage 之后源再变不影响本次运行身份。
    try:
        with open(src_agr, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise SystemExit(
            f"[GATE-FAIL] 142: FASA agreement 源文件不可读（{src_agr} → "
            f"{type(e).__name__}: {e}）——treatment 名必须绑定实际校准"
            f"内容（先跑 qwen3_identify_fre.py + convert_agreement.py），"
            f"fail closed")
    full_sha = hashlib.sha256(raw).hexdigest()
    # 142 sha8 前缀碰撞条件残余修复：物化目标 = 内容寻址进程私有目录，
    # 目录段用完整 SHA256（非 sha8 前缀）——两个不同内容若 sha256 前
    # 8 hex 碰撞且并发首次 stage，sha8 段方案下双方都过 isfile 检查、
    # 后写覆盖先写；full_sha 段下不同内容目录必然不同，天然隔离。
    # sha8 只作 treatment 名显示标识（_fasa_agreement_sha8），不承担
    # 文件隔离保证。
    staged_root = os.path.join(repo_root, "dimension_rankings_staged",
                               full_sha)
    dst_agr = os.path.join(staged_root, "dimension_rankings", args.model,
                           f"{fasa_dataset}_agreement.pkl")
    snap = FasaAgreementSnapshot(
        src_path=src_agr, dst_path=dst_agr,
        sha256=full_sha, sha8=full_sha[:8], raw=raw,
        staged_root=staged_root,
        identity_basename=os.path.basename(os.path.realpath(src_agr)))
    if os.path.isfile(dst_agr):
        try:
            with open(dst_agr, "rb") as f:
                dst_sha = hashlib.sha256(f.read()).hexdigest()
        except OSError as e:
            raise SystemExit(
                f"[GATE-FAIL] 142: FASA agreement 目标读取失败（{dst_agr}"
                f" → {type(e).__name__}: {e}）——无法证明目标与源内容一致，"
                f"拒绝运行，fail closed")
        if dst_sha != snap.sha256:
            raise SystemExit(
                f"[GATE-FAIL] 142: FASA agreement 目标与源内容不一致"
                f"（stale 目标，内容寻址目录被篡改/损坏）：\n"
                f"  源   {src_agr}\n  sha256 {snap.sha256}\n"
                f"  目标 {dst_agr}\n  sha256 {dst_sha}\n"
                f"dimension_rankings_staged/<full_sha256>/ 下应与内容逐字节"
                f"对应，被静默复用的目标与源不符——请删除该 stale 目标后"
                f"重跑，fail closed")
    else:
        os.makedirs(os.path.dirname(dst_agr), exist_ok=True)
        # 142 reopen：原子物化——临时文件 + os.replace（copy2 无锁非
        # 原子，窗口内半写目标可能被 runtime 读到）。内容 = 冻结快照
        # bytes（绝不二次读源——快照与物化目标逐字节一致）。
        # 142 并发残余：同内容双进程并发写同一路径——tmp 名含 PID
        # 互不覆盖、os.replace 幂等（双方内容逐字节相同）；A 后到
        # isfile=True 时走上方 SHA 全量比对，同内容幂等通过。
        tmp_agr = f"{dst_agr}.tmp.{os.getpid()}"
        try:
            with open(tmp_agr, "wb") as f:
                f.write(snap.raw)
            os.replace(tmp_agr, dst_agr)
            # 142 sha8 前缀碰撞残余：写后复核——os.replace 落盘后读回
            # 目标与冻结快照 raw 逐字节比对，不一致即 fail closed。
            # full_sha 目录段下正常时序不可能不一致；此复核防御任何
            # 残余覆盖路径（未来目录段再变更 / 外部进程同路径写 /
            # 文件系统损坏），保证「stage 成功返回 ⇒ 目标内容恒等于
            # 本次快照」这一 runtime 身份不变式。
            try:
                with open(dst_agr, "rb") as f:
                    written_back = f.read()
            except OSError as e:
                raise SystemExit(
                    f"[GATE-FAIL] 142: FASA agreement 写后复核读取失败"
                    f"（{dst_agr} → {type(e).__name__}: {e}）——无法证明"
                    f"物化目标与快照一致，fail closed")
            if written_back != snap.raw:
                raise SystemExit(
                    f"[GATE-FAIL] 142: FASA agreement 写后复核不一致"
                    f"（物化目标在落盘后被覆盖/损坏）：\n"
                    f"  目标 {dst_agr}\n  快照 sha256 {snap.sha256}\n"
                    f"  写后实读 sha256 "
                    f"{hashlib.sha256(written_back).hexdigest()}\n"
                    f"stage 成功返回 ⇒ 目标内容必须恒等于本次快照，"
                    f"fail closed")
        except BaseException:
            # 物化失败清理半写临时文件，不留会被误当目标的残骸
            try:
                if os.path.lexists(tmp_agr):
                    os.remove(tmp_agr)
            except OSError:
                pass
            raise
    # 142 reopen：冻结快照挂 args——命名（get_method_name_with_info 的
    # fasa 分支经 _fasa_agreement_sha8）与 runtime 共用同一内容身份。
    setattr(args, "_fasa_staged_snapshot", snap)
    return staged_root


@dataclasses.dataclass(frozen=True)
class MinferencePatternSnapshot:
    """149 + 149 并发残余（C+ 批）：pattern JSON 内容身份冻结快照。

    入口加载期单次读源构造：完整 sha256 + 8 位段 + raw bytes +
    解析后的 JSON 对象。命名（_cfg<sha8> 段）与 adapter
    （enable_minference 构建 best_pattern）共用同一冻结对象——
    freeze 之后源 config_path 再被替换/删除都不影响本次运行的
    身份与 runtime 实际消费的 pattern（name 恒等于 runtime 内容）。
    frozen dataclass：属性不可再赋值。"""
    config_path: str  # freeze 时解析出的 pattern JSON 路径（报错口径）
    sha256: str       # pattern bytes 完整 SHA256（名字 _cfg<sha8> 段同源）
    sha8: str         # 前 8 hex（treatment 名 _cfg<sha8> 尾段）
    raw: bytes        # freeze 时刻的源 bytes（身份与内容共用同一批）
    pattern: object = None  # json.loads(raw)（adapter 直接消费，不再 open）
    identity_basename: str = ""  # Freeze realpath basename with content identity.


def freeze_minference_pattern_sha8(args):
    """149（TL-MINFERENCE-PATTERN-IDENTITY，GPT 审计 8b56bbd）+ 149
    并发残余（C+ 批）：入口加载期单次读源冻结完整 pattern 身份快照。

    挂 args._minference_pattern_frozen（MinferencePatternSnapshot：sha8
    + raw bytes + 解析后的 JSON 对象）——命名侧与 adapter
    （qwen3_minference_patch enable_minference 构建 best_pattern）共用
    同一冻结对象，adapter 不再二次 open 可变 config_path（旧实现
    freeze 只存 hash、adapter 重新 open 载入 JSON——窗口内源被替换则
    name 绑 A、runtime 读 B，与 142 同构）。缺 config_path / 不可读 /
    畸形 JSON 一律 fail-closed SystemExit（畸形 JSON 的报错从
    adapter 读点提前到入口 freeze 点）：pattern 内容是 treatment
    身份的一部分，MInference 无 config 直跑路径不存在。返回 sha8
    （签名不变，既有消费方兼容）。"""
    cfg_path = getattr(args, "minference_config_path", None)
    if not cfg_path:
        raise SystemExit(
            "[GATE-FAIL] 149: method=minference_vendor 必须传 "
            "--minference_config_path（is_search 在线搜出的 best_pattern "
            "JSON）——pattern 内容决定逐层逐头 sparse prefill，是 treatment "
            "身份的一部分，无 config 直跑路径不存在，fail closed")
    try:
        with open(cfg_path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise SystemExit(
            f"[GATE-FAIL] 149: MInference pattern JSON 不可读（{cfg_path} → "
            f"{type(e).__name__}: {e}）——treatment 名必须绑定实际 pattern "
            f"内容，fail closed")
    # 149 并发残余：畸形 JSON 在 freeze 期即报错（fail closed 提前到
    # 入口）——不再留给 adapter 二次 open 时才炸
    try:
        parsed = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as e:
        raise SystemExit(
            f"[GATE-FAIL] 149: MInference pattern JSON 畸形（{cfg_path} → "
            f"{type(e).__name__}: {e}）——treatment 名必须绑定可被 adapter "
            f"实际消费的 pattern 内容，fail closed")
    full_sha = hashlib.sha256(raw).hexdigest()
    snap = MinferencePatternSnapshot(
        config_path=cfg_path, sha256=full_sha, sha8=full_sha[:8],
        raw=raw, pattern=parsed,
        identity_basename=os.path.basename(os.path.realpath(cfg_path)))
    setattr(args, "_minference_pattern_frozen", snap)
    setattr(args, "_minference_pattern_sha8", snap.sha8)
    return snap.sha8


def _minference_cfg_sha8(args):
    """149：minference_vendor 名字 pattern 内容身份段的唯一来源。

    优先消费入口冻结值（freeze_minference_pattern_sha8，模型加载前
    单次读源——name 与 adapter 实际加载的 pattern 同一批 bytes）；
    未经入口冻结的路径（如直接调命名函数的回归测试）现场读源
    fail closed——降级路径内容身份仍正确，仅读点晚于 runtime（读点
    替换窗口由入口冻结闭合，本函数不为此重复设防）。"""
    frozen = getattr(args, "_minference_pattern_frozen", None)
    if frozen is not None:
        return frozen.sha8
    sha8 = getattr(args, "_minference_pattern_sha8", None)
    if isinstance(sha8, str) and sha8:
        return sha8
    return freeze_minference_pattern_sha8(args)


def _vendor_name_with_hash(args, base, snapshot=None):
    """E155e：vendor 方法名追加 canonical treatment hash 尾段（与 tli
    076 同构）。base = 既有可读段（含 142/149 的 _agr/_cfg/_src 内容
    身份段），追加段不改既有语义字段——既有 vendor 文件名不追溯改动。
    hash 从 vendor manifest（method 条件字段 + 文件内容身份）派生，
    身份单射结构性成立（clusterkv 7 旋钮 / asynctls channel / jssa
    basis 内容全部由 hash 承载，可读段降级为防御性冗余）。"""
    return f"{base}_h{_treatment_hash(args, snapshot)}"


def get_method_name_with_info(args, snapshot=None):
    # 081：snapshot 提供时 tli 名的 _h<hash10> 段从冻结 manifest_json
    # 派生（与 sidecar/receipt 同源）；E155e 起 vendor 分支同样消费
    # snapshot 的冻结 manifest（_vendor_name_with_hash）。
    if args.method == "ffd_vendor":
        from sparse_attn.baseline_runtime import ffd_identity
        identity = ffd_identity(args)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        return f"ffd_vendor_native_{digest}"
    if args.method == "quest":
        return f"quest_{args.quest_block_size}_{args.quest_topk}"
    elif args.method == "quest_vendor":
        # B 轨 vendor（官方源码复用）：Quest 官方口径
        # token_budget/chunk_size 全参数编码进文件名。
        # E155e：追加 _h<hash10> 尾段（manifest 条件字段承载身份）。
        return _vendor_name_with_hash(
            args,
            f"quest_vendor_{args.quest_token_budget}_"
            f"{args.quest_chunk_size}", snapshot)
    elif args.method == "minference_vendor":
        # B 轨 vendor（官方源码复用）：MInference prefill 动态稀疏 + decode dense。
        # pattern 文件（config_path）不同 = treatment 不同（不同模型/搜索口径），
        # 取 basename 编码进文件名；starting_layer 之前浅层 dense。
        # 149（TL-MINFERENCE-PATTERN-IDENTITY，GPT 审计 8b56bbd）：只编码
        # basename 时两份不同内容、同 basename 的 pattern 共享同一 method
        # name（且非 tli 路径 treatment sidecar 不写，无第二道门）。修复 =
        # 追加 pattern JSON 内容 SHA256 前 8 位尾段 `_cfg<sha8>`（内容身份，
        # 与路径无关），追加段不改既有语义字段（既有文件名不追溯改动）。
        # 内容 sha 优先消费入口加载期冻结值（freeze_minference_pattern_
        # sha8，142 同款单次冻结）；未经冻结的路径现场读源 fail closed。
        cfg_sha8 = _minference_cfg_sha8(args)
        pattern_tag = os.path.basename(
            getattr(args, "minference_config_path", "pattern")
        ).replace(".json", "")
        return _vendor_name_with_hash(
            args,
            (f"minference_vendor_{pattern_tag}_"
             f"{getattr(args, 'minference_starting_layer', -1)}_cfg{cfg_sha8}"),
            snapshot)
    elif args.method == "fasa_vendor":
        # B 轨 vendor（官方源码复用）：FASA decode 频域剪枝。校准数据集 +
        # budget/threshold_ratio/min_k/max_k/mode 全参数编码进文件名——
        # 不同校准产物/预算 = 不同 treatment，防同名互覆（与 minference
        # config_path 编码同哲学）。fasa_mode 默认 topk（官方 static 口径，
        # dynamic 消融单独编码）。
        # 142（TL-FASA-CALIBRATION-IDENTITY，GPT 审计 B 批）：参数段只
        # 编码「读了哪份校准配置」，不绑定「agreement 文件实际内容」——
        # 不同校准内容（同 dataset 名下重新标定/换模型重标定）共享同一
        # treatment 身份。修复=追加 agreement 源文件内容 SHA256 前 8 位
        # 尾段 `_agr<sha8>`（内容身份，与 checkout/路径无关），追加段
        # 不改既有语义字段（既有文件名不追溯改动）。
        fasa_dataset = getattr(args, "fasa_dataset", "qasper")
        return _vendor_name_with_hash(
            args,
            (f"fasa_vendor_{fasa_dataset}_b{getattr(args, 'fasa_budget', 1024)}"
             f"_t{getattr(args, 'fasa_threshold_ratio', 0.8):g}"
             f"_k{getattr(args, 'fasa_min_k', 16)}-{getattr(args, 'fasa_max_k', 16)}"
             f"_{getattr(args, 'fasa_mode', 'topk')}"
             f"_agr{_fasa_agreement_sha8(args)}"), snapshot)
    elif args.method == "sparq_vendor":
        # B 轨 vendor（官方源码复用）：SparQ decode top-k + V_mean 插值。
        # 主表口径（用户 2026-10-14 拍板）：k 固定 1024；r 由
        # context_length 反推（r=round(head_dim/(ctx/(2k)))）——ctx 不同 =
        # 不同 treatment，必须编码进文件名（LongBench 默认模型 cap 31500
        # vs RULER 32K/64K/128K 各 L 档位），防跨序列长度同名互覆。
        # ctx 优先消费入口冻结的 _sparq_ctx_effective（pred 两入口在
        # enable_sparq 前 setattr，与 runtime 实际生效值同源）。
        _ctx = getattr(args, "_sparq_ctx_effective", None)
        if _ctx is None:
            _ctx = getattr(args, "sparq_context_length", None)
        if _ctx is None:
            _ctx = getattr(args, "context_length", 31500)
        from pathlib import Path
        adapter = Path(__file__).resolve().parents[1] / 'exp/vendor_baselines/sparq_vendor/qwen3_sparq_patch.py'
        source_hash = hashlib.sha256(adapter.read_bytes()).hexdigest()[:16]
        return _vendor_name_with_hash(
            args,
            (f"sparq_vendor_k{getattr(args, 'sparq_k', 1024)}"
             f"_cl{_ctx:g}_src{source_hash}"), snapshot)
    elif args.method == "moba_vendor":
        # B 轨 vendor（官方源码复用）：MoBA prefill 块级 gate 稀疏 + decode
        # dense。chunk/topk/budget 决定每 query 有效预算，必须进 treatment 名
        # （防不同预算口径同名互覆）。topk 默认 None → 换算
        # max(2, ceil(budget/chunk))，与 enable_moba 同源（同一换算，名称绑定
        # 实际生效 topk，防名/行为失配）。
        _moba_chunk = getattr(args, "moba_chunk_size", 4096)
        _moba_topk = getattr(args, "moba_topk", None)
        if _moba_topk is None:
            _moba_budget = getattr(args, "moba_budget", 1024)
            _moba_topk = max(2, -(-_moba_budget // _moba_chunk))
        return _vendor_name_with_hash(
            args, f"moba_vendor_c{_moba_chunk}_k{_moba_topk}", snapshot)
    elif args.method == "clusterkv_vendor":
        # B 轨 vendor（官方源码复用）：ClusterKV KMeans 聚类稀疏。token_budget
        # /sink/nlist 决定选择器预算与建簇粒度，全编码进 treatment 名（防不同
        # 预算口径同名互覆）。sink 已含在 token_budget 内（官方口径），不再
        # 重复计入；命名同 quest_vendor_<budget>_<sink> 风格。
        _tb = getattr(args, "clusterkv_token_budget", 1024)
        _sink = getattr(args, "clusterkv_sink", 16)
        _nlist = getattr(args, "clusterkv_nlist", 400)
        return _vendor_name_with_hash(
            args, f"clusterkv_vendor_{_tb}_{_sink}_nl{_nlist}", snapshot)
    elif args.method == "asynctls_vendor":
        # C 轨语义复现（arXiv:2604.07815，无官方开源）：三级流水。channel 文件
        # 名（校准内容身份）+ token_budget/block/top_kb 全编码进 treatment 名，
        # 防不同校准/预算口径同名互覆（与 clusterkv 命名风格一致）。
        import os as _os
        _chan = getattr(args, "asynctls_channel_file", None)
        _chan_tag = _os.path.splitext(_os.path.basename(_chan))[0] if _chan else "nochan"
        _kb = getattr(args, "asynctls_token_budget", 1024)
        _bs = getattr(args, "asynctls_block_size", 64)
        _tk = getattr(args, "asynctls_top_kb", 128)
        return _vendor_name_with_hash(
            args,
            f"asynctls_vendor_{_chan_tag}_k{_kb}_b{_bs}_kb{_tk}", snapshot)
    elif args.method == "hisa_vendor":
        # B 轨官方 kernel 语义 torch 等价（COLM 2026 / arXiv:2603.28458v4）。
        # budget（最终 token）/块长/保留块数全编码，防不同口径同名互覆。
        _hb = getattr(args, "hisa_budget", 1024)
        _hbs = getattr(args, "hisa_k_block_size", 128)
        _htk = getattr(args, "hisa_block_topk", 64)
        return _vendor_name_with_hash(
            args, f"hisa_vendor_k{_hb}_b{_hbs}_kb{_htk}", snapshot)
    elif args.method == "jssa_vendor":
        # B 轨官方 vLLM patch 语义 torch 等价（arXiv:2610.19026v1）。budget/rank
        # 全编码（basis 文件名段含校准内容身份，rank 影响子空间维度），防互覆。
        import os as _os
        _jp = getattr(args, "jssa_basis_path", None)
        _jtag = _os.path.splitext(_os.path.basename(_jp))[0] if _jp else "nobasis"
        _jb = getattr(args, "jssa_budget", 1024)
        _jr = getattr(args, "jssa_rank", 32)
        return _vendor_name_with_hash(
            args, f"jssa_vendor_{_jtag}_k{_jb}_r{_jr}", snapshot)
    elif args.method == "twi":
        return f"twi_{args.twi_block_size}_{args.twi_level1_topk}_{args.twi_level2_topp}"
    elif args.method == "tia":
        async_info = "_async" if args.tia_enable_async_topk else ""
        return f"tia_{args.tia_block_size}_{args.tia_level1_topk}_{args.tia_level2_topk}_c{args.tia_level2_cmp_ratio}{async_info}"
    elif args.method == "tli":
        # 【F11 修复（kimi3 清单 2026-10-08）】ab 用「生效值」而非 flag 原值：
        # tli_subspace=full 时 TLIIndexer.__init__ 内部强制 enable_subspace=False
        # → 不写 "A"（原按 flag 原值写 "A"，默认 full 口径下文件名虚标子空间，
        # 与实际 treatment 不符——历史 ARM_CONTRACT 串 tli_64_128_1024_c4_A 的
        # "A" 即此虚标，既有文件名不追溯改动）。
        subspace = getattr(args, "tli_subspace", "full")
        eff_subspace = getattr(args, "tli_enable_subspace", True)
        if subspace == "full":
            eff_subspace = False          # 与 TLIIndexer.__init__ 同源
        elif subspace in ("rope", "nope", "mla_rope"):
            eff_subspace = True
        ab = ("A" if eff_subspace else "") + ("B" if args.tli_enable_kmeans else "") + ("D" if args.tli_enable_layer_skip else "")
        # 【B09 修复（kimi3 清单 2026-10-08）】α/β/γ 是 treatment 的一部分，
        # 必须进输出文件名（a0.25_b0.125_g0.625 风格），否则不同分区配置
        # 共享同 method_name → 同路径 w 模式互相覆盖 / 同名混装不可区分。
        # γ=off（显式传 "off" → None）渲染 "off"；缺省属性按 1.0 兜底——与
        # TLIIndexer.__init__ 的运行时缺省 getattr(args,"tli_gamma",1.0) 同源，
        # 否则程序化构造的 args（无该属性）文件名标 goff 而实际跑 γ=1.0，
        # 文件名/行为失配（B09 族）。
        alpha = getattr(args, "tli_alpha", 0.0)
        beta = getattr(args, "tli_beta", 0.0)
        # 【E155d 参数退役归一】None=未传 → legacy 缺省渲染（既有文件名
        # 逐位不变）；"off" 哨兵 → None → goff（与 E122 γ-off 同渲染）。
        if alpha is None:
            alpha = 0.0
        if beta is None:
            beta = 0.0
        gamma = getattr(args, "tli_gamma", 1.0)
        if isinstance(gamma, str) and gamma.strip().lower() == "off":
            gamma = None
        elif gamma is None:
            gamma = 1.0
        g_str = "off" if gamma is None else f"{gamma:g}"
        # 【10-09 TL-PREFILL-PROVENANCE-001 修复】稀疏 prefill 开关必须进输出文件名，
        # 否则 P-off/P-on 同路径 w 模式互相覆盖（GPT 审计 B09 族新实例）。默认不传
        # flag 时后缀为空 → 在跑链文件名逐位不变。
        p = "_P" if getattr(args, "tli_sparse_prefill", False) else ""
        # 【E138 消融①】decode dense 回退标记进可读名——P on + DD on
        # （prefill-only 稀疏臂）与 P on（双稀疏臂）文件名必须可分（同
        # TL-PREFILL-PROVENANCE-001 族：开关不进名则同路径 w 模式互覆）。
        # 默认关不加后缀，在跑链文件名逐位不变。
        pdd = "_DD" if getattr(args, "tli_decode_dense", False) else ""
        # 【E135】sink/swa 非默认值/归因臂标记进可读名——默认（128/128/关）
        # 不加任何后缀，锚点文件名逐位不变；身份单射由 manifest/hash 恒承载。
        sk = getattr(args, "tli_sink_tokens", 128)
        sw = getattr(args, "tli_swa_size", 128)
        sso = getattr(args, "tli_sinkswa_only", False)
        e135 = ""
        if sso:
            e135 += "_sinkswaonly"
        if sk != 128 or sw != 128:
            e135 += f"_sk{sk:g}_sw{sw:g}"
        # E137：动态 D' gate 标记（默认关不加后缀；eps/rows 非默认时渲染）
        ddg = getattr(args, "tli_dyn_dgate", False)
        if ddg:
            e135 += "_dyndgate"
            dep = getattr(args, "tli_dyn_dgate_eps", 0.05)
            drw = getattr(args, "tli_dyn_dgate_rows", 256)
            if dep != 0.02:
                e135 += f"_eps{dep:g}"
            if drw != 256:
                e135 += f"_r{drw:g}"
        # E122 合并说明：γ=off 自由竞争经 g_str="off" 渲染进文件名
        # （a…_b…_goff），与 γ 数值臂天然区分，无需独立 _goff 后缀；
        # 默认缺省 γ 按 1.0 处理（getattr 兜底，只有显式 off 才是 None）。
        # E155/E155b：固定配额几何 + 双开关标记进可读名（GPT 审计
        # 2026-10-19 阻塞级）。参照系已翻转到新口径——默认（无几何/
        # bypass 开/quant 关）不加后缀；几何显式时 _np/_nt/_nl，旧口径
        # 消融臂显式 _bb0（bypass 关）/ _lq4（quant 开）。身份单射由
        # manifest/hash 恒承载（可读段仅防御性冗余）。
        e155 = ""
        _np = getattr(args, "tli_near_pages", None)
        _nt = getattr(args, "tli_near_tokens", None)
        _nl = getattr(args, "tli_near_len", None)
        # E155d：nt=off 哨兵渲染 _ntoff（自由竞争几何臂），与数值 nt
        # 天然区分（f"{_nt:g}" 对 str 会崩，须先分流）。
        _nt_off = isinstance(_nt, str) and str(_nt).strip().lower() == "off"
        if _np is not None or _nt is not None:
            if _np is not None:
                e155 += f"_np{_np:g}"
            if _nt_off:
                e155 += "_ntoff"
            elif _nt is not None:
                e155 += f"_nt{_nt:g}"
        if _nl is not None:
            e155 += f"_nl{_nl:g}"
        if not getattr(args, "tli_boundary_bypass", True):
            e155 += "_bb0"
        if getattr(args, "tli_l2_quant", False):
            e155 += "_lq4"
        # 076（TL-E121-OUTPUT-ID）：可读段逐位不动（B09 口径），追加
        # canonical treatment hash 尾段 _h<hash10>——全部输出相关参数
        # （far/near method/select、subspace 具体值、投影基、per_q_head、
        # sim/σ/MoBA 选择配置等）排序键 JSON→sha256 的身份段。不同
        # treatment 必不同名（单射），同名互覆/指针错切不再可达。
        # 081：hash 从冻结 snapshot 派生（生产入口传入）——生成窗口内
        # 文件被替换时文件名与 runtime/sidecar 仍同源。
        return (f"tli_{args.tia_block_size}_{args.tia_level1_topk}_"
                f"{args.tia_level2_topk}_c{args.tia_level2_cmp_ratio}_"
                f"{ab}a{alpha:g}_b{beta:g}_g{g_str}{p}{pdd}{e135}{e155}"
                f"_h{_treatment_hash(args, snapshot)}")
    elif args.method == "none":
        return "none"
    else:
        raise ValueError
