import argparse
import math

def _add_tia_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='Twi-Index Sparse Attention')
    group.add_argument('--tia_block_size', type=int, default=64)
    group.add_argument('--tia_level1_topk', type=int, default=128)
    group.add_argument('--tia_level2_topk', type=int, default=1024)
    group.add_argument('--tia_level2_cmp_ratio', type=int, default=2, choices=[2, 4, 8])
    group.add_argument('--tia_enable_async_topk', action="store_true")
    return parser

def _add_twilight_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='Twilight Sparse Attention')
    group.add_argument('--twi_block_size', type=int, default=64)
    group.add_argument('--twi_level1_topk', type=int, default=128)
    group.add_argument('--twi_level2_topp', type=float, default=0.95)
    return parser

def _add_quest_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='quest')
    group.add_argument('--quest_block_size', type=int, default=64)
    group.add_argument('--quest_topk', type=int, default=16)
    # B 轨 vendor（官方源码复用，per-head topk 官方粒度）：Quest 官方口径
    # token_budget=1024（等效 token 预算）、chunk_size=16（chunk 长度）。
    # 与 A 轨 quest_block_size/topk 的关系：quest_vendor 用这两个参数，
    # 官方 local_heavy_hitter_mask 内部按 chunk_size 分组选 topk chunk。
    group.add_argument('--quest_token_budget', type=int, default=1024)
    group.add_argument('--quest_chunk_size', type=int, default=16)
    return parser

def _add_minference_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='minference')
    # MInference vendor（官方源码复用）：prefill 动态稀疏（vertical+slash）+ decode dense。
    # config_path = best_pattern JSON（is_search 在线搜出后复用）；starting_layer 之前
    # 的浅层 dense（官方默认 -1 = 全层 sparse）。
    group.add_argument('--minference_config_path', type=str, default=None)
    group.add_argument('--minference_starting_layer', type=int, default=-1)
    return parser

def _add_sparq_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='sparq')
    # SparQ vendor（官方源码复用）：decode 时 |Q| 的 r 个最大坐标近似注意力
    # 分数 → top-k（k 个 token）→ 全维 QK 精确注意力 + V_mean lerp 插值。
    # 主表口径（用户 2026-10-14 拍板）：k 固定 1024（token budget 与
    # quest/fasa 对齐）；r = max(1, round(head_dim / (context_length/(2*k))))
    # 随序列长度变化（Qwen3-8B head_dim=128 → 32K: r=8；128K: r=2）。
    group.add_argument('--sparq_k', type=int, default=1024,
                       help='SparQ top-k token 预算（主表口径固定 1024）')
    group.add_argument('--sparq_context_length', type=int, default=None,
                       help='SparQ r 反推用的 context_length（默认 None = 跟随任务'
                            ' --context_length，即 r 随序列长度自适应）')
    return parser


def _add_moba_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='moba')
    # MoBA vendor（官方源码复用）：prefill 块级 gate 稀疏（chunk-mean key gate →
    # per-query top-k 选块）+ decode dense。主表预算口径对齐 K2=1024 token/query：
    # topk 默认 None → enable_moba 内 topk = max(2, ceil(budget/chunk))。官方默认
    # chunk=4096 → topk=2 → 预算 8192（官方 efficient 内部 topk-1，topk=1 退化为
    # dense）；严格对齐 1024 须改 chunk（如 256→topk=4），论文标注 budget-normalized
    # 消融。
    group.add_argument('--moba_chunk_size', type=int, default=4096,
                       help='MoBA chunk 大小（官方默认 4096，方法定义参数）')
    group.add_argument('--moba_topk', type=int, default=None,
                       help='MoBA top-k 块数（None = 按 budget 换算；显式传覆盖）')
    group.add_argument('--moba_budget', type=int, default=1024,
                       help='MoBA token 预算口径（默认 1024 与 Quest/SparQ/FASA 对齐）')
    return parser


def _add_fasa_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='fasa')
    # FASA vendor（官方源码复用）：decode 时按频率分组 agreement（Phase 1 校准产物
    # dimension_rankings/<model>/<dataset>_agreement.pkl）选 head_dim 子维度，再
    # core_module_with_padding_old 剪枝到 budget 个 token。budget 默认 1024 与
    # quest/minf 主表预算口径对齐（FASA 论文默认 256 可显式覆盖做消融）。
    group.add_argument('--fasa_budget', type=int, default=1024)
    group.add_argument('--fasa_dataset', type=str, default='qasper',
                       help='Phase 1 校准数据集名（agreement.pkl 的 dataset 段）')
    group.add_argument('--fasa_threshold_ratio', type=float, default=0.8)
    group.add_argument('--fasa_min_k', type=int, default=16)
    group.add_argument('--fasa_max_k', type=int, default=16)
    group.add_argument('--fasa_mode', type=str, default='topk', choices=['topk', 'dynamic'])
    group.add_argument('--fasa_agreement_dir', type=str, default='dimension_rankings')
    return parser


def _add_clusterkv_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='clusterkv')
    # ClusterKV vendor（官方源码复用，KMeans 聚类稀疏）：prefill dense；decode
    # 后期对 prefill key pylibraft KMeans 建簇（build_cluster）→ query×centroids
    # 最近邻选簇到 token_budget−sink → sel_key_indices → +sink 偏移 → gather +
    # 精确 matmul 细筛（cluster_attn_out）。官方 token_budget 是「选择预算」
    # （sink 额外，kv_cache_size = sink + token_budget + window），与 TLI K2=1024
    # 已含 sink+swa 强制区的口径不同——严格对齐总 attend=1024 须换算
    # token_budget（如 768 + sink16 + window240），接入验收时按每 Q-head 实际
    # unique token 数核对（evaluation.md 口径）。默认沿用官方默认参数
    # （token_budget=1024, sink=16, nlist=400, fit_iter=20, head_sel=truc）。
    group.add_argument('--clusterkv_token_budget', type=int, default=1024,
                       help='ClusterKV 选择预算（官方默认 1024，sink 额外；与 TLI '
                            'K2=1024 口径不同需换算核对）')
    group.add_argument('--clusterkv_sink', type=int, default=16,
                       help='ClusterKV sink 区大小（官方默认 16，强制保留前缀 token）')
    group.add_argument('--clusterkv_window', type=int, default=320,
                       help='ClusterKV window 区大小（官方 efficiency 路径默认 320，'
                            'accuracy 路径由 prompt 后最近区隐含；仅统计/换算用）')
    group.add_argument('--clusterkv_nlist', type=int, default=400,
                       help='KMeans 簇数（官方默认 400）')
    group.add_argument('--clusterkv_fit_iter', type=int, default=20,
                       help='KMeans max_iter（官方默认 20）')
    group.add_argument('--clusterkv_head_sel', type=str, default='truc',
                       choices=['truc', 'pad'],
                       help='per-head 选择预算截断策略（官方默认 truc）')
    group.add_argument('--clusterkv_balance', action='store_true',
                       help='Balanced KMeans（ivf_flat，官方默认 False）')
    group.add_argument('--clusterkv_dist_t', type=str, default='cosine',
                       choices=['cosine', 'inner_product', 'l2', 'l1', 'euclidean',
                                'chebyshev', 'canberra'],
                       help='KMeans 距离度量（官方默认 cosine）')
    group.add_argument('--clusterkv_gqa_policy', type=str, default=None,
                       choices=['qavg'], help='GQA 聚合策略（官方默认 None = repeat）')
    group.add_argument('--clusterkv_topk_stat', action='store_true',
                       help='官方 top-k 命中率统计（默认关闭）')
    return parser

def _parse_gamma(x):
    """E122：γ 支持 'off'（大小写不敏感）→ None = 自由竞争模式。

    γ=off 语义（用户 2026-10-10 指令，cavg 探索臂）：取消 L2 near/far 配额
    分割，全部 mid 候选在统一预算 K2_mid 内单池 topk 竞争（消费端
    tli_indexer.compute_mask 的 use_partition 分支）。数值照旧解析为 float。
    """
    if str(x).strip().lower() == "off":
        return None
    return float(x)


def _parse_dgate_rows(x):
    """A4（TL-E137-PARAM-VALIDATION-147）：--tli_dyn_dgate_rows 边界校验。

    标定用 prefill q 尾行数，必须 >= 1；非法值抛 ArgumentTypeError
    （argparse 转 exit 2，与 tli_indexer 构造函数双处校验同口径——
    构造层兜底 getattr 兜底路径）。"""
    try:
        v = int(x)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"--tli_dyn_dgate_rows 必须是整数（标定用 prefill q 尾行数），收到 {x!r}")
    if v < 1:
        raise argparse.ArgumentTypeError(
            f"--tli_dyn_dgate_rows 必须 >= 1（标定用 prefill q 尾行数），收到 {x!r}")
    return v


def _parse_dgate_eps(x):
    """A4（TL-E137-PARAM-VALIDATION-147）：--tli_dyn_dgate_eps 边界校验。

    far 区概率质量损失上界阈值，必须为有限值且 0<=eps<=1
    （NaN/inf/越界一律拒绝——NaN 比较恒 False 会静默禁用 gate）。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"--tli_dyn_dgate_eps 必须是数值（far 区概率质量阈值），收到 {x!r}")
    if not math.isfinite(v) or not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError(
            f"--tli_dyn_dgate_eps 必须为有限值且 0<=eps<=1"
            f"（far 区概率质量损失上界阈值），收到 {x!r}")
    return v


def _add_tli_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='TLI (Two-Level Indexer)')
    group.add_argument('--tli_enable_subspace', type=lambda x: str(x).lower() != 'false', default=True)
    group.add_argument('--tli_enable_kmeans', type=lambda x: str(x).lower() != 'false', default=True)
    # ---- E110：ccluster / sim_greedy（用户 2026-10-07 定义，TASK.md「关于cluster的方法」节）----
    #   far_select/near_select 各自独立可选 L2 选择方式：
    #   4bit       = 细筛分数 topk（现状默认，回归保护）
    #   cluster    = kmeans 簇代表打分 topk（ccluster = (cluster, cluster)；cavg = far 侧已有）
    #   sim_greedy = 增量贪心聚类（余弦相似度 >= --tli_sim 归并，簇心=算术均值增量维护）
    group.add_argument('--tli_far_select', type=str, default='4bit',
                       choices=['4bit', 'cluster', 'sim_greedy'],
                       help='far 侧 L2 选择: 4bit=细筛分数(默认) / cluster=kmeans 簇代表 / '
                            'sim_greedy=增量贪心聚类簇代表')
    group.add_argument('--tli_near_select', type=str, default='4bit',
                       choices=['4bit', 'cluster', 'sim_greedy'],
                       help='near 侧 L2 选择: 4bit=细筛分数(默认,现状) / cluster=kmeans 簇代表 / '
                            'sim_greedy=增量贪心聚类簇代表（ccluster 组合 = near 侧开 cluster）')
    group.add_argument('--tli_sim', type=float, default=0.9,
                       help='sim_greedy 余弦相似度归并阈值（簇心-running-mean 与新 token 的 cos >= sim 则归并）')
    group.add_argument('--tli_sim_dims', type=str, default='subspace',
                       choices=['subspace', 'nope', 'tail', 'full'],
                       help='sim_greedy 聚类维度: subspace=与 cluster/kmeans 路径同一子空间(默认,含压缩维) / '
                            'nope=后 64 非旋转维 / tail=低频尾维(cmp_ratio 控宽) / full=全 128 维')
    group.add_argument('--tli_enable_layer_skip', type=lambda x: str(x).lower() != 'false', default=True)
    group.add_argument('--tli_far_clusters', type=int, default=256)
    group.add_argument('--tli_far_niter', type=int, default=10)
    group.add_argument('--tli_far_blocks', type=int, default=16)
    group.add_argument('--tli_far_tokens', type=int, default=512)
    group.add_argument('--tli_layer_skip_path', type=str, default=None)
    # ---- E64 框架参数化（B 配置帕累托点：bp128+sup_wsvd d8+alpha0.125/beta0.25/gamma0.5）----
    group.add_argument('--tli_far_method', type=str, default='minmax', choices=['minmax', 'avg'],
                       help='far 池 L1 粗筛分数源: minmax=块上界(默认) / avg=块均值')
    group.add_argument('--tli_near_method', type=str, default='avg', choices=['avg', 'minmax'],
                       help='near 池 L1 粗筛分数源: avg=块均值(默认) / minmax=块上界')
    group.add_argument('--tli_alpha', type=float, default=0.0,
                       help='near 区占 mid 长度比（E64 alpha；0=单池老逻辑）')
    group.add_argument('--tli_beta', type=float, default=0.0,
                       help='near 块预算占 K1 比（E64 beta）')
    group.add_argument('--tli_gamma', type=_parse_gamma, default=1.0,
                       help='near 细筛 token 折扣（E64 gamma）；off=自由竞争'
                            '（E122：取消 near/far L2 配额分割，mid 单池 topk 竞争）')
    group.add_argument('--tli_subspace', type=str, default='full',
                       choices=['full', 'rope', 'nope', 'tail', 'random', 'highfreq', 'mla_rope'],
                       help='子空间选择: full=全128维(默认) / rope=前64旋转维 / nope=后64非旋转维 / tail=旧口径低频尾维32(E71/B7/C0复现) / '
                            'mla_rope=DeepSeek MLA 192维布局尾段rope 64维(E142a，E141判决:rope-only与full192持平)')
    group.add_argument('--tli_l2_subspace', type=str, default='same',
                       choices=['same', 'full'],
                       help='L2 细筛子空间选择（E134 子空间矩阵第三臂 sub+full）: '
                            'same=与 L1 粗筛同一子空间(默认,现状) / full=L2 细筛 k_qat 全 128 维量化'
                            '（仅 L1 用子空间而 L2 用全维的组合）')
    group.add_argument('--tli_proj_basis', type=str, default=None,
                       help='sup_wsvd 投影基 .pt（裸 tensor [n_layers,Hkv,128,r]）')
    # ---- E85f：per-layer 静态 pair 选取（E85e 重放 0.849 vs tail32 0.811 的 e2e 判决）----
    group.add_argument('--tli_static_pair', action='store_true',
                       help='prefill 尾 256 q 统计的 pair 幅值 top-16 完整对作 per-layer '
                            '固定子空间（需 --tli_subspace tail；pair 未定前回退 tail32）')
    # ---- E87：top-σ 选择（用户 2026-10-02 定义：区不做两级，细筛分 ≥ sink 分数 max − σ 即选中）----
    # ---- E89：MoBA 复现臂（统一 harness 内公平实现，training-free chunk gate）----
    group.add_argument('--tli_moba', action='store_true',
                       help='MoBA 复现臂：全维 chunk-mean gate 选 top-K2/BS 块全展开，'
                            '无两级无量化；sink/swa 同口径保送')
    group.add_argument('--tli_sigma_select', type=str, default='none',
                       choices=['none', 'far', 'near', 'mid'],
                       help='top-σ 侧: none=两侧两级(默认) / far=far 区 top-σ+near 两级 / '
                            'near=near 区 top-σ+far 两级 / mid=整个 mid 不分区 top-σ')
    group.add_argument('--tli_sigma', type=float, default=8.0,
                       help='top-σ 阈值宽度（细筛原始分尺度，锚 = sink 分数 max）')
    # ---- E103：kv-head 共享消融（审稿 MAJOR：§2.2 GQA 共享 vs per-q-head 独立选择）----
    group.add_argument('--tli_per_q_head', type=lambda x: str(x).lower() != 'false',
                       default=False,
                       help='E103 消融：旁路 GQA 组内 mean 聚合，per-q-head 独立选择'
                            '（mask [H,T]，索引量 ×G；关闭 = 论文 §2.2 kv-head 共享口径）')
    # ---- E135：sink/swa 正交强制区参数化（敏感性扫参）+ 归因臂（用户 2026-10-13 批准）----
    #   历史口径：sink = 前 2 块（tli_indexer sink_blocks=2 ×64=128 tok，硬编码）、
    #   swa = 尾 128 tok（tia_indexer sliding_window_size=128，硬编码）——两者均为
    #   无实验依据的工程约定（文献惯例 + 块对齐）。默认 128/128 与历史行为逐位一致。
    group.add_argument('--tli_sink_tokens', type=int, default=128,
                       help='sink 正交强制区宽度（token 级，默认 128 = 历史 sink_blocks×64）。'
                            'L1/L2 池左界按 floor(sink_tokens/block_size) 块对齐'
                            '（子块 sink 时 sink 所在块参与池竞争，强制区并集语义）')
    group.add_argument('--tli_swa_size', type=int, default=128,
                       help='swa 正交强制区宽度（token 级，默认 128 = 历史 sliding_window_size）')
    group.add_argument('--tli_sinkswa_only', action='store_true', default=False,
                       help='E135a 归因臂：mask 只含正交强制区（sink 头 + swa 尾），mid 全丢，'
                            'L1/L2 两级索引分数全旁路（分数照算但绝不进 mask）——'
                            '回答「功劳归 sink/swa 还是归 indexer」的消融对照')
    # ---- E137：D' gate 动态标定（用户 2026-10-13 指令：救静态 D' gate）----
    #   prefill 尾行观测逐层 far 概率质量 r_l，r_l < eps 的层 decode 期 skip far。
    #   核心假设已由 E60 离线验证：prefill 尾行 far mass 与 decode 尾行 far mass
    #   层间 corr 0.966~0.991（e60_prefill_dynamic_gate.json，mean/max/P95 三口径）。
    group.add_argument('--tli_dyn_dgate', action='store_true', default=False,
                       help='E137 动态 D\' gate：prefill 期逐层标定 far 区概率质量 '
                            '（q 尾行全维 softmax，与 dense prefill 同源），r_l < eps 的层 '
                            'decode 期跳过 far（E111 dgate_short 省算链全复用）')
    group.add_argument('--tli_dyn_dgate_eps', type=_parse_dgate_eps, default=0.05,
                       help='动态 D\' gate 阈值（被跳层 far 平均概率质量损失上界，'
                            'head-mean 口径；必须有限且 0<=eps<=1，'
                            'E137 全量标定 τ=0.05 割 ~4% 层 / 0.10 割 ~18% 层，'
                            '质量-速度旋钮，e2e 双臂判决）')
    group.add_argument('--tli_dyn_dgate_rows', type=_parse_dgate_rows, default=256,
                       help='标定用 prefill q 尾行数（与 observe_prefill_q E85f 同口径，必须 >= 1）')
    return parser

def parse_args():
    parser = argparse.ArgumentParser(description='Sparse Attention Arguments',
                                     allow_abbrev=False)
    parser.add_argument('--method', type=str, choices=['quest', 'quest_vendor', 'tia', 'twi', 'tli', 'none'])
    parser = _add_quest_args(parser)
    _add_twilight_args(parser)
    parser = _add_tia_args(parser)
    parser = _add_tli_args(parser)
    return parser.parse_args()

def add_sparse_attn_args(parser: argparse.ArgumentParser):
    parser.add_argument('--method', type=str, choices=['quest', 'quest_vendor', 'minference_vendor', 'fasa_vendor', 'sparq_vendor', 'ffd_vendor', 'asynctls_component', 'moba_vendor', 'tia', 'twi', 'tli', 'none'])
    parser = _add_quest_args(parser)
    parser = _add_minference_args(parser)
    parser = _add_fasa_args(parser)
    parser = _add_sparq_args(parser)
    parser.add_argument('--baseline-execute', action='store_true', help='explicitly enable new FFD/SparQ candidate execution; plan: python -m sparse_attn.baseline_runtime')
    parser.add_argument('--asynctls_calibration', help='held-out per-layer/head channel JSON, mandatory for component')
    parser.add_argument('--asynctls_blocks', type=int, default=128)
    parser.add_argument('--asynctls_tokens', type=int, default=1024)
    parser.add_argument('--ffd_delta', type=float, default=5.0, help='FFD upstream log2 threshold, variable budget')
    parser.add_argument('--ffd_cache_length', type=int, default=131200)
    parser = _add_moba_args(parser)
    parser = _add_clusterkv_args(parser)
    parser = _add_twilight_args(parser)
    parser = _add_tia_args(parser)
    parser = _add_tli_args(parser)
    return parser
