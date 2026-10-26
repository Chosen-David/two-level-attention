import argparse
import math

def _add_tia_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='Twi-Index Sparse Attention')
    group.add_argument('--tia_block_size', type=int, default=64)
    group.add_argument('--tia_level1_topk', type=int, default=64)
    group.add_argument('--tia_level2_topk', type=int, default=1024)
    group.add_argument('--tia_level2_cmp_ratio', type=int, default=4, choices=[2, 4, 8])
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
    group.add_argument('--quest_token_budget', type=int, default=1024, help='Quest token 预算（官方默认 1024）')
    group.add_argument('--quest_chunk_size', type=int, default=16, help='Quest chunk 长度（官方默认 16）')
    return parser

def _add_minference_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='minference')
    group.add_argument('--minference_config_path', type=str, default=None, help='MInference best_pattern JSON（None = 不启用）')
    group.add_argument('--minference_starting_layer', type=int, default=-1, help='起始 dense 层（默认 -1 = 全层 sparse）')
    return parser

def _add_sparq_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='sparq')
    group.add_argument('--sparq_k', type=int, default=1024, help='SparQ token 预算（默认 1024）')
    group.add_argument('--sparq_context_length', type=int, default=None,
                       help='SparQ 坐标数 r 推算用长度（None = 跟随任务 context_length）')
    return parser


def _add_moba_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='moba')
    group.add_argument('--moba_chunk_size', type=int, default=4096, help='MoBA chunk 大小（官方默认 4096）')
    group.add_argument('--moba_topk', type=int, default=None, help='MoBA top-k 块数（None = 按 budget 换算）')
    group.add_argument('--moba_budget', type=int, default=1024, help='MoBA token 预算（默认 1024）')
    return parser


def _add_fasa_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='fasa')
    group.add_argument('--fasa_budget', type=int, default=1024, help='FASA token 预算（默认 1024）')
    group.add_argument('--fasa_dataset', type=str, default='qasper', help='FASA agreement.pkl 数据集名')
    group.add_argument('--fasa_threshold_ratio', type=float, default=0.8, help='FASA agreement 阈值比')
    group.add_argument('--fasa_min_k', type=int, default=16, help='FASA min retained dims')
    group.add_argument('--fasa_max_k', type=int, default=16, help='FASA max retained dims')
    group.add_argument('--fasa_mode', type=str, default='topk', choices=['topk', 'dynamic'])
    group.add_argument('--fasa_agreement_dir', type=str, default='dimension_rankings')
    return parser


def _add_clusterkv_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='clusterkv')
    # ClusterKV: KMeans 聚类稀疏（prefill dense + decode 簇选择 + 精确细筛）
    group.add_argument('--clusterkv_token_budget', type=int, default=1024, help='簇选择 token 预算（官方默认 1024）')
    group.add_argument('--clusterkv_sink', type=int, default=16, help='sink 区大小（官方默认 16）')
    group.add_argument('--clusterkv_window', type=int, default=320, help='window 区大小（官方默认 320）')
    group.add_argument('--clusterkv_nlist', type=int, default=400, help='KMeans 簇数（官方默认 400）')
    group.add_argument('--clusterkv_fit_iter', type=int, default=20, help='KMeans max_iter（官方默认 20）')
    group.add_argument('--clusterkv_head_sel', type=str, default='truc', choices=['truc', 'pad'], help='per-head 预算截断策略')
    group.add_argument('--clusterkv_balance', action='store_true', help='Balanced KMeans')
    group.add_argument('--clusterkv_dist_t', type=str, default='cosine',
                       choices=['cosine', 'inner_product', 'l2', 'l1', 'euclidean', 'chebyshev', 'canberra'],
                       help='KMeans 距离度量')
    group.add_argument('--clusterkv_gqa_policy', type=str, default=None, choices=['qavg'], help='GQA 聚合策略')
    group.add_argument('--clusterkv_topk_stat', action='store_true', help='top-k 命中率统计')
    return parser

def _add_asynctls_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='asynctls')
    group.add_argument('--asynctls_channel_file', type=str, default=None, help='Double-Sparsity channel 校准文件（None = 拒绝启动）')
    group.add_argument('--asynctls_token_budget', type=int, default=1024, help='最终 token 预算（默认 1024）')
    group.add_argument('--asynctls_block_size', type=int, default=64, help='Stage-1 块长（默认 64）')
    group.add_argument('--asynctls_top_kb', type=int, default=128, help='Stage-1 保留块数（默认 128）')
    return parser


def _add_jssa_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='jssa')
    group.add_argument('--jssa_budget', type=int, default=1024, help='JSSA 每 kv head top-B token 预算（默认 1024）')
    group.add_argument('--jssa_rank', type=int, default=32, help='JSSA 投影子空间秩 r（默认 32）')
    group.add_argument('--jssa_basis_path', type=str, default=None, help='JSSA 校准基 .pt（None = 拒绝启动）')
    return parser


def _add_hisa_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='hisa')
    group.add_argument('--hisa_budget', type=int, default=1024, help='最终 token 预算（默认 1024）')
    group.add_argument('--hisa_k_block_size', type=int, default=128, help='Stage-0 块长（默认 128）')
    group.add_argument('--hisa_block_topk', type=int, default=64, help='Stage-1 保留块数（默认 64）')
    return parser


def _parse_gamma(x):
    """解析 legacy γ：'off'=自由竞争（取消 L2 near/far 配额），数值=float。"""
    if str(x).strip().lower() == "off":
        return "off"
    return float(x)


def _parse_near_tokens(x):
    """解析 near token 配额：'off'=自由竞争（默认），数值=int。"""
    if str(x).strip().lower() == "off":
        return "off"
    try:
        return int(x)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"--tli_near_tokens 须为非负整数或 'off'，收到 {x!r}")


def _parse_dgate_rows(x):
    """D' gate 标定尾行数校验（必须 >= 1）。"""
    try:
        v = int(x)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"--tli_dyn_dgate_rows 必须是整数，收到 {x!r}")
    if v < 1:
        raise argparse.ArgumentTypeError(f"--tli_dyn_dgate_rows 必须 >= 1，收到 {x!r}")
    return v


def _parse_dgate_eps(x):
    """D' gate eps 校验（必须为有限值且 0<=eps<=1）。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"--tli_dyn_dgate_eps 必须是数值，收到 {x!r}")
    if not math.isfinite(v) or not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError(f"--tli_dyn_dgate_eps 必须为有限值且 0<=eps<=1，收到 {x!r}")
    return v


def _add_tli_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='TLI (Two-Level Indexer)')
    group.add_argument('--tli_enable_subspace', type=lambda x: str(x).lower() != 'false', default=True)
    group.add_argument('--tli_enable_kmeans', type=lambda x: str(x).lower() != 'false', default=False)
    group.add_argument('--tli_far_select', type=str, default='4bit',
                       choices=['4bit', 'cluster', 'sim_greedy'],
                       help='L2 far 选择: 4bit=细筛分数(默认) / cluster=kmeans 簇代表 / sim_greedy=贪心聚类')
    group.add_argument('--tli_near_select', type=str, default='4bit',
                       choices=['4bit', 'cluster', 'sim_greedy'],
                       help='L2 near 选择: 同 --tli_far_select')
    group.add_argument('--tli_sim', type=float, default=0.9,
                       help='sim_greedy 余弦归并阈值')
    group.add_argument('--tli_sim_dims', type=str, default='subspace',
                       choices=['subspace', 'nope', 'tail', 'full'],
                       help='sim_greedy 聚类维度')
    group.add_argument('--tli_enable_layer_skip', type=lambda x: str(x).lower() != 'false', default=False)
    group.add_argument('--tli_far_clusters', type=int, default=256)
    group.add_argument('--tli_far_niter', type=int, default=10)
    group.add_argument('--tli_far_blocks', type=int, default=16)
    group.add_argument('--tli_far_tokens', type=int, default=512)
    group.add_argument('--tli_layer_skip_path', type=str, default=None)
    group.add_argument('--tli_far_method', type=str, default='minmax', choices=['minmax', 'avg'],
                       help='far 池 L1 粗筛: minmax=块上界(默认) / avg=块均值')
    group.add_argument('--tli_near_method', type=str, default='avg', choices=['avg', 'minmax'],
                       help='near 池 L1 粗筛: avg=块均值(默认) / minmax=块上界')
    # ── legacy 比例口径（与显式几何 nb/nt/nl 互斥）──
    group.add_argument('--tli_alpha', type=float, default=None, help='near 区长占 mid 比（legacy α，与 nb/nt/nl 互斥）')
    group.add_argument('--tli_beta', type=float, default=None, help='near 块占 K1 比（legacy β，与 nb/nt/nl 互斥）')
    group.add_argument('--tli_gamma', type=_parse_gamma, default=None,
                       help='near token 折扣（legacy γ；off=自由竞争，与 nb/nt/nl 互斥）')
    # ── 固定配额几何（论文 champion 默认）──
    group.add_argument('--tli_near_pages', type=int, default=32, help='near 块数 nb（默认 32 = champion）')
    group.add_argument('--tli_near_tokens', type=_parse_near_tokens, default='off',
                       help='near token 配额: off=自由竞争(默认) / 整数=固定配额')
    group.add_argument('--tli_near_len', type=int, default=4096, help='near 区宽度 nl token（默认 4096 = champion）')
    group.add_argument('--tli_boundary_bypass', type=lambda x: str(x).lower() != 'false', default=True,
                       help='边界页保送 L2（默认开）')
    group.add_argument('--tli_l2_quant', type=lambda x: str(x).lower() != 'false', default=False,
                       help='L2 4bit 假量化（默认关）')
    group.add_argument('--tli_subspace', type=str, default='tail',
                       choices=['full', 'rope', 'nope', 'tail', 'random', 'highfreq', 'mla_rope'],
                       help='评分子空间: tail=低频尾维32(默认) / full=全128 / rope=前64 / nope=后64 / mla_rope=MLA')
    group.add_argument('--tli_l2_subspace', type=str, default='same', choices=['same', 'full'],
                       help='L2 细筛子空间（默认 same = 与 L1 一致）')
    group.add_argument('--tli_proj_basis', type=str, default=None, help='sup_wsvd 投影基 .pt')
    group.add_argument('--tli_static_pair', action='store_true',
                       help='per-layer 静态 top-16 RoPE 对子空间')
    group.add_argument('--tli_moba', action='store_true', help='MoBA 复现臂（无两级无量化）')
    group.add_argument('--tli_sigma_select', type=str, default='none',
                       choices=['none', 'far', 'near', 'mid'], help='top-σ 替代两级')
    group.add_argument('--tli_sigma', type=float, default=8.0, help='top-σ 阈值')
    group.add_argument('--tli_per_q_head', type=lambda x: str(x).lower() != 'false',
                       default=False, help='per-q-head 独立选择（关闭=GQA 共享口径）')
    group.add_argument('--tli_sink_tokens', type=int, default=4, help='sink 宽度 token（默认 4 = StreamingLLM）')
    group.add_argument('--tli_swa_size', type=int, default=128, help='SWA 宽度 token（默认 128）')
    group.add_argument('--tli_sinkswa_only', action='store_true', help='归因臂：仅 sink+swa，mid 全丢')
    group.add_argument('--tli_dyn_dgate', action='store_true',
                       help='动态 D\' gate: prefill 标定 far 质量 < eps 的层 skip far')
    group.add_argument('--tli_dyn_dgate_eps', type=_parse_dgate_eps, default=0.05, help='D\' gate 阈值')
    group.add_argument('--tli_dyn_dgate_rows', type=_parse_dgate_rows, default=256, help='D\' gate 标定尾行数')
    group.add_argument('--tli_far_skip_backend', choices=['legacy', 'compact'], default='legacy',
                       help='far-skip 后端（E175 ring buffer = compact）')
    group.add_argument('--tli_far_skip_cache', choices=['retain', 'evict'], default='retain',
                       help='far-skip 缓存模式')
    group.add_argument('--tli_cuda_graph', action='store_true', default=False, help='CUDA Graph whole-step replay')
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
    parser.add_argument('--method', type=str, choices=['quest', 'quest_vendor', 'minference_vendor', 'fasa_vendor', 'sparq_vendor', 'ffd_vendor', 'moba_vendor', 'clusterkv_vendor', 'asynctls_vendor', 'hisa_vendor', 'jssa_vendor', 'tia', 'twi', 'tli', 'none'])
    parser = _add_quest_args(parser)
    parser = _add_minference_args(parser)
    parser = _add_fasa_args(parser)
    parser = _add_sparq_args(parser)
    parser.add_argument('--baseline-execute', action='store_true', help='explicitly enable new FFD/SparQ candidate execution; plan: python -m sparse_attn.baseline_runtime')
    parser.add_argument('--ffd_delta', type=float, default=5.0, help='FFD upstream log2 threshold, variable budget')
    parser.add_argument('--ffd_cache_length', type=int, default=131200)
    parser = _add_moba_args(parser)
    parser = _add_clusterkv_args(parser)
    parser = _add_asynctls_args(parser)
    parser = _add_hisa_args(parser)
    parser = _add_jssa_args(parser)
    parser = _add_twilight_args(parser)
    parser = _add_tia_args(parser)
    parser = _add_tli_args(parser)
    return parser
