"""【阶段 2c】CUDA Graph decode 共享状态（E139 先例同款模块级单例）。

背景：pred_ruler_batch 的 TLI batch decode（_tl_decode_batched +
TLIStaticCache）整步 capture 时，decode 步间变化的数据必须经
「预分配固定地址 buffer + replay 前 copy_ 注入」传入，否则 capture 时
烘焙的 tensor 地址/launch 参数在 replay 时读到旧值。

本模块持有 graph 模式开关与 KV 写入位置 buffer：
- pos_buf [B] long：本步每行新 token 的绝对位置（= seq_lens[b]-1），
  decode 循环每步 copy_ 更新 → TLIStaticCache.update 的 index_copy_ 用
  tensor 索引写新列（replay 时 pos_buf 内容变化 → 写新位置）。
- 其余每步输入（input_ids / position_ids / 4D additive mask / TLI mask /
  valid_length）由 decode 循环与 patch 侧各自的预分配 buffer 承载。

非 graph 模式（pos_buf 为 None）：update 回退 Python 逐行构造 pos，
行为与现状逐位一致（铁律）。
"""
from __future__ import annotations


class _GraphState:
    def __init__(self):
        self.enabled = False       # graph 模式开关（decode 循环 capture/replay）
        self.pos_buf = None        # [B] long：本步 KV 写入列（capture 外每步 copy_）
        self.valid_len_buf = None  # [B] long：本步每行有效长（prepare_mask valid_length）
        self.bs = None             # 当前 batch（buffer 形状校验用）


_GRAPH_STATE = _GraphState()


def graph_state():
    return _GRAPH_STATE


def enable_graph(bs):
    st = _GRAPH_STATE
    st.enabled = True
    st.bs = bs


def disable_graph():
    st = _GRAPH_STATE
    st.enabled = False
    st.pos_buf = None
    st.valid_len_buf = None
    st.bs = None
