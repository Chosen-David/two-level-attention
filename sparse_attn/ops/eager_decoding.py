"""请求边界的延迟错误标志与 fail-closed 生命周期 guard。"""
from contextlib import contextmanager

import torch


_ERR_FLAG = {}  # device -> [1] int32；0=正常，其余取值见 _raise_for_code。


def _err_flag(device: torch.device) -> torch.Tensor:
    buf = _ERR_FLAG.get(device)
    if buf is None:
        # 持久标志必须比 inference_mode 请求活得更久。
        with torch.inference_mode(False):
            buf = torch.zeros(1, dtype=torch.int32, device=device)
        _ERR_FLAG[device] = buf
    return buf


def eager_decoding_check_errors(device=None, clear: bool = True):
    """在请求边界一次性读取并清空延迟错误标志。"""
    if device is None:
        for buf in list(_ERR_FLAG.values()):
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
    if code == 4:
        raise RuntimeError(
            'ordered attention: invalid selected token IDs, counts or live '
            'lengths; request rejected')
    if code == 1:
        raise RuntimeError(
            'eager_decoding_attn: 选择 mask 存在全 False 行，softmax 将产生 '
            'NaN；请检查 indexer 的 sink/SWA 强制选择契约。')
    if code == 2:
        raise RuntimeError(
            'eager_decoding_attn: mask 某行选中数超过 NFA_E127_W_UPPER 上界；'
            '本次请求已 fail-closed 作废。')
    if code == 3:
        raise RuntimeError(
            'eager_decoding_attn: cu_seqlens_k 契约破坏；本次请求已 '
            'fail-closed 作废，请检查调用方传入的长度元数据。')


@contextmanager
def eager_decoding_request_guard(device=None):
    """清理旧标志，并保证请求失败或返回前消费本次延迟错误。

    标志按设备共享；同一进程内的并发请求不具备错误归因隔离，调用方必须
    串行化使用该 guard 的请求边界。
    """
    for buf in list(_ERR_FLAG.values()):
        buf.zero_()
    try:
        yield
    finally:
        eager_decoding_check_errors(device)
