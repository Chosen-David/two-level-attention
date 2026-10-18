# JSSA 校准脚本 transformers 4.56.2 兼容补丁（2026-10-18 Opus）

`third_party/JSSA/`（vendored，gitignore 不入库）下两处本地兼容修改，记录于此可追溯：

## 1. calibration/chunked_moe.py —— MoE 导入容错

```python
try:
    from transformers.integrations.moe import (
        ALL_EXPERTS_FUNCTIONS,
        grouped_mm_experts_forward as _orig_grouped_mm,
    )
except ImportError:
    # transformers 4.56.2 无 transformers.integrations.moe（更新版本引入）。
    # chunked_moe 仅服务 MoE 模型校准；dense 模型不需要——置 None，
    # install() 增加 `or ALL_EXPERTS_FUNCTIONS is None` 早退守卫。
    ALL_EXPERTS_FUNCTIONS = None
    _orig_grouped_mm = None
```

## 2. calibration/jssa_calibrate.py L200/L256 —— get_interface → get

`ALL_ATTENTION_FUNCTIONS.get_interface(impl, eager)` 在 4.56.2 不存在；
等价旧 API `ALL_ATTENTION_FUNCTIONS.get(impl, eager)`（按名取 attention
函数、缺省回退），两处替换，语义不变。

## 校准产物（不入库，路径记录）

- `third_party/JSSA/bases/qwen3_8b_32k.pt`：8×~32K LongBench 拼接序列
  （calibration/data/longalign_calib.jsonl，DeepSeek 轨 prepare_longbench_calib.py
  产出，LongAlign-10k 网络不可得的离线替代）prefill 累积 G_Qm/G_K，
  M = TopEig_32(G_Qm/tr + G_K/tr)，36 层全，M[0] 形状 (8, 32, 128)。
- 部分样本不足 32K（13427/16468 tokens，数据池限制）——gram 按 token
  加权累积，如实披露；校准长度选择与 32K 部署上下文匹配
  （JSSA 论文默认 8×131072，64K/128K 对比档如需再出长校准）。
- 冒烟：jssa_vendor/smoke_qwen3_jssa.py ALL PASS（含 T3 预算拉满=dense
  语义锚点铁律）；校准脚本 1×4096 冒烟 + 8×32K 正式各通过。
