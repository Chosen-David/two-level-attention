# E141：先修正 Q/K 重建，再决定 MLA 子空间与层跳过

## 范围与结论

- 代码基线：`87029997b9edf63cabd2fe53a05bb10eb9623be4`，增量 `f0b0308..8702999`。本轮生产 selector/kernel 未变化，重点检查新增 E141 trace 及 E140 审计回应。
- 实验分支 `e138-ablation` 仍为 `e1edf9c316fc4190dd0c29bc5d47477136f8bdab`；未合并实验分支，也未启用未验收 block-list 候选。
- **TL-E141-QK-RECONSTRUCTION-20261015，P1，confirmed（源码契约 + CPU 实际函数复现）**：E141 手动重建既漏掉 attention 前的 input RMSNorm，又漏掉 DeepSeek RoPE 的通道重排。当前 probe 的概率参考不是所声明的官方模型注意力概率，不能据此接受 rope-only / sink+swa 层跳过。
- 不宣称真实模型 GPU 对拍完成，不宣称修后质量/提速。远端实验机器实际加载的自定义模型文件版本未提供；下述官方版本必须与部署副本再核对。

## 定位与根因

路径均相对本仓库，行号绑定上述基线。

| 定位 | 违反契约 | 触发和实际行为 | 预期行为 |
| --- | --- | --- | --- |
| `agent_doc/guide/paper_figs/trace/e141_mla_nope_probe.py:73-80` | 用 `hidden_states[L]` 直接投影并不等于该层 self-attention 的输入 | 任意非恒等 input RMSNorm；Q 以及 KV 的输入发生变化，含 token 范数和学习的逐维权重影响 | 先应用该层 `input_layernorm`，或捕获真实 attention 入口张量 |
| 同文件 `:85-90` | cos/sin 来源一致不足以保证 RoPE 一致 | 64 维 RoPE 在不同 token 位置的旋转；遗漏输入通道的偶/奇重排，跨位置 QK 点积和概率会变化 | 使用部署版本自身的 `apply_rotary_pos_emb`，保留其布局转换与 position_ids |
| 同文件 `:99-107,145-161` | 错误 Q/K 又进入 full192 概率 oracle 和 mid-mass 跳过判定 | 所有 arms 相对于错误参考计算 retained mass；低于 `1e-9` 的 query 被丢弃 | 先通过真实 forward 的 logits/probability parity，再生成选择质量结论 |

2026-10-15 核验官方模型源文件：
[DeepSeek-V2-Lite-Chat / modeling_deepseek.py，固定版本 85864749](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/85864749cd611b4353ce1decdb286193298f64c7/modeling_deepseek.py)。文件 SHA256 为 `7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c`；main 与该固定版本下载字节一致。

该文件的 `DeepseekV2Model.forward` 在调用 decoder layer **之前**记录 hidden states（约 1531 行），decoder layer 内先 input RMSNorm 再 attention（1255 行）；attention 的 rotary helper 在乘 cos/sin 前做通道重排（339-370 行）。这是两个独立缺项，不是允许的 bf16 舍入误差，也不能由后面的 `kv_a_layernorm` 补偿。概率对拍进一步排除了“只差无害的坐标置换”的解释。

## 有界可运行复现

新增 `exp/trace/verify_e141_qk_reconstruction.py`，只抽取原 probe 的函数和可信官方源文件的 RMSNorm / rotary 定义，不加载模型权重、不运行完整模型。诊断失败退出 **1**，不是把复现 bug 写成测试通过。

```bash
curl -L --fail https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/raw/85864749cd611b4353ce1decdb286193298f64c7/modeling_deepseek.py -o /tmp/e141_modeling_deepseek.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python exp/trace/verify_e141_qk_reconstruction.py --upstream-source /tmp/e141_modeling_deepseek.py
```

环境：Python 3.12.14，PyTorch 2.8.0+cpu，x86_64，1 CPU thread。无 CUDA 设备、驱动或 GPU 编译器验收；无 kernel 配置/性能数字。运行时以本地既有 CPU torch 路径加入 PYTHONPATH，不安装新依赖。

输入：seed=20261015，float32，B=1/T=5/hidden=8/H=2，NOPE=128/ROPE=64，KV latent=512；不同 token 范数及非恒等 RMSNorm 权重。RoPE 使用非零、非均一测试角度；这不是 YaRN 真实 checkpoint 的质量测试。诊断程序把 norm、RoPE 两种误差分离：第二项将已归一化输入送入原函数，并将该层 norm 置为 Identity，仅隔离 RoPE。

- probe SHA256：`b7565a5acbb3af0701f339ea93870b7cff95379572764e0f80ac3908380e9930`
- fixture（输入、投影参数、norm 权重、cos/sin 连续字节）SHA256：`ea873ffc4ee5544e014d9fc54ee311b6d95fbe9badc4ba3e597bee9caf383ffb`

原始短输出（普通 Python 与 `python -O` 一致，exit=1）：

```json
{
  "norm_q_nope_max_abs": 15.283309936523438,
  "rope_only_q_max_abs": 2.6790432929992676,
  "rope_only_k_max_abs": 2.23616886138916,
  "rope_only_probability_max_abs": 0.03163066506385803,
  "parity": false
}
```

另外在**私有临时副本**补上这两处，保持其他函数不变，相同 fixture 四项误差全为 `0.0`、parity=true、exit=0（普通 Python 与 `-O`）。该反事实验证说明 CPU 夹具能够区分修正前后；它不是生产修复发布或模型验收。

## 历史数据影响与修复建议

1. `exp/trace/results/e141_mla_nope_trace.json` 及其 verdict 需要标为“待重建 oracle 后重跑”。当前文件有 20 cells，layers 为 4/8/12/16/20，总计 157 个被保留 query；8 个 L12/L16 cells 全跳过。未记录下来的 query 不能等价于概率精确为零，更不能证明所有层可跳过。
2. verdict 的 L24 结论没有本次 raw JSON /脚本支持；原数据没有 L24。应修正元数据，不把未测试层当通过。
3. 暂停从该 trace 得出“nope-only 被驳回”“rope64 与 full192 等价”“天然零成本层跳过”。**没有证明这些假说最终为假**；证明的是现有构造不能支撑判决。192→64 是维数比，不能当端到端 3x 实测。
4. 最小修复：第 73 行先应用 `model.model.layers[L].input_layernorm`；RoPE 对接部署模型自身 helper（包含布局变换、position_ids、YaRN 参数），不要仅复用 cos/sin。更稳健方案是在真实前向短输入上 hook attention 输入/中间 QK，并用真实 forward 输出建立 fail-closed parity gate。
5. 重测先短输入、多个位置、bf16/float32、非连续布局，验证逐 head logits、概率及 head-mean oracle；再跑原 2 tasks × 2 samples × 5 layers × 16 queries，记录每个跳过 query 的实际 mid mass、finite 状态和原因，而不是将阈值跳过称“精确 0”。实际模型 revision/源码 hash 必须入 metadata。
6. 所有子空间臂必须重测。核验 oracle 后再考虑 selector 减维、单索引或层跳过，并分别测选择质量与完整请求成本。本轮没有依据修改生产 prefill/decode 算法、预算、chunk2048 或开启 overlap。

受保护路径：`agent_doc/README.md` 将 guide 标为人类发布、AI 只读，当前用户也要求尊重既有编辑边界。因此本轮**没有修改 guide 探针、图或原始实验数据**；发布的是独立诊断器与审查报告，源修复仍待该路径获授权的维护者完成。也没有通过复制一份正式 probe 到其他目录来绕过权限。

## 旧发现复查与后续

- TL-E140-QUOTA-ROUNDING-20261015：8702999 已修正首个零 far 配额到 156，并修正对应标签/总预算描述；本轮确认 diff，不重复旧报告。
- TL-E140-MASS-ORACLE-20261015：判决已降级为 proxy 观察；原 probe 的 logits/probability oracle 尚未统一，不标为算法已修复。
- E140 `get_qk` 也直接取 raw hidden states，是后续接口核验线索；本报告未进行 Qwen3 实际模型对拍，不据此追加其全部历史质量结论。
- Graph 实际 dispatch 证明与 block-list GPU 差分/测速仍未完成，检查点保留。未运行 GPU、真实 LongBench、全模型 TTFT，未占用远端设备；CPU 诊断不替代这些验证。
