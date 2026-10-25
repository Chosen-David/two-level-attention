# main prefill因果修复与dev decode mask门的安全整合

按用户授权，仅整合 main `366fd384ebd18c9caf294f9114be144bb2040fb4`
到原 dev `dc083b20af02f7a4b8d4d062a3de2ee10fb5fd16`。fresh fetch确认没有
其他writer已整合；非force只发布原dev，不合main。

## 精确范围

- 保留main fresh multi-token prefill、mask=None时的上三角因果屏蔽，以及
  输入4D浮点/device/准确shape校验。显式prefill additive mask（含-inf）照常消费。
- 保留dev已缓存前缀的decode约束及全零值校验，非法mask在投影、cache.update、
  indexer.clear/freeze/prepare_mask前拒绝。拒绝非零mask不是实现padding支持。
- 合并时将Tensor类型、strided layout检查置于全局mask.ndim等属性访问前，
  Python list/dict不再可能先触发AttributeError。decode全零检查未移到prefill。
- 只扩已有测试：prefill list/dict/sparse mask，拒绝后cache和indexer状态不变；
  decode新增sparse mask负例。沿用main因果对拍、替换后缀不影响前缀、batch2、
  bf16、strided hidden，以及dev非零/NaN/Inf/bool/int/错长度/设备等负例。
- Qwen Fig1、归档校验器、workflow、eager_decoding及batch_select与原dev字节一致。
  kernel、选择预算128+128、实际guide、实验矩阵、既有论文PDF和数据均不改。

main旧报告的“decode非零mask未修复”描述其当时main状态；本次整合后的dev
通过提前拒绝维护无padding契约，不宣称MLA算法恢复或完整padding支持。

## 源码身份、测试与独审

合并后 `sparse_attn/patches/deepseek_v2_attn_patch.py`：

- SHA256 `5e0a718401c14470f5458a6a6c71fd705823e892c3d585c845fe0b1f93072cf9`
- Git blob `e7a18d95e5592d91154d4266aa986a0bdc758b81`

证据：`exp/paper/evidence/mask-integration-cpu-20261015/`，含逐文件身份、
精确命令、日志和独审结论。CPU普通/-O ops全套各37项通过；Qwen Fig1/归档
回归各14项通过。独立只读reviewer的mask/lifecycle/异构KV子集各17项通过，
无阻塞问题。没有把测试替身当真实HF cache或GPU验收。

剩余限制：真实权重/HF cache/CUDA、长序列内存和性能未验证。继承main的
None prefill遮罩需要O(L²) bool临时空间；decode值检查可能同步设备，未做
性能结论。Qwen归档仍缺，paper_ready=false不变。本轮零GPU/付费/Claude/
SGLang操作，没有重试取消的稿件上传。
