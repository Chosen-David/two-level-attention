# E141 collector 修复与 CPU 核验（2026-10-15）

状态：采集正确性修复已完成；替代源码上的 CPU 回归通过。**固定官方源码身份
核验仍受阻**，不得据此宣称指定官方模型对拍通过或放行实际采集。本轮不恢复
MLA 算法研发，不合 main，不修改历史数据或 verdict。

## 范围、授权与来源

用户本轮明确授权修复 E141 collector，包括原 guide/trace 下的采集代码；此授权
覆盖该具体代码文件，不扩展到实际人类 GUIDE 或其他 guide 内容。唯一写入者
为本任务；独审只读。开发基线 `7cfcc1d7dd5528111ad7a4220c7fa9e6ee29ca63`。

从 main `13be23cb1da409a4a63c0d009af34c8fbf167f95` 定向复用原 collector、
`exp/trace/verify_e141_qk_reconstruction.py` 和对应 reconstruction audit。
诊断器与旧审查文档保持原字节，不重造已有诊断、不整分支合并。

官方要求源文件固定 revision `85864749cd611b4353ce1decdb286193298f64c7`，
SHA256 `7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c`。
本环境对 Hugging Face 原始/resolve/源码 Git 请求均返回 HTTP 403，没有得到
该文件。没有把另一份源码当成官方固定文件，也没有下载模型权重。

## 修复内容

1. `get_mla_qk` 对 decoder 保存的层输入先应用该层 input_layernorm；RoPE
   旋转前按官方契约将偶/奇坐标重排为两个半区。固定 B=1、无 padding、无 cache、
   position_ids 从零起；拒绝训练模式、非法层号和不支持的投影几何。重建关闭梯度，
   验证有限值；没有创建 CPU 路径的 CUDA 清理调用。
2. 复用已修 E140 的 disjoint token 记账：near 在 mid 右端、SWA 恰好 128，
   短序列 sink/SWA 重叠不重复计数。末尾不足 64 的页参与选择，包括只有 1 token
   的页；候选使用输入 tensor 的 device。
3. 不再丢弃 skip query：保存位置、实际 mid/ref TopK mass、阈值、原因、O1 和
   reference IDs。`<1e-9` 不称为“精确零”或“下溢证明”。记录所有 query 的支持
   数量，并将已使用 query 的预算饱和/受限指标分别汇总。
4. 指标名称明确分母，overlap recall 以 reference token 数为分母；保存 selected/
   candidate/reference IDs、数量、输入 token hash 和实际子空间坐标。
   full192 明确是 logit proxy，P-topK 才是聚合质量 oracle。方差臂明确使用全序列
   future keys，是离线诊断；expanded NoPE 不等于 shared latent。
5. 默认仅 plan，须 `--execute` 才能采集；执行前要求身份声明和本地固定源码 hash，
   加载后再次检查实际 attention 源码。仅 local_files_only；新输出 v2，已有输出
   拒绝，最终 exclusive create。权重/tokenizer/data 身份声明仍未独立验证。

## 已完成验证与确切限制

证据目录：`exp/paper/evidence/e141-collector-cpu-20261015/`。

本地可用对照为 HISA 仓库 `dc9f3a3b7938bd26e165b0c99e3b6ea627d1ecfc` 的
`TransMLA_NeurIPS_2025/models/deepseek_v2_lite/modeling_deepseek.py`，SHA256
`040246d626f2bb94075177c9cc17999fc1641a96bdb4eaa45a9efd3cce4df0aa`。
此文件含 TransMLA 变更，不等于所要求的官方源文件。

- 复用原诊断器，在该替代源码的 RMSNorm/rotary 定义上，修后四项误差为 0；
  历史 collector 普通/-O 均失败退出 1。
- 新 `verify_e141_native_forward.py` 复用原 AST loader，执行该源码真正的
  `DeepseekV2Attention.forward`，捕获实际 Q/K、logits 和概率；不是再手算一份
  同源 collector 当 oracle。随机参数、非恒等 RMSNorm、非连续 hidden、17 个
  position，CPU fp32、普通 RoPE。Q/K/logits 最大差为 0，head-mean 概率最大差
  `5.960464477539063e-8`。
- 默认核验器只接受官方 hash；替代源码必须显式传其 `--require-sha256` 才可做
  有限范围比较。默认对替代源码普通/-O 都拒绝，退出 1。
- 分别移除 input RMSNorm、移除偶奇重排以及使用原历史 collector，实际 forward
  数值门禁普通/-O 都失败退出 1。不是以“复现缺陷”为成功的 assert 测试。
- 5 项 E141 边界测试及 8 项受影响 E140 accounting 测试，普通/-O 均通过；
  包含 partial-page 受限预算、分母、零/小于阈值但非零质量、skip/饱和记账、
  身份/输出碰撞在 CUDA/model 前拒绝。独立只读审查见 review.md。

这些验证**不覆盖固定官方源码、完整 decoder/model、YaRN、bf16、预训练权重、
GPU 或真实 LongBench**。不宣称层可跳过、NoPE/rope-only 科学结论、速度或质量。
后续先恢复指定官方源码并匹配 hash，运行两个 verifier 普通/-O；真实模型和
设备补采仍需独立授权及身份核对。此前已有数据无法从本轮小夹具重建。

复现（仅现成本地 CPU 环境，不安装依赖）：

```bash
PYTHONPATH=/tmp/tla-cpu-deps OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_e141*.py' -v
PYTHONPATH=/tmp/tla-cpu-deps OMP_NUM_THREADS=1 python exp/trace/verify_e141_qk_reconstruction.py --upstream-source /path/to/modeling_deepseek.py
PYTHONPATH=/tmp/tla-cpu-deps OMP_NUM_THREADS=1 python exp/trace/verify_e141_native_forward.py --upstream-source /path/to/official/modeling_deepseek.py
```

同样命令加 `python -O` 验证失败门不依赖 bare assert。反例和全部退出码见
`runs.json`。历史 JSON/verdict 的固定 main blob/hash 收据见
`identity-and-history.json`；原数据没有复制、修改或删除。

发布前只读发现 main 已前进至 `7aa0c1518e1fdda8f35a884326dbbaefb72cc060`，
含独立的 E142a MLA 接入提交。该增量未合入、未执行，也不属于本轮采集修复验收。
