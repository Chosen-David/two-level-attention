# E140 collector CPU 核验与短 advice（2026-10-15）

结论：v2 collector 已通过真实 Hugging Face Qwen3 随机小模型的 CPU fp32
对拍，可作为后续受控补采的代码基础；不能据此验收旧 observation 图、预训练
Qwen3-8B、bf16/GPU 或论文科学结论。本轮只发布原开发分支，不合 main。

## 修复与验证

- `get_qk` 拒绝训练模式、非单序列/空/非整数输入、重复或越界层号、异常
  Q/K/RoPE schema、非有限值和不合法 GQA 几何。使用实际 q_norm/k_norm 输出
  与 attention 收到的 RoPE，冻结采集副本。hook 分步登记，部分注册失败或
  forward 失败也清理自己创建的 hook，保留调用方 hook。
- 概率入口拒绝无效尺度和不匹配 dtype/device；O3 拒绝空、重复、非法子空间
  坐标及偏离冻结全维尺度的输入。没有改变生产 near-average/far-minmax 规则，
  没有恢复动态 alpha/beta/gamma 或 D′。
- `tests/test_e140_hf_cpu.py` 直接构造随机模型，无 from_pretrained、下载或 CUDA。
  torch `2.14.1+cpu`、transformers `4.57.3`，seed 140；2 层、4 Q heads/2 KV heads、
  head_dim 128、641 tokens、非默认 theta 12345、非均匀 input_layernorm 权重。
  collector 接收故意错误的 theta 参数，以确认它真正使用 forward 提供的 RoPE。
- Oracle 包装 HF 原生 `apply_rotary_pos_emb` 和 `eager_attention_forward`，逐层
  比较全量 Q/K，以及 8 个 query 位置的因果概率。Q/K 最大绝对误差 **0**，
  组概率最大绝对误差 **1.1920928955078125e-7**。历史漏 input_layernorm 的
  重建在 position 0 误差 **1.6065808534622192**，会触发数值失败。
- 13 项受影响回归在普通和 `python -O` 下均通过；额外进程内把 collector
  rotate_half 替换成 zeros_like，两个模式均明确数值失败并 exit 1。
  身份声明缺失、已有输出、错误输入与 hook 异常路径都有负测试。

证据：`exp/paper/evidence/observation-cpu-20261015/` 内含两种模式日志、
机器可读身份回执（config、随机权重、输入、collector/HF 源码哈希）、注入失败
日志和历史文件哈希核验。独立只读审查未发现阻断项；见该目录 review.md。

复现环境已有本地依赖，禁止为此下载权重：

```bash
PYTHONPATH=/tmp/tla-hf-cpu:/tmp/tla-cpu-deps:$PWD OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover -s tests -p 'test_e140*.py' -v
PYTHONPATH=/tmp/tla-hf-cpu:/tmp/tla-cpu-deps:$PWD OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -O -m unittest discover -s tests -p 'test_e140*.py' -v
```

## 仍需保留的边界与下一步

1. 随机小模型身份回执不等于真实模型/tokenizer/dataset revision 验证；实际
   collector 的 `diagnostic-unverified-identity` 状态不升级。未来补采前必须
   独立核对本地权重、tokenizer、数据与请求的固定版本，而非只相信环境变量声明。
2. 此 O3 fixture 的 mid=385 小于 K2=768，只验证全选、部分页和分母，未证明
   预算受限排序；O3 仍是 uniform-page-average 诊断，不是生产 selector 对拍。
3. 未获得旧实验对应 raw Q/K 或 per-token 概率，旧 JSON 与 O_observation 图
   保持原字节/哈希，不能用改标签或插值修复。后续真实补采需新输出名与 raw Q/K
   回执，以文档/序列为独立单位，不把 query×head 当独立样本。
4. 未测 bf16、GPU/Graph、峰值内存或性能。投影快照 clone 的资源成本也未测，
   不作性能提升声明。任何真实 GPU 补采须另获授权；本轮无 GPU、付费、Claude、
   SGLang 或已取消 MLA 工作。shell/仓库可用不代表外部执行服务已修复。

发布基线 `a03937277d0e12103ac3c2ed04921ee742437f2f`；核查时远端 main 为
`87029997b9edf63cabd2fe53a05bb10eb9623be4`，未合入其新增内容。唯一写入者为
本任务，独审仅只读；只对 `gpt_dev/paper-aligned-runtime-20261015` 非 force 推送。
