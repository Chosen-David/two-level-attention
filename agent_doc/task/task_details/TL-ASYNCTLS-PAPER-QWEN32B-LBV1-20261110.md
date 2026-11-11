# TL-ASYNCTLS-PAPER-QWEN32B-LBV1-20261110

## 目标

按用户 2026-11-10 指令，先修正 AsyncTLS 的论文精度语义，再在 Qwen3-32B 上运行 LongBench v1。模型、代码、校准、输入、配置及输出身份均须固定；新结果在独立验收前保持 `pending`。

论文依据：`agent_doc/advice/other_paper_pdf/AsyncTLS.pdf` §4.1–4.2。实现范围是同步执行的选择算法复现，不包含异步 CPU offload、resident cache 或增量块传输，也不开展性能测试。

## 既有结果检索与复用判定

- Llama-3.1-8B 旧近似臂 `llama_vendor_lb_v1_20261109` 的 16/16、3750 请求及 AVG16=31.81 已独立验收，保留为旧实现历史结果，不覆盖、不重写。
- `exp/results_longbench_vendor/Qwen3-32B/pred_asynctls/` 为跨 KV group 共享选集、同步当前块集及 question 逐 token 稀疏的旧口径产物，只可参考任务协议，不能复用为本次修正后的成绩。
- `exp/vendor_baselines/asynctls_vendor/calib/qwen3-32b_qasper_channels.json` 是每层一份全局 channel 的旧格式；新语义要求每层、每 KV group 各一份 channel，必须重新校准。
- 因此本次代码路径相关臂全部重跑，新校准和新输出使用新身份，不与旧产物混池。

## 实现与验收范围

1. Stage-1 按每个 KV group 独立打分和选择 top blocks，不再跨 KV head 聚合。
2. channel 校准仅在该 KV group 的 G 个 query head 内聚合，格式为 `[layer][kv_group][channel]`；旧全局格式显式拒绝。
3. Stage-2 按每个 KV group 独立计算组内 softmax 均值和 top token，Stage-3 使用各组自己的全维 K/V 选集。
4. 完整 prompt 一次性 dense prefill，并以 prompt 最后位置初始化各层 `M_{t-1}`；decode 的 token 细筛使用上一步块集，同时保存当前 `M_t` 供下一步使用。
5. 当前 decode token 必须进入候选并至少保留一次；最终有效选集不超过 `kt`，无效 padding 不参与 softmax 或预算计数。
6. LongBench 的 AsyncTLS 路径不得再把 question 后缀逐 token 送入稀疏 forward；其他 baseline 的 prompt 协议不变。
7. Qwen 与 Llama 共享 helper 必须同步回归；覆盖 INT4 16 级、per-group 独立性、上一步块集、当前 token 去重/预算、full-budget dense 等价和稀疏形状，普通及 `python -O` 均执行。

## 实验门槛与交付

- 源码回归通过后才扫描 GPU；只用实测空闲卡，不杀其他任务，精确 PID 互斥。
- 固定新代码后生成 Qwen3-32B per-group channel，记录校准数据、模型、代码和文件 SHA256。
- 真实 32B smoke 通过后启动 LongBench v1 英文 16 任务；沿用 `kt=1024、B=64、kb=128` 和 greedy 协议。
- 点火后检查进程、显存与日志；保存输入身份、逐请求输出及 run-cap 证据。正式结果须由独立检查重算人口与官方分数后才可判决。

## Progress

- 2026-11-10：任务建立；完成论文/源码差异定位与旧结果复用判定。源码修正、校准和正式测试尚未完成。
