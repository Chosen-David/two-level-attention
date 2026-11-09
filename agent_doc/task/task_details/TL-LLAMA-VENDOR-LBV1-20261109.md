# TL-LLAMA-VENDOR-LBV1-20261109

## 目标与范围

按用户指令补齐 Llama-3.1-8B base 在 LongBench v1 英文 16 任务上的 Quest、JSSA、AsyncTLS 精度数据，复用已验收 FullKV/NFA，不重复生成锚点。每臂 3750 请求；13 个任务各 200 行，`multifieldqa_en=150`，`lcc/repobench-p=500`。统一 `cap=31500`、`TS=11052030`、`pred_postfix=_official` 和 raw LongBench prompt，不使用 Instruct chat template。

三种 vendor 均保留原生 decode 选择规则，nominal token budget=1024；NFA 的 K2=1024 含 sink4+SWA128，不把该强制区强加给 vendor。Quest 使用官方 forward；JSSA 使用 rank32 校准子空间；AsyncTLS 属于论文语义复现而非官方源码。

## 最终结果

状态：**complete / usable-with-scope**。逐任务五臂分数、预算口径、代码与校准身份见 [summary.json](../../results/llama_vendor_lb_v1_20261109/summary.json)。

| 方法 | 完成任务 | 请求数 | AVG16 |
|---|---:|---:|---:|
| FullKV（复用） | 16/16 | 3750 | 38.43 |
| NFA（复用） | 16/16 | 3750 | 38.49 |
| Quest | 16/16 | 3750 | 31.02 |
| JSSA | 16/16 | 3750 | 33.67 |
| AsyncTLS（INT4 修复版） | 16/16 | 3750 | 31.81 |

这些是同输入、同人口的描述性分数；本轮未做配对显著性检验，不写“等价/持平”结论。

## 关键实施与修复记录

- 基线 harness：`d258060b792137a282bd5695d92fabe0f0ee95d1`；Llama vendor adapters 与 AsyncTLS INT4 修复已合入 `088cecad`。
- 语义审查发现 AsyncTLS INT4 缺少 signed offset，旧映射把 16 级输入的 8–15 饱和到 7。旧 queue/worker 按精确 PID 停止，正式预测写出数为 0，产物隔离到 `invalid_pre_int4fix_20261109/`。
- 修复为 `q=round((x-mn)/scale+lo)`、`x_hat=(q-lo)*scale+mn`；16 级与常量回归在本地/pc02 普通及 `python -O` 均为 5/5。生产 patch SHA256=`2807dcd3ef92f7e88d640b49d8e1a6601eaa1136a3abd9d417d12e0d189e7a94`，与当前 main 一致。
- 修复版真实 8B qasper smoke：2/2、全部非空、runcap complete；预测 SHA256=`ab0b2b6efa7a66fc2a03bb877e39637434def08514300968a3b34fa7955a5f74`。smoke 不抵扣正式人口。
- JSSA basis 为 LongAlign 8 条目标 131072-token 序列、rank32；SHA256=`5aacbb5483c5658480e565f315e25127f1c82d09147d39041b20a07d9d001082`。AsyncTLS channel 为 qasper 前 8 条、每条最多 4096 token、dc32；SHA256=`eeb0c16ff916ff7569a54124f5ac1d31d5ff791094130198a4a5e0a4a2bdb443`。
- pc02 独卡互斥队列：Quest/GPU2、JSSA/GPU0、AsyncTLS/GPU1+GPU3 两片。500 行任务置于链头且分卡；最终队列回执均为 `complete-pending-independent-validation`，四卡随后自然释放，无停车动作。

## 独立验收

2026-11-10 重新从原始 JSONL/runcap 逐行验收，而非采用 runner 汇总：

1. 三 vendor 每任务唯一 complete best-file，人口、`_id` 顺序/唯一性、answers SHA、预测 SHA 与 sidecar 全部对齐。
2. 每臂 strict runcap=16、input identity v2=16、legacy=0；完整 3750 请求。
3. 官方 `sample_score` 独立重算，逐任务分数及 AVG16 与官方 `result.json` 完全一致。
4. FullKV/NFA 原始预测同步重算，分别保持 38.43/38.49。
5. 五臂在全部 16 任务上的 `source_fingerprint`、`prompt_template_sha256`、`actual_input_ids_ordered_sha256` 一致。

独立报告：`/home/wangyuanshuo02/ak_verify_results/llama_vendor_lb_v1_20261109/validation/all_vendor_full16_independent.json`

- 报告 SHA256：`c157c74cc6ef6e79a71ae40b4ff0106870d43d606a3920d79393c4dc33ee08aa`
- source identity SHA256：`e9efb94c23c617e5fa9db7c92c9afb875c16c983b9c2f71f5e8c38f9b36a581e`
- 官方 result SHA256：Quest `021d15d084024171054da3f4d176c014c96e681ef363907deb17e00af654f667`；JSSA `a378aa2fb249589b11c8faefbe31f01e19e5f9b69935212f67912fb27e5ded78`；AsyncTLS `27d89f7ce26097e793a4a98f22a19b30255a00673456db02a164ba3e5bbdfad3`

原始预测、sidecar、runner receipt、官方评分与独立报告保存在 `/home/wangyuanshuo02/ak_verify_results/llama_vendor_lb_v1_20261109/`；git 只收录精简汇总和本记录，不收模型、校准张量或原始大预测包。
