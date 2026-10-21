# LBv1 输入身份 v2：源行、模板与实际 token 闭环

问题ID：`TL-LBV1-INPUT-IDENTITY-20261020`。实现基线：
`561070b751cac59879664e096b9011ea83303d38`。本文件是
[2026-10-20 修前证据](2026-10-20_lbv1_input_identity_by_gpt.md)的后续检查点；
旧证据不改写。

## 修前失败与根因

修前 `pred.py` 只对有序的 `answers + length` 投影取 SHA-256 前12位，
`context`、问题、其它源字段及模板均不进入样本ID或run-cap身份。固定输入
SHA `b6e7faaca3b44c4a1e923cb28deb9819b35dfb67c413fc7c6f488c7984ff0ede`
的4个反例（context、question、同投影行交换、Unicode正文）都产生不同prompt，
却保持指纹 `8a44386d7d5a` 和相同逐行ID。原始观察文件 SHA-256 为
`0039612a6d3d9162e8eb97b9e9937c432edaf55739324ffe074dd47eaff9c73e`。

## 实现

- `benchmark/LongBench/run_cap_manifest.py:123-184` 定义
  `longbench-input-identity-v2`：每个有序完整源行采用键排序、紧凑UTF-8 JSON，
  以8字节小端长度分帧后累计 SHA-256；模板单独记录 SHA-256。完整
  `v2-<64hex>` 源指纹进入样本ID、run-cap payload及写门身份。
- `benchmark/LongBench/pred.py:319-350` 在模板、截断、question split及BOS处理
  后，对实际送模的 prefix+question token ID 按 signed-int64 little-endian
  逐样本哈希，再搬到CUDA。这样同时移除了原先GPU tensor上的BOS `.item()`
  同步；未实测速度，不能据此声称性能提升。
- `benchmark/LongBench/run_cap_manifest.py:401-452,533-583` 将逐样本token hash、
  有序聚合hash与编码版本写入sidecar，并校验source/template/token数量及闭包；
  任一单侧篡改 fail-closed。
- `benchmark/LongBench/eval.py:126-153,442-486` 验证后把v2身份逐文件写入
  `result.json._meta.runcap.input_identity`。旧run-cap继续按原schema读取，但
  显式计为`legacy_strict`，不冒充已有完整输入证明。
- `exp/trace/backfill_run_cap_manifest.py` 不从预测JSONL伪造v2源正文或token证据；
  新回归固定 `dataset:v2-<64hex>:row` 在缺原生sidecar时拒绝补制。

## 验证

环境：CPython 3.12.14，Linux 6.18.55 x86_64；当前节点无`torch`、无
`nvidia-smi`。执行：

```text
python -m unittest tests.test_lbv1_input_identity_v2 tests.test_backfill_identity_format -v
python -O -m unittest tests.test_lbv1_input_identity_v2 tests.test_backfill_identity_format -v
```

两种模式均为15/15通过，覆盖生产入口接线、context/question/Unicode/任意元数据/
顺序、模板、token编码、source/template/token篡改、legacy不重解释及v2禁止backfill。
普通与`-O`的`compileall`及`git diff --check`通过。相邻
`benchmark/LongBench/test_e116a_gate.py`在导入评分器时因环境缺`jieba`而阻断，
未计为通过。

## 未验证边界与历史影响

当前节点没有实际LongBench qasper源文件、Qwen3 tokenizer/model或GPU，因此没有
生产token序列、模型输出质量、完整TTFT、显存、kernel数或速度数据。本改动是
身份正确性修复，不作性能收益主张。历史JSONL缺源正文、模板快照及实际token，
只能维持legacy/unknown；本提交不回填伪证据，也不能据此判断既有87格是否污染、
冠军是否改变。任务保持未完成，待有授权生产资源时做真实同输入跨臂闭包、模板/
tokenizer变化拒混、新旧sidecar聚合及历史影响抽查。
