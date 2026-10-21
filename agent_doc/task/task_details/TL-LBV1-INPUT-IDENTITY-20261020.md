# TL-LBV1-INPUT-IDENTITY-20261020

Date: 2026-10-20

## Plan

来源：固定配额T2审查与用户要求保持输入身份/公平对照。v1指纹仅绑定
answers+length，不能认证问题/context。先源码绑定复现和独审；生产修复
须版本化身份方案、真实输入与新旧sidecar/CI兼容验收，不改写历史证据。
复用现有可选input_ids_sha256编码，不为补元数据自动开启timing或GPU作业。

## Progress

基线5276987：实际AST+官方qasper模板4反例/3对照，普通/-O输出相同。
见[中文证据与待修方案](../../advice/2026-10-20_lbv1_input_identity_by_gpt.md)。
格式化prompt变化与指纹/ID不变已确认。

基线561070b：已实现并验证
`longbench-input-identity-v2`，绑定有序完整源行、模板及逐样本实际token ID；
新身份进入样本ID、run-cap写门/校验与eval聚合元数据，旧sidecar保持legacy，
禁止从预测JSONL伪造v2 backfill。标准库普通/-O定向回归各15/15通过，详见
[实现证据](../../advice/2026-10-21_lbv1_input_identity_v2_by_gpt.md)。

任务保持未完成：当前节点无torch/GPU、真实LongBench源和模型，尚未验证生产
Qwen3 token闭包、端到端输出/性能、新旧聚合及历史污染/T2冠军影响。
