> Historical design from PR #5. Preserved for review, not current implementation status.
> See agent_doc/results/pr-unification-20261027/validation.md for current disposition.

# AsyncTLS 对标建议：先闭合实现身份，再比较 resident 三方

日期：2026-10-23。来源：用户本轮要求详细测试设计、PR和advice。状态：待审设计；零GPU执行。

[完整分阶段方案](../../research/docs/ASYNCTLS_COMPARISON_PLAN_20261023.md)
接续 S-T023 和旧 AsyncTLS 阅读方案；不更改旧结果、当前模型预算或生产代码。

本轮固定 main a7bcdddf727aa99196d925b9f106d785ce9273d4。
关键新增缺口：现有RULER/LongBench分支引用的 asynctls_vendor adapter及校准脚本不在提交树；
tls_attn实现存在但未绑定作者官方来源/许可/当前baseline入口。先恢复可核字节与许可，不能把CLI选项当可运行复现。

建议采纳顺序：
1. A0身份/可运行性；A1真实张量语义；A2固定选集attention和缓存；A3各方法eager/Graph同法等价。
2. R主轨仅GPU resident、相同batch和真实输入，FullKV/LoDA-SA/AsyncTLS-style；P论文机制与N同保护预算适配分开。
3. A4完整请求成本，A5 smoke，A6逐任务配对确认，A7同配置联合判断。现有32/64/128K只复用匹配证据，不自动重跑。
4. Offload容量吞吐及更大batch单列候选；不把bs6/bs1当同批延迟，也不把上一时刻候选当输出必然不变。

E152v4当前归档支持混合执行口径的TPOT比值，不支持graph-vs-graph或完整请求收益。
404feb17运行身份与a7bcdddf审查源码须分别记录，宿主未记录行为保持未知。
本方案的统计容差是待冻结提案，不是已有非劣判决；质量smoke/mass/CPU测试不能替代确认集。

设计PR：[Draft PR #5](https://github.com/Chosen-David/two-level-attention/pull/5)。此前PR #4通用Graph设计继续有效，本PR只补AsyncTLS比较契约。
本建议不启动作业、不改变权限、不请求Claude、不修改人类指南；prepare_mask门禁候选另行保全。

