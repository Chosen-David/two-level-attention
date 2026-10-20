# Task-ID: TL-EXPLICIT-FREE-PROTECTED-20261019

Date: 2026-10-19

## Plan

来源：用户持续审查/修复独立TwoLevel仓库授权，保护区计入总预算合同。
仅修复显式nt=off绕过保护区预算校验；保留合法配置与历史数据。
先保留修前失败，再最小构造期拒绝、普通/-O专项和相邻回归、独立复核、
发布前fetch及发布后远端SHA/文件树核验。GPU修改与性能宣称不在此有界验收内。

## Progress

- 基线836281d；修前普通/-O各18个非法off子用例未拒绝。
- 新增6行host校验；普通/-O每模式专项3项+相邻4项+batch守卫6项通过。
- 独立1080个构造场景：144预期新增拒绝、612接受状态一致、324同拒绝。
- 完整分析、输入hash、原始短日志及环境见
  [advice](../../advice/2026-10-19_explicit_free_protected_budget_by_gpt.md)。
- host契约修复已验证。GPU/实模/性能无新增结论；全依赖与真实选集、
  prefill→decode和历史非法配置筛查/重测继续待办。
