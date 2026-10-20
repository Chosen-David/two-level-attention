# TL-EXPLICIT-L1-BUDGET-20261019

## Plan

来源：用户独立TwoLevel代码审查/修复授权；不涉及Agent或SGLang。
基线 `6f006b1527c122ca942db4860a81047d027b8715`，干净独立检出
twolevel-audit-49，本地flock串行（不声称跨机器排他）；读取TASK、只读guide、
相关advice及main/实际e138-ablation分支。此前scratch检出本轮不可用，重新clone，
已发布历史从Git恢复；不假定未发布私有检查点仍存在。

假设：显式near页配额大于K1会突破L1竞争预算；只补构造守卫，不改合法选择算法。
先保存修前失败，再普通/-O回归及独立只读复核，发布前fetch，普通main发布后回读。

## Progress

证实构造器接受超额配置；24个超额子用例修前失败，修后4项host测试与相邻6项
batch守卫测试普通/-O均通过。独立复核支持5页对K1=4的对齐几何反例及修复边界。
完整输入、环境和限制见
`../../advice/2026-10-19_explicit_near_page_budget_by_gpt.md`。

## Pending

构造期正确性修复已验；生产GPU选集/输出、模型质量、性能与状态交接未验。
若发现历史near_pages>K1配置，核对后按纠正预算重测；不据此自动否定所有旧实验。
发布及远端SHA/内容回读由主控完成，不以此任务记录替代发布确认。
