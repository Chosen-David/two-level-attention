# TL-PREFILL-MASK-CONTRACT-20261019

基线3b0cfea1dac5598b44bfa28818b020c5a7e61227；e138-ablation只读2ed6a1feb78a7b5bcf3cac922b4510c3968714a5。本轮独立clone，串行flock持有，旧检出无关字体文件保持原样。未发现适用AGENTS.md/根TASK.md；已读人类GUIDE（只读）、任务索引、prefill及decode相关advice。当前用户授权覆盖本独立库及验证后main直推，不涉及旧E127隔离分支或其他仓库。

## 冻结计划

问题：reference与candidate逐组prepare_mask消费点未检查mask协议，单槽宽度可在reference广播成全前缀；错误batch/query/head维度可能被截取或广播；返回block_size被忽略。现有prefill实现文档要求token级bool选择，真实TLI返回block_size=1。块级selector不能默认为token级；本轮明确拒绝，不新扩展Quest的prefill算法。

修复方案：共享轻量metadata helper核验Tensor/strided/bool/device、[1,1,Hkv或H,T_pad>=prefix]、正整数block_size=1，然后返回原切片view。无归约、.item、复制或新kernel。保留合法padding、非连续mask、GQA共享/per-query-head选择；不改变预算、组末Q语义、chunk2048、sink/SWA或_inc缓存。仅两个逐次selector消费点接入；内部batched_group_sels路径不改，后续GPU验证单独记录。错误发生在selector调用后，不承诺回滚其内部状态；已有oracle旗标finally应继续清理。

候选取舍：自动repeat块mask会新增此前未声明的通用block-prefill执行/保护语义，超出本次安全门范围，不采用；仅extent检查仍会把错误头数等广播成合法输出，不完整。门禁仅metadata开销O(1)，零设备同步；性能未测不宣称收益。

任务链：最小反例及修前失败 → helper及两处接入 → 普通/-O新测试+ops相邻回归 → 独立复核 → fresh fetch/仅明确文件普通发布/CAS回读。CPU PyTorch2.8.0+cpu、Linux6.18.55，CUDA不可用；不调未知远端GPU或新增资源。不重复已修decode extent问题，这是另两个prefill消费入口。预算本轮最多一项修复，CPU测试各120s，无性能搜索。

测试覆盖：两runtime+candidate分组、空/短/非整块、非法形状/dtype/layout/device/extent/block、合法block尾padding及stride、共享/per-query-head/GQA，确定性独立绝对位置输出参考，真实TLI sinkswa路径和oracle异常清理。旧ProtectionIndexer fixture若返回full-token mask却标page_size，修正其声明为1，保留page配置参数用于原sink回归；不得放宽原数值断言。

## Progress

- 已完成红绿：修前实际singleton错误未拒绝+51个非法metadata进入mock后错误；修后54非法子例、24合法子例及实际singleton反例通过。
- ops普通/-O各84方法通过；E151c真实CPU TLI路由和E139相邻契约普通/-O各23方法通过。独立15项metadata/view、54随机logit、2条batch回退/尾组检查通过，结果review无阻塞。
- 中文简报：`agent_doc/advice/2026-10-19_prefill_mask_contract_fix.md`；证据：`agent_doc/results/prefill-mask-contract-20261019/`。未将原始64KB traceback入库，保留hash及短日志。
- 发布前再次fetch并CAS/普通快进；远端提交SHA与tree回读由主控发布检查点记录，不在提交前预称成功。
- 有界修复CPU验收；GPU/真实模型/性能、内部batch输出契约及候选选量容量策略仍未验证。真实TLI CPU路由测试不能替代GPU。持续任务不停止，下一步优先当前未完成的候选内L2/紧凑索引及合法资源下GPU验证。
