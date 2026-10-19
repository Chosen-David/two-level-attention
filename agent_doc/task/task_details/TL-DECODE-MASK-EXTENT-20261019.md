# TL-DECODE-MASK-EXTENT-20261019

范围：独立two-level-attention；用户恢复实现/测试/文档及main直推授权。本轮修复decode消费端不完整mask广播，保留prefill、selector、kernel及既有预算语义。人类guide只读。

任务链：复现 → host覆盖检查 → 正常/-O专项与相邻CPU回归 → 独立复核 → fresh fetch与普通main发布/回读。生产GPU不在本机可验证资源中，不使用历史卡数推定当前授权空闲设备，不新增资源。

基线5549339870004cb9ffb5dba6f20a88c89ac48547；e138-ablation只读核验2ed6a1feb78a7b5bcf3cac922b4510c3968714a5。本机旧TwoLevel检出经工作区清理已不存在，重新克隆；保留无关未跟踪字体临时文件，不暂存。

## Progress

- 修前4测试方法出现15失败、9错误，包含真实torch.where单槽广播错误输出。
- 增加6行metadata门；普通/-O各19 eager与6 HF padding测试通过。合法块尾padding继续支持。
- 独立复核PASS：48合法随机逐头float64参考对照、12非法extent通过；8例spy确认实际CPU gather，最大误差2.384185791015625e-07。报告及脚本见 `../../results/decode-mask-extent-20261019/`。
- 发布前fresh fetch确认main仍为基线5549339870004cb9ffb5dba6f20a88c89ac48547；仅提交本任务实现、测试和证据，发布结果以远端包含本记录的提交为准，不预写成功。详见 `../../advice/2026-10-19_decode_mask_extent_fix.md`。
- 未完成：真实selector/GPU/模型质量与性能验证，增量缓存和prefill热点优化；不因本次有界修复关闭持续任务。
