# TL-E140-ASSET-PROVENANCE-20261015

## Plan

用户明确授权本轮窄修 E140 当前 SVG/PNG/PDF 的可见历史诊断警告与资产身份清单；这覆盖所列 guide 图资产、renderer 和 validity 说明，不授权更改 GUIDE、论文结论、默认参数或数据。沿既有 dev 非 force 交付，main 只读；无 GPU、模型下载、付费、Claude、SGLang 或 MLA/动态研究。

## Progress

- fresh main `44a98682538209e3fcd44cd12b703012f277b58b` 整合到 dev，基线 `037f269f3c5c0095b383b2a4489b84a70e7b7291`。他人的 SparQ/E148 更新原样保留，未新增验证其科学结论。
- 原 v1 manifest 的五条 `artifacts` 身份全部保留，逐项核对其字节可从 `096414368d2f7c102cb526a62c222e181b97620c` 恢复；后续未标注图另以基线 commit/blob/SHA256 冻结。新图使用独立 `current_assets` 身份，状态始终为 `historical-diagnostic-not-corrected`。
- 三格式只在原图边界外添加历史诊断警告；PNG 原图像素不变，SVG 原绘图子树不变，PDF 原页保留为矢量嵌入。既有 JSON/TSV 未改，旧图未被称作修正数据。默认 renderer 仍先退出且不写文件；显式历史 opt-in 后续导出也带警告。
- 新 CPU 正负测试覆盖旧/新身份、可见文本及原图保存、篡改/有效性升级拒绝、重复标注拒绝、默认 renderer 拒绝且无写入、共享 opt-in 警告导出。既有 collector 的 8 项回归普通及 `-O` 通过；资产 6 项两模式通过。独审与完整日志随交接证据提供。
- 未验：完整历史 renderer 因缺少其引用的 `e140_o4_select_trace.json` 及宿主 style/lint 依赖未执行；manifest 明确登记缺失输入，测试只执行共享警告导出助手及默认拒绝路径。首次 apply 在输入预检时退出且未写图，随后改为显式记录缺失；未补造该 JSON。GPU、模型、修正后科学证据与正式论文有效性仍未验收。
