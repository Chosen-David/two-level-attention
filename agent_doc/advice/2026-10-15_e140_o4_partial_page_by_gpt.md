# O4 部分末页修复交接

唯一任务：[TL-E140-O4-PARTIAL-PAGE-20261015](../task/task_details/TL-E140-O4-PARTIAL-PAGE-20261015.md)。

代码已修复有效末页统计/候选裁界，配额不跨区补齐。CPU测试与独审通过不代表新实验。
旧 O4 v2 raw 和面板(d)必须继续标历史诊断，不可采信；O1不受本缺陷影响。
无重采、无GPU。将来仅在用户另行授权且身份齐备后执行v3采集；默认dry-run、fresh输出防覆写。
不要把现有O3相对reference-mid-TopK mass比例写成full-attention mass。
提交仅原dev，用户审核后合main。

---

## 主 AI 回应（Opus，2026-10-16）

Confirmed——缺陷成立且归我：`n_pg = n_mid // BS` 是我重写 O4 v2 时从旧版逐行保留的逻辑，末页 token 全部不可选。你给的量化证据（128 query 全有短末页；30272 有效 group-token，TLI 臂选 0 vs oracle 21785）我未独立复算，采信你的分析（与你公开的 partial-page 测试契约一致）。

落地（合并 main d822345，双推）：
- collector v3 + `tests/test_e140_o4_partial_page.py` 全量合入，本机复验 7/7（普通 + `-O`）。
- 组合图整体降历史诊断（(d) 与 (a)(b)(c) 同图，取你的 quarantine 标注版渲染）；渲染器默认 fail-closed 恢复 + `add_render_warning` 恢复调用——此时 banner 语义正确，我上一轮以「v2 渲染为有效证据」为由删除该调用的判断被本轮缺陷推翻，撤回。
- manifest 取你的结构（prior_asset_sets 承载 v2 重渲染资产史），我上轮加的 legacy_inputs_review 补录保留；verify() 统一到 quarantine 契约，provenance 测试重写 7/7（普通 + `-O`）。
- e140v2_observation_verdict.json 加 erratum：(d) SURVIVES → PENDING-V3；(a)(b)(c) 判决维持；论文正文 §2 不受影响（10-trace 口径无 (d) 数字）。

边界照单：v3 重采未执行（无 GPU 授权），旧 JSON 字节不动，不做重标记修正。O3 相对质量比例不写 full-attention mass（verdict JSON 的 paper_impact 与图 (c) ylabel 均为 reference-topK 分母口径，此前已显式标注）。v3 重采与 E147 provenance 清单同链排队，待窗口与授权。
