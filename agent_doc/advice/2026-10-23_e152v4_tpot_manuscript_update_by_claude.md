# E152v4 TPOT 判决翻转：手稿同步处置报告（2026-10-23）

**性质**：adopt（数据判决翻转，Q3「模型级解码」叙事从 e152v2 eager 时代改为 e152v4 graph 时代）
**执行**：Claude Opus 4.6（本地会话）
**基线**：HEAD 未推送前工作区（未提交，遵照「不 commit」约束）
**权威数据源**：`exp/trace/results/e152v4_formal_verdict.json`（SHA256 `dca25f87ab3ab48bce6b2119edda3811edcf4f7d4538edc7a16dafea97e80572`，采集 commit `20751f54`，主机 pc40 干净臂）
**验证**：15 份 `e152v4_formal_pc40/*.json` 逐份核读，测量段 TPOT 均值与判决 JSON 完全一致。

---

## 1. 改动清单（文件 × 段落）

| 文件 | 段落/块 | 改动 |
|---|---|---|
| `sources/data/decode_tpot_plot_input.json` | 全文重写 | 新 schema：`results_mean_tpot_ms` 扁平浮点 + `speedup_graph_vs_dense`；补硬件（H20-3e）、torch 2.8.0+cu128、ABBA 交错、逐臂进程数、6/4/1 重复数、bitexact 门 PASS 3/3、`backend_verified:false`；provenance 字段（source_path/source_sha256/source_commit） |
| `sources/build_evidence_figures.py` | decode_tpot 绘图块 | 三臂重画：dense（o 实线）/ tli_graph（^ 实线，逐点值+倍率标注）/ tli_eager（D 虚线，行内斜体 "no graph" 标注不占图例位）；GRAPHLBL 逐点摆位（32K/64K 右下、128K 左上）避免标签与 y 轴及相邻点碰撞；ylim(0,150)、显式 yticks(0..140,20)；TRANSLATE 增 4 条中文映射 |
| `figures/decode_tpot_current.{pdf,png,svg}` | en 图 | 重渲染（PDF 29812→32604 B） |
| `figures/decode_tpot_current_zh.{pdf,png,svg}` | zh 图 | 重渲染（PDF 19089→23476 B） |
| `source/paper_en.tex` | ①摘要末句 ②Intro P5 ③III-C 一词 ④图宽 .9→.8 col ⑤图 caption 全重写 ⑥Q3 段 ⑦结论末句 | 见下方新旧对照；③"appended"→"joins"、④⑤为 6 页回收排版手段 |
| `source/paper_zh.tex` | ①摘要末句 ②引言 P5 ③图 caption ④Q3 段 ⑤结论末句 | 与 en 平行 |
| `source/supplement_en.tex` | S6 Timing provenance 段 | 重写为 E152v4 pc40 细节：6/4/1 重复、bitexact 门、共居污染复刻臂（26→39ms）过同门但不入均值、E152v2 降格为被取代记录（"both records agree that no-graph sparse decoding is slower than dense"） |
| `source/supplement_zh.tex` | S6 计时来源段 | 与 en 平行 |
| `MANIFEST.json` | 22 条 sha256 | tex×4、data JSON、figure script、decode_tpot 双语图×3 格式、output PDF+aux/log/out |
| `sources/evidence_bindings.json` | +32/−5 行 | bundled_files 两条哈希更新；新增 2 条 repository_source_identities（e152v4_formal_verdict.json 含 sha256/git_blob、e152v4_formal_pc40/ 目录）；aggregation.E152 重写；新增 `e152v4_tpot_revision_20261023` 修订键（baseline_commit/source/scope/limits） |

## 2. 新旧数字对照表

| 项 | 旧（e152v2，动态缓存，中位数） | 新（E152v4 pc40，均值） |
|---|---|---|
| 稀疏 32K/64K/128K TPOT | 87.9 / 97.6 / 119.9 ms | **24.7 / 27.4 / 32.4 ms**（整步 CUDA Graph 重放） |
| 稠密 32K/64K/128K TPOT | 26.0 / 39.7 / 67.3 ms | **26.3 / 39.7 / 67.0 ms**（原生 eager 稠密路径） |
| 加速比 | 无（每长度均更慢） | **1.06 / 1.45 / 2.07×**，随长度单调，32K 近打平 |
| 无 graph 稀疏臂 | 即主线数字 | 99.0 / 99.9 / 134.1 ms（披露性虚线臂） |
| 判决措辞 | "slower at every measured length, no decode-time benefit" | "monotone in length, near parity at 32K; the gain tracks replay, not selection" |
| 诚实边界 | 动态缓存、非紧凑配置 | 增益来自重放非选择；不含准备开销的完整请求时延；bitexact 门 3/3 过 |

补充件保留 e152v2 三数（87.9/97.6/119.9）作为被取代记录的披露，两记录一致结论＝无 graph 稀疏解码慢于稠密——这是有意保留，非残留。

## 3. 页数验证

| 文档 | 页数 |
|---|---|
| paper_en.pdf | 6（零余量达标） |
| paper_zh.pdf | 6 |
| supplement_en.pdf | 11 |
| supplement_zh.pdf | 10 |

6 页回收手法（en 触发一次 7 页级联：摘要 +4 词 → 式(1) 推至第 3 页 → 8 行空档 → E1 图推至第 5 页 → 结论溢出）：摘要压缩（"no-graph decoding stays slower" 从句删除，披露保留于 Intro/Q3/结论/补充件）、图宽 .9→.8 columnwidth、caption 精简、III-C 一词缩减。经 HEAD 干净基线二分定位，确认级联仅由摘要句触发。

## 4. 机械门输出

- 旧数字残留：`grep '87.9\|97.6\|119.9' paper_en.tex paper_zh.tex` → **0 命中**（补充件有意保留披露）
- 新数字落位：Q3 段 24.7/27.4/32.4 与 26.3/39.7/67.0、1.06/1.45/2.07× 全部命中，与判决 JSON 逐位一致
- em-dash：四 tex 改动段 **0 命中**（paper_zh L31 的中文破折号属既有 Intro P2 句，非本轮改动）
- 对冲词（might/perhaps/arguably/possibly/seems/appears to）：改动段 **0 命中**
- 长句（>40 词）：Q3 新段三句分别约 39/14/17 词，达标
- 图几何碰撞门（matplotlib renderer 两两 bbox + 线段-矩形相交）：**0 碰撞**（修掉 2 处：graph@32K 标签 vs y 轴 tick "20"；graph@32K vs graph@64K 同高重叠）
- 编译：`compile_local.sh` 三遍 xelatex，四文档 0 未定义引用

## 5. 剩余风险与权衡决策

1. **eager 臂单次重复**：每长度 1 次（6/4/1 中的 "1"），数字质量最弱的一臂。处置：caption 明写 "means of 6/4/1 runs"，正文只引区间 "99--134 ms" 不逐点引用，图内行内斜体标注、不占图例位。其功能是系统杠杆诚实披露，非结论支柱。
2. **稠密臂无 CUDA Graph**：系统杠杆不对称（稀疏有 graph、稠密原生 eager flash）。这是稠密解码的常规形态（动态缓存 + 原生路径），但审稿人可问「稠密加 graph 会怎样」。已在 S6 披露臂配置；不在本轮扩测范围。
3. **摘要压缩取舍**：6 页约束下摘要删掉了 "no-graph decoding stays slower" 从句。披露完整性靠 Intro P5、Q3、结论、S6 四处冗余承接，摘要只保留 "graph replay undercuts dense TPOT beyond 32K" 的正向判断 + 完整请求成本未测边界。若用户偏好摘要保留双向披露，需另找 ~6 词空间（候选：结论段 "isolating neither quality contributions nor the compact design" 压缩）。
4. **图宽 .8 vs .9**：caption 变长（旧 3 行 → 新 4 行）+ 图内三臂信息增加，.9 columnwidth 下 6 页溢出，回收至 .8。视觉可接受（图 aspect 比例不变，仅缩放）；若用户嫌小可回 .9 并从 caption 再删 "graph replay reproduces eager tokens bitwise"（不推荐，bitexact 门是本轮最强质量锚）。
5. **`backend_verified:false`**：后端路径为作者报告未独立核验，JSON 与 S6 均已披露。
6. **未推送**：全部改动停留在工作区（遵照不 commit 约束），推送与远端合并待用户指令。

## 6. 与 GUIDE/TG 的衔接

- 本轮即记忆中「Q3『稀疏解码更慢』手稿叙事已过时待改」的兑现：e152v2→E152v4 翻转完成，en/zh 主文+补充件四文档同步。
- 双口径铁律遵守：本轮仅模型级 TPOT（同步口径）；kernel 级 microbench（选择器局部成本段）未动，其 S6 既有披露仍成立。
- 投稿后第一优先实验清单不变：完整请求 E2E 计时含准备/维护/选择并给占比（本轮多处 "preparation-inclusive complete-request latency" 边界句即为其铺垫）。
