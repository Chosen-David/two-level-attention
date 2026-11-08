# Qwen3-8B 独立 Near/Far 0.05 全量最终中文报告

报告时间：2026-10-30T00:32:03.271407+08:00。PROJECT_ROOT=/home/wangyuanshuo02/two-level-attention。

全量1300配对、3900生成、64分片已完成。质量点估计确有下降：Far-only AVG5下降0.082408个百分点，独立Near+Far下降0.085406个百分点；Qasper两臂均下降0.683604个百分点，配对bootstrap 95%区间均低于0。AVG5区间含0，只能说总体差异尚不确定，不能称等价或绝对无损。没有事前等价界值。

本报告消费两个独立Codex会话的全量核验，验收范围是固定预测、输入、评分、统计、保存的门控/缓存/执行回执；未独立重跑GPU生成。正式独立报告见文末，完成标记仅在报告与最终字节绑定收尾后写入。

## 固定协议与覆盖

Qwen3-8B；TLI gates-off（Full）、Far-only 0.05（Far）、独立Near+Far 0.05（Dual）。prefill末min(256,L)个query，head mean后query mean，各分区严格<0.05，decode冻结。算子优化关闭，上下文cap=31500。Full不是dense attention。源码固定3e4a6885604ac3e719ee92a4906f87b532535314。旧100样本screen未并入。

Qasper/HotpotQA/Musique/GovReport各200，RepoBench-p500；唯一行全覆盖，无partial、重复或失败分片。QA为F1，GovReport为ROUGE-L，RepoBench-p沿既定difflib代码相似度；下表分数统一乘100，差值单位为百分点。AVG5是五任务等权平均，不是1300样本混合平均。

## 质量结果

| 任务 | n | Full | Far | Dual | Far−Full | Dual−Full | Dual−Far |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qasper | 200 | 48.486065 | 47.802461 | 47.802461 | -0.683604 | -0.683604 | +0.000000 |
| HotpotQA | 200 | 59.164875 | 59.414875 | 59.414875 | +0.250000 | +0.250000 | +0.000000 |
| Musique | 200 | 35.961537 | 35.962687 | 35.962687 | +0.001149 | +0.001149 | +0.000000 |
| GovReport | 200 | 33.626052 | 33.640465 | 33.625477 | +0.014413 | -0.000575 | -0.014988 |
| RepoBench-p | 500 | 63.960000 | 63.966000 | 63.966000 | +0.006000 | +0.006000 | +0.000000 |
| **AVG5** | 1300 | 48.239706 | 48.157297 | 48.154300 | -0.082408 | -0.085406 | -0.002998 |

原协议配对bootstrap：5000次、NumPy default_rng(42)，任务顺序及随机数消耗序沿冻结分析脚本；AVG5按任务分层重采样。以下保留探索性区间，未做多重比较校正。

| 任务 | Far−Full 95% CI | Dual−Full 95% CI | Far下降/提升/相同 | Dual下降/提升/相同 |
|---|---|---|---|---|
| Qasper | [-1.400376, -0.152087] | [-1.387867, -0.146292] | 11/2/187 | 11/2/187 |
| HotpotQA | [+0.000000, +0.750000] | [+0.000000, +0.750000] | 0/1/199 | 0/1/199 |
| Musique | [+0.000000, +0.003448] | [+0.000000, +0.003448] | 0/1/199 | 0/1/199 |
| GovReport | [-0.323394, +0.357139] | [-0.331371, +0.343745] | 82/66/52 | 83/66/51 |
| RepoBench-p | [-0.514050, +0.552000] | [-0.528000, +0.538000] | 13/14/473 | 13/14/473 |
| AVG5 | [-0.284729, +0.118843] | [-0.285422, +0.125730] | — | — |

Dual−Far宏平均差为−0.002998pp，95% CI [-0.008993, +0.000000]；差异全部来自GovReport98。该样本Full/Far为30.390738、Dual为27.393225，下降2.997513pp，使GovReport任务均值下降0.014988pp。不能把Near低mass当作质量无影响的保证。

## 多跳 Musique 单独分析

200/200完整配对，Far和Dual均为35.962687，Full35.961537；差+0.001149pp，95% CI [0,0.003448]。0条评分下降、1条提升、199条相同；194条输出token完全相同，6条输出发生变化。Far与Dual全部200条评分一致。

唯一评分提升是0-based row82，F1由0.0666667到0.0689655；标准答案是“ease of use and enhanced support for Plug and Play”，生成仍在回答财富/亿万富翁，不能解释成多跳推理答对或能力提升。其余输出变化row63、74、133、151、154评分不变；评分不变也不保证语义等价。Musique56层11触发Both，三臂输出完全相同且评分0。因此本轮未观察到Musique评分下降，但不证明一般多跳任务无损。

## 门控是否实际发生

所有分母为实际检查过的层—请求数；36层×1300=46800/臂。Near-only与Both确实触发，但覆盖稀少。Dual Near-only=33/46800（0.070513%），涉及19/1300请求（1.461538%）；Both=2/46800（0.004274%），涉及2/1300请求（0.153846%）。含Both的Near激活共21/1300；Far激活1040/1300（80%）。

| 臂 | Keep层数 | Near-only层数 | Far-only层数 | Both层数 |
|---|---:|---:|---:|---:|
| Full | 46800 | 0 | 0 | 0 |
| Far | 41011 | 0 | 5789 | 0 |
| Dual | 40978 | 33 | 5787 | 2 |

以下每格为“层数/涉及请求数”；每任务完整的三臂Keep/Far等表见独立路径报告。

| 任务 | 层分母/请求分母 | Dual Near-only | Dual Both | Dual Far-only |
|---|---:|---:|---:|---:|
| Qasper | 7200/200 | 1/1 | 0/0 | 901/113 |
| HotpotQA | 7200/200 | 17/3 | 1/1 | 581/189 |
| Musique | 7200/200 | 6/6 | 1/1 | 433/194 |
| GovReport | 7200/200 | 1/1 | 0/0 | 1041/149 |
| RepoBench-p | 18000/500 | 8/8 | 0/0 | 2831/395 |

Both为HotpotQA135层7和Musique56层11，分别保存评分1和0，三臂输出均相同。Dual与Far的46800/46800个Far决策一致，Near增加没有改写Far判定。

## 缓存一致性、实际路径与限制

独立重建1300输入token、核3900输出解码、140400条末步TLI回执和421200个层缓存快照。72协议源码文件及额外分析/评分依赖匹配冻结源。保存的mean、严格阈值决策、节点容量、顺序、可见性、token守恒均通过所查不变量。11条输入发生既定截断（GovReport3、RepoBench-p8），无隐藏改口径。

调用链为prefill_frozen_skip→qwen3_attn_patch→frozen_region_tli→frozen_region_cache：Near-only保留不可见staging，不参与attention，老化后进入Far；Far-only物理淘汰Far并在后续老化时淘汰越界token。11个Near-only层—请求记录18页/1152 token进入Far，GovReport98占8页；Both只保留sink/SWA。tail32用于选择，最终attention使用完整128维K/V，不能误称attention仅32维。

Near-only层KV节点容量比均值1.0，attention-token比均值0.689318；全体Dual冻结节点KV比0.954845、attention-token比0.957547。这些只描述节点容量及工作量代理，不是实测加速比或整卡显存收益。

独立复用第二轮396缓存性质用例、400门控组合、20mean参考对照，并核绑定；本轮未重新生成GPU预测。原始GPU Q/K、逐decode选择集合、迁移后GPU数值、完整权重历史哈希及部分kernel环境仍未独立覆盖。64个native_full_parity标记不等于dense attention一致。现有verification_plan是简版实质清单，未追补或伪造正式历史schema。有限检查不能保证绝对无bug。

## 独立证据与本轮动作

- [独立评分/统计/导出报告](ai_supervision/round_20261029T161704/independent_scores/REPORT.md)。
- [独立输入/门控/缓存路径报告](ai_supervision/round_20261029T161704/independent_paths/REPORT.md)。
- [独立会话身份及报告哈希](ai_supervision/round_20261029T161704/independent_sessions.json)。
- [1533项最终资产哈希映射](ai_supervision/round_20261029T161704/final_artifact_bindings.json)与[收尾复核](ai_supervision/round_20261029T161704/binding_closure.json)。
- [监督轮次报告](ai_supervision/round_20261029T161704/ROUND_REPORT.md)。

同一1300请求集合摘要：fd595e95527362034fe2bf859061d93060c378cf3b179ce38ed21b98de8fa431；1533项扩展绑定摘要：901e031bbf6110667ebcdabacad7c03447e3750cea84910deddb84206b5471f5。字节复核无变化。

本轮从1291/3877推进至1300/3900，较上轮交接1271/3818增加29配对/82生成；正常自动接力、自动validate→analyze→export完成。三机精确PID/GPU/log/heartbeat/落盘多次核验，汇总锁竞争未破坏互斥，未抢锁。没有修改生产源码/阈值、另派GPU、重启E184或使用本机GPU1。无须质量重跑；本轮负结果按既定范围交付。

收尾记录：两个独立会话exit 0，最终链接与1533资产再次核对通过。文档提交f3d4d2a8dc18cc3597778b9902c58ca63717c045已推送并核验bee/origin；冻结被测源未改变。完成条件已满足，ai_supervision/COMPLETE.json登记verified_complete。
