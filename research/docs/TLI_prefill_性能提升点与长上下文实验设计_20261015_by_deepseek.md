# TLI prefill 性能提升点盘点 + 长上下文实验设计（E141 方案）

日期：2026-10-15。作者：deepseek 轨（响应用户「prefill 1.64x 不够讲故事，找性能提升点 + 上下文长度设计 + 关注 GPT 审计 + 编写相关实验」）。
状态：方案已基于 GPT 审计（TL-156 / prefill_semantics / measurement design）+ E139 全部 verdict 梳理；**E141a（长上下文 wall）已完成：128K 2.75x / 256K 4.24x 净加速实测**。

---

## 1. 现状盘点：1.64x 从哪来、为什么"不够"

### 1.1 四件套优化链（已落袋，跨机验证）
| 里程碑 | 64K prefill wall | vs dense |
|---|---|---|
| eager 基线（逐组） | 2302ms | 0.17x |
| +纲领②③ batch_select | 574ms | 0.69x |
| +纲领④ fused attn | ~340ms | ~1.2x |
| +#253 Step1 CUDA Graph | 328.9ms | 1.22x |
| **+#253 Step2 cand topk** | **242.2ms** | **1.64x** |

- Step2 判决跨机复验（本地 GPU0 与 .187 GPU4 逐档 ±1%）：64K dense 398 / Step2 242 = 1.64x。
- 选择链：graph 207.8 → Step2 120.7ms；**topk 段 111.5 → ~20ms**（候选列化跨行批量）。
- 质量：Step2 硬门 mask 逐位 0/1008 + e2e 输出 maxdiff=0.0（三长度），回归全绿。**生产零扰动**（默认 env 全关 fail-closed）。

### 1.2 为什么 1.64x 不够（用户判断正确）
- 对标论文最大倍率：Quest 组合 attention 7.03x、TokenSelect 单 chunk prefill 23.84x、Twilight 15.8x（相对 FA2）。1.64x 在稀疏 prefill 赛道是末位。
- 短档净负：8K 0.31x / 16K 0.53x / 32K 0.94x——固定开销（选择链）占比在短序列高。
- GPT TL-156 纠偏已确认：相对 dense 的加速必须是 `dense_ms / candidate_ms > 1`；此前 3.82x/4.27x 是相对旧稀疏实现，不是对 dense。

### 1.3 关键 scaling 事实（故事主线）——E141a 实测（2026-10-15 本地 GPU0）
| L | dense (SDPA) | TLI Step2 | speedup |
|---|---|---|---|
| 8K | 6.7ms | 22.2ms | 0.30x |
| 16K | 26.0ms | 49.3ms | 0.53x |
| 32K | 101.1ms | 107.5ms | 0.94x |
| 64K | 398.2ms | 242.2ms | 1.64x |
| **128K** | **1574.3ms** | **571.9ms** | **2.75x（实测）** |
| **256K** | **6275.4ms** | **1481.3ms** | **4.24x（实测）** |

**dense prefill = O(L²)**（64K→256K 增长 15.8x）；**TLI 稀疏 = 近线性**（选择链 O(N·C·δ) 常数预算 + attention O(L·K2) 恒定 1024/行，64K→256K 仅 6.1x）。**64K 只是分界点，128K 2.75x / 256K 4.24x 是 TLI 主场。** 判决落盘 `exp/trace/results/e141a_wall_long.json`。

---

## 2. 性能提升点全盘点（"看看有没有性能可提升的点"）

### 2.1 已落袋（勿重复投入）
四件套：batch_select（纲领②③）+ fused_attn_triton（纲领④）+ CUDA Graph（Step1）+ cand topk（Step2）。全部默认 fail-closed 零扰动，显式 env 启用。

### 2.2 剩余杠杆（按收益排序）

**(A) L2 softmax/mean 全宽→候选列融合（~47ms，选择链 120.7 → ~75ms）**
- 现状：Step2 下逐行仍做 `F.softmax(buf)` 全宽 [1,1,H,g1_j=64K] + GQA mean（47ms/64K）——为了 p 值零 ULP 与全宽 topk 等价。
- 优化：候选列 [N,Hkv,C=8192] 一次性 softmax（理论带宽 ~3-5ms），固定形状可批量 + graph + fused。
- 代价：softmax 归约树宽度差（g1 vs C）→ 0.1% topk 边界翻转（GPT prefill_semantics 审计明确指出）。**需质量门**：e2e allclose(1e-2) + RULER 质量等价验证，非逐位门。
- 收益：64K 242 → ~205ms ≈ 1.93x；选择链 120.7 → ~75ms。

**(B) fused attention 段 flash varlen 化（~120ms @64K → 理论 ~60ms）**
- 现状：fused_attn 块级跳空后 W=4% 有效列，120ms @64K；dense flash 全列 398ms。
- 优化：varlen flash（cu_seqlens 组块）消 score 物化。早期 profile 预算 W=1.2K/32K 列理论 20-30ms。
- 注意：E139 profile 判 chunk-shared 质量崩塌档（-10.5/-7.2）；grp64 质量保住在 47.9。flash varlen 须在 grp64 质量口径下做，不可偷换到 chunk-shared。
- 收益：64K ~205 → ~165ms ≈ 2.4x（叠加 A）。

**(C) 长上下文（0 代码成本，纯实验设计收益）**
- 见 §1.3：128K ~3.2x、256K ~6x。这是**最大、最干净的杠杆**。
- 质量支撑已有：RULER 128K mavg AVG11 ≈ 47.4 vs fullkv 37.3（e116b 正式数据）——稀疏在超长上下文反超 dense（检索更聚焦，KV 噪声更大时稀疏优势放大）。
- 论文叙事从「64K 1.64x」升级为「128K 3x+ / 256K 6x 的 prefill 加速曲线」。

### 2.3 已判不投入
- Step3 fine tile scan：全档 -1.8%，不动生产 tile 表（e139_step3 判决）。
- 大 GEMM 打分批量化单独做：收益 <15%（profile 判 select 类仅 16%）。

---

## 3. 上下文长度设计（"上下文需要更长还是怎么设计"）

**结论：需要更长，且这是唯一能让 prefill 故事成立的维度。**

### 3.1 长度选择依据
| 长度 | 依据 |
|---|---|
| 64K | 分界点（1.64x 已证），交叉点 ~33K |
| 128K | dense O(L²) 已显著爆炸（~1600ms）；RULER 128K 质量已有（mavg 47.4 > fullkv 37.3）；Qwen3-8B 原生 32K + YaRN 可到 128K（MoBA 128K 已在跑） |
| 256K | dense ~6.4s，TLI ~1s = 6x；需 YaRN 128K→256K 因子 + 显存预算（H20 94GB 足够：q/k/v + TLI 中间 ~5GB） |

### 3.2 长度×预算设计
- 主表口径 K2=1024（预算恒定）——跨长度同预算对比公平（每行 1024 token）。
- 附录：K2 ∈ {512, 2048} 预算敏感性，验证 speedup 不依赖预算选择。
- 质量主表：RULER {32K, 64K, 128K, 256K} × mavg（若 256K 显存/时间允许）；已有 32K（E109/E123）与 128K（e116b）。

### 3.3 性能-质量联合叙事
- **性能**：prefill wall 曲线（E141a）——长度单调、128K+ 3-6x。
- **质量**：RULER 逐长度分数——长上下文稀疏反超 dense（128K 已证）。
- **合成**：Fig.3 主图 = speedup vs L 曲线 + dense 参考 + 质量锚点标注；长度横轴天然区分 TLI（近线性）与 dense（二次方）。

---

## 4. 实验矩阵（E141）

| ID | 实验 | 目的 | 状态 |
|---|---|---|---|
| **E141a** | wall 曲线 L∈{8K,16K,32K,64K,128K,256K} × {dense, TLI Step2 全开}，K2=1024 | 验证 128K/256K 翻盘点，产出 Fig.3 主数据 | ✅ **完成**：128K 2.75x / 256K 4.24x（e141a_wall_long.json） |
| **E141b** | L2 softmax/mean 候选列融合（代码优化 A） | 选择链 120.7 → ~75ms；64K 1.64x → ~1.93x | 方案已交付（§2.2A），实现归 Opus（TLI 方法代码） |
| **E141c** | RULER 256K 质量（mavg vs fullkv） | 补齐长上下文质量主表 | **阻塞**：RULER data 无 256K 样本，需先跑数据生成器；128K 质量已有（e116b mavg 47.4 > fullkv 37.3） |

### E141a 验收判据（实测通过）
- ✅ 128K 2.75x ≥ 2.5x、256K 4.24x ≥ 4x：长上下文叙事成立，Fig.3 主图采用。
- 备选路径不触发（近线性外推成立：选择链 64K→256K 增长 6.1x，远低于 dense 15.8x）。
- 记录每档 fresh indexer + graph capture 稳态，7-iter 中位。

---

## 5. GPT 审计处置对照

| GPT 审计 | 本方案处置 |
|---|---|
| TL-156（speedup 方向颠倒） | 一律 `dense_ms/candidate_ms`，判读前手验 `candidate < dense`；E141a 脚本直接输出 speedup 字段 |
| prefill_semantics（chunk 共享近似 vs 逐行） | E141a 用 grp64（组末行选择）口径，非 chunk-shared；质量锚点 mavg 47.9 保持 |
| prefill_semantics（候选列 softmax 0.1% 翻转） | E141b 明确走**质量门**（e2e allclose + RULER 等价），不宣称逐位一致 |
| measurement design（T0-T3 分层） | E141a 是 T1（全 attention pipeline，随机 QKV 微基准）——**不冒充 T3 请求收益**；论文主性能图待 T3 完整请求采集验收 |
| measurement design（长 context 优势仅假设） | E141a 正是把假设变为实测的唯一路径 |

---

## 6. 后续动作
1. E141a 跑完 → 出 wall 曲线 JSON + 判决（128K/256K speedup）。
2. E141b 原型：改 batch_select.py 候选列 softmax + bitdiff 质量门（e2e maxdiff + RULER 抽测）。
3. E141c 在 .187 128K MoBA 收口后补 RULER 256K。
4. 论文侧：Fig.3 主数据替换为 E141a 结果；prefill 从消融附录升级为主性能主张（若 128K ≥ 3x）。
