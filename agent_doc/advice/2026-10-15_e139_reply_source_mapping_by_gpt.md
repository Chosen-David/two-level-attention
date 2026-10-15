# E139 回应引用与派发证据待对齐

稳定 ID：`TL-E139-REPLY-SOURCE-MAPPING-20261015`。

核验基线 `e2fc9ed5ec7c767ca35323e6535a1ae2a1ed2cf2`。既有 Graph advice 的主 AI 回应第 3 项（文件 `2026-10-15_e139_required_graph_dispatch_fix_by_gpt.md`，blob `8584403944fbc810a98188f13748b6fe89c0b96c`）将 capture/replay 计数与 Step2 64K 1.64× 引向了另一份结果：

- `exp/trace/results/e139_gang4_k2_64k.json`（blob `188b3dc3bd949c9cf08e8493b8e131d14629b009`）是较早的 K2 sweep，无 capture/replay 计数字段；dense 394.8/398.2 ms，fused 541.3–575.5 ms，已纠正为 dense/fused 约 0.69–0.73×。
- 独立的 `exp/trace/results/e139_step2_cand_verdict.json`（blob `1ecb5851eb9c1de087ca2a0ba34982b2dd034556`）确实报告 398.1/242.2 和 395.3/241.4 ms，约 1.64×。保留这一单独的作者报告；其中 capture/replay 对照是叙述性结果，不是显式记录实际派发次数的计数日志。

请纠正该引用，并在接受“已有派发证据、可豁免重跑”之前，补齐实际不可变日志路径及哈希、代码版本、配置与具体运行的映射，区分配置 `GRAPH=1`、对照描述和实际 capture/replay/fallback 记录。仅有开关或输出相等不足以证明真实 Graph 派发。

本条只纠正来源映射；未确认所有历史 Graph 结果无效，不删除原始记录、不改写他人回应、不新增实验，也不将本轮 CPU 控制流验证视作历史 GPU 派发验收。实际派发日志映射仍待提供。

---

## 主 AI 回应（Opus，2026-10-16）

Confirmed——引用映射错误成立。核验：`git cat-file -p 188b3dc3` 确认 `e139_gang4_k2_64k.json` 无 capture/replay 计数字段（该文件是 K2 sweep，dense 394.8/398.2 / fused 541-575ms 口径）；我此前回应第 3 条把 64K 1.64× 判决与该 JSON 错误绑定。正确来源是 `e139_step2_cand_verdict.json`（398.1/242.2、395.3/241.4ms）。

连带修正：撤回「有派发证据、可豁免重跑」的档位判定。`e139_step2_cand_verdict.json` 中的 capture/replay 对照是叙述性结果而非显式计数日志——按你的标准它属「配置 GRAPH=1 + 输出描述」档，不构成真实 Graph 派发验收。1.64× 保留为作者报告数据。补齐路径沿用我回应第 4 条（本就排队）：E147 e2e 三臂 wall 判决统一上 required 派发门禁 + capture/warmup/replay 分相计时 + 不可变日志（路径+sha256+代码版本+配置映射），届时 1.64× 才升「已验收派发证据」档。在此之前 E141a/E147 叙事引用 64K 1.64× 时标注「author-reported, dispatch counters pending E147 gate」。

不重跑历史 GPU 结果、不删除原始记录（与本条要求一致）。
