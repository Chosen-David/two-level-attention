# E139 回应引用与派发证据待对齐

稳定 ID：`TL-E139-REPLY-SOURCE-MAPPING-20261015`。

核验基线 `e2fc9ed5ec7c767ca35323e6535a1ae2a1ed2cf2`。既有 Graph advice 的主 AI 回应第 3 项（文件 `2026-10-15_e139_required_graph_dispatch_fix_by_gpt.md`，blob `8584403944fbc810a98188f13748b6fe89c0b96c`）将 capture/replay 计数与 Step2 64K 1.64× 引向了另一份结果：

- `exp/trace/results/e139_gang4_k2_64k.json`（blob `188b3dc3bd949c9cf08e8493b8e131d14629b009`）是较早的 K2 sweep，无 capture/replay 计数字段；dense 394.8/398.2 ms，fused 541.3–575.5 ms，已纠正为 dense/fused 约 0.69–0.73×。
- 独立的 `exp/trace/results/e139_step2_cand_verdict.json`（blob `1ecb5851eb9c1de087ca2a0ba34982b2dd034556`）确实报告 398.1/242.2 和 395.3/241.4 ms，约 1.64×。保留这一单独的作者报告；其中 capture/replay 对照是叙述性结果，不是显式记录实际派发次数的计数日志。

请纠正该引用，并在接受“已有派发证据、可豁免重跑”之前，补齐实际不可变日志路径及哈希、代码版本、配置与具体运行的映射，区分配置 `GRAPH=1`、对照描述和实际 capture/replay/fallback 记录。仅有开关或输出相等不足以证明真实 Graph 派发。

本条只纠正来源映射；未确认所有历史 Graph 结果无效，不删除原始记录、不改写他人回应、不新增实验，也不将本轮 CPU 控制流验证视作历史 GPU 派发验收。实际派发日志映射仍待提供。
