# TL-E155D-SCORE-UNITS-20261020

来源：用户固定配额审查、根 TASK T2、PAIRING 已记录的量纲未决项。
基线：4ea101c9eca2ef9c4a852db7a5dfbd04db061c6b。

完成：先复现 7 项中 5 失败；输出前转换 CI 到 AVG5 的 score points，
显式记录单位，保留采样/显著性/配额。普通/-O 各 32 项、相邻 17 项及
24 组基线差分通过。[验证](../../results/e155d-units90/verification.md)、
[影响与边界](../../advice/2026-10-20_e155d_score_units_by_gpt.md)。

下一步：真实快照重新汇总并核对外部消费单位；source/staging 一致快照与
answers/input 身份另行处理。near-width 与 T7 GPU 门禁保持未完成。
不将本任务的 host 验证计为整个 T2 或 GPU 验收完成。
