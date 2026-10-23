# TL-ASYNCTLS-COMPARISON-20261023

## 来源与范围
用户要求更详细的 AsyncTLS 测试对标设计、独立 PR 和 advice。设计基线 a7bcdddf727aa99196d925b9f106d785ce9273d4。
不启动GPU、付费或新矩阵，不改生产方法/default，不合main；prepare_mask候选独立保全，不混入。

## 稳定方案
[分阶段设计](../../../research/docs/ASYNCTLS_COMPARISON_PLAN_20261023.md)：A0来源闭合→A1语义→A2 attention/cache→A3 Graph等价→A4成本→A5 smoke→A6确认集→A7联合判决。
[建议](../../advice/2026-10-23_asynctls_comparison_design_by_gpt.md)。接续S-T023、旧阅读方案、PR #4，不重写历史。
验收当前设计须独立审查和真实CLI/路径静态核对；后续硬件阶段逐项绑定授权及原件，本文不授予执行许可。

## Progress
- 2026-10-23：实际重读AsyncTLS v1 §4–5，未核官方repo/commit/license；提交树缺现有CLI引用的adapter/校准脚本。
- 独立审查要求分清M_t/M_{t-1}、dense层与sparse预算、硬件授权、公开路径隐私、eager诊断计时范围；作者已修订。
- 本PR仅测试设计与advice，未执行GPU、质量或性能测试；后续实现/运行状态均保持待验。

