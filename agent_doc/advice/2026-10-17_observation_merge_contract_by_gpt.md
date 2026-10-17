# Observation 合并后的验证契约不一致

审查快照：`Chosen-David/two-level-attention` main `d1489fc8a1280eec4a263b9c66b4b08e7e2f51bb`；相关合并为 `ca4ca8fdd9c43c8aa55e9572cd6bf0f07812fc81`。本条是只读核验，不改渲染器、原始数据、实验配置或运行中任务；没有 GPU、新采集或整套回归。

## 确认失败：validator 与实际 manifest 不匹配

- `exp/paper/e140_asset_provenance.py:22–23,95–99` 要求 `e140-historical-validity-v4` / `whole-bin-diagnostic-gate-blocked`。
- `agent_doc/guide/paper_figs/observation/e140_validity.json:2–3` 实际保留 `e140-historical-validity-v3` / `redesigned-v2-inputs-current`。
- 从上述 SHA 下载两个文件到隔离目录，核对 Git blob 分别为 `3157e7620cefdc05aa5192306b121d0c73e8c3a7`、`a44d0cab800094f5553b6c4255bff66f7f122557`，运行 `PYTHONDONTWRITEBYTECODE=1 python -B exp/paper/e140_asset_provenance.py --root .`，实际退出码 **1**，在 `verify` 第 99 行抛出 `ValueError: historical validity status/schema changed`。

该失败发生在读取任何资产哈希之前，因此隔离目录未下载整套资产不影响这一复现。此结果只证明当前验证入口与 manifest 冲突，不代表资产损坏，也不是完整 provenance 验收。

## 其他确定的源码契约冲突（未执行这些测试）

- `tests/test_e140_asset_provenance.py` 多个用例先调用 `asset.verify(ROOT)`，会遇到上述同一前置失败。第 94 行还要求渲染器含 `add_render_warning(fig)`，当前 `agent_doc/guide/paper_figs/figs/fig_o_observation.py` 没有该调用。
- `tests/test_observation_bins.py:73–88` 导入渲染器并调用 `renderer.build_data(source)`；保留的出版渲染器没有 `build_data` 定义。
- `test_default_renderer_fails_closed` 要求默认失败、stderr 含 `No files written` 且不改资产；保留的渲染器第 503–508 行是顶层导出，第 554–555 行写 TSV，并无该 diagnostic opt-in 契约。它还使用第 73、78 行的宿主绝对路径。故不能把该测试当成无副作用检查直接运行；实际首次失败可能受环境依赖影响。本轮没有运行渲染器或整套测试。

## 与已有建议的关系及最小修复方向

`agent_doc/advice/2026-10-16_observation_plot_correctness_by_gpt.md` 的「Opus 核验与合并裁决」已明确选择出版 renderer/assets/validity v3，未采纳 diagnostic renderer/v4。这次发现的是该裁决没有同步到 verifier/tests，并非再次要求撤下出版图。

建议维护者明确并落实选定的验证契约：若保留出版 v3，调整 verifier 和对应回归测试，使其验证该版本的真实资产、数据源和出处；诊断 v4 若继续保留，应具有独立入口及自己的测试，不能让同一路径同时承诺两套互斥行为。不要仅为过测试把 manifest 改写成 v4 或删掉身份校验；不要直接运行会覆写出版资产的诊断 provenance 记录脚本。之后在隔离输出目录执行匹配版本的 CPU 回归，记录精确 SHA、命令和退出码；已有分支成功日志不能代替最终合并树的回归。

本条没有发现或声称任何运行中实验受影响，不支持停止任务或重跑 GPU。验证契约修复也不等于 E6 gate 科学有效性获得认证；该边界仍需独立证据。
