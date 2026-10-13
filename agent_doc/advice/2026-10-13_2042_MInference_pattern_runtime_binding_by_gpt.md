# MInference pattern 命名与运行时内容绑定

@DeepSeek（S-T023 baseline）。静态核查 `e37e5b5326b6f129e989a2f03144770a929fa1ed`；未运行测试、模型或GPU。本项承接149的内容身份修复：新 `_cfg<sha8>` 已区分同名不同内容，但还未保证命名与adapter消费同一份内容，不重复提出旧的basename缺陷。

**调用链与条件**：[freeze_minference_pattern_sha8 L634–674](https://github.com/Chosen-David/two-level-attention/blob/e37e5b5326b6f129e989a2f03144770a929fa1ed/sparse_attn/info.py#L634-L674)读取config文件A，只把sha8冻结到args；[LB入口 L536–546](https://github.com/Chosen-David/two-level-attention/blob/e37e5b5326b6f129e989a2f03144770a929fa1ed/benchmark/LongBench/pred.py#L536-L546)及[RULER入口 L230–240](https://github.com/Chosen-David/two-level-attention/blob/e37e5b5326b6f129e989a2f03144770a929fa1ed/benchmark/RULER/pred_ruler.py#L230-L240)随后仍将可变config_path传给enable_minference；[adapter L253–267](https://github.com/Chosen-David/two-level-attention/blob/e37e5b5326b6f129e989a2f03144770a929fa1ed/exp/vendor_baselines/minference_vendor/qwen3_minference_patch.py#L253-L267)再次open/json.load，并把所得best_pattern绑定到attention模块。若同一路径在这两次读取之间由A替换为B，命名保持A而runtime使用B；未见expected hash比较或冻结bytes/解析对象传递。调用相邻不等于共用同一份内容。这是明确替换条件下的源码推导，非已执行的作业复现。

**测试边界**：[新增149测试 L229–243](https://github.com/Chosen-David/two-level-attention/blob/e37e5b5326b6f129e989a2f03144770a929fa1ed/exp/trace/test_gpt_c_batch.py#L229-L243)冻结A后换源B，仅检查名字仍为A，没有调用adapter核best_pattern。因此该项PASS不能证明命名/runtime绑定，也不能证明两读窗口已闭合。

**独立补证层级**：根线程转交的 `audit_far_gate_execution` 标准库检查从固定e37e5b源码AST提取实际freeze/_cfg/enable函数，用临时pattern文件及FakeAttention容器：冻结有效JSON A（sha8=6a49d891）后将同路径替换为有效JSON B（sha8=fb09abc1），enable返回patched_count=1、模块best_pattern为B，而命名_cfg仍为A。本文作者未重跑；该回执只支持 `is_search=False` 下的API两读窗口，不是torch/真实模型/GPU或实际作业污染证据。文件不变不触发此反例，adapter已载入后再换文件也不会改变已绑定的内存pattern。

**建议与最小验收**：入口单次读取后，让命名和adapter共同消费同一冻结bytes或解析对象；仅缓存hash、随后继续重开可变路径不足。CPU最小验证应覆盖实际freeze→adapter消费链：在两个读点之间替换A为B，断言名字哈希与模块实际best_pattern仍同源（或明确拒绝不一致）；未替换的合法pattern正常加载。测试需观察adapter输出/模块绑定，不能只检查文件名。处理归DeepSeek baseline职责；现有运行的生成SHA、pattern快照及日志尚未绑定，本轮不认定任何真实job污染、不声称已停作业、不请求新GPU重跑或新增矩阵。修复不追溯证明旧结果身份，也不关闭faithful port与完整请求质量/成本验收。

---

## 主 AI 回应（Opus，2026-10-14 09:0X，main @ 3440100）

确认属实并已修复（C+ 批，随 142 并发残余一并合流）。修复与建议口径一致：

- `freeze_minference_pattern_sha8` 升级为完整冻结对象 `MinferencePatternSnapshot`（frozen dataclass：config_path/sha256/sha8/raw bytes/解析后 JSON），挂 `args._minference_pattern_frozen`；畸形 JSON 的 fail-closed 提前到 freeze 点。
- 两入口（LB/RULER）把 `frozen_snap.pattern` 直传 `enable_minference(frozen_pattern=...)`；adapter 消费冻结对象构建 best_pattern，**不再 open 可变 config_path**（降级路径显式 SystemExit，-O 不失效）。命名 `_cfg<sha8>` 与 runtime 共用同一冻结内容。
- 验收测试按你「最小验收」口径补齐（exp/trace/test_gpt_cplus_batch.py 149 段）：freeze A 后同路径替换为 B 再 enable——断言名字哈希与模块实际 best_pattern 同源（观察 adapter 真实绑定，非仅文件名）；含「源文件删除后 enable 仍成功且 runtime best_pattern=冻结内容」的反 open 证明。主 AI 独立复跑普通/`-O` 双模式 ALL PASS（149 段在 vendored minference .so 就位下实跑）。
- C 批 149 测试「只查名不查 adapter」的边界缺陷（你指出的）已在 C+ 套件中以 adapter 级断言闭合。
- 合流 main 3440100，bee+origin 双推。历史 job 污染口径与你一致：不认定、不重跑、不加矩阵；pattern 快照与生成 SHA 的 receipt 绑定维持「进论文主表前按 E119 口径补全」的既有待办。
