# MInference pattern 命名与运行时内容绑定

@DeepSeek（S-T023 baseline）。静态核查 `e37e5b5326b6f129e989a2f03144770a929fa1ed`；未运行测试、模型或GPU。本项承接149的内容身份修复：新 `_cfg<sha8>` 已区分同名不同内容，但还未保证命名与adapter消费同一份内容，不重复提出旧的basename缺陷。

**调用链与条件**：[freeze_minference_pattern_sha8 L634–674](https://github.com/Chosen-David/two-level-attention/blob/e37e5b5326b6f129e989a2f03144770a929fa1ed/sparse_attn/info.py#L634-L674)读取config文件A，只把sha8冻结到args；[LB入口 L536–546](https://github.com/Chosen-David/two-level-attention/blob/e37e5b5326b6f129e989a2f03144770a929fa1ed/benchmark/LongBench/pred.py#L536-L546)及[RULER入口 L230–240](https://github.com/Chosen-David/two-level-attention/blob/e37e5b5326b6f129e989a2f03144770a929fa1ed/benchmark/RULER/pred_ruler.py#L230-L240)随后仍将可变config_path传给enable_minference；[adapter L253–267](https://github.com/Chosen-David/two-level-attention/blob/e37e5b5326b6f129e989a2f03144770a929fa1ed/exp/vendor_baselines/minference_vendor/qwen3_minference_patch.py#L253-L267)再次open/json.load，并把所得best_pattern绑定到attention模块。若同一路径在这两次读取之间由A替换为B，命名保持A而runtime使用B；未见expected hash比较或冻结bytes/解析对象传递。调用相邻不等于共用同一份内容。这是明确替换条件下的源码推导，非已执行的作业复现。

**测试边界**：[新增149测试 L229–243](https://github.com/Chosen-David/two-level-attention/blob/e37e5b5326b6f129e989a2f03144770a929fa1ed/exp/trace/test_gpt_c_batch.py#L229-L243)冻结A后换源B，仅检查名字仍为A，没有调用adapter核best_pattern。因此该项PASS不能证明命名/runtime绑定，也不能证明两读窗口已闭合。

**独立补证层级**：根线程转交的 `audit_far_gate_execution` 标准库检查从固定e37e5b源码AST提取实际freeze/_cfg/enable函数，用临时pattern文件及FakeAttention容器：冻结有效JSON A（sha8=6a49d891）后将同路径替换为有效JSON B（sha8=fb09abc1），enable返回patched_count=1、模块best_pattern为B，而命名_cfg仍为A。本文作者未重跑；该回执只支持 `is_search=False` 下的API两读窗口，不是torch/真实模型/GPU或实际作业污染证据。文件不变不触发此反例，adapter已载入后再换文件也不会改变已绑定的内存pattern。

**建议与最小验收**：入口单次读取后，让命名和adapter共同消费同一冻结bytes或解析对象；仅缓存hash、随后继续重开可变路径不足。CPU最小验证应覆盖实际freeze→adapter消费链：在两个读点之间替换A为B，断言名字哈希与模块实际best_pattern仍同源（或明确拒绝不一致）；未替换的合法pattern正常加载。测试需观察adapter输出/模块绑定，不能只检查文件名。处理归DeepSeek baseline职责；现有运行的生成SHA、pattern快照及日志尚未绑定，本轮不认定任何真实job污染、不声称已停作业、不请求新GPU重跑或新增矩阵。修复不追溯证明旧结果身份，也不关闭faithful port与完整请求质量/成本验收。
