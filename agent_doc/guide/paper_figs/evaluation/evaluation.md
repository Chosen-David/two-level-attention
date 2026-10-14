关于实验部分，按照顶会可视化审美做出如下图

最终论文方法与其他基线的对比：
测全量
1. LongBench v1的14个英文任务+2个代码任务（如果主张中英文通用需要补齐全部任务），
2. LongBench v2的全部503题
3. RULER 32k 64k 128k的上下文
最终论文模型：Qwen3-8B, Qwen3-32B（验证规模变化）和LLama（跨家族验证）， 
扩展模型实验 DeepSeek-V2-Lite-Chat(则是验证扩展到nope+rope的MLA，以及跨家族验证)暂时先不做。


对比基线怎么比呢，我们是prefill和decode分别加速多少这样，其他方法分别加速多少这样？

对比baseline: 
HISA, Quest, FA(full attention也即我们的fullkv), FFD(faster than flash)
他们各自的github链接一览：

时延相关的数据呢


