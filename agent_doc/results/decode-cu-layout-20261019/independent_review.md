# 独立复核：decode cu_seqlens 指针布局

结论：可作为低风险 host ABI 正确性修复发布，非性能优化；GPU 尚未验证。

基线：bb14bc8a0347f1e440c78d397a3e2019e4a16c82。仅审阅生产 diff，未修改生产实现、测试或文档。

## 调用链与根因

- Qwen3 patch 的 decode 分支在 prepare_mask 后将同一 cu_seqlens_k 传给 eager_decoding_attn；Llama3、DeepSeek patch 也有此调用。
- eager_decoding_attn → _eager_decoding_attn_gather → 满足 T>=16384/block_size=1、布局及 dtype 等条件时进入 _eager_gather_triton。
- wrapper 仅对非 int32 做转换；int32 间隔 view 直接透传。kernel A 使用 tl.load(CUK) 和 tl.load(CUK+1)，没有 cu 的 stride 参数。因此逻辑上 [0,T] 的 gapped int32 view 被读成 [0,夹层值]，可误置错误码 3。
- 新的 stride(0)!=1 分支在 launch 前 contiguous，保持逻辑长度值与 int32 ABI，不改变 attention、选择或预算。

## 风险与常规路径

- stride 查询是 host 元数据，不引入 .item() 或设备同步。
- 连续 int32（包含非零 storage_offset）保持原对象；不增加复制。
- 常规 prepare_cu_seqlens 通过 cumsum 返回 int64 连续数组；原有 int32 cast 保持，新增分支不再复制其结果。
- gapped int32 的两元素复制是恢复既有 ABI 所必需，不能宣传提速。
- 使用正常 PyTorch 当前 stream 操作，没有引入跨 stream 或额外缓冲复用；未验证 CUDA allocator/实际 launch 行为。
- 未扩展到非法 rank、cu 数组元素不足、溢出或跨 device 输入；这些不是本补丁的覆盖声明。

## 自行验证

环境 torch 2.8.0+cpu。

1. `python -m unittest sparse_attn.ops.tests.test_eager_cu_layout -v`：最终重新执行 3 tests OK（0.027s，含 public dispatch 和直接 wrapper 子例）。新增反向测试的逻辑 cu 为 [0,16383]、stride=2，而底层物理邻接值为 [0,16384]；正确行为是错误码 3 并在检查阶段抛出包含 cu_seqlens_k 的 RuntimeError。测试显式关闭辅助 stride 断言，避免旧代码仅因测试布局断言失败而被误认为契约拒绝；新实现两种调用入口均正确拒绝。
2. `PYTHONPATH=. python agent_doc/results/decode-cu-layout-20261019/independent_review_probe.py`：30/30 布局通过。独立 probe 使用真实 wrapper + 单位 stride 指针 ABI double；dtype int32/int64 × stride 1/2/3/7/31 × storage_offset 0/1/5。检查物理邻接读取等于 [0,16384]、输入底层 storage 未改变，以及连续 int32 对象身份不变。

这不是 CUDA/Triton 数值 attention 测试、模型测试或性能测试。生产常规 cu 连续路径不会触发该旧缺陷；不可据此否定既有模型实验或报告速度收益。
