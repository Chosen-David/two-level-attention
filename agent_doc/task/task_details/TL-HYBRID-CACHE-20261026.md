# TL-HYBRID-CACHE-20261026

用户授权：在独立 branch 开发按层 cache 分配并提 PR；不合并 main。
追加授权：支持大 batch、CUDA Graph，升级同一个 PR #9。
基线：`900f79b4c714164e5e0b36a62c4e330bc739a346`。

## 已实现

- Qwen3 在通用 cache update 前分发 compact consumer，eager/static 每步只写一次 KV。
- 静态 cache 先冻结 gate；far-off 层直接分配固定 sink + near/SWA 共用 ring，far-on 层分配线性 buffer。
- 请求间重新判断布局；旧 ring 不再接收完整 prompt 的线性 copy，far-on 转换恢复线性存储。
- E152v4 GraphRunner 传递显式 cache_position、注册自有无 padding mask，compact 请求重填时释放旧 graph/output 并重新 capture。
- benchmark 分开记录主 KV 字节数、释放 DynamicCache 后的 allocated 显存和全过程峰值。
- compact 模式暂时拒绝未接入 ring consumer 的 Llama/MLA；不改变 legacy 默认值。
- `HybridBatchCacheBuilder` 逐请求消费无 padding prefill，立即冻结并保存该请求每层的 gate/indexer/pair 状态，随后释放完整 prefill KV。
- patched compact 层按请求独立分配 ring 或线性 buffer；不以同层最大容量给所有请求分配矩形全宽存储。未 patch 的原生层使用普通 batch 线性 cache。
- `HybridBatchRunner` 将 token、位置、有效长度、alive、mask 保存在固定地址；统一驱动 staged/graph decode，EOS 后所有层的对应 KV 行冻结。
- RULER batch 入口自动分流 compact+evict；支持不等长请求、混合 far gate、每个 batch 重新构建/捕获，取消此路径的 16K 下限。
- graph selector 修复短序列 sink/SWA bypass 页重复、K2 未按真实长度截断、L1 quota 大于容量和非页对齐容量越界；超额 L2 quota 回到完整 token 概率域补齐零分候选。

## 存储与语义边界

保留既有 compact selector：容量 `sink + max(near_len, page_size) + swa + page_size - 1`。
4096/128/4/64 几何对应 4291 槽位；最后 63 槽覆盖 near 左边界页对齐。
保留整个 near 候选区，而非只保留当前 K2；SWA→near 只改变逻辑归属，不复制 KV。
near/SWA ring、全宽 K/V、子空间工作区和 L1/L2 预算均沿用已有定义。

支持范围：Qwen3、head_dim=128、逐 token、dense prefill、冻结 gate。混合 batch 没有硬编码 BS 上限，CPU 实测至 BS=32；GPU 可承载规模尚未测量。
DynamicCache retain/evict 保持 B=1；旧 B=1 static 入口仍用 `fill_static_from_dynamic(..., model=model)`。
batch static 使用 `HybridBatchCacheBuilder.add(dyn, length)`，必须在每行 prefill 后立即调用，再 `finish()` 交给 `HybridBatchRunner`。每次 `inject(tokens, lengths, alive)` 对存活行长度 +1、结束行保持长度；`capture()` 在首步 inject 后调用，`run()` 执行 staged 或 replay。
RULER batch 使用现有参数加 `--tli_far_skip_backend compact --tli_far_skip_cache evict --batch-size 32`，保持固定 near_len/pages 几何；`TLI_CUDA_GRAPH=1` 开 capture，`=0` 用相同 selector 的 staged decode。capture 失败显式报错，不静默标成成功。
batch 输入可左 padding，入口先剥除 padding 逐行 prefill，decode 有效 KV 在各自前缀。不得中途换请求、复活已结束行或改变布局；新 batch 必须新建 runner/graph。

compact 原有 sink 边界选择与 legacy 不完全相同。本次对拍采用相同 compact 选择规则的完整历史参考，不宣称等价于 legacy 或 FullKV。
far-on batch 使用原有 graph selector；常规预算已对拍 eager 完整历史。超额预算耗尽非零候选时，等零分 topk 在固定宽与变长域中的 tie 顺序可能不同，不宣称该组合与 legacy token 序列逐位相等。
attention/indexer 仍逐请求发射操作，整模型 projection/MLP 是 batched，graph 记录整个 batch；未实现融合 batch attention kernel，不声称大 BS 吞吐随 BS 线性提升。far-on 层仍需全历史 KV/index，dense prefill 的峰值需求仍存在。
本次没有加入分页池、KV 量化、attention gather 融合，也没有宣称 GPU 加速或实模质量达标。

## 验证与剩余项

详见 [验证记录](../../results/hybrid-cache-20261026/validation.md)。

- [x] CPU 真实 HF cache + 随机小型 Qwen3：多次环绕、短 prompt、保留/淘汰 KV 对拍、重复请求和 gate 切换。
- [x] 分配与生命周期：跳过层禁止创建全量 static buffer，旧 KV 可释放，拒绝无 model 重填，旧 graph/output 失效。
- [x] 相邻 mask、padding、batch context、baseline、graph contract 回归。
- [x] BS=1/2/8/32、混合动态 gate、不等长 prompt、多层 patch、EOS 冻结、重复 batch、inference_mode、短/非页对齐容量以及超额 L2 quota。
- [ ] GPU 整模型 capture/replay 与 E127、batch 两项硬门。
- [ ] 真实 Qwen3 32K/64K/128K：质量、主 KV/workspace/峰值显存、TPOT、fill/capture 成本。

新增 CUDA 测试在无 GPU 时明确 skip。GPU 门禁与质量验收通过前，不升级主表叙事。
