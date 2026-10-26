# TL-HYBRID-CACHE-20261026

用户授权：在独立 branch 开发按层 cache 分配并提 PR；不合并 main。
基线：`900f79b4c714164e5e0b36a62c4e330bc739a346`。

## 已实现

- Qwen3 在通用 cache update 前分发 compact consumer，eager/static 每步只写一次 KV。
- 静态 cache 先冻结 gate；far-off 层直接分配固定 sink + near/SWA 共用 ring，far-on 层分配线性 buffer。
- 请求间重新判断布局；旧 ring 不再接收完整 prompt 的线性 copy，far-on 转换恢复线性存储。
- E152v4 GraphRunner 传递显式 cache_position、注册自有无 padding mask，compact 请求重填时释放旧 graph/output 并重新 capture。
- benchmark 分开记录主 KV 字节数、释放 DynamicCache 后的 allocated 显存和全过程峰值。
- compact 模式暂时拒绝未接入 ring consumer 的 Llama/MLA；不改变 legacy 默认值。

## 存储与语义边界

保留既有 compact selector：容量 `sink + max(near_len, page_size) + swa + page_size - 1`。
4096/128/4/64 几何对应 4291 槽位；最后 63 槽覆盖 near 左边界页对齐。
保留整个 near 候选区，而非只保留当前 K2；SWA→near 只改变逻辑归属，不复制 KV。
near/SWA ring、全宽 K/V、子空间工作区和 L1/L2 预算均沿用已有定义。

支持范围：Qwen3、head_dim=128、B=1、逐 token、无 padding、dense prefill、冻结 gate。
DynamicCache 可用 compact retain/evict；TLIStaticCache compact 仍要求 evict，并通过 `fill_static_from_dynamic(..., model=model)` 初始化。
当前 batch harness 的逐行填充不在此次接入范围；静态可运行入口为 E152v4 GraphRunner。
不得在同一请求中改变 backend/gate/geometry；新请求重填并重新 capture。

compact 原有 sink 边界选择与 legacy 不完全相同。本次对拍采用相同 compact 选择规则的完整历史参考，不宣称等价于 legacy 或 FullKV。
本次没有加入分页池、KV 量化、attention gather 融合，也没有宣称 GPU 加速或实模质量达标。

## 验证与剩余项

详见 [验证记录](../../results/hybrid-cache-20261026/validation.md)。

- [x] CPU 真实 HF cache + 随机小型 Qwen3：多次环绕、短 prompt、保留/淘汰 KV 对拍、重复请求和 gate 切换。
- [x] 分配与生命周期：跳过层禁止创建全量 static buffer，旧 KV 可释放，拒绝无 model 重填，旧 graph/output 失效。
- [x] 相邻 mask、padding、batch context、baseline、graph contract 回归。
- [ ] GPU 整模型 capture/replay 与 E127、batch 两项硬门。
- [ ] 真实 Qwen3 32K/64K/128K：质量、主 KV/workspace/峰值显存、TPOT、fill/capture 成本。

新增 CUDA 测试在无 GPU 时明确 skip。GPU 门禁与质量验收通过前，不升级主表叙事。
