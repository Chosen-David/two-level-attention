# TL-NFA-PRODUCTION-P0-P8-20261103

## 任务来源与边界
- 用户指令：优先完成 P0–P8 修复与优化，再统一测试性能；Graph 32K 仍未达 2× FA2 时继续迭代。
- 起始提交：`40756b6c925ad0caa5741ccbc305299fb8189567`（本地 `main`）。
- 单 Agent 串行、in-place 修改；保留其他轨道 WIP，不启动子 Agent。
- 保持 dense prefill、NFA 两级选择、冻结 gate、GQA、paged KV、seed/sink/SWA、Graph 地址稳定与真实多请求语义。

## 执行链
- [x] P0：选择组与消费者一致，逐 KV group 对拍 L1/L2/逻辑与物理选集/attention。
- [x] P1：prefill 批量 paged KV 写入。
- [x] P2：gate/core-token 流式统计，删除无消费者工作。
- [x] P3：增量页摘要及页复用生命周期。
- [x] P4：L2 tile 并行与中间量方案。
- [x] P5：排序、去重、union 与 workspace。
- [x] P6：四态 skip 轻路径与物理 KV 回收。
- [x] P7：grouped/split-KV 主 attention 对比，数值变化单独归类。
- [x] P8：eager 同步、Graph 生命周期、metadata 与槽复用。
- [ ] 全量正确性回归后，统一测量 prefill/gate/freeze/capture/首 token/稳态 TPOT/显存与多 BS、多长度、真实多请求。

## 当前状态
- 进行中：统一端到端正确性与性能测量。
- P0/P1 针对性统一回归 29/29 通过；P1 覆盖非连续 request slot、非整页尾部、双层的 CPU/GPU 全 arena 对拍。
- P2 gate/core-token CPU 与 CUDA 对拍 26/26 通过，流式实现不再物化全头尾部概率矩阵。
- P5 去重表改为 int32 generation tag，并覆盖回卷重建；P6 both-skip 仅执行 canonical protected/seed union，覆盖 eager launch 重放与 Graph 的当前 decode KV 写入。
- P7 默认 grouped 保持不变；split-KV 仅作实验参数。合成 K=1156 微基准中 Graph split 更快、eager split 更慢，且相对 grouped max-abs-diff 为 0.0039–0.0078，后续若启用必须独立做端到端 token identity 验收。
- P8 按唯一 skip policy 计算 visible bucket、页表无变化时省略全宽 H2D、seed 消费使用固定 request-id buffer、重复 capture 直接复用；修复轻路径把旧 paged write 录进 eager 计划而覆盖当前 KV 的问题。
- P0–P8 统一回归 80/80 通过，含真实 Qwen B1/B2/B4、eager/Graph、多请求与动态 EOS token identity。
- 旧 head-0 共享选集性能结果不作为正确基线。
- 性能结论须绑定源码 SHA、配置、硬件、依赖、命令、原始计时与 GPU hygiene。

## HISA 官方实现对照（2026-11-03）
- 来源：`TransArch/HISA_COLM_2026` commit `815ea014f6001ad05f04c4715b0d5e4e7134e9e1`；`tile-ai/tilelang` commit `fa69f7fa817cacab1af36e9e616c6aaba6ee115b`；`xuyufei-a/sglang_hisa` 的 `hisa_pr` commit `e399d9f33fecfe50b90fa65e3e0da01ea1572f8f`。论文目标为与原 DSA 选集平均 IoU >99%，不提供 NFA 所需的 bit-exact/token-identical 保证。
- 两级候选细化：HISA 以 block mean-pool + coarse MQA 选 block，再对入选 block 做 token-level sparse MQA；对应 NFA 的页摘要/L1 与 paged fine-score/L2。可移植的是分阶段数据流与工作区组织，不能替换 NFA 的 min/max/sum、far/near、bypass、quota 与补洞规则。
- 部分 Top-K 与有序归并：HISA 使用 fast radix Top-K，避免通用全排序；对应 `_nfa_filtered_topk_kernel`。可移植 tile 部分 Top-K、两阶段归并和按实际预算裁剪工作量，但必须保持 signed-BF16 key、同分优先新 token、`-inf`、quota/补洞及逐 KV-group 选集完全一致。
- 选择后立即生成消费坐标：HISA 将 Top-K 与 slot→token 坐标变换、可选 page-table gather 融合；对应 NFA 的 middle 解码与 `_final_union_physical_kernel`。可减少重复 gather/坐标物化，但不能省略 owner/generation 去重、protected/seed canonical union 和物理页映射。
- Graph 稳定工作区：HISA 跨层共享 schedule metadata，保留 DeepGEMM padded row stride 并以 view 读取有效列；对应 NFA 的固定 bucket/workspace 与 launch plan。可继续消除无意义 contiguous/copy，保持 capture 地址稳定。
- 数值边界：HISA 的 FP8、DeepGEMM/TileLang 与 `+inf` boundary maintenance 均不直接进入 token-identical 生产臂；任何会改变选集的 HISA 语义只能进入独立实验臂。
