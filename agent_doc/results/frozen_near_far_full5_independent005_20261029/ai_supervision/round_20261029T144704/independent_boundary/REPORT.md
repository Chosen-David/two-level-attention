**结论：`reference_boundary` 的本轮 CPU 合成性质子范围通过，可关闭对应缺口；真实 8B 跨页迁移覆盖与实际生成复现仍未关闭。全量 1300 配对实验保持 `pending`，本报告不签发全量验收。**

本轮作为用户指定的独立审查者，仅执行只读 CPU 检查，所有 `exec` 均为 `login=false`。未写文件、调用 GPU、生成模型输出、安装依赖或创建调度器；未重跑已通过的 300 项评分。执行证据保留在工具日志，报告由外层保存。

**源码身份与实际执行**

远端 pc02 冻结目录：

`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/frozen005_full_source_20261029`

实际 HEAD 为 `3e4a6885604ac3e719ee92a4906f87b532535314`。检查前后，`manifest_rank0.json` 所列 **72 项源码 SHA256 全部匹配**；cache、gate、TLI adapter、生成入口四个关键文件还逐字节匹配该提交的 Git blob。

本地 worktree HEAD 为 `42b635a868d801701db27d0b7d01b52d92e1cdd0`，因此没有以本地 HEAD 冒充冻结源。执行均指向远端冻结目录。执行环境为 Python 3.9.9、PyTorch 2.8.0+cu128，张量均在 CPU，设置 `CUDA_VISIBLE_DEVICES=''`、`PYTHONDONTWRITEBYTECODE=1`、单线程及 `python -B`。

**独立检查结果**

| 检查 | 实际结果及可关闭范围 |
|---|---|
| 指定参考脚本 | 重新执行退出码 0；32 步迁移、5 个阈值输入、8 个 query 对照通过，未采信原 `_result.json` 的 pass 代替执行。 |
| cache 扩展参考 | 3 组 geometry、初始长度 1–33、四种 skip 组合，共 **396 用例、16,236 状态快照**通过。每例含冻结状态及 40 次 decode 更新。 |
| Near-only | staging 不进入 gather；K/V 数值和可见位置与独立位置集合一致；跨页后原 staging token 正确转入 Far 并重新可见；无永久丢弃，冻结前后节点分配字节不变。 |
| Far-only / Both | Far-only 删除 Far，Near 老化越界后永久丢弃；Both 仅保留 sink/SWA。检查了位置、K/V、可见数及 `stored_tokens + permanently_dropped = logical_length`。 |
| 短序列与保护条件 | 覆盖不足 sink/window 的序列、sink=0、page=1；重复同决策冻结允许，改变冻结决策和超过生成容量均拒绝。 |
| 独立门限 | **400 组合**通过，包含四种模式、区域存在/缺失、0、1、恰好 0.05 及其上下相邻浮点数。Near/Far 分别判断，**等于 0.05 不跳过**；设置不同 p95 也未改变 mean 模式判断。 |
| 末 query mean | 长度 1、2、7、20、300，含末 256 query；四种 chunk 共 **20 对照**通过。参考使用独立 Python 标量点积、因果 softmax 和分区求和，不调用生产 `bounds`。最大 mean 绝对误差 **3.003013754332429e-08**，低于检查容差 `2e-7`。 |

未发现上述测试范围内的生产 bug。

静态调用链也支持可见性结论：`frozen_region_cache.py:114` 冻结并隐藏 staging，`:167` 执行老化迁移，`:194` gather 排除不可见节点；`frozen_region_tli.py:69` 的 decode 消费 cache 返回的可见 K/V，`:56–62` 将选择 mask 投影至可见位置。**本轮未执行真实 TLI GPU attention/kernel，因此该部分属于静态路径核查，不是 GPU 数值认证。**

**真实 8B 覆盖仍有明确限制**

本轮重新读取旧 screen 的 100 个原始请求并统计，dual 的 3,600 个层—请求决策为：

- keep：3,272；Far-only：326；Near-only：1；Both：1。
- `musique/0079` 层 30：staging 从 4,116 增至 4,122，Far 始终 12,348，仅 6 次 decode 更新，**没有真实跨页老化迁移覆盖**。
- `musique/0056` 层 11：Both 冻结后仅 sink/SWA；永久丢弃数从 14,493 增至 14,496，与 3 次更新一致。

原始文件清单摘要与前次报告一致。这些是保存记录的独立复核，**不是重新生成验证**。Near-only/Both 各仅占 `1/3600`，不能据此宣称广泛质量覆盖，更不能用 screen 替代全量 1300。

**仍需保留的缺项**

1. 全量到齐后，独立核对 64 分片、1,300 唯一配对、3,900 臂结果、输入/配置身份、实际 gate 覆盖及最终数据绑定。
2. 全量独立评分、任务等权 AVG5、配对差和 Musique 单独评估仍待完成；本轮未重复旧 screen 的评分。
3. 真实 8B Near-only 跨页迁移、原始模型 Q/K 的 profile 对照、适用的生成复现仍缺。本次小张量 CPU 检查不替代它们。
4. 原生模型 parity 标记未在本轮独立执行验证；模型权重完整绑定、历史生成依赖闭包也未补齐。
5. 现有 `verification_plan.json` 是实质验收要求清单，但不是完整形式化合同及六项结构化协议。本轮不补造事前合同，**也不因此否定已授权、实际完成的只读检查**。

`full` 仍是 **TLI gates-off 基线**，不是 dense full-attention decode；节点容量与可见 token 数不支持实际峰值显存或速度结论。

**文件 SHA256 绑定**

以下 `R` 为用户指定本地结果根；`F` 为上述远端冻结源。远端全量 `manifest_rank0.json`：

```text
b29e2989b1f8d379102079d6d1e9020e5aa9620f70be29b16e27f1b03e3d576a
```

| 文件 | SHA256 |
|---|---|
| `R/dispatch_plan.json` | `8aedcafb7fb3346c03d0899196234594f4931097d586a5d6aeb018ffbcca8441` |
| `R/verification_plan.json` | `6bd9754a06f7c820645e60da2df4ce197c36b00da9e0ccd34c80e74f69268fc9` |
| `R/ai_supervision/FULL_VERIFICATION_HANDOFF.md` | `b5599c9317f940661cf5541f0e70f1e12b84fc30fa6a9e4ca351e94b5b15a068` |
| `targeted_boundary_reference.py` | `f482fcf3c5a77e24d137c81ff75a17e61424ecba41a090a2a1c987cb9f2cd039` |
| `targeted_boundary_reference_result.json` | `51e6bb4206d0952eafa9f899bb31e8ce4a6cdd11d5635b8090a9473155d72e0d` |
| `independent_screen/REPORT_access_retry.md` | `534bfc9e9850c09ac84edc2d80bc1bd20de313244ab0e9b99bec14ec285c67a9` |
| `independent_screen/SCORER_FOLLOWUP.md` | `660e2b92df8ea20f4dd901854c5f487b57d7b16fd171dfb603181b6d63430911` |
| `F/sparse_attn/ops/frozen_region_cache.py` | `1d5c46c25d29a907b6e584247b57256347af34cfa4f49d2c1125875c0692daa9` |
| `F/sparse_attn/ops/frozen_region_gate.py` | `a5ae01fea121425f908053821984ebd23c1f51a12f1a4fc18f02a90b0bf2d065` |
| `F/sparse_attn/ops/frozen_region_tli.py` | `62842a6200f45878a994e6bcca11b97ee45b4fed1d3ee4b0fa9787875ceef446` |
| `F/benchmark/LongBench/prefill_frozen_skip.py` | `11daac6077e0985103b0f416c5749ae68ac0df8d1b5e1f259c6ce2cc584260a4` |

表中四个轮次文件均位于 `R/ai_supervision/round_20261029T141704/` 下。重新读取的旧 screen 100 文件清单摘要为：

```text
6404104e4c5fcd711415924b3d125b89df59be1a594d44404db51d6dd23c93ea
```

本报告只允许将所列 **CPU 参考/边界性质记为限定通过**；整体实验及全量独立验收继续保持 `pending`。