# 独立 verify_experiment_result — RULER MultiKey2 64K

结论：**formal100 = partial pending，未获全量科学消费许可。** 本会话实际执行 CPU 核验，最新固定副本包含 **9/100 条完整记录（ID 0–8）、27 个臂、972 个请求层回执**。这些已固定记录的输入身份、分数、停止协议、门控与缓存回执检查通过；只可消费为局部审计证据，不可用于部分样本质量结论、100 条完成声明、无损或性能结论。

审查会话：用户显式授权的独立会话 `independent_review_20261030T0243`；与生产者分离。本报告不冒用历史 `/root/verify_pool_initial` 的身份。PROJECT_ROOT 为 `/home/wangyuanshuo02/two-level-attention`，源码只读工作树为 `/home/wangyuanshuo02/two-level-attention-frozen-mk2`。使用 gpu-experiment-dispatch 技能和 project_document/result_reuse/result_validation 契约。人工指南只读；用户本轮禁止监督器、派发及进程控制的约束优先。只写当前审查目录，全部 exec 使用 login=false。

## 固定证据和实际执行

最终依据为 `snapshot_20261030T024846618545Z/`；时间戳表示快照任务开始时间，各文件随后逐一读取，不假称全目录在同一瞬间原子冻结。只对 status=complete 的记录保存原始字节，读取前后字节相同才纳入；未完成样本未复制到验收集合。最终快照捕获 ID 0–8，尚缺 ID 9–99。生产可能继续增长，本报告不绑定活日志或活目录状态。

早期快照 `snapshot_20261030T024601522882Z/` 保留 7 条完整记录，排除了当时 running 的 ID7；最终快照是新增证据，并未覆盖早期记录。两轮都读取并重建全部100条输入；没有把烟测重复计入正式样本。两个快照都保留 CAPTURE_BINDINGS.json，记录原始路径和副本 SHA256。

实际运行命令（CUDA_VISIBLE_DEVICES=''、PYTHONDONTWRITEBYTECODE=1）：

```bash
/home/wangyuanshuo02/two-level-attention/.venv/bin/python -B /home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_qasper_mk2_64k_20261030/independent_review_20261030T0243/check_independent.py
/home/wangyuanshuo02/two-level-attention/.venv/bin/python -B /home/wangyuanshuo02/two-level-attention/agent_doc/results/frozen_qasper_mk2_64k_20261030/independent_review_20261030T0243/cpu_boundary_checks.py
# cwd = 只读源码工作树；禁用 pytest 缓存
/home/wangyuanshuo02/two-level-attention/.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_frozen_ruler_protocol.py tests/test_frozen_region_cache.py tests/test_frozen_region_tli.py
```

独立脚本直接调用冻结 `validate(..., allow_partial=True)`，目标是审查目录中的副本，未向生产目录写 validation/summary。烟测副本也实际运行冻结 validator。两次独立检查、CPU 边界程序、pytest 均退出0。环境记录见 REVIEW_ENVIRONMENT.json；测试结果32 passed。新写的 CPU 边界程序完成252次实际64K长度缓存逐元素核验（4种门控×63步）、108种阈值/区域存在性组合、8种真实小型CPU Qwen forward 的停止案例。小模型使用随机权重和强制 logits，仅验证机制，不是 Qwen3-8B 质量实测。

## 六项验收及范围

| 契约项 | 本轮已通过 | 尚未检查/未完成 | 全量状态 |
|---|---|---|---|
| implementation | 阅读 RULER 入口、生成循环、post-RoPE profile、独立门控、cache迁移、Qwen路由和TLI adapter；32项CPU回归及额外边界检查 | 实际生产全部层的Q/K张量、每一步mask/KV内容没有保存，无法独立复算全部执行 | scoped pass；完整GPU语义不能认证 |
| reference_boundary | 全4种门控的CPU缓存token位置/K/V内容参考对照；严格小于.05；EOS/newline首token忽略停止、后续停止、上限截断；烟测原始native parity回执存在 | 没有第二次GPU生成；native侧独立token轨迹未单独保存，不能把boolean回执当本会话重新生成 | 部分参考通过，GPU重生成未检查 |
| data_integrity | 全部100行原始JSONL、唯一ID和选行顺序；100行token长度/哈希重建；9条完整记录映射；78份源码hash与commit对象逐份一致；模型配置三份hash | 91条正式样本尚缺；全部权重未哈希；生产时tokenizer资产未全部在manifest绑定 | inconclusive（full100） |
| numerical_sanity | 27个输出重新decode及独立不区分大小写子串命中分数；972层strict gate/守恒；冻结validator的finite、profile一致性、byte检查；统计合成夹具通过 | 正式100条均值、CI、升降/相同计数尚未计算或验收 | partial pass；全量pending |
| measurement_validity | 已完成样本三臂同输入/协议；K1=128/K2=1024/tail32、YaRN2、batch1、BF16、raw prompt无chat无截断；统计按请求配对 | 9条不是预定100条完整样本；不能质量外推；无计时/性能测量验收 | inconclusive（full100） |
| reproducibility | CPU独立tokenization、decode/score、门控/几何复算；源码冻结validator重跑；可重跑独立脚本和固定依赖 | 无完整独立GPU复现、无全模型权重认证、无全部100条结果 | scoped CPU pass；全量pending |

**检查失败：本轮实际执行的检查未发现失败或重大bug，未生成 FINDINGS.md。** 这不是绝对无bug声明。未检查项目没有标成通过或N/A；本协议六项均不允许用N/A绕过。

## 身份、协议和输入

正式 identity：`dd0e84bc766e7131d0ef12f935535693eaf28821a05fc09bc84af7b4dfea9214`。
源码 HEAD：`de2a1c1670b82bebcde6c0c3aa02fb8ee1431568`，开始和检查结束关键源码hash匹配，工作树status为空。78份manifest源码与Git提交对象逐一匹配。run_contract绑定的validation_plan SHA匹配。独立脚本另保存tokenizer实际文件与必要验证依赖；这只能固定本次检查字节，不能追溯证明生产时未绑定文件始终相同。

数据：`niah_multikey_2.jsonl` SHA256 `52b587a924defa07d36091824e3128310f657bca34a4f011c062879e757c5063`。全100条按原顺序选择0–99，无重复、无重采样。实际读取全部原始input/outputs并逐行tokenize，不只是检查首行。每条长度在60000–65536，输入+64不超过YaRN有效81920。所有重建长度/哈希/answers见最终快照 `all100_inputs.json`；9条成品与之精确对应。

停止ID `[151645,198]`；max_new_tokens=64，stop_after_tokens=2，首个token不因EOS停止但也不屏蔽EOS；之后首次EOS或newline结束。所有27条输出长度2–64、内部无应已停止的token、结束位置合法，与重新decode的文本一致。三臂首token一致。评分独立实现为 `命中答案个数 / 答案总数`（逐答案lower后子串匹配）；保存的score为0–1，报告统计转换为0–100，差值单位pp。

主比较是 Dual0.05−Far0.05；full是 **TLI gates-off** 辅助臂，不是dense FullKV。标准预算tail32维度为48–63及112–127。RULER入口关闭operator、fine gather、sparse prefill、dense decode和CUDA graph优化。Near和Far分别以末256 query的head-mean profile与strict .05比较；不合并为joint budget。近区缺席不触发，profile不随decode更新。

## 原始触发与缓存回执（仅覆盖审计）

| 已验收副本内臂 | Keep层 | Near-only层 | Far-only层 | Both层 | Near触发请求 | Far触发请求 |
|---|---:|---:|---:|---:|---:|---:|
| full | 324 | 0 | 0 | 0 | 0/9 | 0/9 |
| far_0.05 | 324 | 0 | 0 | 0 | 0/9 | 0/9 |
| dual_0.05 | 287 | 37 | 0 | 0 | 9/9 | 0/9 |

37个Near-only层全部保存attention不可见的near_staging，未谎称省resident KV，全部观察到Far token增长；stored+dropped=logical、attention排除staging、各区域token几何和capacity/bytes检查通过。27臂profile与对应full的最大差为0。最后一步TLI scorer/selector名称、tail维度、所选KV head token数、逻辑/可见长度与final cache匹配。这些是最后一步原始回执的一致性，不是每步内部实际attention mask的独立记录。

真实64K已完成样本无Far-only/Both触发，不能认证这些分支已被生产结果覆盖。CPU合成边界覆盖这两种状态，但不能替代真实模型的触发覆盖。烟测为独立的1条/3臂/108层，Near-only6层全部Far增长；其native parity回执仅作源代码实施过比较的记录，未独立GPU重生成。

## 全量统计脚本与消费门禁

`check_independent.py` 可在相同授权边界内重跑，每次创建新的snapshot子目录。未满100时不调用summarize、不发布正式样本均值或CI。满100时要求 GENERATED身份和样本数正确、完整唯一100条/300臂和全部CPU核验通过，再执行冻结validator与summarize，并独立复算以下三组统计：Dual−Far、Far−full、Dual−full。

使用配对请求差值pp，B=10000，seed42，percentile [2.5%,97.5%]；三组按冻结实现顺序共享同一连续RNG流。逐次独立重采样均值与冻结向量化结果比较。核对各臂均值、CI、improved/degraded/same_score、same_generated_ids和门控层数/触发请求覆盖。`cpu_boundary_checks.py` 已在非退化100行合成夹具上实际验证这三组统计；`synthetic_statistics_fixture/summary.json` 明确仅为程序测试，不可冒充正式结果。

本轮正式数据未满100，因此未来全量分支尚未在真实100条上执行。脚本即使未来通过也只写“CPU checks passed; reviewer admission required”，独立审查者仍需检查新证据和限制后作新验收。本报告不预先验收未来数据。用户3pp容忍界只可在完整数据上作预设工程描述；CI含0不能解释成等价/无损。

## Qasper、未验证项与交接

只盘点 `frozen_near_far_v1_16_independent005_20261030`，最终固定inventory观察到43/200完整Qasper记录（早期41）。此处inventory是逐文件读取时的状态记录，不是Qasper输入/分数/权重验收。本轮优先RULER，未重新计算Qasper200分数，未复制旧full5、未混入旧样本、未重复派发；Qasper专项结论继续pending。既有reuse_decision适用于避免重复生产；旧报告不替代当前200验收。

没有GPU调用、GPU重新生成、派发/停止进程、创建监督器或修改共享源码。本轮未读取并hash所有模型权重；manifest主要绑定config、tokenizer_config和generation_config，不能据此认证8B权重。未保存真实Q/K或全量attention probabilities，无法独立复算生产attention mass；仅检查其数值域、跨臂一致性和阈值决策。native parity不是dense质量基线。无GPU性能、端到端速度、峰值显存或普遍无损结论。

可消费scope：固定9条RULER的CPU输入/score/协议/回执审计、CPU机制与统计程序回归证据。不可消费scope：formal100科学结果、当前Qasper200专项结果、未触发真实分支泛化、attention mass真实性、权重身份认证、速度/显存收益、统计等价。

稳定证据由同目录 `EVIDENCE_SHA256.json` 逐文件绑定，包含本报告、固定副本、检查脚本和已结束CPU输出；排除活session.log、生产活日志、CLI FINAL_RESPONSE.md及哈希清单自身。不存在把生成原始数据或机械自检升级成完整独立验收的授权。
