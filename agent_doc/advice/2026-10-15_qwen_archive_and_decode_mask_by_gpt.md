# Qwen Observation 归档复用门与 decode mask 契约

来源：用户恢复任务，主论文 Observation 仅 Qwen、MLA 另存诊断；只发布
`gpt_dev/paper-aligned-runtime-20261015`，由用户审核合 main。
对应唯一任务索引中的 `GPT-PAPER-20261015`。

## Qwen：缺失保持缺失

本执行环境未挂载宿主归档。检查 `~/.archive/trace-dumps/qwen3-8b`、
`/tmp/trace/qwen3-8b`、`/home/wangyuanshuo02/.archive/trace-dumps/qwen3-8b`
均不存在。Git 仅有校验和，不能从 27GB/592 文件的历史清单推断本机可用。
没有新增采集、模型权重下载、GPU、付费或实验臂。

新增 `exp/paper/audit_qwen_archive.py`，限定 hotpotqa/musique 各 0/1、层 8/24：
8 个 PT + 4 个 meta.json。其相对路径与原始 MD5 完整列在
`exp/paper/evidence/qwen-reuse-mask-20261015/archive-local.json`，本轮全部 missing。
校验器逐文件流式核 MD5/SHA256，校验后才以 CPU、weights_only=True 读取张量，
核 Q/K/GQA 形状、有限值、S、qpos 和 16 中点锚点。不加载模型，不访问网络，
不输出 question/answer 正文。退出 2 表示缺文件，3 表示身份/结构不通过；
即使退出 0 也只是字节/结构通过，`paper_ready=false`，来源身份仍未验证。

最短执行 advice（仅在已有归档的授权主机上运行）：

```sh
python exp/paper/audit_qwen_archive.py --root /EXISTING/qwen3-8b > /FRESH_PATH/qwen-audit.json
```

先只提供这 12 个文件，勿搬整个归档或重采。继续补 checkpoint/config、tokenizer、
dataset 行/文档重叠、确切输入 IDs/模板和实际 HF SDPA 源码/scale 身份。
旧 meta 的 prompt 标签及短摘录不足以填这些空缺；MD5 也不证明模型来源。
16 anchors 是 **prefill 采样**，不得改称 decode；256 tail 不混入候选子集。

采用补充的两份 Library 设计/审计输入；两份完整在线读取成功，但下载失败，
没有声称本地字节可读或核过所给 SHA256。仓库 collector/索引已独立查看。
已有 e140 actual-forward hook 与 --save-qk 入口保留，不重复造采集器。

后续有合格数据才实现/发布主图：借鉴 ClusterKV 的逐 query dense reference、
固定页边界及等预算集合比较，不挪用其模型结论。先逐 Q-head causal softmax，
再 KV-group 平均；可见 keys=0..q，sink/SWA 各128取并集、页64保留部分页。
优先距离 mass/每有效 token 密度，以及 near/far token/page 的区域内90%集中度；
空/零质量区域为 undefined，保留各 sequence/layer 和贡献 query 数。
四个 source-sequence 的独立性待核，不能把128行当独立样本。
旧 E140 JSON 和 fig1_motivation 聚合均不复用。unpaired random32 不填
paired-random32 占位；无 V 不声称 output-vector 误差或任务质量验证。
此次按缺数据停止条件不生成数值、离线复算图或新实验矩阵。

## TL-E142-DECODE-MASK-GUARD-20261015

确认 main 新的单 token 生命周期修复后，decode 仍忽略任意 mask。
本次在投影和 cache.update 前检查已缓存前缀的 decode 契约：B=1、q_len=1、
use_cache=True；mask 必须 None 或同设备、准确 [1,1,1,S] 形状、strided 浮点
全零张量。padding、bias、NaN/Inf、bool/int、错形状/设备均明确拒绝。
拒绝不修改 cache 或 indexer。首步单 token dense/clear 与 main 异构 KV 守卫保留。
这是输入契约修复，不是恢复 MLA 算法研发。

全零值核验可能触发 CUDA 同步；本次无 CUDA/graph/性能验收，不宣称零开销。
None 仅表达调用方无 mask；不从中推断外部 tokenizer 是否曾 padding。

证据目录 `exp/paper/evidence/qwen-reuse-mask-20261015/`：修前3个测试方法17个
失败子例；修后普通/-O ops套件27项通过（消除了导入测试类的重复收集）。
CPU 测试运行真实 forward 函数，模型/cache/indexer为替身，非真实 HF cache 或
pretrained/GPU 验收。新归档测试5项含 missing、checksum、schema、query身份负例。
扩展 collector 回归中 E141 finite-verifier 因旧 `/tmp/tla-hisa-official/` 外部
源码缺失而失败，日志保留；没有放宽默认源码 hash 门或用替代源伪装通过。

main 合并冲突：保留 dev 修复后的 E141 collector；verifier 合并 main 的 optional
rotate_half 与 fixture-local module table；CLI同时保留 FFD/SparQ门与 MoBA，
SparQ源hash命名保留。原始实验数据和人类 GUIDE 均未改写。
GUIDE只核文件SHA，未读取/输出可能含凭据的正文；目标仓库无 AGENTS.md，
宿主 sglang 两个只读指南路径不存在，按当前明确用户约束执行。

独立只读 reviewer 未发现本轮阻塞：normal/-O 归档/基线/E141 collector/E140
Observation 各33项通过；消除重复导入后 mask+归档各8项通过。完整受影响
回归 normal/-O 各49项通过（含实际随机小型HF Qwen CPU模型；不下载权重），
详见 affected-*.log。缺外部源码的 finite-verifier 失败单独保留在 collector-*.log，
不将这两组日志混称全部回归通过。reviewer 未写代码或推送。
