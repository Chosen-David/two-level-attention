# Sub/sub：32 个有效评分坐标与物理张量宽度的历史绑定

- ID：`DAC-SS-ACTIVE-COORDINATES-PHYSICAL-WIDTH-20261015`
- 日期：2026-10-15；关联现有任务 S-T022/E134。
- 范围：公开仓库静态源码与已提交 sidecar 的来源说明；不修改实验数字、代码、指南或任务状态。
- 核查基线：main `bb3991f1bf982f8be491bb8a4b907af8e1214a74`；开发分支 `6f1848ad9111ee8096bd6ef3b3020ddfbf2e1d24`。

## 与已有记录的增量

[已归档 E134 判决](https://github.com/Chosen-David/two-level-attention/blob/6f1848ad9111ee8096bd6ef3b3020ddfbf2e1d24/agent_doc/advice/archive/2026-10-13_0050_E134_subspace_matrix_verdict_by_deepseek.md#L35-L44)和 [S-T022](https://github.com/Chosen-David/two-level-attention/blob/bb3991f1bf982f8be491bb8a4b907af8e1214a74/agent_doc/task/task_details/S-T022.md#L34-L57)已记录旧的零填充 128 宽实现、后续紧凑化及性能重判。本条不重复要求实现紧凑 32 维；只补充五份 sub/sub 质量运行回执与旧物理布局的明确绑定。39 个活跃 advice/README 路径及相关 E134 归档中，未发现同一五回执绑定说明。

## 源码事实与适用分支

质量回执共同记录 producer `35b39bd8e43cf478bb4f0ab64a8457f08d87c864`。该提交的 [tli_indexer.py](https://github.com/Chosen-David/two-level-attention/blob/35b39bd8e43cf478bb4f0ab64a8457f08d87c864/sparse_attn/indexer/tli_indexer.py#L499-L508)，Git blob `f57c7e961f728d1352476e886ec195d81119c066`：
- L273–295：tail、cmp_ratio=4 选择 `[48..63,112..127]`，共 32 个坐标；L1 对选中坐标建统计，属于紧凑 L1。
- L499–508：非投影 L2 先 `zeros_like(k)` 保留 128 宽，再把这 32 个坐标的量化值散写进去；L337–340、L390–400 的增量缓存同样保留 128 宽。
- [L1034–1082](https://github.com/Chosen-David/two-level-attention/blob/35b39bd8e43cf478bb4f0ab64a8457f08d87c864/sparse_attn/indexer/tli_indexer.py#L1034-L1082)：非投影 Q 保持全宽；Triton 路径显式检查 K 宽 128，回退路径也做全宽收缩。故“32 个有效评分坐标”不等于“历史 L2 张量物理宽 32”。

五份 treatment manifest 共享 blob `cd5bdef9fdd938b4e9dd981e94c33cbaed856c85`，明确记录 tail、cmp_ratio=4、enable_subspace=true、proj_basis=null、static_pair=false、layer_skip=false、moba=false、near/far_select=4bit。它们对应上述普通非投影分支，而非可选投影或 gate 分支。每份 manifest 与下面回执同 stem，后缀为 `.jsonl.tli_manifest.json`。

## 五份质量回执绑定

以下链接均固定到已核查的 main SHA；回执共同记录上述 producer，并绑定原预测文件：
- [gov_report，cap 62c6940f74](https://github.com/Chosen-David/two-level-attention/blob/bb3991f1bf982f8be491bb8a4b907af8e1214a74/exp/trace/results/e134_l1sub/pred_mavg_l1sub/gov_report-tli_64_128_1024_c4_Aa0.25_b0.125_g0.625_h0a1708cf4e-1012175451.jsonl.runcap.json)：receipt blob `1f4033575dede7c275a9ba7dd69becb2567df1b5`；预测 SHA-256 `0f5810deca8abc1b6527144f6bfa2e76dd25ceec80037adf63fe1d8eba2aafdc`。
- [hotpotqa，cap dda5feef09](https://github.com/Chosen-David/two-level-attention/blob/bb3991f1bf982f8be491bb8a4b907af8e1214a74/exp/trace/results/e134_l1sub/pred_mavg_l1sub/hotpotqa-tli_64_128_1024_c4_Aa0.25_b0.125_g0.625_h0a1708cf4e-1012180214.jsonl.runcap.json)：receipt blob `6c47664b2538a22b4e2f39e5a29ea59400decf12`；预测 SHA-256 `5af21b846142f93dc2f940dd3e6977482b73c48a2558c9b90274ac8c7ae1ec07`。
- [musique，cap cd90f25f17](https://github.com/Chosen-David/two-level-attention/blob/bb3991f1bf982f8be491bb8a4b907af8e1214a74/exp/trace/results/e134_l1sub/pred_mavg_l1sub/musique-tli_64_128_1024_c4_Aa0.25_b0.125_g0.625_h0a1708cf4e-1012180214.jsonl.runcap.json)：receipt blob `eb6840a5fef691aaa360c5e2512a6a8f4432f5eb`；预测 SHA-256 `971b686aef480100324a12dfcbedd3e5a3ee809c098adb9470d8563b58e342f1`。
- [qasper，cap a224375bf0](https://github.com/Chosen-David/two-level-attention/blob/bb3991f1bf982f8be491bb8a4b907af8e1214a74/exp/trace/results/e134_l1sub/pred_mavg_l1sub/qasper-tli_64_128_1024_c4_Aa0.25_b0.125_g0.625_h0a1708cf4e-1012180214.jsonl.runcap.json)：receipt blob `6f54671a1f5ca262f6dc8f33c1fd56d9f9f08b03`；预测 SHA-256 `6a805c2a512fe086cac4b7f2a284b5a762c1c7b7a7c63e7da6200b026a88218c`。
- [repobench-p，cap 4643ba839a](https://github.com/Chosen-David/two-level-attention/blob/bb3991f1bf982f8be491bb8a4b907af8e1214a74/exp/trace/results/e134_l1sub/pred_mavg_l1sub/repobench-tli_64_128_1024_c4_Aa0.25_b0.125_g0.625_h0a1708cf4e-1012175451.jsonl.runcap.json)：receipt blob `0e014327a50658968072705157143c4f80ad87bd`；预测 SHA-256 `d62fca03c04aa94f68999abd5c19424a65725c62383ab508735a507f63459f57`。

五份回执的 producer script SHA-256 均为 `235cc092a456e4e913a6453c6bc06e9f4351d2c506bb7747d7fcd358f9a7e5bb`，与该 producer 提交中的 [benchmark/LongBench/pred.py](https://github.com/Chosen-David/two-level-attention/blob/35b39bd8e43cf478bb4f0ab64a8457f08d87c864/benchmark/LongBench/pred.py)实际字节相符。这里只核验回执绑定与已提交源码；不据此证明运行时工作树无修改，亦不声称重新计算了全部预测字节或评分。

## 后续紧凑实现与结论边界

[后续实现](https://github.com/Chosen-David/two-level-attention/blob/2c393c5615a66554729ccd00202a45c173582f2f/sparse_attn/indexer/tli_indexer.py#L633-L637)固定提交 `2c393c5615a66554729ccd00202a45c173582f2f`、blob `5f4e255cb2d4b8098a770a9e412bd8211211f215`，在 tail32 非投影、subspace-mode L2 中直接生成紧凑 K，并在 [L1216–1221](https://github.com/Chosen-David/two-level-attention/blob/2c393c5615a66554729ccd00202a45c173582f2f/sparse_attn/indexer/tli_indexer.py#L1216-L1221)使用相应紧凑 Q。[E134 性能重判](https://github.com/Chosen-David/two-level-attention/blob/bb3991f1bf982f8be491bb8a4b907af8e1214a74/exp/trace/results/e134_perf_reverdict.json)（blob `91e9fe7782cda0c8b48c58e9dbe3b317ca87ba60`）记录的是这一后续来源，不能反向证明五份旧质量运行物理紧凑。

当前 main 的同一实现文件 blob 为 `82a40be369bb7bddd05602d7dce48a417d99d609`，[L857–861](https://github.com/Chosen-David/two-level-attention/blob/bb3991f1bf982f8be491bb8a4b907af8e1214a74/sparse_attn/indexer/tli_indexer.py#L857-L861)与 [L1441–1446](https://github.com/Chosen-David/two-level-attention/blob/bb3991f1bf982f8be491bb8a4b907af8e1214a74/sparse_attn/indexer/tli_indexer.py#L1441-L1446)仍有紧凑 tail32 路径；显式 full-L2 与兼容回退存在 128 宽例外，不概括为所有路由均紧凑。

处理建议：沿用 E134/S-T022 的既有记录，只把历史质量来源标注为“32 个有效坐标，L2 物理 128 宽零填充表示”，把紧凑布局限定到相应实现版本。它不应被改标为 sub/full，不使已有任务均值失效，也不构成跨实现位级等价、后续成本与旧质量可直接合并的证据。下一步优先复用身份匹配的现有预测、producer/输入/scorer 回执，补齐报告中的来源对应；本条不申请新矩阵或重复运行。
