# TL-VENDOR-FROZEN-MANIFEST-20261020：新增 vendor hash 重读已冻结源

基线 `f622c907e5af2be7ffa674e5bcb64936c395306b`；2026-10-20。仅本独立仓库，固定配额与kernel未改。

## 问题与实际路径

`sparse_attn/info.py:resolve_treatment_snapshot` E155e新增vendor文件身份分支直接重开原始文件。
但MInference在LongBench `load_model_and_tokenizer` / RULER加载阶段已经调用`freeze_minference_pattern_sha8`；FASA已经`stage_fasa_agreement`。两个入口只有TLI传递完整TreatmentSnapshot，vendor后续名字/sidecar/receipt仍现场解析manifest。

触发：加载冻结A → 原始文件替换为B/删除 → 名称与manifest生成。旧`_cfg`/`_agr`段仍代表A，新增hash/manifest却代表B或null，甚至抛出源缺失错误。运行内容与结果身份失配；旧数据是否受影响必须核对实际源变更与冻结内容，不能一概判为污染。

## 修复

manifest优先消费已有MinferencePatternSnapshot/FasaAgreementSnapshot的完整sha256；freeze/stage同时保存解析后的basename，防后续符号链接替换改变名称元数据。未冻结的独立身份查询继续读取当前源并保留缺失拒绝规则。TLI固定配额及其他vendor的manifest路径不变。

## 验证与边界

- 修前5项：2失败、2错误、1通过；日志`agent_doc/results/vendor-frozen80/before.txt`。原内容hash为`1fa96eaaf7cd5c81782e3c973bb3d4f4c191cdbb5821d9cd4f5fcce9ddb1b8dd`，替换内容hash为`65fb8a77a8202b314c61cd0c62b5a75dd67200acac01a94c44cb6795f2d79e73`。
- 修后扩展7项，普通/-O各通过；覆盖两vendor替换/删除、未变源、符号链接替换、未冻结内容变化与缺失拒绝。保留修前日志，不改历史实验。
- 固定配额`test_explicit*.py`普通/-O各17项通过；batch显式配额guard普通/-O各6项通过。既有E155e vendor身份7项通过（同下述无Torch加载方式）。
- 当前Python3.12.14/Linux x86_64，无Torch、无nvidia-smi。本轮执行整份info.py AST，仅去除torch import并注入禁止任何tensor访问的哨兵；不替代Torch导入、模型加载或GPU验证。实际执行生产元数据函数、真实临时文件冻结/staging、SHA/名称/manifest比较。测试入口`python -m unittest discover -s tests -p test_vendor_frozen_manifest.py -v`。
- 独立复核见`agent_doc/results/vendor-frozen80/independent.md`。无速度、显存、质量收益主张；仅消除已复现身份漂移。不存在性能测量输入或kernel配置，不编造GPU数据。

## 固定配额后续

已有TL-EXPLICIT-NEAR-WIDTH-20261019三行候选继续待实际Torch/GPU选集验证；不将host表达式复现作为生产kernel验收。T7稳态bench仍需已授权空闲独占GPU窗口，本轮未触碰在飞任务或远端GPU。保留外层chunk2048和全部预算契约。
