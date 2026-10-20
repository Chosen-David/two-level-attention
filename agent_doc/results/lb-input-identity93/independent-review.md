# LB-INPUT-IDENTITY-93 独立只读复核

2026-10-20；基线 `5276987c8df987d0620d023cded19a7a8b78e396`。
结论：**确认 v1 数据身份遗漏格式化 prompt 的正文来源**。这是可复现的身份契约缺口；未证明历史实验受污染，不是 GPU/质量或性能结论。生产身份修复尚未实施、未验收。

## 实际链路

- `benchmark/LongBench/pred.py:909–914`：仅对逐行 `answers + length` 的 JSON 求 SHA256 前12位；不读取 `input/context/all_classes`。
- `pred.py:247`：实际模板使用 `prompt_format.format(**json_obj)`；官方 qasper 模板消费 `input/context`。`:448` 的 v1 `_id` 使用 `dataset:data_fp:row_idx`，`:955` 将同一 data_fp 传入 run-cap。
- `run_cap_manifest.py:193–216,289–306`：cap identity 仅沿用这个 dataset_fingerprint，不重新绑定正文。`:116–139` 的 producer 身份是脚本 SHA/commit/path，不是数据内容身份。
- 结果文件 SHA、行数与 sidecar 校验保护**预测产物字节**；input_tokens 记录计数，不能代替输入内容身份。可选 eval manifest 验 ID/answers hash，也不补正文。`analyze_e155d_grid.py` 的 ID 集合和暂存字节绑定不读取原问题/上下文，不能在事后补出遗漏字段。

**相反证据与覆盖限制**：库内确有可复用输入hash机制。`exp/paper/receipt.py:83–96`的`QwenRequestCapture`把material/question token拼接为little-endian int64并记录`input_ids_sha256`；`pred.py`可将receipt写入`paper_request_capture`。但`--paper-capture`默认off，timing仅支持Qwen3 manual路径并拒绝samsum/DeepSeek generate等路径；本次已核对eval/grid分析器均不消费`paper_request_capture/input_ids_sha256`。因此“默认v1样本/CI身份未闭合”成立，不能泛称全库没有inputhash或有receipt的全部数据也不可补证。未审计外部数据管理流程，不声称外部冻结数据不存在。

## 独立执行

只使用标准库：AST提取实际 `_fp_src`、data_fp、v1 `_id` 和模板格式化表达式；读取官方 `config/dataset2prompt.json` 的 qasper 模板，实际调用纯标准库 `cap_identity_hash`。没有导入 pred.py 的 Torch 栈，没有读取 guide 凭据段/collector，没有联系历史主机。

两行均为 `answers=["Alice"], length=100, all_classes=[]`：A=`input="Who won?", context="Alice won."`；B=`input="Who lost?", context="Alice lost."`。其余固定相同。

```text
fingerprint_equal = True; fingerprint = ab202ae83eaa
sample_id_equal = True; sample_id = qasper:ab202ae83eaa:0
different_formatted_prompts = True
prompt_sha256(A) = 13f30ec0d38910093e9a4dc71b167da9895bfa06ffe78263d5271e63a1da67da
prompt_sha256(B) = d50fbdf4c02842338462ce078adfd1910ef2967e8abe2a15a90efdf071c5bdd4
cap_identity_equal = True; cap_identity_hash = bdc214da0c
```

cap 参数固定为 Qwen3-8B/qasper/tli、timestamp=fixed、effective/protocol cap=32768、type=protocol、max_new_tokens=128。这里是格式化后、tokenization/截断/chat-template 前的文本，未执行真实 tokenizer 或模型；不能称为已测模型输出差异。旧指纹的相同不是12位截断碰撞，而是 hash 输入本来相同。

另独立读取并重跑主控 `probe.py`：AST提取各表达式必须唯一，否则报错；官方模板实际加载。普通及`-O`输出均逐字节等于`observed.json`，4反例/3对照均符合源代码。行交换例限定为两行answers/length投影相同的实际fixture，不泛称任意行交换都保留指纹。代码使用显式异常而非依赖assert，`-O`未移除探针判断。

- probe SHA256：`dc4fdaa18d92556f0a5501a5d100af23fcad4b46cd081f924f6eccee17323bd4`
- observed SHA256：`0039612a6d3d9162e8eb97b9e9937c432edaf55739324ffe074dd47eaff9c73e`
- 合成输入 SHA256：`b6e7faaca3b44c4a1e923cb28deb9819b35dfb67c413fc7c6f488c7984ff0ede`

| 实际读取源 | SHA-256 |
| --- | --- |
| pred.py | `b54693249fb1648979301f476556784cb3b81ba5a053437f255b06562643d2f3` |
| run_cap_manifest.py | `dec2f18bd45ac56c40a215311b0064706c485fd2d057036811a4be1bb78e371d` |
| eval.py | `e6636fe34c7b400c65903611fc5641af57a6912962a855e9c76509d2f203583e` |
| analyze_e155d_grid.py | `c719089b3c4e279abc1a42430d116802b901929054f936ec52100abbd4f44658` |
| exp/paper/receipt.py | `d8da614deeb1ce9949f16dfd183b8a79c99ecb19e5b133878b2a0a146d8f97cd` |

## 后续修复门槛

宜定义显式 v2 输入身份：冻结实际数据字段/行序及模板来源，逐例正文与评分目标绑定，再明确它与截断后token输入身份的关系。优先核对并复用现成`input_ids_sha256`编码规范，按source/render/tokenizer/chat/截断层次绑定；不能只给旧指纹加长度。写入`_id`/run-cap/消费者的版本规则必须一致。不能为补身份直接强开timing采集：其hook/计时扰动及模型路径覆盖都需独立验证；有receipt者可定向核查，off历史仍不能自动补出token身份。

保留旧 IDs、sidecars、原始测量，不追溯重写为“已绑定输入”。默认不混用 v1/v2；能找到原始冻结输入者按明确迁移协议补证，找不到者标未知，不能从预测文本猜回问题。新采集使用新身份/路径并复核缓存，避免旧产物误复用。

必要回归：正文/问题/类别/答案变化各自失效，合法相同数据稳定，行序/Unicode/换行与模板变更的约定明确，采集后替换源不改变已消费绑定；真实 producer→sidecar→eval→paired CI 需对应版本门禁。当前只读反例不代替生产依赖、tokenizer/模型或历史冠军重评。

本角色仅新增本文；没有修改源码、TASK、guide、既有数据或其他报告，没有提交/推送。无“历史结果全错”或“必须重跑全部 GPU”的结论。
