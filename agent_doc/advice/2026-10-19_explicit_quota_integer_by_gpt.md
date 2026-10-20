# TL-EXPLICIT-INTEGER-20261019：固定配额禁止静默截断小数

## 范围、触发与根因

重新 fetch 的 main 为 `bf9b07994b187cdebfd5070d94b80d27292e73ec`；
`e138-ablation` 为 `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`，仅核验身份，
不合并或借用实验结果。持有本机审计锁，在新独立 worktree 验证；保留旧工作树
的候选及他人文件，不修改人类 GUIDE 或其他仓库。

固定几何的 `nb/nt/nl` 是页数、token 数、token 区宽，不是比例。
`arguments.py:197–210,297–307` 已通过 CLI 类型转换拒绝小数字符串。
但直接 Python 构造 `TLIIndexer(args)` 绕过 argparse：基线
`indexer/tli_indexer.py:381–385,424–425,447` 使用 `int()`，会将
`-0.5` 截成 0，绕过非负与容量检查；正小数也被悄悄截断。
例如 bs=64、K1=4、K2=256、sink=SWA=64、nb=nt=0、nl=1024，
分别将 nb、nt、nl 替换为 -0.5 或 0.5，六个构造均原本成功。
nt 已存为截断整数；nb/nl 原值暂存，消费时再 int，错误不会因暂存原值消失。
这不是 CLI 已能接收小数的声明。

静态路径：`patches/patch.py` 通过 indexer_type_dict 构造 TLIIndexer；
prefill 的 reference/candidate 逐组入口与 decode patch 经 `prepare_mask`
消费同一 indexer。候选代码的 `nt_near_pages/:2187`、
`nt_near_tokens/:2572,2608`、`nt_near_len/:625,1105,1526,2119`
均用整数配额。实际模型派发未运行，不能将静态可达当设备验证。

## 修复与兼容边界

`sparse_attn/indexer/tli_indexer.py:384–402` 在构造期统一规范三项配额。
整数协议使用标准库 `operator.index`；保留原有整数字符串及 nt 的大小写/空格
宽容 off 哨兵。None 仍交给已有几何齐备性守卫；原有非负、总预算、容量与
比例/几何互斥守卫保留。浮点（包含 1.0）、原生 bool、NaN/Inf 等非法输入
改为带字段名的 ValueError。直接 Python 调用者若传 1.0，应明确改传整数1；
不再以 int() 的宽松转换默默改变调用者输入。

选择严格整数协议而非 round/floor：后两者会改变配置，且不能恢复计数类型合同。
不新增依赖；仅构造期常数次 host 操作，不进入每 token 热路径。
未改合法整数配置的算法、预算、归一化、topk ties、GQA、保护区、chunk2048、
tile、量化、缓存、streams 或 kernel。整数字符串和 __index__ 对象现在统一
存为 Python int；本次只承诺构造器兼容，不扩大整个 CLI/结果命名支持类型。

## 实际验证

新增 `tests/test_explicit_quota_integer.py`，沿用完整构造函数 AST 加载器：
执行 Indexer→TIAIndexer→TLIIndexer 全部构造体，metrics 工厂替身；
Torch 任意属性访问即失败。不是完整模块导入，不是张量/选集/GPU 模拟验收。
覆盖三字段×八类非法值、整数/整数字符串/__index__、零/内部合法几何、
off 多种拼写与缺项。测试文件 SHA256：
`1471684f6eaeb00bd5638f1505c9d4257e1ef2671adb2b2eae29a95bc9a479a5`。

```text
python -m unittest discover -s tests -p test_explicit_quota_integer.py -v # 修前
Ran 3 tests; FAILED (failures=21, errors=3)

python [-O] -m unittest discover -s tests -p 'test_explicit*py' -v
Ran 10 tests; OK                         # 普通/-O各一次
python [-O] -m unittest discover -s tests -p test_partition_ratio_contract.py -v
Ran 3 tests; OK                          # 普通/-O各一次
python [-O] sparse_attn/ops/tests/test_batch_explicit_quota_guard.py
Ran 6 tests; OK                          # 普通/-O各一次
python [-O] agent_doc/results/explicit-quota-integer-20261019/verify.py
6 truncated fractional inputs before -> 6 rejected after
108 identical controls (90 accepted, 18 rejected) # 普通/-O一致
git diff --check                         # 通过，无输出
```

`verify.py` 从 git 取基线构造函数，与候选比较八个相关实例字段或错误类型/文本；
81个整数几何配置（block32/64/128，K1=1/4/128，nb=0/1/4，nt=0/32/off）
和27个 legacy 比例配置不变。不是整个对象字典、不是 selector 输出比较。
非独立代理复核；该差分与前述单测均由本执行者运行，预期来自整数计数合同。
输入矩阵 SHA256：`f12004f7b402a9eed2dc1c070ef343fc46b20889f326f23dbcbdc36c0e07d76a`。
候选生产文件 SHA256：`1ac64c9c4fa1d5c1885a29ed2c481edfab003f1266b6199daa079c29726a17e1`。
小型原始失败/通过摘要及六个前后实例见同目录 `host-evidence.json`。

环境：Linux x86_64 容器，AMD EPYC 9V74，9逻辑CPU，Python3.12.14 /
Clang22.1.3。当前未安装 Torch/Triton，PATH 无 nvidia-smi；GPU架构、驱动、
CUDA编译器、JIT/kernel配置无本轮验证值。未测模型质量、速度、TTFT、峰值内存、
kernel数或host同步数；此为恢复输入合同的正确性修复，不宣称性能提升。

## 历史影响和后续任务

仅需排查直接 Python 构造传小数/布尔等计数的历史调用；本轮未证明任何历史
实验触发，不更改既有数据。CLI 正常整数配置不因这项问题自动失效。
非法配置应纠正为预期整数后重测，不能以旧截断结果代表原请求。

完整依赖导入、真实 prefill→decode 与 GPU 回归仍待可用环境。此前
TL-EXPLICIT-NEAR-WIDTH 的子页区宽修复仍仅留档，未趁本次合入；全前缀L2、
显式配额批量选择和 compact attention 的收益仍需同语义整链实测。
复用 research-implement-optimize 技能：先红证、最小 host 修复、普通/-O差分，
无设备不晋升 kernel 候选；本轮无需外部算法替换或新增工具。
