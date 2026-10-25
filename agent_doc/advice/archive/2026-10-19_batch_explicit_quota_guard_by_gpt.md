# TL-BATCH-EXPLICIT-QUOTA-20261019：显式 near 配额误入旧批量公式

## 问题与修复

基线 `a2ef9163796e822379b3f592b1aeeea431039436`。
E155 在 `sparse_attn/indexer/tli_indexer.py:359-370` 新增显式 near 页数、
token 数与窗口长度；`:2000-2002`、`:2068-2072`、`:2453-2457`、`:2489-2493`
分别消费覆盖值。与此同时，`sparse_attn/ops/batch_select.py` 基线
`:369-380`、`:439-443` 仍仅按 α/β/γ 计算，`:239-282` 门禁没有排除新配置。

触发：candidate grouped prefill，启用 `TLI_PREFILL_BATCH_SELECT=1`，其余
batch 门禁满足，且显式 near 配额已设置。构造器甚至会为显式配额把非正
α/β补成1，因而不能靠旧 `alpha>0 and beta>0` 门禁排除。
默认 batch 开关为0；本问题不是所有默认请求均受影响。

一个公式差异例：S=1024、block64、sink64、SWA64、K1=4、K2=256、
α=β=.25、γ=.5，显式 pages=2/tokens=16/len=128。
旧 batch 使用 near 长224、near页1、near token名额32；参考使用128、2、16。
这是源码公式反例，不是 GPU 实测选集或模型质量数据。

发布前main推进到 `1814ea932641afe3e930e6339677bd51472a2033`，新增E155d
构造期几何/比例互斥与约束。已重放本次变更并复查：batch及caller未变化，
问题仍在。上述混配例仅对应初始基线；最新合法输入应仅传pages/tokens/len，
构造器将运行时α=β=γ归一为1。相同S/K1/K2/保护区、显式2/16/128时，旧
batch公式变为near长896、near页4、near token名额128，参考仍为128、2、16。
新增测试直接使用该运行时元数据；不声称混配CLI能通过新构造器。
`nt=off`则归一gamma=None，旧门禁已拒绝，不作为本次新增缺陷。

修复仅在 `batch_select.py:262-268` 加入三个 `is None` 条件：任何显式值
（包括0）均在张量/索引/graph工作前回退，保留缺字段旧实例与全None配置。
`eager_prefill.py:480-492,535-540` 已有逐组回退，再由 `:462-464`
`prepare_mask` 消费新语义；required-graph入口则明确报错，不能伪装测速成功。
未改变外层chunk2048、算法选择粒度、预算、sink/SWA或kernel。

独立只读复核确认上述调用链。没有额外禁用 boundary_bypass：当前门禁要求
sink/SWA整块且入口要求group_end整块，两个跨界余数均0；复核枚举3072组
整数几何无反例。l2_quant由共享prepare_index生成k_qat，batch直接消费，
未发现需另加门禁的依据；这不是该精度开关的GPU等价性认证。

## 实际验证与未验证边界

新增 `sparse_attn/ops/tests/test_batch_explicit_quota_guard.py`：标准库解析并
原样编译三个生产函数AST，使用CUDA形状元数据；任何torch属性访问硬失败。
不导入/替代生产selector，不生成张量，也不把模拟CUDA元数据称为GPU测试。
覆盖三个字段单独非None、显式0、合法成对配置、缺字段/全None对照、公共
回退、required graph错误，以及boundary/quant开关门禁未被扩大。

初始基线5项测试修前24子用例失败；加入E155d合法运行时配置后，使用最终
6项测试、从最新origin/main读取修前函数AST再次执行，失败25子用例。
最终修前/后短日志（生产函数提取机制相同，仅修前使用git show文本）：

```text
python sparse_attn/ops/tests/test_batch_explicit_quota_guard.py
Ran 6 tests
FAILED (failures=25)   # 修前：原24项 + E155d归一配置

python sparse_attn/ops/tests/test_batch_explicit_quota_guard.py
Ran 6 tests
OK
python -O sparse_attn/ops/tests/test_batch_explicit_quota_guard.py
Ran 6 tests
OK
git diff --check      # 无输出，通过
```

8项配额矩阵按JSON sort_keys、紧凑分隔符编码的SHA256：
`d65f1545bb9989dd68ba4de96134f5f5ce6069bc265ac9c9607b7e6be70d8fc4`。
这是配置hash，不存在张量输入hash。环境Python3.12.14，Torch未安装，
nvidia-smi不可用；GPU架构/驱动/CUDA/Triton/kernel/JIT配置未验证。
未跑完整依赖测试、实际CUDA选集/输出、prefill→decode交接或真实模型。

本次是低风险host派发正确性修复。性能、质量、峰值内存、kernel数、host同步
变化均未实测；逐组回退可能增加选择成本，不能声称提速。历史 batch+显式配额
实验需要重测；纯旧比例模式、默认batch关闭的历史结果不因本问题自动失效。
不修改任何历史数据。后续如实现批量显式配额，须同输入/预算/精度核验真实
GPU选集、输出、因果/保护带、状态交接和完整请求耗时，再撤销对应门禁。
