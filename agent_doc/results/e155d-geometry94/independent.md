# E155d treatment identity 独立验收

2026-10-20；独立上下文，只读验收。结论：**未发现阻断本次有界修复的 finding**。
适用父目录（含 `/`）和仓库均未发现 AGENTS.md。基线 HEAD/origin main 均为
`8d541d4001b184c2c14062e5e7303dce9b7b12a2`，Python 3.12.14。
读过检查点、根 TASK、agent_doc 任务索引及 snapshot 详情/advice；未读取或执行
collector，未读取人类 guide，未联系历史机器、启动 GPU/外部任务或发布。
本复核只新增本报告，没有修改源码、测试、任务、advice 或未知字体文件。

## 固定审查对象

| 文件 | SHA256 |
|---|---|
| exp/trace/analyze_e155d_grid.py | 85ae9d442a09fc19fec9cd564929b52a81cf5cff11da2b611bc57d75655c4e2f |
| tests/test_e155d_geometry.py | 8db3506b6dc9815fb8b06c1c1308d0db15a0c23885c47195ac0e010caf693fae |
| tests/check_e155d_geometry.py | 18f4b4e61819c40cf55389b5b9613390d9a61af1702d947a745135d2b20a308d |
| tests/test_e155d_snapshot.py | 4839d07a0b4110324fcbeb9bd04408e0367bb726e856d24a5371db040efc8ff0 |
| tests/check_e155d_snapshot.py | f1c154df05a643d8d1760d65f609947ab7fd612eb8976c819e2e70aeab040d6b |
| sparse_attn/info.py | b5ee1055a6fb27c4aa540887ec16ec4ca0c617fc27f2e46d6db29beba5d62ed4 |

复核结束时再次核验前五个文件哈希未变。源码 diff 仅新增消费者门禁、结果 provenance
及失败明细；未修改算法预算、selector、kernel、generator 或真实实验产物。

## 实际执行与结果

工作目录为本仓库；所有 Python 命令均带 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests`。

```bash
python --version
python -m unittest -v test_e155d_geometry test_e155d_snapshot test_e155d_grid_collection test_e155d_pairing test_e155d_score_units
python -O -m unittest -v test_e155d_geometry test_e155d_snapshot test_e155d_grid_collection test_e155d_pairing test_e155d_score_units
python tests/check_e155d_geometry.py
python -O tests/check_e155d_geometry.py
python tests/check_e155d_snapshot.py
python -O tests/check_e155d_snapshot.py
python tests/check_e155d_score_units.py
python -O tests/check_e155d_score_units.py
```

- 57 项普通/-O全部通过，分别 0.589s/0.548s，其中几何专项16项。
- 真实 `info.py` producer→消费者普通/-O均通过：86 geo + 1 CTRL，51个错误但合法
  label 均拒绝。检查了 `load_info()`：只删除 AST 的 torch import，任何 tensor属性访问
  都 raises；这是真实 metadata producer 的 host 验证，不是 GPU、模型或实际评分验证。
  manifest输入SHA256：`70b6b0f0f99cf7f3e213cd98169540a648ca7e5b28db6e3a6eae8f9575dbd241`。
- 全入口24组、量纲24组差分普通/-O均通过。全入口输入SHA256：
  `3aaf366af9095b401cb7517f1767b986d99093e03091adf1b3be7d9f3125f8f9`。
  原 checker 仍针对其既定 b418e22；另用内存 `read_text().replace(BASE, 8d541d...)`
  后 `exec(compile(...))` 重跑相同 checker 普通/-O，对本次直接基线也各24组通过。
  仅忽略新增 provenance/空错误字段；非空 evaluation_errors 必须失败，评分/CI比较未放宽。
- 保留 `.agent-runs/geometry94/before-v2.log` 原修前证据，8 method / 7 fail / 0 error；
  文件SHA256 `28c1c35ac89e23757e8b8de5a0b03582819134da4983b4b12b200ded635fab26`。
  另用 `git show 8d541d4001b184c2c14062e5e7303dce9b7b12a2:exp/trace/analyze_e155d_grid.py`
  在内存恢复消费者，执行原七个错误接受场景及87合法格控制（控制只检查既有分数，不要求新字段），
  独立再次得到8 method / 7 fail / 0 error。基线消费者SHA256：
  `c719089b3c4e279abc1a42430d116802b901929054f936ec52100abbd4f44658`。
- 额外 inline Python 探针不修改测试：9类 schema/畸形输入（array、坏JSON、坏UTF-8、
  bool页数/bool块宽、null method、CTRL bool alpha/字符串beta/缺alpha）均在eval前拒绝；
  合法缩进重排JSON保留分数且记录实际原始字节SHA；eval中删除CTRL manifest只隔离该格，
  其它合法格继续得到分数。探针使用 `GeometryTests` 临时夹具和合成 scorer。

## 独立判断与边界

逐项核对类型而非 `==` 独判，避免 bool/int、float/int、数字字符串误认；CTRL三几何值必须
显式 null，alpha/beta须数值0。producer canonical JSON与消费端一致，缺省TLI manifest本来
不含method，兼容缺省而拒绝明确非TLI合理。评估前后检查不仅看canonical hash，也比较原始
manifest字节SHA，因此同对象重新排版发生在评分中也不能悄然通过。缺失、畸形、变更走
per-cell eval_error，main新增evaluation_errors提供可追溯原因；无错误格进入geo排序/CI。
snapshot夹具只补合法metadata与真实命名结构，原预测替换、CI绑定断言保持。

非阻断限制必须保留：这仅验证标签、声明manifest及文件名的一致性，不证明执行过对应
GPU配置，也不验证全网格非几何字段统一、模型/输入/tokenizer身份或历史质量结论。保留
现行10hex（40bit）filename hash协议，不是防恶意伪造；完整原始manifest SHA提供追踪而非
事后认证。重复host去重仍只比较预测字节，选中副本manifest单独校验。共享staging并发、
ABA写回、后续CI阶段metadata外部变化、路径非持久归档及失败后旧verdict仍可能留存，
未由本补丁闭合；运行成功状态和evaluation_errors必须一起核对。实际T2重评分、GPU、
模型与性能均未验，不因host回归通过升级这些主张。
