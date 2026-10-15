# TL-E139-GRAPH-SCHEMA-158：补齐 Graph 输入形状与 Q dtype 隔离

审查基线：main `ceb8fa8cdef24941ca09085be4b19b6f70c06b88`。
该提交合入 E138/E139 实验链及 E140 几何缓存键修复。本次没有合入此前
`perf/prefill-compact-block-traversal` 的 GPU 待验证候选。

## 新增确认问题与修复

严重度 P2，状态：host 缓存路由缺陷 confirmed，本次修复并 CPU 回归；
实际 CUDA 输出差异、GPU 速度和显存变化仍未验证。

路径：基线 `sparse_attn/ops/batch_select.py` 的 `_graph_cached_run`
（86行起）及 `batched_group_sels` 缓存键（608行附近）。E140 已绑定
α/β/γ、保护区、预算等几何，但仍有两个独立遗漏：

1. 没有绑定输入 tensor shape，H/Hkv/G 也不在原键中。相同 L/预算的
   Hkv8→Hkv1（G 相同）调用可命中旧条目；`copy_` 会广播单例 head 维度，
   不报错，旧图仍返回旧 Hkv 形状的静态输出。
2. 原始 Q dtype 不在键中。`b_q` 已被强制转为 float32；保持 K dtype
   不变、改变 Q dtype 时，原 `_dt_fp` 不变。但 `_body` 捕获 `qdtype`
   用于 coarse/fine 分数存储和舍入，不能复用旧 dtype 的图。

最小修前证据（真实 CPU tensor + CUDA API test double，仅验证路由）：
```text
shape: actual (2,8,4,32), expected (2,1,4,32)
dtype: actual torch.float32, expected torch.bfloat16
optional tensor -> None: cache entries 1, expected 2
real batched_group_sels key: Q bf16 and Q fp32 produce equal keys
Ran 5 tests: FAILED (failures=4); identical-schema reuse passed
```
其中 helper 的 dtype/optional 测试是防御层测试，不能说现有调用者所有
dtype/None 切换都会触发（E140 已覆盖部分 dtype）；生产侧新增遗漏是
head shape 和原始 Q dtype。真实 key 测试执行原 `batched_group_sels`
建键代码，仅替换 CUDA eligibility、index preparation 和 graph executor。

修复：helper 查找前绑定输入名、shape、dtype、device、layout、None 状态；
caller 另加原始 `qdtype`。全为 host 元数据，不读 tensor 值、不增加
`.item()`、D2H 或 GPU kernel。tensor 身份不入键，保留跨层/跨实例复用；
stride 不入键，因为 `copy_` 将同形值写入 capture 原静态布局。
replay/copy 异常时移除完整 schema 对应条目，再由原 caller 回退；不误清
其他配置缓存。不改 chunk2048、group、L1/L2、mask、topk、预算或算子。

## 验证

新增 `sparse_attn/ops/tests/test_graph_cache_schema.py`：6项，包含尺寸广播、
dtype转换、optional状态、跨实例复用、replay错误后精确清理及实际调用点
Q dtype 键隔离。加既有 prefill shape/protection 回归共15项。

```sh
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s sparse_attn/ops/tests -v
PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -O -m unittest discover -s sparse_attn/ops/tests -v
```
```text
normal: Ran 15 tests in 0.155s; OK
-O:     Ran 15 tests in 0.258s; OK
git diff --check: passed
```

环境：Linux x86_64，Python3.12.14，PyTorch2.8.0+cpu；无 nvidia-smi、
无 `/dev/nvidia*`，CUDA/驱动/架构/Triton kernel配置不适用。测试使用固定
ones/zeros，无随机数。fixture manifest：
`ones:fp32,bf16:[2,8,4,32],[2,1,4,32];zeros:q=[1,256,8,128],k=[1,256,2,128];ends=[192,256];sink=swa=64;alpha=.25,beta=.25,gamma=.5;k1=4,k2=128`
SHA256：`041fe5137f2bf9d2fddc5836d458bf9b0435e4d5c08e1762f989ea2ae49963ab`。
测试文件 SHA256：`51dde5057f29f80fd329b42dd1d8748dfc9238523ed132d582ac17b9731e5dba`。

CPU test double 的 replay 不执行 CUDA，不能充当 GPU 对拍、编译或内存检查。
本次是缓存兼容性修复，不是加速结果；不同规格新增 capture 是必要成本，
同规格复用保留。未测端到端质量、TTFT、峰值显存或 host 增量开销。

## 历史影响与下一验收

没有证据表明已报告的固定 Qwen3-8B 32/8-head、统一 bf16 实验触发了本问题，
故不撤销其结果。应重测同进程切换 head 数、GQA组数、Q dtype 的场景：
A capture→B replay候选→新鲜B eager逐位比较，再A复用；记录新cache条目数、
真实派发和失败回退。现有 E140 α/γ/sink/k1 五组 GPU 门也应在目标机器重跑。

旧问题复核：E140 的几何字段修复已在新 main 存在，远端报告有 G1-G5 GPU
结果，本次未重跑；不能据此宣称覆盖本次 shape/Q dtype 场景。topk v2
探针已新增，明确把零概率边界排除在等价保证外。旧 print-only 门禁问题仍
需单独处理：当前 `e139_bs_graph_bitdiff.py` 仅打印 diff，
`e139_step2_cand_bitdiff.py` 末尾仅打印 TOTAL FAIL；与回应文档中“全部
sys.exit(1)”的说法不一致，不能把该旧问题标记 fixed。
