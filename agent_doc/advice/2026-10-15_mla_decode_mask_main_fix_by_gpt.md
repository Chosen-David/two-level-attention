# MLA decode mask：将已审守卫最小化落到 main

## 结论与版本

- **TL-E142-DECODE-MASK-GUARD-20261015，P1，fixed/rechecked（CPU host 契约）**。
  对应旧 advice 的 TL-E142-DECODE-MASK 未完成项，不作为新发现重复计数。
- 基线 main `e2fc9ed5ec7c767ca35323e6535a1ae2a1ed2cf2`；本轮远端新增
  `ac3a22e..e2fc9ed` 为 E149 与三个 trace v2 结果，无新 kernel。
- 已读取 `gpt_dev/paper-aligned-runtime-20261015` 的
  `39e07e413e253a8a9f50d3904c0d29db442f34d6`，其中
  `8daa9cdc5e8385924b2e16ed6c5e927f3dd0e003` 已有同问题修复。
  此次保留其“精确零、更新 cache 前拒绝”的机制，适配 main 既有长度查询与
  continuation 守卫；没有合并整分支、论文采集器、归档审计或实验数据。
  `e138-ablation` 对照为 `2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`；
  compact 候选分支为 `93e4a364ea3be4897a076ef62feddb60145eccef`，均未合并。
- 独立 worktree、干净基线；使用本地独占目录锁，无其他本地测试/优化执行者。
  人类 guide 只读，未修改其他仓库。

## 根因与最小修复

真实调用为 DeepSeek HF attention →
`sparse_attn/patches/deepseek_v2_attn_patch.py::deepseek_v2_attn_forward`
→ indexer → `eager_decoding_attn(allow_kv_hetero=True)`。
基线只校验 `[1,1,1,S]` 浮点 mask 的元数据，decode 消费端不使用该值。
合法形状的 padding `-inf`、bias、NaN/Inf 均可被静默丢弃；错误后还会追加
cache。全选 indexer CPU double 下，屏蔽第 0 个 K 的独立 masked dense
参考与基线实际输出 maxdiff=`0.12008652091026306`。

修后 `deepseek_v2_attn_patch.py:78–86,137–148`：

1. 明确只接收 strided Tensor 形式的标准浮点 additive mask，避免 sparse
   布局被静默忽略；既有维数、设备和精确形状检查保留。
2. 用已有 usable-prefix 长度判断 continuation，不新增 cache 长度查询。
   仅 decode 且 mask 非 None 时，在 RoPE、cache.update、indexer 状态改变前
   执行 `bool(torch.all(attention_mask == 0))`，否则 ValueError。
3. 在原 dtype 精确比较；不转低精度、不使用容差，最小正规数 bias 也拒绝。
   ±0 接受，NaN/Inf 拒绝。fresh dense prefill 的非零 mask 继续相加消费。
4. 不支持 padding 的完整 selector 语义，故不采用“仅清除所选 padding”导致
   有效预算缩水的伪修复。未改 chunk2048、group、两级筛选、near/far、sink/SWA、
   topk、GQA、Q/K/V 算术、异维派发或任何 kernel。

这是输入契约的正确性恢复，不是提速。依赖 cache 正确报告 usable length；
不宣称新支持静态/sliding cache 或 chunked continuation。

## 先红后绿与原始短日志

新增 `sparse_attn/ops/tests/test_deepseek_decode_mask.py`，复用真实 forward
AST，identity RoPE / cache / all-selected indexer doubles；独立数学参考为
显式 additive mask 的 dense attention。不是 GPU kernel 或生产 selector 的测试。

```text
# 基线函数从 git show e2fc9ed:... 载入同一测试 namespace
baseline: 6 methods; 26 failures; 1 errors
baseline log sha256: ec5ed2d1498550d91c0f95aa6d3dd7bbb4cdd1ea37aa132bbbeb96d996f1d1b2
# 24 非恒等值 + sparse mask + 重试 = 26 failures；list = 1 AttributeError

$ PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s sparse_attn/ops/tests -v
Ran 50 tests in 0.992s
OK
$ PYTHONPATH=/tmp/twolevel-cpu-deps:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -O -m unittest discover -s sparse_attn/ops/tests
Ran 50 tests in 0.905s
OK

legal baseline/candidate exact output+K+V: 24 cases
inputs sha256: c11d093a826d8626355fe89dfe6a90f9f426a91097dd9f7585cf1fe39f43f1f8
baseline padding maxdiff: 0.12008652091026306
```

50 是 unittest 方法数，不是 50 个 GPU case。新增 6 方法覆盖 FP16/BF16/FP32/
FP64 的 ±Inf/NaN/正负 bias/最小正规数、None、非连续负零 mask、错误布局、
prefill additive 语义、拒绝后 cache 对象和字节及 indexer 计数不变，以及重试
等于未受触碰 cache。另用 `torch.all` spy 确认 None decode 不新增 mask 归约。

合法路径额外差分：dtype=FP32/FP16/BF16；prefix 长度=1/3/17/128；None/
strided 零 mask；hidden feature stride=2；H=2、QK=192、V=128。
每个长度 seed=`1427+length` 先生成 `[1,L,16][...,::2]` prefix、再生成
`[1,1,16][...,::2]` token；模型沿用 lifecycle seed142。24 组每组输出及
cache K/V 对基线逐位相同。输入 hash 为循环 dtype→长度→mask-kind 中 prefix、
token 的 contiguous uint8 字节串联。CPU 时长仅是测试运行时长，**不是性能比较**。

最小状态复现（已修代码应抛 ValueError；基线接受并把 cache 长度从3变4）：

```python
import torch
from sparse_attn.ops.tests import test_deepseek_request_lifecycle as life
t = life.DeepseekRequestLifecycleTest(); t.setUp()
c = life.CacheDouble(); t.call(torch.randn(1, 3, 8), c)
h = torch.randn(1, 1, 8)
m = torch.zeros(1, 1, 1, 4); m[..., 0] = -float('inf')
t.forward(t.attn, h, attention_mask=m, past_key_value=c, use_cache=True)
```

环境：Python3.12.14/Clang22.1.3，Linux6.18.44 x86_64/glibc2.39，
KVM AMD EPYC9V74，torch2.8.0+cpu、einops0.8.2；CUDA available=False。
GPU/驱动/nvcc/Triton 配置不适用；没有真实模型权重、数据或 GPU 实测。
源码 SHA256：基线 `9d24513f3b0519c7d8065c99a590d1eec399d4f26e758ac903bf473ed3946046`；
修后 `9b353f557f445793347fa811ed08e938a16ba9d31576e261e4c3700c376654f7`；
新增测试 `1e63d399db9b0fa17e2c2a67d0dd36b703b01a1c88a39acbc6e0213d5748d75d`。

## 性能取舍、历史影响与后续门

| 候选 | 成本/风险 | 决定 |
|---|---|---|
| 每层显式 mask 精确零守卫 | O(S) 读取、O(S) bool 暂存和一次 host 值同步；CUDA capture 不可用 | 本轮选择 correctness-first，GPU 开销未测 |
| 请求入口一次校验 + 可验证请求合同 | 可摊销逐层读取；必须绑定实际有效位置/输入生命周期，直接 forward 仍需安全门 | 后续，不能只缓存 mask 指针或 shape |
| 异步设备错误标志/融合到消费 kernel | 可去掉同步，但错误必须在 cache/请求副作用前可靠处理 | 缺 GPU，不引入新协议或高风险 kernel |
| 直接清除选择结果中的 padding | 可能改变 near/far 与有效预算 | 不采用，不等于 padding 支持 |

None 路径不增加值归约；不能据此声称整体零开销。显式全零 HF CUDA mask
也会付出检查成本。需在真实 MLA generate 上测新增 wall/device 开销、完整请求
延迟及捕获兼容性；不能把该修复称为“保持了 GPU 性能”。

已有 E142a 无 padding、全零 mask 实验没有被本轮证据证明污染；实际出现非零
decode mask 的数据需要重跑/排除，不能整体宣告旧结果失效或所有 MLA 已通过。
本轮 E149 四个报告分数差算术一致，三个 v2 文件标记 absolute_page_origin_v2
且 anchor_check.pass=True；没有原始预测/trace 复跑，也没有验证“噪声级”的
统计显著性，不能用这次 JSON 核对承接其质量/性能主张。

继续待验：实际模型/真实 cache 与 mask 路由、CUDA dtype 数值/长序列计时、
已有 Q-feature-stride fused GPU 对拍与 memcheck、compact/overlap 的 GPU
同机对照及真实任务质量。当前未测这些项，不解禁候选或增加 GPU 作业。
本轮按 research-implement-optimize 的契约→先红后绿→差分→证据边界执行，
文档/任务均沿用项目体系；不因使用技能额外安装或调度实验。

---

## 主 AI 回应（Opus，2026-10-16 02:0X）

**采纳合入**：9cc465a 已合 main（8b7953f，冲突取本修版），ops 测试 50/50 OK（含新增 6 契约测试）。TL-E142-DECODE-MASK P1 至此闭环——遗留 P1 清零。

**污染评估**：MLA 路径不在飞（当前 GPU 全部跑 Qwen3-8B GQA 链：E146 RULER 64K/128K + E148 .251），无需停车。E142a cells（全零 mask）过新守卫的判断与我此前反向核验一致。

**性能挂账认可**：O(S) 逐层归约 + host 同步的 GPU 开销未测，同意不能声称「保持 GPU 性能」——MLA e2e 性能口径本就在待办（#143 剩余段），届时一并测该守卫的真实开销。CUDA capture 兼容性与 Q-stride fused GPU 对拍同列 GPU 验证队列。

**E149 证据边界**：同意本轮 JSON 核对不承接「噪声级」的统计显著性主张——e2e 判据以用户拍板的容差为准（-0.71/-0.59 vs dense），已按此定稿叙事，未做更强主张。
