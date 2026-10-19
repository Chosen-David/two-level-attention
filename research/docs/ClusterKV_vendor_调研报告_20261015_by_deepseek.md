# ClusterKV vendor 接入调研报告（#16）

日期：2026-10-15（deepseek 轨）
范围：官方仓库 `third_party/ClusterKV`（sjtu-zhao-lab，commit c7380819588771e8c5dadacdbd33942d7c53adea）
结论：**官方无独立 mask/gate 提取函数**（不像 MoBA 有 `moba_attn_varlen`），但 accuracy 侧选择器
`cluster_attn_out` 可复用，按「统一后端机制轨」接入。

---

## 一、官方代码两条路径

### 路径 1：accuracy 侧（官方 LongBench pred.py 实际走的路径）
- 文件：`accuracy/cluster_attention.py` + `accuracy/patch.py`
- 流程：`patch.py` 的 `enable_attention_eval` 把 `forward_cluster` 绑定到 `LlamaAttention.forward`（
  仅支持 LlamaAttention / glm4 SelfAttention，**Qwen3 的 attention 不是 LlamaAttention → 官方对
  Qwen3 不生效，必须写 qwen3_clusterkv_patch.py 适配层**）
- 选择器本体：`cluster_attn_out()`（纯 torch）：
  - 聚类建簇：`build_cluster()`（pylibraft KMeans，nlist=400/niter=20）
  - 选择：query × centroids 余弦相似度 → 取最近邻簇 → 簇内累计达到 `token_budget−sink` → 输出
    `sel_key_indices`（per-head 选中 token 索引）
  - 细筛：`sel_key_states/value` = sink 区 + 选中 token + prompt 后最近区 → torch matmul 精确注意力
  - `use_search_kernel=True` 时用 pybind `_clusterkv_knl.search_indices`；**False 时用纯 torch
    （pad_sequence/stack）可绕开编译内核**
- 依赖：pylibraft（KMeans）+ rmm，正在安装
- **这就是可提取的 selector**：`cluster_attn_out` 构造的 `sel_key_indices` 即 mask/gate，
  与 TLI 统一后端机制轨（evaluation.md:123「相同 Q/K 回放与最终 attention backend，只隔离
  selector」）完全匹配

### 路径 2：efficiency 侧（端到端系统）
- 文件：`clusterkv/clusterkv_models/ClusterKVAttention.py` + `clusterkv_utils/clusterkv_controller.py`
- 形态：`ClusterKVController(token_budget, sink, window, window_nlist, offload, ...)` 端到端
  controller，prefill 用 flashinfer paged KV（dense），decode 用 `get_neigh_c`/`get_sel_indices`
  选 token + flashinfer decode
- 依赖：flashinfer 0.5.2（已装）+ pybind 编译内核 `_clusterkv_knl`（kernel/ 目录，需 cmake 构建）
- 预算语义：`kv_cache_size = sink + token_budget + window`（token_budget 是选择预算，sink/window 额外）

---

## 二、接入方案

按 evaluation.md 的统一后端机制轨（轨 2）：**复用官方 accuracy 选择器**
（`build_cluster` + `cluster_attn_out` 的 sel_key_indices 构造），细筛走统一后端。

- 新文件：`exp/vendor_baselines/clusterkv_vendor/qwen3_clusterkv_patch.py`
  - 胶水层：把官方 `forward_cluster`/`cluster_attn_out`/`build_cluster` 绑定到 Qwen3Attention
  - 官方核心文件（`cluster_attention.py` 相关）与官方逐字节一致（md5 冻结铁律），只改适配层
  - `use_search_kernel=False` 纯 torch 提取，绕开 pybind 编译
- 参数：`--clusterkv_token_budget`（主表 1024）/ `--clusterkv_sink` / `--clusterkv_nlist` /
  `--clusterkv_niter` / `--clusterkv_head_sel`（truc/pad）/ `--clusterkv_gqa_policy`（qavg）
- pred_ruler.py / pred.py 加 `clusterkv_vendor` 分支（同 moba_vendor 模式）
- arguments.py 加 `_add_clusterkv_args`

## 二.五、接入方案落定（2026-10-15 晚，编译受阻后改道）

**编译内核受阻**：官方 kernel/ 需 cmake 构建，但本机网络三处阻塞——
raw.githubusercontent.com（RAPIDS.cmake 下载 0 字节）、release-assets
（CPM.cmake 超时 300s）、github.com git 协议（cccl clone 卡 15 分钟 80K）。
编译链路（rapids-cmake → CPM → raft → cccl → pybind）每一步都依赖不可达源，
vendor 工作区 cmake 配置已到 raft 阶段仍无法继续。

**改道方案（官方文件零改动）**：官方 accuracy 侧 `cluster_attn_out` 本有
`use_search_kernel` 开关——False 分支是官方自己的纯 torch 提取路径
（pad_sequence/stack，cluster_attention.py:156-176）。官方 `_clusterkv_knl
.search_indices` 的 CUDA 语义（kernel/cluster/src/test.cu
KMeansSearchIndicesKernel）= 把每 head 前 num_need_clusters 个簇内的 token
索引按簇填充到 sel_key_indices。因此：

- vendor 包 `clusterkv/_clusterkv_knl.py` 提供纯 torch `search_indices`
  （语义与官方 KMeansSearchIndicesKernel 一致，簇内逆序填充）；
- 官方 `accuracy/cluster_attention.py` 逐字节一致直接 import（md5 冻结），
  build_cluster + cluster_attn_out 完全官方代码；
- 编译内核完全绕开，不触发任何 GitHub 下载。

**依赖版本偏差（记录）**：pypi 无 pylibraft-cu12 24.10（官方 pyproject 锁定），
NGC 镜像不可达。改用 pypi 的 pylibraft-cu12/rmm-cu12 **25.6.0**（KMeans
fit/ivf_flat API 保持兼容，已冒烟验证）。ClusterKV 选择算法与 pylibraft 版本
无关（KMeans 建簇语义一致），偏差仅工具链版本，记录待核。

**编排脚本修正**：run_ruler_clusterkv.sh / run_longbench_clusterkv.sh 原含
`--clusterkv_offload 0`（accuracy selector 路径无 offload 参数，efficiency
路径才有）→ 已删除。参数与 arguments.py `_add_clusterkv_args` 对齐。

## 二.六、依赖安装定稿（2026-10-16）

**阻塞解决：torch + pylibraft + rmm 三库共存。**

问题链：
1. clusterkv conda env（py311）有 pylibraft/rmm **24.10.0**（官方 pyproject 锁定版本）但无 torch；
   fluentllmenv（py311）有 torch 2.8.0+cu128 但无 pylibraft。
2. pypi 的 pylibraft-cu12 **25.6.0/26.10.0 均不含 cluster/neighbors 绑定**（Cython 接口在
   pypi 发行版被移除，包内只有 common/random/sparse）——官方代码 `from pylibraft.cluster
   import KMeansParams, fit` 必须用 conda 24.10 包。pypi 无 24.10 wheel。
3. 直接注入 clusterkv env 失败：torch(cu128) 先加载 fluentllmenv 的 libcublas.so.12（soname
   相同，12.8 版，**缺 cublasSetEnvironmentMode 符号**），conda libcusolver.so.11 加载时符号
   解析失败（`undefined symbol: cublasSetEnvironmentMode, version libcublas.so.12`）。

**定稿解法**（已验证 import + KMeans fit 冒烟通过）：
- 官方 accuracy 侧 `cluster_attention.py` 的 `build_cluster`（KMeans 建簇）用 conda 24.10 的
  pylibraft/rmm：`PYTHONPATH` 优先 clusterkv env site-packages（完整 cluster/neighbors 绑定）。
- `LD_PRELOAD` 强制预加载 clusterkv env 的 libcublas.so.12（12.9.2.10，含 cublasSetEnvironmentMode），
  使全局符号表先有该符号，cusolver11 解析成功；`LD_LIBRARY_PATH` 同时给 clusterkv env lib
  （conda cublas12/cusolver11/cusparse12/curand10/libraft/libstdc++6.0.36）与 fluentllmenv
  nvidia lib（cublas/cusolver/cusparse/cuda_runtime）。
- 固化脚本：`exp/vendor_baselines/clusterkv_vendor/ckv_env.sh`（source 后 `PYBIN` 即
  fluentllmenv python，`PYTHONPATH` 已含 pylibraft 24.10 与 local_pylibs）。
- 编排脚本 `run_ruler_clusterkv.sh` / `run_longbench_clusterkv.sh` 已改 source ckv_env.sh。

**版本偏差记录**：官方 pyproject 锁 pylibraft 24.10（conda 即此版本，反而与官方一致）；pypi
25.6/26.10 已弃用 Cython 绑定接口，不可用。无版本偏差。

**冒烟证据**（KMeans fit，X=[256,128] cuda，n_clusters=16）：
`centroids: torch.Size([16,128])`，与官方 build_cluster 同款 `fit(cluster_params, head_keys)`
签名（返回 3 元组，官方丢弃后两项）。

## 三、预算口径对齐注意（2026-10-16 官方代码实测）

- 官方 accuracy `cluster_attn_out` 最终 gather = `[sink 区, 选择区(token_budget−sink),
  prompt 后最近区]` → **总 attend = token_budget + 最近区**（sink 已含在 token_budget 内）。
  decode 单 token 最近区≈1 → 默认 token_budget=1024 总 attend ≈ 1025。
- TLI 的 K2=1024 已含 sink128+swa128 强制区（选择预算 768），总 attend=1024。
- 对齐总 attend=1024 → `clusterkv_token_budget=1023, sink=16`；或直接用官方默认 1024
  （Δ≈1 token）并在主表记录实际 unique token 数（evaluation.md 口径）+ 构成差异披露
  （ClusterKV 强制区=仅 sink 16，TLI= sink+swa 256）。接入验收时 microbench 核对。

## 四、阻塞与下一步

- [ ] pylibraft-cu12 + rmm-cu12 安装（进行中）
- [ ] qwen3_clusterkv_patch.py 胶水（绑定官方 selector 到 Qwen3Attention）
- [ ] pred_ruler.py / pred.py `clusterkv_vendor` 分支
- [ ] arguments.py `_add_clusterkv_args`
- [ ] 预算对齐换算 + 实际 unique token 统计核对
- 编排脚本已就绪：`benchmark/RULER/run_ruler_clusterkv.sh` + `benchmark/LongBench/run_longbench_clusterkv.sh`
