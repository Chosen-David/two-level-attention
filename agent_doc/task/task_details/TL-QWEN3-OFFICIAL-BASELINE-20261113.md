# [TL-QWEN3-OFFICIAL-BASELINE-20261113] 纯官方 Quest/JSSA 高精度实现重跑 LongBench

Task-ID: TL-QWEN3-OFFICIAL-BASELINE-20261113
Date: 2026-11-13

## Plan

### 目标与验收

按用户最新指令，删除 Quest/JSSA 的旧 vendor/NFA 接入，直接绑定主仓库 `third-party/` 中的官方克隆，重新测量 Qwen3-8B × LongBench v1 英文16任务的精度与完整请求性能；AsyncTLS 保持论文语义复现，旧 GLM AsyncTLS 任务随后恢复。

- Quest：官方 `quest.utils` + 编译后的 `quest._kernels`，FP16、B1、paged KV、page/chunk=16、token budget=1024。不得调用 `evaluation/quest_attention.py` 或 Torch decode fallback。
- JSSA：官方 JSSA `34ce5abd82453b16ca6d5f16a1a8ae50fac5b920`，指定 vLLM `39910f2b25aacc09f5e7f166cdf0030b19f8b9e8` 加 `jssa_vllm.patch`；budget=1024、rank=32、FP8 projected-key、gather-free indexed-attend。不得调用 HF/Torch 语义移植。
- 两条 runner 均不得导入 `sparse_attn`、旧 `exp/vendor_baselines` 或 `benchmark.LongBench.pred`。
- 输入保持 LongBench 中部截断、Qwen3 chat、31500 cap、greedy、任务官方 max-new 与逐样本输入哈希。
- 用户 2026-11-13 最新顺序：两条官方源码链路真实 smoke 通过后，**先用本地双 H20 出对齐 NFA 的性能数据，再出精度数据**。LongBench 性能采用 B1、sample0、1 warmup、3 repeats、固定33 token、EOS不提前停止；使用 `T(33)-T(1)` 两点法分离 decode，完整 host wall 与 kernel/profiler 数据分开。
- 性能验收 additionally 覆盖独立长输入 256K/512K，统一比较 NFA、Quest、JSSA、AsyncTLS 与 FullKV FlashInfer，K2=1024、AsyncTLS K1=128；预期 NFA 快于 baseline、baseline 快于 FlashInfer。若不满足先查实现路径、预算、Graph/warmup、计时口径和 GPU 独占性，不把异常数字直接写成结论，也不为凑排序启用 fallback。
- 缺官方扩展、backend、环境或发生回退时 fail-closed，不生成“官方高性能”结果。实验结果在不同 owner 独立验收前保持 `pending-independent-verification`。

### 实现方法

- 新入口 `benchmark/official_baselines/longbench_official.py` 只复用数据文件与 LongBench JSON 配置，不复用 NFA 顶层 harness。
- `qwen3_quest_kernel.py` 仅适配 Qwen3 Q/K norm、GQA layout、RoPE 参数和 HF 接口；官方 estimate/top-k/paged attention 保持原实现，并记录真实调用计数。
- JSSA 在独立 Python 3.12 环境运行 patched-vLLM，核验模块路径、官方 backend 文件、补丁树 diff 与校准基哈希。
- 旧 `qwen3_lbv2_vendor_20261112` 和 `qwen3_lbv1_vendor_perf_20261112T01` 是历史 vendor 口径，不能自动复用为本任务新结果。

## Progress

### 2026-11-13：独立入口与环境准备

- 固定官方身份：Quest `01c1623bf9395009520874e989e29f683203b357`；JSSA `34ce5abd82453b16ca6d5f16a1a8ae50fac5b920`。
- 已写独立精度/性能入口与 Qwen3-Quest CUDA 接口层；AST 审计无 NFA/vendor/Torch参考路径导入，回归 `tests/test_official_longbench.py` 3/3 PASS，Python 编译和 `git diff --check` PASS。
- Qwen3-8B 校准基已核验：36层、首层 `[8,32,128]`、FP32；生产传入 rank=32、budget=1024。
- 本机双 H20-3e 点火前均1MiB；远端资源池所有可见机器驱动均为550.127.08/CUDA 12.4。官方 JSSA README 要求 Python 3.12 与支持 CUDA 12.9/13.0 的预编译运行时，故必须先完成真实导入和 kernel smoke，不能假定驱动兼容。
- 本机无 Python 3.12，uv 的 GitHub standalone 下载超时；已从 Anaconda 官方源安装独立 CPython 3.12.8，并创建 `/home/wangyuanshuo02/venvs/jssa-official-py312`。
- Quest 子模块和 vLLM 正在通过 SSH 克隆，已观察到对象数据增长；尚未构建或完成真实 kernel smoke。

### 2026-11-13：官方源码落地与高性能 fail-closed 门禁

- Quest 顶层及嵌套子模块已对齐官方固定提交；vLLM `39910f2b25aacc09f5e7f166cdf0030b19f8b9e8` 已浅取并只应用官方 `jssa_vllm.patch`。
- `longbench_official.py` 新增 JSSA 运行时门禁：缺少 `torch.ops._C.top_k_per_row_decode`、未执行 FP8 projected-key store、未捕获/调用 radix Top-K 或 gather-free indexed-attend，以及进入 gather/BF16/unprojected fallback 均立即失败；包装只做验收计数，不替换官方算子。`tests/test_official_longbench.py` 5/5、Python 编译和 `git diff --check` PASS。
- 修复独立 runner 的多分片 manifest 覆盖与零样本分片处理；正式结果仍未产生。
- Quest 的 HTTPS 下载失败留下0字节 RAPIDS/CPM 文件。RAPIDS 24.02 已改走官方 SSH 浅克隆 `bf3faf0e8f56792ad74ec3304bb7222b7c3fff2f`；CPM 按官方 `publish.yaml` 将 v0.38.5 源码中的版本占位符替换为 `0.38.5`，所得 MD5=`c98d14a13dfd1952e115979c095f6794`，与 RAPIDS 官方校验值逐字一致，未绕过完整性检查。
- RAFT 的 CCCL/RMM/CUTLASS/cuCollections/fmt/spdlog 依赖改用固定版本官方浅克隆与 CPM 本地源覆盖，避免重复访问不可达的 GitHub HTTPS；尚未完成配置和 CUDA 编译。
- JSSA 独立环境 `.venv` 指向 `/home/wangyuanshuo02/venvs/jssa-official-py312`，后续 Python/uv 命令经该环境执行。`torch==2.11.0` 主包已安装；内网缺 `cuda-toolkit==13.0.2` 和指定 CU13 NVIDIA 库，基础依赖已由 uv 安装，CU13 官方包正从公网 PyPI 获取。安装成功不等于 Torch/CUDA 可运行，仍需 import 与真实 kernel smoke。
- `wheels.vllm.ai` 与 `download.pytorch.org` 在本机及 pc02 均网络不可达；指定提交的 cu129 wheel 尚未取得，不得用其他提交的 PyPI wheel 冒充。

### 2026-11-13：Quest 进入官方 NVCC 编译并定位 fmt/spdlog 兼容问题

- CUTLASS 普通浅克隆两次因 `early EOF` 失败后，改用同一官方 v2.10.0 的 blob-filter + sparse checkout，HEAD=`fc9ebc645b63f3a6bc80aaefde5c063fb72110d6`；只检出编译所需头文件和 CMake 规则，没有换版本。
- RMM/cuCollections 子项目独立下载的 `RMM_RAPIDS.cmake`/`CUCO_RAPIDS.cmake` 已预置为同一份官方 RAPIDS 24.02 文件；未修改其子项目源码或完整性校验。
- RAFT 24.02 在 generate 阶段引用 CUDA 12.9 已移除的 `CUDA::nvToolsExt`。通过仓库外 `/tmp/quest_cuda12_nvtx_compat.cmake` 创建空 imported interface 后配置和 generate 成功，真实进入 `approx_attn.cu`、`batch_prefill.cu`、`bsk_ops.cu`、`topk.cu`、`rms_norm.cu`、`estimate.cu`、`page.cu` 编译；传入了 `RAFT_NVTX=OFF`，但实际 NVCC 命令仍含 `-DNVTX_ENABLED`，变量展开尚待核对。
- 首次 Ninja 构建最终退出码1，8个官方 CUDA 对象全部在相同的 spdlog/fmt 头文件处失败：fmt 10.1.1 在 NVCC+C++17 下根据 `core.h:93-106` 将 `FMT_CONSTEXPR` 置空，而 spdlog 1.12.0 `common.h:349-352` 仍以 `constexpr` 调用 `buf.data()`。仓库外最小复现确认：原 C++17 配置失败；保持 C++17 并显式 `FMT_USE_CONSTEXPR=1` 编译通过；改试 C++20 反而出现更多 fmt constexpr 错误，未采用。
- 已将 `FMT_USE_CONSTEXPR=1` 通过同一仓库外 CMake 注入文件仅添加到 CUDA 编译定义，未改 Quest CUDA 源码、fmt/spdlog 头文件或算法。首次重配 `bx1826q3i` 因相对 RAFT 路径被 CMake 按 build 目录解析而在配置阶段失败，未进入编译；已改用绝对依赖路径启动 `bjcg8bsnh`，尚未生成 `_kernels.so`。`quest/ops/CMakeLists.txt` 仍仅为本机构建临时使用 `/usr/bin/gcc/g++`，构建结束后必须恢复官方 gcc-11 两行。
- JSSA 公网 CU13 依赖任务 `buay6fmix` 仍在运行，最近只明确完成 `nvidia-cufile` 下载；不能把目录大小或下载中状态记为安装/CUDA可用。vLLM v0.30.0 标签实为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，与指定提交不符，已排除该替代路径。

### 2026-11-13：Quest 原生 GQA 扩展、官方回归与 warmed smoke 闭环

- 显式 `cmake --build ... --target _kernels -j8` 已成功链接真实官方扩展；默认 all-target 的额外 RAFT host 对象在 fmt warning-as-error 处失败，与 `_kernels` 数值代码无关。发现 CMake 只写构建目录、不会自动覆盖 Python 包目录旧 `.so`，已部署新产物并以运行时 SHA 验收，当前扩展 SHA256=`de7d6e32052fd63206c7ba20b54d9040c40f8dc6617eb2a392c80e0bcc1b6b1f`；`quest/ops/CMakeLists.txt` 已恢复官方 gcc-11 两行。
- 原生 GQA 不能仅改 buffer 头数：官方 `paged_kv_t` 原路径同时以 KV head 索引物理 KV 和每个 query head 独立的 Top-K 页表，且该 fork 的 `GROUP_SIZE>1` decode 与32头 reference 不一致。最终实现每个 query head 独立 CTA，`qo_head_idx` 负责 Q/O/独立页表，`kv_head_idx=qo_head_idx/group_size` 负责共享物理 KV；等头数默认路径不变。
- 聚焦32头重复KV reference vs 8头原生KV candidate 对拍真实通过：RoPE、物理KV、32头metadata、prefill、estimate、Top-K indices、decode sparse attention 和 page indices 均逐位一致；`tests/test_official_longbench.py` 在显式 CUDA 门禁下为6/6 PASS。
- 官方 `pyproject.toml` 虽固定 Transformers 4.45.2，但 `test_rope.py` 的 reference 仍调用旧 `LlamaRotaryEmbedding.forward(x, seq_len=...)` API。4.44.2/4.38.2 均在 reference API 处失败，并非 CUDA 数值失败。最终只在 `/tmp/quest-transformers436-clean` 隔离使用 Transformers 4.36.2 + tokenizers 0.15.2，未改生产依赖；当前 GQA 扩展六组官方 tests 真实结果为 **199 passed、28 skipped**（含 RoPE 64/64）。
- 当前扩展的真实 Qwen3 warmed smoke 已完成：`narrativeqa` sample0、1 warmup pair、固定33 token、B1、budget=1024/page16。runtime 明确32个query/metadata heads、8个物理KV heads、HF仅第0层保存长度cache；官方调用计数为 prefill144、estimate2176、Top-K2176、sparse-attn2304、fallback0。
- 新旧 smoke 输入哈希均为 `7b0573de...bc8927`，33个 token IDs 与 EOS 位置7/19/31完全一致，输出 SHA256 均为 `e03e24067ccbcd1b7c1d624a940ee615f54143a20bf81069e30cc2f91659edde`，证明原生GQA与第0层HF长度缓存没有改变该请求的生成语义。新 smoke 文件=`/tmp/quest_official_smoke_gqa_warm_20261113/quest_performance.json`；单次 TPOT=21.286889ms 只作 smoke，不是正式性能结论。
- JSSA 原 CU13 下载任务 `buay6fmix` 已因30s网络超时失败；使用相同固定版本和 `UV_HTTP_TIMEOUT=600` 重启为 `bk0uq1kgd`，截至本段仍在下载，Torch/CUDA、patched-vLLM、官方tests和真实smoke均未验收。

### 2026-11-13：256K 官方 Quest 容量闭环与长上下文 mask 修复

- 首次256K集成探针在真正进入 Quest prefill 前失败：HF eager mask interface 物化 `[1,1,262144,262144]` 布尔 causal mask，真实 CUDA OOM 为尝试分配64GiB。将模型加载的 `attn_implementation` 改为 `sdpa`，仅让 HF mask interface 使用 `is_causal` 快路；36层 attention 仍全部由官方 Quest 接管，不执行 SDPA attention。聚焦 CPU 回归为6 tests OK（显式 CUDA 对拍按门禁跳过）。
- 第二次256K探针 `bwcqomo6r` 真实成功：262144输入token、生成2 token、36次官方 prefill、34次 estimate/Top-K、36次 sparse attention、fallback=0。runtime 仍核验32个query/metadata heads、8个物理KV heads、单层HF长度cache及当前扩展 SHA256=`de7d6e32...`；该冷启动2-token探针只证明容量与官方路径，不能当作性能结果。
- 新 mask 快路下31.5K warmed smoke `baukyko7k` 已完成，输入哈希 `7b0573de...bc8927`、33个输出token SHA256=`e03e24067ccbcd1b7c1d624a940ee615f54143a20bf81069e30cc2f91659edde`、EOS位置7/19/31均与原路径一致；调用计数prefill144、estimate2176、Top-K2176、sparse-attn2304、fallback0。单次TPOT=21.441936ms仍只作smoke。
- 512K同协议容量探针已启动为 `bchahykjb`，本段写入时尚未收取结果。JSSA CU13依赖 `bk0uq1kgd` 仍有实际公网连接但尚未安装完成；官方 wheel 元数据URL复验继续连接超时，官方 tests/smoke 仍未验收。

### 2026-11-13：源码交付形态与512K分块修复迭代

- 主仓库以 `.gitmodules` 固定 Quest `01c1623b...` 与 JSSA `34ce5abd...` 官方克隆；Quest 当前六个原生GQA修改文件保存为 `benchmark/official_baselines/quest_native_gqa.patch`，并已用 `git apply --check --reverse` 验证补丁与当前工作树精确对应。`.venv`、公网下载缓存、编译目录和 `_kernels.so` 不进入主仓库；运行时仍以扩展 SHA256 和官方提交/补丁哈希验收。
- 首次512K分块复测 `botwc8w0w` 越过此前的 decoder-layer RMSNorm，但在4D `[B,L,H,D]` q_norm 处再次 OOM：FP32转换与 `pow(2)` 各需8GiB。分块辅助函数已修正为对3D/4D输入都沿 token 维1切分，3D与4D Qwen3RMSNorm 聚焦对拍均逐位一致，7 tests OK（显式CUDA测试按门禁跳过）。第三次512K探针 `bhqj2mbhc` 已真实进入GPU计算，当前约137.6GiB/100% util，尚未完成，不能提前记为成功。
- JSSA 内网镜像 dry-run 明确返回 `cuda-toolkit[cublas]==13.0.2` 无可解版本；公网下载仍在低速传输，官方 wheel 站点仍超时。环境未闭环，不生成JSSA性能或精度结果。

### 阻塞与下一步

1. 收取512K官方 Quest 容量探针，随后建立带warmup的256K/512K正式性能测量；冷启动探针不计入TPOT。
2. 取得同一 vLLM 提交的官方 cu129 预编译 wheel，完成 patched-vLLM editable 安装并运行官方 JSSA kernel tests；若官方运行时无法获得或550驱动不支持，如实记录阻塞，不退回 Torch，也不换不匹配的 wheel。
3. 两条官方路径真实 smoke 通过后删除旧 Quest/JSSA vendor 路由和文件，保留 AsyncTLS；按用户顺序先在本机双卡跑性能，再跑精度。
4. 新结果落盘、官方评分、哈希 manifest 和独立验收前不宣称收官；GLM AsyncTLS 已完成1200/3750的断点保持不动。
