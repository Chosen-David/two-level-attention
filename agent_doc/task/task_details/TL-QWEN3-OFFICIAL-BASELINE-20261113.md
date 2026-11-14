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

### 2026-11-13：512K 官方 Quest 容量闭环与 WIP 源码推送

- 第三次512K探针 `bhqj2mbhc` 已真实完成：输入524288 token、输出 `[11,11]`、36次官方 prefill、34次 estimate/Top-K、36次 sparse attention、fallback=0。runtime 核验32个query/metadata heads、8个物理KV heads、单层HF长度cache、65536-token分块及扩展SHA256=`de7d6e32...`；冷启动2-token总耗时997.4411583691835秒，只证明容量与官方路径，不作为性能结果。原始任务输出已固化为 `agent_doc/results/qwen3_official_long_context_20261113/quest_512k_capacity_probe.log`（SHA256=`dd81abf6f1b5f58501b51d7614139fc3f7cf6054e615e1a38701b0017655d602`），结构化回执为同目录 `quest_512k_capacity_probe.json`；原始探针未记录可复用输入哈希，正式性能必须重新生成并登记对齐长输入。
- 按用户指令，独立runner、Qwen3-Quest接口、原生GQA补丁、官方子模块gitlink、fail-closed路由、聚焦测试与本任务文档已提交并正常推送 `origin/main`，commit=`41006cf8298167eed4b9bac386dfaa2a6c24bc85`。提交前Python编译通过、普通源码 `git diff --cached --check` 通过、GQA补丁 `git apply --check --reverse` 通过；聚焦回归7 tests OK（常规环境显式CUDA对拍跳过1项，CUDA对拍此前已独立通过）。GLM任务索引、profiler SQLite、`.venv`、构建目录、下载缓存和 `_kernels.so` 未进入提交。
- JSSA CU13依赖任务 `bk0uq1kgd` 最新输出仍只有各CUDA库下载项，未出现安装完成或真实import结果；继续按未完成处理。

### 2026-11-13：四项审计修复、真实 smoke 与 JSSA 环境复核

- RULER 已切换到独立官方入口 `benchmark/official_baselines/ruler_official.py`；Quest/JSSA 共用不可变 generation、单指针和 producer-yarn-config-v2 回执，旧 `pred_ruler.py` vendor 分支 fail-closed。32B/TP2 未验收脚本同样 fail-closed，不再误用 Qwen3-8B 单卡适配。
- LongBench 新输出已生产完整 `.runcap.json`，覆盖完整分片、过滤子集、空过滤和中断状态；真实 `samsum` smoke 首次仍生成128 token，定位到换行会并入其他 BPE token（如 `".\n"`），仅停止 token198 不足。现 Quest 使用 HF `stop_strings=["\n"]`，JSSA 使用 vLLM `stop=["\n"]`，并保留独立换行 token 停止和首 token EOS 抑制。修复后同一样本生成8 token，严格 runcap 与未修改官方评分端通过；单样本分数80.0仅为 smoke，保持 pending，不是全任务质量结果。
- Quest 新增 `native` 与 `hf-position-embeddings` 两种显式 RoPE 模式；native 遇 YaRN fail-closed，HF 模式消费 Qwen3 按 effective YaRN 生成的 `(cos,sin)`，Q/K 各旋转一次后进入官方 Quest KV/attention。真实32K native 与64K YaRN factor2 RULER 单样本均生成128 token、答案 `4226067`、36层官方路径、fallback0，运行时 RoPE 验证和未修改 generation 消费者均通过；结果在 `agent_doc/results/qwen3_official_ruler_smoke_20261113/`，仍只是单样本 smoke。
- 用户提供 wheel `third-party/JSSA/.venv/vllm-0.21.1rc1.dev192+g39910f2b2.cu129-cp38-abi3-manylinux_2_28_x86_64.whl` 复核仍只有13107200字节、SHA256=`68b5bcbe26c570d10729a09120b9da66d6ccd91a22d288ad5c9a0b18d34a1030`，`zipfile.BadZipFile`，缺中央目录，属于截断文件；未删除、覆盖或安装。pc40/pc31/pc32/pc02 对 `wheels.vllm.ai` 的固定提交 cu129 下载均明确连接超时，本机 HTTP/HTTPS 也不可达，内网 PyPI 无该开发版，WebFetch 受网络安全策略阻断。当前没有可用的完整固定提交 wheel，JSSA 官方 tests/smoke 仍未验收，不换提交、不换 CUDA 变体、不退回 Torch。
- 四项修复后的聚焦回归曾为 LongBench 18 tests OK（CUDA对拍按门禁跳过1项）+ RULER 4 tests OK。新增256K/512K长性能 RoPE 测试后为 **20 tests OK（skipped=1）+ 4 tests OK**。

### 2026-11-13：长性能旧口径失败证据与 YaRN 对齐修复

- 旧256K性能只在 repeat0 落盘一次 `TPOT=27.92742312885821ms`，repeat1 发生 CUDA illegal memory access，JSON 保持 `status=in_progress`；旧512K在 warmup 的 `T(33)` 阶段于 `v_proj` 表面栈报 `CUBLAS_STATUS_EXECUTION_FAILED`。异步表面栈不等于实际出错 kernel，两个错误也不无证据合并根因。
- 进一步审计发现旧 `long_context_official.py` 未加载/修改 HF config，仍强制 native RoPE，与 NFA/FlashInfer 的256K factor8、512K factor16不一致。因此旧27.927ms既缺 repeats，又属旧 RoPE 口径，不得作为对齐后的正式 TPOT。
- 现长性能入口在 `OfficialEngine` 构造前调用统一 `configure_rope(..., yarn=True)` 和 CPU preflight，设置 `model_config`、Quest `hf-position-embeddings` 与 JSSA `hf_overrides`；构造后分别验证真实 HF model 或 patched-vLLM effective config，并把 `rope_policy`、`rope_preflight`、`runtime_rope_verification` 写入性能 JSON。测试覆盖两档 factor、两种 method 的前置字段与构造后验证对象。
- 双 H20 在无其他 compute process 后启动同步 CUDA 诊断：256K `bo2njg4fk`（GPU0，1 warmup+2 repeats），512K `b46nxkq7n`（GPU1，1 warmup+1 repeat），`CUDA_LAUNCH_BLOCKING=1`，输出分别为 `quest_256k_yarn_sync_diagnostic.json` 和 `quest_512k_yarn_sync_diagnostic.json`。启动时 Python PID=3681507/3681505；最早检查尚在模型加载前、GPU各1MiB，不记为已点火或已完成。诊断不是 clean 性能测量，即使完成也不以其同步耗时作为正式 TPOT。

### 2026-11-13：同步 CUDA 栈与长序列最小复现边界

- 同步诊断已收割：256K 完整模型真实失败于 `qwen3_quest_kernel.py` → `quest.utils.decode_estimate` → `estimate.cu:77`，512K 真实失败于 `decode_sparse_attn` → `approx_attn.cu:143`，均为 `an illegal memory access was encountered`。旧 `v_proj`/cuBLAS 栈不再作为根因证据；尚未修改 CUDA 或宣称修复。
- 仓库外 `/tmp/quest_long_minimal_diag.py` 直接调用当前真实 `quest.utils`/`_kernels`，不执行 HF 模型或 Torch attention。256K 单层 native（32 query/metadata heads、8物理KV heads）与32物理KV reference 均完成2步真实 append/estimate/Top-K/sparse decode；随后256K/512K native 均完成36层、两请求、每请求2步 decode 和 `clean_states()` 后页池重用。结果分别为 `/tmp/quest_long_minimal_{256k,512k}_native_l36_r2.json`，状态 complete。它们只排除“长度、36层大地址空间或跨请求页重用单独必然触发”，不是模型精度或性能结果。
- 已通过合成实验与失败现场仍有三个关键差异：合成输入为16对齐的262144/524288且仅2步；真实性能输入为261935/523827且33步；生产 controller 在第0/1层使用大 page budget，第2层 `end_forward()` 后以 budget64、`updateTensor=False` 重启。sparse planner 的 group 语义不一致目前仅为候选，不能替代根因证据。
- 新一轮仅加入上述生产变量：双卡分别执行真实非对齐长度、36层、两请求、33步 decode 和生产 dense→sparse controller 生命周期；逐层同步并记录 KV/metadata seqlen、last-page length 和页数。后台任务为256K `b2c9kf701`、512K `bsh0agtgp`，临时 JSON 为 `/tmp/quest_long_minimal_{256k,512k}_native_l36_r2_d33_prod.json`。点火前双卡各1MiB、无compute process；两档均已 **complete**，每档2请求×33步全部完成，256K跨过了首个decode后的新页边界，512K跨过了后续新页及metadata页边界。该结果仍未复现完整模型故障，下一判别变量是尚未执行的真实官方 `prefill_forward`，不能据此宣称完整模型问题已消失。
- 已在同一最小脚本中单独新增 `--prefill-attention`，保持上述非对齐长度、生产生命周期、两请求和33步不变，只在每层 append 后增加真实官方 `prefill_forward` 并逐层同步记录。新任务为256K `bqf65ah01`、512K `bu7y8ixpn`，输出 `/tmp/quest_long_minimal_{256k,512k}_native_l36_r2_d33_prod_prefill.json`；本段写入时尚未完成。该实验用于判别 prefill 是否造成后续 decode 内存损坏，不测量或报告其同步耗时为正式 TPOT。

### 2026-11-13：memcheck 有效复核与精确 T(1)→T(33) 请求日程判别

- 256K `bqf65ah01` 已确认 complete：真实官方 `prefill_forward`、非对齐261935、36层、两请求各33步全部通过，不能把真实 prefill 单独视为充分触发条件。512K `bu7y8ixpn` 已完成首请求36层 prefill 与33步 decode，第二请求仍在运行，当前不记整项完成。
- 首次256K memcheck `bjtccke04` 的 `ERROR SUMMARY: 1 error` 是初始化阶段 `ExpandableSegment::map → cuMemCreate → CUDA_ERROR_NOT_PERMITTED`；其后真实36层、两步 decode 均完成。该错误不能解释为 Quest 越界。去掉 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后，`bzkpdxvaz` 对相同真实 Quest 合成路径完成两步 decode，`ERROR SUMMARY: 0 errors`；原始日志=`/tmp/quest_long_minimal_256k_sanitizer_no_expandable.log`，回执为同前缀 JSON。此处仅表示该配置未检出非法访问，不证明完整 HF 路径无 bug。
- 复核同步栈发现旧合成实验尚未对齐 warmup 请求日程：完整模型第一次 `T(1)` 只执行 prefill、没有 decode，失败发生于紧随其后的第二次 `T(33)`；旧实验首请求却先执行33步。仓库外诊断脚本新增 `--decode-schedule`，在其它变量不变时启动256K `0,33` 精确日程实验 `btcum86t4`，保留真实官方 prefill、36层、原生GQA和生产 budget 生命周期，输出=`/tmp/quest_long_minimal_256k_native_l36_r2_d0_33_prod_prefill.json`。该实验用于检验页池释放/重用顺序，结果未收割前不修改 CUDA、不宣称根因。

### 2026-11-13：精确请求日程与512K真实 prefill 均未复现

- 256K 精确日程实验 `btcum86t4` 已以退出码0完成，`/tmp/quest_long_minimal_256k_native_l36_r2_d0_33_prod_prefill.json` 为 `status=complete`、`decode_schedule=[0,33]`、两请求 decode 计数 `[0,33]`。它保留真实官方 prefill、非对齐261935、36层、原生GQA与生产 dense→sparse controller 生命周期；因此 `T(1)` 首请求无 decode、随后 `T(33)` 的页池释放/重用日程本身仍不是充分触发条件。
- 512K 真实 prefill 实验 `bu7y8ixpn` 的后台任务查询随后失效，但直接读取 `/tmp/quest_long_minimal_512k_native_l36_r2_d33_prod_prefill.json` 已确认 `status=complete`、两请求 decode 计数 `[33,33]`；第二请求并未失败。至此512K非对齐长度、36层、真实官方 prefill、生产生命周期和两请求33步的组合也未复现完整模型故障。
- 两项完成后双卡均回到1MiB、无 compute process。旧 native-RoPE 与新 HF-YaRN 的完整模型均在第二次 `T(33)` 失败，故不能把 YaRN 单独当作根因；现有合成结果也不证明完整 HF 路径安全。下一轮只在仓库外包装记录完整模型的 request/decode/layer、Q布局与页状态，先确定真实失败现场，不修改生产 Python/CUDA、不重建扩展。

### 2026-11-13：完整模型精确现场、256K 增强链路闭环与单变量判别

- 256K 原始完整模型追踪 `b9khigwyx` 失败于 request1/decode31/layer5 `decode_estimate`；失败 Q 为连续 FP16 `[1,32,128]`、storage offset=0，cache 为 KV/metadata seqlen=261967/16373、KV页16373、metadata页1024、budget64。512K 原始追踪 `b5tb67mow` 随后真实失败于 request1/decode17/layer2 `decode_sparse_attn`，Q同样连续且 offset=0，KV/metadata seqlen=523845/32741、KV页32741、metadata页2047、budget64；错误仍由 `approx_attn.cu:143` 抛出。原始512K日志=`/tmp/quest_512k_full_trace.log`，SHA256=`c8bf2e2e7e1dd7697b4e289a60d9b1146adb2cc0d9692d6f6a7c1d8ee57e9e53`。
- 256K 增强追踪 `b43l2drr7` 已完整完成 warmup pair 和1个正式 repeat：4次完整请求、prefill144、estimate2176、Top-K2176、sparse-attn2304、fallback0，正式请求33 token且正值计时字段齐全；最终 JSON=`/tmp/quest_256k_full_trace_sync_perf.json`，SHA256=`8137024b11fd07f5e75f29305ec5e0d3b96ee849f2c4f47caa18d1c9a6b48e59`，日志 SHA256=`ef749f1c37c869f713fd3f476a44b0d48e6eff999960be4540f3a0f4ded35fe5`。其75.27577574364841ms TPOT包含逐官方算子同步、逐层finite/index检查及D2H，**不是clean正式性能结果**。
- 增强256K在原失败点 request1/decode31/layer5 变为 estimate/Top-K/sparse全通过，Top-K物理页索引无越界、sparse输出无非有限值。日志共136条 `estimate_ok`；仅 request1/decode31/layer4–35 的32条 estimate score 出现非有限值（单层178000–506349个），request1/decode30及request3/decode31的已记录estimate均无非有限值。目前只统计 NaN/±Inf 合集，不能将非有限值直接认定为NaN、未初始化内存或illegal-memory-access根因。
- 为区分显式同步和额外张量检查，新增仓库外仅同步脚本 `/tmp/quest_full_model_trace_sync_only.py`（SHA256=`e178fa5c49e92e56581d9f51216f7905dff9beae295360f07987aae6a5145ae9`），保持相同生产模型、输入、YaRN、warmup日程和逐官方算子同步，但删除全部finite/index/min/max检查；256K任务=`bn7nc31ev`。同时以当前增强脚本启动512K任务=`bmh7fx09h`，用于确定512K原失败点在相同检查配置下的表现。两项仍在诊断阶段，未修改生产Python/CUDA、未重建扩展，kernel SHA256仍为`de7d6e32...bcc1b6b1f`。
- 256K仅同步任务 `bn7nc31ev` 已真实失败于 request1/decode31/layer4 `decode_estimate`，日志 SHA256=`772c695da5def43002df562c6f98e4f770eb078817f050fc0118399f3934d823`。因此显式同步本身不足以消除故障；该层也恰是增强日志首个出现非有限estimate score的层，但二者因果关系尚未证明。随后 `bfjrk73ub` 只保留与增强版完全相同的estimate finite检查，仍在同一 request1/decode31/layer4 estimate 失败，日志 SHA256=`1b3fc37e0a5a95fd0fe94aafb44136aab4ef1dfd3e6736b1f4507b99853e19e1`；仅estimate检查也不足以避免故障。
- 新任务 `b9pr9us51` 在相同同步和日志条件下只保留Top-K物理页范围/min/max检查，不做estimate或sparse张量检查，脚本 `/tmp/quest_full_model_trace_topk_check.py`。该臂已真实失败于 request1/decode31/layer5 `decode_estimate`，与原始包装失败点一致；日志=`/tmp/quest_256k_full_trace_topk_check.log`，SHA256=`4dab2e8f19d629ed23bc281da6dfb9d70d401369d665af7f7b81a8d34f04a037`。因此仅Top-K索引检查也不足以避免故障。
- 增强日志进一步显示 request1/decode30/layer4 与 decode31/layer4 的 estimate score 形状都为 `[32,16372]`，前者 `nonfinite=0`，后者为346211；两步共享同一新元数据项，decode31 是对该项的再次增量更新。此现象只是下一复现边界，尚不能区分元数据损坏、estimate 数值越界或更早内存破坏。GPU0释放后已启动仅sparse finite检查臂 `bmy03p2yu`，脚本=`/tmp/quest_full_model_trace_sparse_check.py`，日志=`/tmp/quest_256k_full_trace_sparse_check.log`；512K增强任务 `bmh7fx09h` 继续运行，二者点火后均确认真实进程、显存增长和日志进展。

### 2026-11-13：单变量故障矩阵闭合与真实 estimate 输入快照

- 仅sparse finite检查臂 `bmy03p2yu` 已真实失败于 request1/decode31/layer5 `decode_estimate`，与原始包装和仅Top-K检查臂一致；日志=`/tmp/quest_256k_full_trace_sparse_check.log`，SHA256=`a5e9820c12dfd83c36a144e8c70e0522380ddb975e08d2ca30dbbb60c35116dd`。该臂未记录任何 sparse 非有限事件，且失败前 request1/decode31/layer4 的 estimate/Top-K/sparse 均通过。至此仅同步、仅estimate检查、仅Top-K检查、仅sparse检查都不能消除故障，只有256K全同步+全检查臂曾完成，不能将检查扰动当作修复。
- 512K全同步+全检查臂 `bmh7fx09h` 已真实失败于 request1/decode18/layer29 `decode_estimate`，错误仍由 `estimate.cu:77` 抛出；日志=`/tmp/quest_512k_full_trace_sync.log`，SHA256=`bbe1dd469935447d54450fcbcbdd9f4d90894e57c43ce72be34100e5c2528911`。更早从 request1/decode1/layer4 起，多层 score 形状 `[32,32739]` 且 `nonfinite=1047648=32×32739`，即100%非有限；decode18 对应 `[32,32740]`、`nonfinite=1047680=32×32740`。未检出Top-K物理页越界或sparse输出非有限，但尚未区分NaN、±Inf与未初始化输出，不能据此把根因归为Top-K或sparse。
- 新增仓库外一次性快照脚本 `/tmp/quest_full_model_capture_estimate.py`，只包装真实官方调用并同步，在原 estimate 前保存目标层的Q、整层metadata、indices、indptr及页标量；若原 estimate 成功，再保存score并分别统计NaN/+Inf/-Inf。512K任务 `bqlqvqqxt` 目标 request1/decode1/layer4，256K任务 `bb2z4zeoz` 目标 request1/decode31/layer4；后者日志=`/tmp/quest_256k_capture_estimate.log`。两项只用于故障复现，不是性能测量，尚未取得快照前不记为完成。
- 新增仓库外短进程重放脚本 `/tmp/quest_replay_estimate_snapshot.py`，直接调用同一真实 `quest._kernels.estimate_attn_score`，记录输入有限性、物理页范围、输出初始化和跨repeat逐位结果。后续将分别用 empty/固定填充输出判别“kernel产生非有限值”与“输出未完整写入”；若短进程不复现，则保留完整HF路径更早内存破坏的假设。生产Python/CUDA与本地GQA补丁均未修改，真实扩展SHA256仍为 `de7d6e32052fd63206c7ba20b54d9040c40f8dc6617eb2a392c80e0bcc1b6b1f`。
- 256K目标快照任务 `bb2z4zeoz` 已真实完成并主动退出，目标 request1/decode31/layer4 的本次 estimate 成功，score 的 NaN/+Inf/-Inf 均为0。输入快照=`/tmp/quest_256k_r1_d31_l4_estimate_snapshot.pt`，SHA256=`16a9de0a025e4102601d72557d7ff011dbbad68367bf8db3faa361e5b4d67afd`；score=`/tmp/quest_256k_r1_d31_l4_estimate_snapshot.score.pt`，SHA256=`8eccdf242c766b1ca795dbd99c44eb29332129162e9a82d21270830d7f145efa`；日志 SHA256=`271c155fc74ee7630b54c2e0395beb918b8d0a1e7951a8555d18640a1157ed44`，快照脚本 SHA256=`cb82e9cd9e82decdde9f2fd1d77801c0dd7ed8a62156a0e845d0889ea1f285aa`。这是一份健康现场，只证明相同 request/decode/layer 坐标不是充分触发条件，不能替代此前故障现场。
- 首个重放版本误把物理 metadata 页表长度当作逻辑条目数，使用 `output_len=indices.numel()-1`，将本应为 `[32,16372]` 的输出错误分配为 `[32,1023]`；真实 kernel 按逻辑几何写出，造成该短诊断进程自身越界。因此 `/tmp/quest_256k_r1_d31_l4_replay_empty.log`（SHA256=`f292368f93aa01e88b6de39e0fa50af38bea4b3bbb32db3b2bc4af4e904763fa`）**整次作废，不作为Quest证据**。正确几何为 `(metadata_indices.numel()-1)*page_size + metadata_last_page_len - 1`；本例物理页1024、page16、last_page_len5，对应逻辑 metadata seqlen16373、输出长度16372。
- 修正后的重放脚本 SHA256=`1b2655c6e39dc5add4166375596850dc0745955bf09f7c2b302332d8de09f9da`，正确日志=`/tmp/quest_256k_r1_d31_l4_replay_empty_v2.log`，SHA256=`de0e76475048ff039405ca336f3b398b1ecb63cb158f9add13edc306e826b581`。Q的4096个元素、整个metadata buffer及134127616个有效引用元素均有限，有效值绝对值最大55.5；indices为1024个唯一物理页、范围0–1023、越界0，indptr=`[0,1024]`。真实 estimate 输出 `[32,16372]` 全有限，3次repeat逐位一致，与完整模型健康score逐位一致（differing=0），8个抽样位置的独立FP32参照经FP16舍入后也逐位一致，包含最后位置16371。结论仅限该健康输入在独立短进程中稳定、正确、可重复。
- 为捕获真正故障而非再次取得健康输入，新增 `/tmp/quest_full_model_capture_nonfinite.py`（SHA256=`d7338fc4e8a434f303c62a1c3a4522b2b2e42852bdd57c1e3115e19c64060597`），保留全同步、estimate finite、Top-K索引与sparse finite检查，在首个非有限estimate后保存同类输入快照和NaN/+Inf/-Inf分类并退出。256K任务=`b7tee5c40`、日志=`/tmp/quest_256k_capture_nonfinite.log`；512K目标快照任务 `bqlqvqqxt` 继续运行。本段记录时两者均已进入第二请求prefill但尚未生成目标快照，均未完成。

### 2026-11-14：512K真实NaN现场闭环与上游前向定位

- 256K首个非有限捕获任务 `b7tee5c40` 未等到非有限检查触发，已在 request1/decode31/layer3 `decode_estimate` 直接报 illegal memory access；request1/decode31/layer2 的estimate/Top-K/sparse均通过，Top-K索引无越界、sparse非有限计数0。未生成 `/tmp/quest_256k_first_nonfinite_estimate_snapshot.pt`，失败日志 SHA256=`0ea2e8524560961c8afe8e31849e5f4c0409dfa4ccb466348371ad85acb0d77f`。256K真实故障输入仍未捕获。
- 512K目标快照任务 `bqlqvqqxt` 已真实完成：request1/decode1/layer4 的score共1047648个元素，NaN=1047648、+Inf=0、-Inf=0。输入快照 SHA256=`978f38722eddb0b61360ecda576ef769d4afb8bab09b66ca56810b0bdf07f7cf`，score SHA256=`f9dbe2b48fd8e1aea4b8fed69561f469ba4a6ea87c7ac257729df2465a0fb011`，捕获日志 SHA256=`40388a3b4efe14d7df794602c8471f9959bbcda780c06405facc4e997e761758`。
- 修正后的短进程已以退出码0重放该512K真实输入，日志=`/tmp/quest_512k_r1_d1_l4_replay_empty.log`，SHA256=`9bb6e4a52643bbd620b2c05f6192e9b94c9723d5f84c18e14bf2a186718a9e04`。**Q的4096个元素全部为位型`0x7fff`的NaN**；整个metadata buffer及268206080个有效引用元素全部有限，有效绝对值最大57.28125；2047个物理页索引唯一且范围0–2046，indptr=`[0,2047]`。真实estimate的3次输出原始字节SHA均为`621b64bdd00b3c7b3b9316c2cec9f5ae758dca7ad477956afaa82c72d78537cf`，与完整模型保存score的原始字节SHA完全相同；独立FP32参照也全为NaN。脚本中的 `torch.equal`/`!=` 对NaN返回假分歧，不能解释为数值不一致，原始字节SHA已闭合。故这次score全NaN是上游Q已损坏后的必然结果，不是estimate凭空产生NaN，也不是metadata或页表非法所致。
- 口径更正：此前512K增强日志的 `sparse_ok` 只在 `decode_idx>=30` 时发出，decode1没有sparse有限性记录；因此不能声称decode1的sparse输出已验证有限，只能说当时未记录sparse非有限事件。新诊断 `/tmp/quest_full_model_capture_first_forward_nonfinite.py`（SHA256=`2f5ca713dac5873ef5e8085d6e90a2a375cc302435be6e0d13c791d2ca91123e`）已在第二请求decode的小张量上逐模块检查输入/输出，目标是在首个NaN处保存模块名、输入、输出和前一有限模块；长prefill不做逐元素检查。任务=`bz5uty8bo`，尚未取得结果，生产Python/CUDA仍未修改。
- 256K逐模块v1任务 `bqx11apn6` 未生成NaN快照，已在某个 `self_attn.o_proj` 的 `torch.isfinite(...).all().item()` 检查处暴露 CUDA illegal memory access；v1没有检查前模块名日志，无法从异常栈确定request/decode/layer坐标。失败日志=`/tmp/quest_256k_capture_forward_nonfinite.log`，SHA256=`a298a933de205eeeb80588218c3ab4285ef1646757ca4d6e72dba4ddc9417e8b`。
- 为不影响仍在运行的512K v1字节身份，另建 `/tmp/quest_full_model_capture_first_forward_nonfinite_v2.py`（SHA256=`9c7781474ea1497fde9d7bf096f60784feaedea516b28b82a0901c9f2acc9d94`），仅在每次finite检查前增加模块名事件。256K v2任务 `brrn601l7` 已真实失败于 **request1/decode12/layer6 `self_attn.o_proj` 输出检查**，未生成 `/tmp/quest_256k_first_nonfinite_forward_v2.pt`。失败前request1/decode12的layer0–5被检查模块、layer6 `input_layernorm`、`q_proj/q_norm`、`k_proj/k_norm` 和 `v_proj` 的输入/输出均有限；最后事件位于日志6650行，随后6743行抛出illegal memory access。失败日志=`/tmp/quest_256k_capture_forward_nonfinite_v2.log`，SHA256=`19c3ac9f73737f256d90dc048efae9e397bfe61c2be2bef35491de130801a2dc`。该栈只说明错误在 `o_proj` 输出同步检查时被暴露，实际非法访问仍可能来自此前Quest sparse/append/其他异步kernel或 `o_proj` 本身，不能据此定罪 `o_proj`。
- 512K逐模块任务 `bz5uty8bo` 已结束并生成 `/tmp/quest_512k_first_nonfinite_forward.pt`。首个非有限边界为 request1/decode1/layer4：`self_attn` 输出4096个元素全部有限，随后 `post_attention_layernorm` 输入含29个NaN、输出4096个元素全部NaN。29个NaN集中在hidden维3624–3698的局部区间，符合相邻内存被局部覆写后继续经residual/RMSNorm传播的现象。日志 SHA256=`d883ad9ab201dbcb4b942f97bb75f98309a592733f035a0b957c881f78a27c4b`，快照 SHA256=`edf88ff3a860e9596a83435326611d19e5bf4cb7d78ee3c593d090ed7d042027`。

### 2026-11-14：官方 RAFT Top-K workspace 越界根因与动态修复

- 生产 `InferenceController.begin_forward()` 原先把 `topk_buf` 固定为32头×1024个FP16元素=65536 bytes；但真实 `_kernels.topk_filtering` 进入 RAFT `radix_topk_one_block_kernel` 后，按实际score长度重新计算每头 `buf_len=round_down(len/(ratio*8),128)`，其中FP16 value+Int32 index的ratio=6，并按 `num_heads*buf_len*2*(2+4)` bytes寻址和写入。生产Top-K输入长度与 `decode_estimate` 完全相同，为 `metadata_cache.seqlen-1`；因此256K的16372需要98304 bytes，512K的32739/32740需要245760 bytes，均超过旧固定空间。官方CUDA benchmark本来会随seq_len分配临时空间，Python controller却未同步该逻辑；官方 `test_topk.py` 的最大长度1110也未覆盖此问题。
- 仓库外独立脚本 `/tmp/quest_topk_workspace_canary.py`（SHA256=`da2b8f580a52f789f649595e766862583f92ff3621629f265be2d029f21a2c18`）直接调用固定真实扩展 `quest._kernels.topk_filtering`，不导入NFA/HF模型，并在workspace后放置1MiB、值0x5A的guard。原始日志 SHA256=`16b24d426661b8caf53538add51233490ec902ab5f92d51ec346950c8e7b43ff`；修复后复跑日志 `/tmp/quest_topk_workspace_canary_after_fix.log` SHA256=`ba72c40f221f5e1b04af6a08c2f19f13375fe11ca3820289496fb9a7e21c4a9b`。真实结果如下，排序后Top-K值最大误差在所有格均为0，说明仅检查Top-K数值不能发现内存破坏：

| score长度 | workspace模式 | 分配bytes | RAFT需要bytes | guard改写bytes |
|---|---|---:|---:|---:|
|1110|固定8192|65536|0|0|
|1110|按公式|0|0|0|
|8192|固定8192|65536|49152|0|
|8192|按公式|49152|49152|0|
|16372|固定8192|65536|98304|5150|
|16372|按公式|98304|98304|0|
|32739|固定8192|65536|245760|21778|
|32739|按公式|245760|245760|0|
|32740|固定8192|65536|245760|21776|
|32740|按公式|245760|245760|0|

- 现已在 `quest/utils/controller.py` 增加与当前RAFT公式一致的 `_topk_workspace_shape`，生产分配长度严格使用 `metadata_cache.seqlen-1`；保留官方RAFT Top-K、页预算、GQA和全部选择语义，没有增加fallback。`quest/tests/test_topk.py` 已改用同一空间公式，并新增8192/16372/32739/32740四组显式预期字节数+1MiB guard+排序值回归。聚焦结果为 **53 passed、15 skipped**；主仓库官方LongBench/RULER回归为 **23 passed、1 skipped、4 subtests passed**。`quest_native_gqa.patch` 已同步包含控制器与测试修改，反向 `git apply --check --reverse` 通过；运行时原始 `git diff` SHA256=`edb9e2d579687781a9b293ed87dc2445664db25118811bc5a8227b5ea6792b8f`，仅为消除主仓 `diff --check` 的空白上下文告警而规范空行后的交付补丁 SHA256=`4e460c0c4e57f013a554f147bdc40a53fe29d522cc1d36ef9c721908cbbaeac9`，两者源码语义一致。
- 本轮只修改Python控制器和测试，没有修改或重建CUDA；真实扩展SHA256仍为 `de7d6e32052fd63206c7ba20b54d9040c40f8dc6617eb2a392c80e0bcc1b6b1f`。直接越界机制已闭环，但完整模型故障是否全部消除仍须生产路径验收。已分别在双卡启动256K `bh481o2ew` 与512K `bfw49n4sr`，均为原始 `long_context_official.py`、1 warmup pair+1 repeat、无同步注入/无finite检查/无profiler；输出分别为 `quest_{256k,512k}_workspace_fix_validation.json`，日志在 `/tmp/quest_{256k,512k}_workspace_fix_validation.log`。两路已核验真实进程、显存增长和模型加载日志，完成前不记修复验收通过，也不将其单次计时作为正式性能。
- 六组官方Quest CUDA tests已在修复后完整重跑：**203 passed、28 skipped**（原199 passed+新增4 guard）；显式 `RUN_QUEST_CUDA_TESTS=1` 的原生8物理KV头 vs 32重复KV头逐位对拍也再次通过。
- 256K完整模型验证 `bh481o2ew` 已以退出码0完成：原失败坐标及后续warmup pair+1 repeat均通过，JSON `status=complete`，实际33 token、EOS未提前停止，输入261935 token，输入IDs SHA256=`05325d14fa6868ab22b6964d160adaa1968323ca80e396261b97ac75b55c093d`，输出IDs SHA256=`68978028f4e21f7f8e72c19fcfffc4630c548ec085ed0e3532916778bb99fd03`；官方调用计数prefill144、estimate2176、Top-K2176、sparse-attn2304、fallback0。JSON SHA256=`bdb3ad40540d4659af2cf5ae26998e114b7d4bd38fdbbfe1d17e5aaffb61445e`，日志 SHA256=`3e4112723739490c5904f57b55c9989ac89d6e5e01ae9a5c68b30ce087058466`。这证明256K完整生产路径的原始illegal-memory-access/NaN故障在本轮相同输入与请求日程下已消除；单repeat TPOT=29.177278ms只作修复验证记录，不替代3 repeats正式性能。512K `bfw49n4sr` 仍在运行，不能提前记为完成。
- 256K修复验证通过后，已在空闲GPU0启动正式3 repeats任务 `bjxfdg20n`，输出 `quest_256k_workspace_fix_performance.json`、日志 `/tmp/quest_256k_workspace_fix_performance.log`；协议保持1 warmup pair、33 token、无profiler/无诊断同步。任务刚点火时尚在初始化，完成并逐项验收前不报告正式TPOT。512K仍只运行修复验证，不提前启动其正式性能。

### 2026-11-14：双长度修复验收、256K正式性能与结构复核

- 512K完整模型修复验证 `bfw49n4sr` 已以退出码0完成：`quest_512k_workspace_fix_validation.json` 为 `status=complete`，实际输入523827 token、生成33 token、EOS未提前停止，输入IDs SHA256=`50d3339d268c4a2d6dcfc058e5c1640e663c633adf1515455f913c9e63ef8426`，输出IDs SHA256=`d5207f6c7d5053bff8bdcd275485df30ecfc83f0a8e2738d9f0cc838680518ed`；官方调用计数prefill144、estimate2176、Top-K2176、sparse-attn2304、fallback0。单次验证TPOT=46.951066702604294ms，仅证明本轮相同输入与请求日程下原始故障未复现，不替代正式3 repeats。JSON SHA256=`0041e120b9f33a0a2e4170d7dcbc0115843f00b5535718fe30a2981dee6a8c94`，日志 SHA256=`23db01adb76a1ae22a8d90b7b451f72fcbefc99408eb3015ec383fb92a923e45`。
- 256K正式任务 `bjxfdg20n` 已以退出码0完成1 warmup pair+3 repeats：三次TPOT分别为29.122937936335802、30.046682339161634、28.950894717127085ms，中位数 **29.122937936335802ms**；每格均为33 token、EOS未提前停止、无profiler/诊断同步，官方调用计数prefill288、estimate4352、Top-K4352、sparse-attn4608、fallback0。结果=`quest_256k_workspace_fix_performance.json`，JSON SHA256=`7d9e6edc29bcd7fcc411bdfa618be83e5db30ea289a32954d84e0866535cda38`，日志 SHA256=`c238d96b4bc7938fc66fc1611409f28f9030d57d6b79ec138ca9fa00bea1bad6`；继续保持 `pending-independent-verification`。
- 三份回执已用只读结构验收脚本逐项通过：完成状态、pending标记、repeat数、输入长度、33个token IDs、EOS语义、正值计时、TPOT=`decode_after_first_wall_ms/32`、YaRN factor8/16、官方Quest提交、运行时diff、扩展SHA、预算/page/头数、fallback0和官方调用计数均PASS。此前验收断言失败是脚本把每个pair的decode计数误写为 `34×32×2×pairs`；实际每个pair只有 `T(33)` 执行32步decode，`T(1)` 仅prefill。正确公式为prefill=`36×2×pairs`，estimate/Top-K=`34×32×pairs`，sparse-attn=`36×32×pairs`，其中pairs=warmup pair数+repeats（本轮为1+repeats）；三份JSON的实际计数原本就正确。
- **跨repeat数值差异尚未判别**：同一256K输入下，修复验证输出SHA=`68978028f4e21f7f8e72c19fcfffc4630c548ec085ed0e3532916778bb99fd03`，正式repeat0/1/2分别为`eade8c4372aec6436e387109fcb1b558dd4bba42eee91271a3862d09cfae2212`、`2b2da1a14cc5032526d8b5d68892a1e95fb726ed6085815c85e4b879eaea51ea`、`9e8eeccdb84c1377f4082cfa75ea5de3d6f11ef3951e56e42ad1c7838cc649b4`，三个正式repeat互不相同且均不同于验证。当前只确认无崩溃、完整33 token、官方高性能路径和计时字段完整；不得据此宣称同臂greedy数值确定性或精度已验收。逐token复核进一步显示：验证与repeat0首次在索引6分歧，验证与repeat1/2及repeat0与repeat1/2首次在索引5分歧，repeat1与repeat2首次在索引7分歧；不是仅哈希字段或末尾token差异。生产入口明确 `do_sample=False`，且每次请求在 `finally` 中调用 `reset_quest_controller()` → `clean_states()` 释放KV/metadata页，故不能用随机采样或遗漏reset调用直接解释。该差异保留为开放问题，需另行区分官方kernel/HF greedy非确定性、请求间状态复位完整性及残余内存/数值问题。
- 上述验收后实测本机GPU0/GPU1均为1MiB、0% util且无compute process。随后已在GPU0启动512K正式1 warmup pair+3 repeats任务 `bvq5mfh44`，Python PID=3838731，输出=`quest_512k_workspace_fix_performance.json`、日志=`/tmp/quest_512k_workspace_fix_performance.log`；仍使用原始clean入口，不加入同步、finite检查或profiler。点火后已核验真实进程、模型分片加载日志及GPU0约138.2GiB/100% util，GPU1保持空闲；完成前不记正式结果。

### 2026-11-14：Quest LongBench16 正式性能闭集

- GPU1正式性能任务 `bvytlw1r5` 已完成：英文16任务 × sample0 × 3 repeats 共 **48/48 cells**，每格均为固定33 token、EOS未提前停止、完整 `T(33)-T(1)` 两点计时。逐任务3次TPOT取中位数后再做16任务宏平均，**macro TPOT=19.365306608960964ms**，48请求总中位数TPOT=19.283101079054177ms；结果=`agent_doc/results/qwen3_official_longbench_perf_20261114/quest_performance.json`，日志=`/tmp/quest_official_longbench_perf_20261114.log`。JSON SHA256=`970c9d7e1a4505ff9efe2174d9d898a395a46fb3fe70deb962ea3029f8901908`，日志SHA256=`1c8740e29911f0a1743ed80388ebc9eef62ce352ed79101ac45060deced85aa2`，继续保持 `pending-independent-verification`。
- 只读复核全部通过：16个任务及48个 `(task,sample0,repeat)` 键完整且顺序正确；每任务3格的输入身份一致，16份原始JSONL SHA256均与回执一致，并用同一tokenizer/prompt独立重算16个sample0的token数与input IDs SHA256全部一致；输出IDs SHA256逐格重算一致，TPOT严格等于 `decode_after_first_wall_ms/32`，逐任务中位数与宏平均重算一致。64个pairs（16任务×4 pairs）的官方调用计数为prefill4608、estimate69632、Top-K69632、sparse-attn73728、fallback0，分别严格等于 `36×2×64`、`34×32×64`、`34×32×64`、`36×32×64` 和0；Quest提交、运行时diff、扩展SHA、32 query/metadata heads、8物理KV heads、page16/budget1024也全部匹配。
- 数值确定性开放问题仍需保留：13/16任务的3次输出完全一致，但 `qmsum` 的repeat0与repeat1/2首次在索引20分歧，`samsum` 的repeat0与repeat1/2首次在索引10分歧，`passage_count` 的repeat1与repeat0/2仅在索引32分歧。它们与256K跨repeat分歧共同说明当前不能宣称官方Quest greedy逐token确定；本轮回执只作为完整clean性能证据，不将输出一致性或质量标为通过。

### 2026-11-14：Quest LongBench16 全量精度四分片点火

- 性能闭集后实测本机GPU1、pc40双卡与pc31 GPU0空闲，pc31 GPU1及pc02/pc32各卡均有其他计算进程，未抢占；pc31 GPU1的GLM AsyncTLS进程保持不动。Quest当前真实路径仅支持B1，因此没有伪造BS2/BS4。全量质量采用 `source_order[rank::4]` 确定性分片：rank0=本机GPU1（任务 `bcaiekj17`，宿主PID=3848676），rank1=pc40 GPU0（PID=762540），rank2=pc40 GPU1（PID=762811），rank3=pc31 GPU0（PID=2891064）。四个进程均在点火后通过宿主 `ps`、约27GiB显存/真实GPU利用率和逐请求日志三重验证，rank0已产生 `narrativeqa` source0/4/8...，rank1/2/3分别产生source1/2/3模4序列，无重叠。
- 远端使用隔离worktree基线 `41006cf8298167eed4b9bac386dfaa2a6c24bc85`，叠加当前质量/runcap入口补丁SHA256=`b18a233f17287db67a1810d2c0142a34ff54d69d87e73b9feca348c1704dee7b`；Quest均固定 `01c1623bf9395009520874e989e29f683203b357`，运行时diff SHA256=`edb9e2d579687781a9b293ed87dc2445664db25118811bc5a8227b5ea6792b8f`，扩展SHA256=`de7d6e32052fd63206c7ba20b54d9040c40f8dc6617eb2a392c80e0bcc1b6b1f`。pc40/pc31快照均在 `/tmp/qwen3-official-quality-20261114`，入口导入与三项身份已核验；不影响远端既有主工作树和其他AI任务。
- 本机输出=`agent_doc/results/qwen3_official_longbench_quality_20261114`；pc40 rank1/rank2分别输出到 `~/ak_verify_results/qwen3_official_longbench_quality_20261114{,_gpu1}`；pc31 rank3输出到 `~/ak_verify_results/qwen3_official_longbench_quality_20261114`。日志分别为 `/tmp/quest_official_longbench_quality_s{0,1,2,3}of4_20261114.log`。质量入口保留首token屏蔽EOS、samsum换行停止、任务官方max-new、逐样本输入身份和runcap；四分片全部完成、合并3750请求、官方评分及独立验收前不报告AVG16或标任务完成。

### 2026-11-14：round-robin canonical 合并器与 narrativeqa 首任务收割

- 新增 `benchmark/official_baselines/merge_longbench_quality.py`：逐分片强制核验 complete runcap、结果SHA/行数、标准 `source_order[rank::world_size]`、rank/world/源人口闭包、逐行 `_id` 与源索引、行内input/provenance、implementation、模型/方法/cap/producer一致性；重读原始JSONL和prompt模板重算完整源身份，按源索引恢复JSONL及tokens/truncated/input-hashes，并在锁内原子生成canonical结果与runcap。聚焦测试覆盖逆序输入、合法空rank、重复/缺rank、记录身份错配、逐样本provenance漂移和原始源漂移；普通与 `python -O` 均为 **23 tests、1 skipped、OK**，`py_compile` 与 `git diff --check` 通过。首轮正向测试失败是mock生产使用 `{context}` 而测试误传仓库正式samsum模板，已让mock配置与生产模板一致，未放宽合并门禁。
- 已真实收割本机rank0、pc40 rank1/rank2、pc31 rank3的narrativeqa完整分片并成功合并 **200/200**：canonical=`agent_doc/results/qwen3_official_longbench_quality_20261114/pred/narrativeqa-quest_official_b1024_p16.jsonl`，SHA256=`71ae69771c46fb46af849c80a21ee920feadf47cbfda4ac46624aa46b843e6c6`；canonical runcap SHA256=`65f8bf73ffbe0dd2702dd80632f9efa5b840bbb937cca6b7d3b3bccba3f54b2b`。真实 `benchmark.LongBench.eval --expect-count 200` 严格消费通过，strict_validated=1、legacy=0，官方分数 **27.69**，结果=`narrativeqa_result.json`，SHA256=`1422c7ea1291eac9a6fc8dd67028341686d4d49d7483fb8e8f6d8dc668b6abdd`。这是单任务pending结果，16任务未齐前不报告AVG16。
- 继续收割后，当前已有 **12/16任务、2350/3750请求** 完成四分片canonical合并与官方评分：2wikimqa=42.95、gov_report=33.38、hotpotqa=58.64、multi_news=24.59、multifieldqa_en=53.53、musique=36.84、narrativeqa=27.69、qasper=47.98、qmsum=23.26、samsum=44.20、trec=71.50、triviaqa=90.05。聚合评分入口严格通过12份runcap与v2输入身份，`strict_validated=12`、legacy=0、cap-comparability-key=`6688f41924`；结果=`result_12tasks.json`，SHA256=`e2ec01707c26a54b580cc542fee7bcd7cb83b2d6e870036c3681c69ca298e7bc`。其余passage_count、passage_retrieval_en、lcc、repobench-p仍在生成，以上均保持pending，不冒充AVG16。
- 512K正式任务 `bvq5mfh44` 仍在运行，回执当前 `status=in_progress` 且仅repeat0落盘：TPOT=54.550930857658386ms、33 token、EOS未提前停止、官方计数与单个已完成pair一致；其余repeat未完成，故不记正式中位数或闭集。四个质量分片继续真实推进，pc31 GPU1的GLM AsyncTLS未接管。

### 2026-11-14：合并器最终回归与 LongBench16 全量质量闭集

- 最终版合并器保留行内 `input.input_ids_sha256` 的 JSON 紧凑序列哈希口径，不错误要求它等于 runcap 逐样本 Int64 小端字节哈希；新增漂移测试篡改 runcap 的 `actual_input_ids.per_sample_sha256` 后，由 `read_run_cap_manifest()` 的 `ordered_sha256` 闭包 fail-closed。最终普通与 `python -O` 聚焦回归均为 **23 tests、1 skipped、OK**，`py_compile` 和 `git diff --check` 通过；既有 `run_cap_manifest.py:801` 的 `ResourceWarning` 保留，不影响测试结果。
- 本机rank0、pc40 rank1/rank2、pc31 rank3四个进程均以 `[ALL DONE]` 正常结束，pc40/pc31双卡已释放，pc31 GPU1的GLM AsyncTLS仍未接管。远端全部64个分片及runcap已收割；其余passage_count、passage_retrieval_en、lcc、repobench-p均通过完整四rank闭包、源身份、逐样本provenance和源顺序合并。新canonical结果SHA256分别为：passage_count=`b8a21e4735694eafc72901c0300a1f5c52ad0de99a368aefcabda1910b7b9fc5`、passage_retrieval_en=`d3f9064bacea70c9977935da42a3de864f7197cb6b210a12bfb6a6eb391e50c8`、lcc=`9a7747cb9012a86e9b3e701a0b247c281a18e6889a68986c3111f5cfc26afce8`、repobench-p=`cc253547053afaa87214bf8f9aa5ea5b92e121c0e4364136ffd69719544e297f`；对应runcap SHA256依次为`5b25dcbefe4e7e2ba1053000812fa3da281366a0cc6482ff0480004a7ae87e4a`、`ab6679329d48dc9710a3a24a6ed9580d041fc6e1f5f60a1bd11d89529ee02a5c`、`ccb529a16bba72c69cfe844cf9f6fcb0532d5a22c1b915f6a3982086567a33fe`、`e9b8714ed013b69bf6930b4450cd4ae785d2af2ac078011dc109a585a75ff915`。
- 16份canonical及sidecar已复制到隔离目录 `/tmp/qwen3-quest-lb16-canonical-20261114-l1NmYf`，由未修改的 `benchmark.LongBench.eval` 严格消费；**16/16任务、3750/3750请求、strict_validated=16、v2_validated=16、legacy=0**，cap-comparability-key=`6688f41924`。最终 `result.json` SHA256=`4dcc12741206be45dc15ec447a4008f1311260531bc1026c75ce3ac3030f950e`，16任务分数为：2wikimqa=42.95、gov_report=33.38、hotpotqa=58.64、lcc=67.46、multi_news=24.59、multifieldqa_en=53.53、musique=36.84、narrativeqa=27.69、passage_count=2.50、passage_retrieval_en=100.00、qasper=47.98、qmsum=23.26、repobench-p=63.76、samsum=44.20、trec=71.50、triviaqa=90.05；逐任务分数宏平均重算为 **macro AVG16=49.270625**。首次评分命令误把 `--output-path` 当作结果文件路径而在 `os.listdir` 处失败，未生成结果；随后按其实际“预测目录”语义重新执行成功，未修改评分端或数据。
- 全量质量与Quest正式性能均保持 `pending-independent-verification`；greedy跨repeat差异仍是开放问题，不用本次完整人口与官方评分掩盖该问题。512K正式任务 `bvq5mfh44` 仍在真实GPU计算，尚未闭集。

### 2026-11-14：512K 正式性能闭集与质量证据归档

- 512K正式任务 `bvq5mfh44` 已完成1 warmup pair+3 repeats，回执 `status=complete`。三次TPOT分别为54.550930857658386、83.82336678914726、88.74027710407972ms，中位数 **83.82336678914726ms**；每格均为523827输入token、固定33生成token、EOS未提前停止，YaRN factor16及运行时验证通过。4个pairs的官方调用计数为prefill288、estimate4352、Top-K4352、sparse-attn4608、fallback0，严格符合 `36×2×4`、`34×32×4`、`34×32×4`、`36×32×4` 和0。只读结构复核已重算全部TPOT、request wall、输出IDs SHA、summary中位数及调用公式并PASS；结果=`agent_doc/results/qwen3_official_long_context_20261113/quest_512k_workspace_fix_performance.json`，JSON SHA256=`3d653abde14b8a7cb4f152cc12f4f206f3778fa43cf1b69e12d3df429f7d869a`，日志 SHA256=`fc765e9060f02613388cf2d8d887589cfefb7ad4b1c72f2def05205216b68399`，继续保持 `pending-independent-verification`。
- 512K三次greedy输出仍不一致：repeat0与repeat1/2首次在索引7分歧，repeat1与repeat2首次在索引9分歧；repeat1/2的TPOT也明显高于repeat0。上述性能离散与数值差异完整保留，不用中位数掩盖，也不宣称greedy确定性或质量数值一致性通过。
- 四个质量分片日志与manifest已归档并复核SHA256：manifest rank0/1/2/3分别为`07a698176556bcd809f562f41dece6eeda565cba03a632e2b375e3536749a21d`、`fae45c928c6febedd80a5add7da5940b4216c23d9e8f4c989d2bd6d66672d57b`、`a13d1b441b363b96fb22bb60a92e0a0dc3a70f96c2cff561e2d24dc095af400d`、`f1aa35578f49f83751e28814d74e2d2cb72285bba6a68d82b13c19aebcfd7a21`；日志rank0/1/2/3分别为`a5b826369e7e3d3375979841414f835422eca67a9a0a145e58ca9b401642945e`、`eb278db8b28de98ddfb8a06b0797ad6befda259bbbc58e3bdfaf846cd3c6fc1d`、`e71c6911e238c6dcdef5d01b1e6028651942cf22c420621545c9dd79039950b9`、`ebd60bc3e927bc217bc06bc8c99d1ba85c7f1283a516d69482e9855900f10e0d`。最终 `result.json` SHA256仍为`4dcc12741206be45dc15ec447a4008f1311260531bc1026c75ce3ac3030f950e`。

### 2026-11-14：NFA/FlashInfer 同两点法入口与真实 smoke 点火

- 新增 `benchmark/efficiency/benchmark_long_context_two_point.py`，让NFA/FlashInfer复用官方输入、`timed_pair`、固定33 token、EOS不提前停止及 `T(33)-T(1)` complete-request host-wall协议；不把旧 `benchmark_nfa_end_to_end.py` 的逐步CUDA-event结果混入新表。入口固定B1，decode循环不创建逐步CUDA Event、不读取GPU token，请求wall停止后才做一次D2H；NFA使用真实GraphRunner计数，FlashInfer使用真实prefill/append/wrapper调用计数，任何固定长度、Graph或backend计数不符均fail-closed。
- 新增 `tests/test_long_context_two_point.py`，覆盖NFA/FlashInfer × eager/Graph × T(1)/T(33)、Graph fallback与调用计数拒绝及干净harness AST门禁。pytest仅安装到隔离目录 `/tmp/qwen3-two-point-test-deps`，未改生产环境；普通与 `python -O` 均为 **11 passed**。为确保 `-O` 不跳过测试自身的裸assert，AST门禁已改为显式异常并另做零裸assert审计；`py_compile` 与 `git diff --check` PASS。pytest在 `-O` 下仍打印其通用断言警告，但本测试模块没有会被优化删除的裸assert。
- 主机回归后实测双卡均1MiB、无compute process，分别启动256K最小真实 `T(1)+T(33)` smoke：NFA任务 `bj64tzm02`（GPU0，宿主PID=3885266）与FlashInfer任务 `bc9xgrpj0`（GPU1，宿主PID=3885267），均为Graph、warmup0、repeats1，输出在 `/tmp/nfa_two_point_smoke_20261114/`。点火后已核验两个真实进程、双卡各约72.2GiB/100% util及模型加载日志；两项smoke现均已正常完成。FlashInfer的T(1)/T(33)回执分别为0/72次真实decode wrapper调用，T(33) capture1、replay32、fallback0，单次smoke TPOT=17.498255241662264ms；NFA同样真实完成32步Graph replay、fallback0，但每请求新建cache并重新capture导致 `decode_after_first_wall_ms=333858.8105365634`、smoke TPOT=10433.087829267606ms。两者都只是warmup0/repeat1的功能与协议探针，不计入正式性能结果。

### 2026-11-14：NFA 固定 arena / Graph 复用修复与二次 smoke 点火

- 将NFA入口改为同一固定 `GraphReadyPagedRegionCache` arena跨请求复用，T(1)/T(33)共享geometry与固定buffer身份；decode通过既有 `session_graph_runner` 池复用Graph，请求边界释放scheduler并清理cache request。Graph capture/replay/fallback改为每请求增量，只有首个冷decode请求允许capture1，后续正式T(33)必须capture0、replay32、fallback0；同时记录pool hit/miss和prepare时间，不改变完整请求host-wall计时区间。
- 主机回归覆盖 `T(1)→T(33)→T(1)→T(33)`：cache仅构建一次、首个T(33) capture1、中间T(1)三项Graph增量为0、第二个T(33) pool命中且capture0/replay32、两请求均清理scheduler。首轮测试另发现首个无runner的T(1)对空计数字典取键失败，已将累计capture/replay/fallback初始化为三个零值；最终普通与 `python -O` 均为 **13 passed**，零裸assert、`py_compile` 与 `git diff --check` PASS。入口SHA256=`62308ee0d29e63c36ab61e985ede1e719b7e3d7c136a956829d4120453a21f4a`，测试SHA256=`c8ce7822113437443300fcb085ff698f01a02386096847b8c84aba319c2611b1`。
- 修复后实测双卡均1MiB、无compute process，已在GPU0启动warmup1 pair+repeat1的真实256K二次smoke，任务 `b1iwy9331`、宿主PID=3902007，输出=`/tmp/nfa_two_point_smoke_20261114/nfa_256k_graph_reuse.json`，日志=`/tmp/nfa_two_point_smoke_nfa_graph_reuse_256k.log`。点火后已核验真实Python进程、GPU0约72.3GiB/100% util及模型注册日志，GPU1保持空闲；该配置会真实经过两个T(33)，只有正式请求capture0且32次replay、0 fallback后才启动正式3 repeats。

### 2026-11-14：NFA 二次 smoke 闭集与 256K 正式双臂点火

- 二次smoke任务 `b1iwy9331` 已正常完成，`status=complete`，真实执行warmup1 pair+repeat1共四个请求。warmup T(33)为capture1、replay32、fallback0、pool miss1，prepare=1248.936302959919ms；正式T(1)的三项Graph增量均为0，正式T(33)为 **capture0、replay32、fallback0、pool hit1、miss0、prepare=0.05964934825897217ms**，session累计hit/miss=1/1且固定arena标记为true。该smoke单格TPOT=25.648872600868344ms，只证明复用与计时协议，不替代正式3 repeats。JSON SHA256=`b537dfd91b872e831eccda27cbb76221a06846ba757f72694293e7ef18be8efc`，日志SHA256=`17d1d73420a953de610acdc66f6b1a4bcd3a3b0174f6f6b42b4a2fda8a896830`；结束后双卡均释放至1MiB。
- 随后已在空闲双卡启动256K正式1 warmup pair+3 repeats：NFA任务 `bcpn8hsuh`（GPU0，宿主PID=3906287）与FlashInfer任务 `b731hd2et`（GPU1，宿主PID=3906288），输出分别为 `nfa_256k_two_point_performance.json` 和 `flashinfer_256k_two_point_performance.json`，均位于 `agent_doc/results/qwen3_official_long_context_20261113/`。点火后已核验两个真实Python进程、双卡各约72.2GiB/100% util及模型加载日志；只有三格全部落盘、状态complete并逐请求复核Graph/backend计数后才记正式中位数，所有数据继续保持pending-independent-verification。

### 2026-11-14：NFA/FlashInfer 256K 正式闭集与 512K 双臂点火

- 256K正式任务 `bcpn8hsuh`（NFA）与 `b731hd2et`（FlashInfer）均已完成1 warmup pair+3 repeats，回执 `status=complete`。NFA三次TPOT=25.721563026309013/26.340891607105732/26.20357763953507ms，中位数 **26.20357763953507ms**；FlashInfer三次TPOT=24.93689046241343/26.931512402370572/27.605653507634997ms，中位数 **26.931512402370572ms**。与Quest 256K中位29.122937936335802ms同表时，NFA相对Quest/FlashInfer的加速比为1.1114107522629315/1.0277799761868096；这只是三次中位数的pending结果，FlashInfer repeat0仍快于NFA全部单格，不挑选单格包装排序。
- 独立只读结构复核PASS：三臂输入IDs SHA一致；NFA warmup T(33) capture1/replay32/fallback0/miss1，三个正式T(33)均capture0/replay32/fallback0/hit1/miss0，中间T(1)无Graph增量，session累计hit3/miss1且固定arena=true；FlashInfer每格T(33)均真实prefill append36、Python decode append/wrapper72、capture1/replay32/fallback0。所有请求33 token、EOS未提前停止，TPOT=`decode_after_first_wall_ms/32`、summary中位数与输出IDs SHA均重算通过。NFA三格输出SHA=`1dace37149533ca25632a6ec2907f8201418ecaf79df91767ba655856952e014`；FlashInfer三格输出SHA=`49afde0d3c7be3a964c847245ff0a6d8e44e68d9bf69dd355b8f04026d68c58a`。
- 256K结果身份：NFA JSON SHA256=`1a0887d4b6a9219cb7f9731cc48b59c0b44874adf7c680e31b6d69f63a9c9059`、日志SHA256=`2740b1c65e542e4862fde057b938b5a50509bc9f483be6a7c0ed6d7cd3d78ca6`；FlashInfer JSON SHA256=`41dfb3d641c5b62a39b44c70f156a8ecc48fe3e44118409a011c8ec75eb60424`、日志SHA256=`40c7f560cbb43ace37dad758ce01be2f6d9e72db27c94230190e587f09da50fc`。结果继续保持 `pending-independent-verification`。
- 双臂结束后双卡均释放至1MiB、无compute process。随后启动512K正式1 warmup pair+3 repeats：NFA任务 `bnbmqxr0k`（GPU0，宿主PID=3914393）与FlashInfer任务 `b7gjak5je`（GPU1，宿主PID=3914387），输出分别为 `nfa_512k_two_point_performance.json` 和 `flashinfer_512k_two_point_performance.json`。点火后已核验两个真实Python进程、双卡约119.9GiB/100% util、模型5/5分片加载及实际输入523827 token；截至本段仍无 `[LONG PERF]` 完成行，不提前登记TPOT，也不重复点火。

### 2026-11-14：LongBench16 NFA/FlashInfer 同协议入口准备

- 新增 `benchmark/efficiency/benchmark_longbench_two_point.py`，直接复用官方LongBench原始数据、`build_input`、固定33 token、`timed_pair` 和 `perf_summary`，并复用已验收的 `RequestTwoPointEngine` 真实NFA/FlashInfer backend回执；不与旧CUDA-event口径混池。每个任务创建独立固定cache arena/Graph session，任务边界释放池，避免不同输入长度错误复用Graph； warmup与正式pair的请求日程保持T(1)→T(33)。
- 新增 `tests/test_longbench_two_point.py`，覆盖两任务独立arena、每任务1 warmup+2 repeats的六请求日程、输入复用、回执人口、任务边界pool释放、固定33 token与TPOT，以及提前EOS/短输出fail-closed。与既有长上下文测试合跑，普通和 `python -O` 均为 **15 passed**，两测试模块零裸assert，`py_compile` 与 `git diff --check` PASS。入口SHA256=`48e26c63776f43c82c375c12b03a7b419e2d28796e866fe6383aa829acc2d5c1`，测试SHA256=`51c2a6fd4d8e2df039f1d01360eef42c4ad728e6e66da525af87eea54e8b3ac3`。尚未执行真实GPU smoke，不能登记LongBench性能结果。

### 2026-11-14：FlashInfer LongBench16 闭集、NFA smoke 与正式任务点火

- pc40真实NFA narrativeqa smoke已完成，`status=complete`，输入31397 token且input/data SHA与Quest、FlashInfer完全一致。warmup T(33) capture1/replay32/fallback0/miss1，正式T(33) **capture0/replay32/fallback0/hit1/miss0**，中间T(1)无Graph增量；固定arena=true，session hit/miss=1/1，NFA配置为tail、cmp ratio4、K1=128、K2=1024、page64、near4096、skip=false，无dense/Torch decode fallback。单格TPOT=12.458378914743662ms，仅作功能/协议验收。smoke JSON SHA256=`aa00762b42330540416845482cc17f127482f9ebe94822aa59e5d055ed47afa6`，日志SHA256=`bd0cf9427df437500cae0c702aae2251e7ad0d35bfa2c4776c83d6b986ef5d88`。
- FlashInfer LongBench16正式任务已完成 **16/16任务、48/48 cells、128/128请求回执**。只读结构复核PASS：逐任务输入/data SHA均与Quest一致；每任务8个请求，T(1)仅prefill append36且无decode/Graph调用，每个T(33)真实prefill append36、Python decode append/wrapper各72、capture1/replay32/fallback0；page64、固定33 token、EOS不提前停止、TPOT公式和summary均重算通过。macro TPOT=**15.49168636847753ms**，all-request median=15.434834058396518ms；Quest macro 19.365306608960964ms，Quest/FlashInfer=1.250045098277065，即当前口径Quest较慢，负排序如实保留。Quest官方page16/FP16与FlashInfer page64/BF16 Graph的实现差异常驻披露，不能把该比值解释为单一kernel加速。48格每任务三次输出SHA各自一致。JSON SHA256=`e11a7aea4753de51aff46e02e373810727b83567b146d8f33b412929afcdd891`，日志SHA256=`798f4c3345b7ec77dcfd7bac4e049b4a4eed4c3803135543cb02a80b2529ce1f`，本地证据目录=`agent_doc/results/qwen3_nfa_flashinfer_longbench_perf_20261114/`。
- FlashInfer结束后pc40双卡均1MiB；随后在GPU0启动NFA LongBench16正式1 warmup pair+3 repeats，真实Python PID=797740，输出=`~/ak_verify_results/qwen3_nfa_flashinfer_longbench_perf_20261114/nfa/nfa_performance.json`，日志=`/tmp/nfa_longbench_two_point_perf_pc40.log`。点火后完成真实进程、GPU0显存/ compute PID、模型注册与 `[PERF]` 三重验证；首次收割已完成narrativeqa 3/3格，TPOT=12.580/12.613/12.631ms，整项48格尚未闭集，不登记正式macro。
- 本地512K双臂仍各自由原进程运行且双卡约128GiB/100% util；当前只落盘repeat0：NFA TPOT=**185.48804777674377ms**、FlashInfer TPOT=**39.58947886712849ms**。NFA正式repeat0回执为capture0/replay32/fallback0/hit1，FlashInfer为append/wrapper72、capture1/replay32/fallback0，输入均523827 token、固定33 token、EOS未提前停止。NFA相对256K的异常增长和相对Quest/FlashInfer的负排序不隐藏，待三格闭集后按真实Graph/host路径诊断；当前不得登记512K中位数或重复点火。

### 2026-11-14：512K 两格中间收割与两点法 prefill 波动边界

- 本地512K双臂仍由原进程运行，均已落盘repeat0/1但尚未完成repeat2。NFA两次TPOT=**185.48804777674377/61.052299570292234ms**，FlashInfer两次TPOT=**39.58947886712849/40.14186095446348ms**；两臂两格各自的输出IDs SHA均一致，输入均为523827 token、固定33 token、EOS未提前停止。NFA正式repeat1为capture0/replay32/fallback0/pool hit1，FlashInfer每格T(33)均为prefill append36、Python decode append/wrapper72、capture1/replay32/fallback0。两格均值只是JSON的interim字段，不登记为正式3-repeat中位数。
- NFA两次 `T(33)` 完整请求wall为948292.1482622623→948243.4614375234ms，仅变化约**−48.687ms**；对应 `T(1)` TTFT为942356.5307334065→946289.787851274ms，变化约**+3933.257ms**。decode差分5935.617528855801→1953.6735862493515ms的变化约−3981.944ms，几乎完全由T(1) prefill变化解释。因此185→61ms的主要离散不能写成decode kernel从185ms降到61ms；官方两点法保持不变，绝对差分仍须与真实prefill差、Graph外host/metadata和selector实现差异分别诊断。
- 旧512K NFA CUDA-event回执属于 `84bd19ee`，其32步算术均值11.618834972381592ms、中位数11.268815994262695ms；当前HEAD相对旧回执的关键selector变更为 `b97af7ef`，包含L1/L2活跃区动态分支与region改造。旧no-skip结果保留原源码身份、并未用新kernel重测，故该提交只列为高优先级性能回归嫌疑；正式任务运行期间不修改生产代码、不附加profiler、不启动重复GPU测量，也不宣称根因已定位。
- pc40 NFA LongBench16已正常完成并释放双卡：**16/16任务、48/48 cells、128/128请求回执**（JSON按16个任务分组，每组8请求）结构复核PASS。逐任务输入token/data SHA均与Quest、FlashInfer一致；每任务warmup T(33) capture1/replay32/fallback0/miss1，三个正式T(33)均capture0/replay32/fallback0/hit1/miss0，中间T(1)无Graph增量，session hit/miss=3/1且固定arena=true；tail/cmp ratio4/K1=128/K2=1024/page64/near4096/no-skip及无dense/Torch fallback均核验通过。逐格33 token、TPOT公式、输出IDs SHA、逐任务中位数和宏平均均重算一致。macro TPOT=**11.319210680085234ms**，all-request median=11.50084612891078ms；相对Quest/FlashInfer macro分别为 **1.7108354245081638×/1.3686189617208253×**。JSON SHA256=`0ed20ad61e99149065453b93fc7dbaae614451b8619f571a68f1c829e2880e57`，日志SHA256=`0131b036b3f2451b669babbd88f51e42e72d9eb3fff85e1d8383a62899d166e6`，本地证据=`agent_doc/results/qwen3_nfa_flashinfer_longbench_perf_20261114/nfa_performance.{json,log}`，继续保持pending-independent-verification。

### 2026-11-14：NFA/FlashInfer 512K 正式双臂闭集

- FlashInfer正式任务 `b7gjak5je` 已完成1 warmup pair+3 repeats，`status=complete`，三格TPOT=**39.58947886712849/40.14186095446348/41.54795524664223ms**，正式中位数 **40.14186095446348ms**。只读结构复核PASS：输入与Quest、NFA同为523827 token且data/input IDs SHA一致，YaRN factor16、page64、固定33 token、EOS未提前停止、输出IDs SHA和TPOT公式均重算通过；三格输出SHA均为 `3c8463409c492cdc0a631b7465c5a0a88b2e5d168c4f23d815b48d6173066bce`。
- 8/8请求回执满足日程：T(1)仅prefill append36且无decode/Graph调用；四个T(33)均为prefill append36、Python decode append/wrapper各72、capture1/replay32/fallback0，fixed split=431页。summary逐任务中位数与宏平均重算一致；Quest 512K中位83.82336678914726ms，Quest/FlashInfer=**2.0881783952227733×**，该负排序如实保留。JSON SHA256=`e6c182f765c5dbca559556e93b790b26d1f971e163722f22c1a32d1c10ae1ad5`，日志SHA256=`2e62367d2c661f5bd675c25d494a93ed3fba661f8ee84babb52a07daeecc1e9f`；GPU1已释放至1MiB，结果保持pending-independent-verification。
- NFA正式任务 `bnbmqxr0k` 随后由原进程正常完成，`status=complete`，三格TPOT=**185.48804777674377/61.052299570292234/58.51614009588957ms**，正式中位数 **61.052299570292234ms**。8/8请求回执结构复核PASS：仅warmup T(33) capture1/replay32/fallback0/miss1，三个正式T(33)均capture0/replay32/fallback0/hit1/miss0，中间T(1)无Graph增量；session hit/miss=3/1、固定arena=true，tail/cmp ratio4/K1=128/K2=1024/page64/near4096/no-skip及无dense/Torch decode fallback全部核验通过。输入、YaRN、固定33 token、EOS、输出SHA、TPOT公式和summary中位数均重算一致，三格输出SHA=`3f56b2bad5eacd973868c52ff81a4e1612956cea04e06b59d3d03ee46a805f4e`。
- 512K正式中位排序为FlashInfer **40.1419ms** < NFA **61.0523ms** < Quest **83.8234ms**。NFA相对Quest为 **1.3729764051334001×** 加速，但NFA/FlashInfer=**1.5209135331206827×**，即FlashInfer比NFA快约52.09%，未满足预期的NFA最快排序，负结果完整保留。NFA三格T(33)完整wall仅相差约48.687ms，而repeat0的T(1) TTFT比repeat1低约3.933秒；repeat1/2的TPOT为61.0523/58.5161ms。两点法prefill差分敏感性与旧CUDA-event口径差异继续作为开放问题，不能用该协议字段直接宣称纯decode kernel耗时或把 `b97af7ef` 定为根因。
- NFA JSON SHA256=`3bbf1df8e878657a516d0ddb0406aa80aab8556e5f99014ef1009e6c01253d59`，日志SHA256=`df2201547602f5dbbeb78704cacb21a38288a78fee4597767d64a95cb69afc70`。双臂进程均已退出，本机GPU0/GPU1均释放至1MiB且无compute process；所有结果保持pending-independent-verification。

- 双臂闭集后在本机无GPU计算进程环境重跑聚焦回归：`test_long_context_two_point.py`、`test_longbench_two_point.py`、`test_official_longbench.py`、`test_official_ruler.py` 普通与 `python -O` 均为 **41 passed、1 skipped、4 subtests passed**；四测试模块零裸assert，相关Python `py_compile` 与 `git diff --check` PASS。当前Quest改动只涉及Python控制器，本轮未重建或修改官方CUDA扩展。
- 六组结果目录已生成 `manifest.json` 并逐项重算SHA256复核PASS，排除临时锁、SQLite和pycache：long-context 19文件，manifest SHA256=`c29619625d842f0ef43d97e424d0aef34f642912a025c1add1a57e190d54c9b6`；Quest LongBench性能2文件，`dcf91d31ebc88fd53e26f7b038016edcdb6a2a4b7ac000544e877c47df9517a5`；NFA/FlashInfer LongBench性能6文件，`e6950f08a38b0ebacbd94c634e8f37cd770f68d965768fcdf273757580c873a1`；Quest LongBench质量171文件，`aa68b851e1eea59d646062bebbeeedcc966ed80db47ebacd6c0eb1043174cc40`；LongBench smoke 7文件，`40a21421af56502e028b634b330ebed0e8ea0b6ec41834f5d73fc3c5499fd479`；RULER smoke 6文件，`27c22a02b6888e223cc2fc7866e1643429a55d078d2bcaad51189a92b976c8ba`。manifest本身不自哈希，全部结果仍为pending-independent-verification。

### 2026-11-14：NFA/FlashInfer LongBench 真实质量入口与四臂 smoke 闭集

- 新增 `benchmark/efficiency/benchmark_longbench_quality.py`，直接复用官方原始数据、`build_input`、31500 cap、`dataset2maxlen`、runcap/v2输入身份和官方评分消费格式，但**不复用固定33 token性能回执**。NFA与FlashInfer分别使用自己的 `nfa_{eager,graph}`、`flashinfer_{eager,graph}` 实现标签和producer；每请求独立NFA cache或FlashInfer page64 runtime，真实执行EOS、min-new、任务stop-token与samsum stop-string。NFA请求边界释放Graph session pool；该入口只生成质量预测，不报告clean TPOT。
- `sparse_attn/scheduler/scheduler.py` 新增共享 `StopStringMatcher`，在CPU回执上复用HF `StopStringCriteria`，覆盖独立换行token与合并BPE token；真实Qwen tokenizer验证 `。\n` 为单token 8997，不能用独立token 198代替stop-string。`sparse_attn/engine/frozen_runner.py` 增加真实停止原因、逻辑/物理decode计数及慢回执下cache/trace回滚；物理Graph/eager计数不回滚。NFA质量验证要求 `nfa_decode_steps=generated_tokens-1`、`submitted_decode_forwards>=逻辑步数`，replay/fallback按真实物理提交数验收。
- `benchmark/official_baselines/longbench_official.py` 增加可选质量身份注入和逐任务 `.receipts.json` 持久化；完整预测SKIP时恢复并逐请求核对task/source-index/input身份，缺失或漂移fail-closed，部分预测重跑先丢弃旧回执。manifest记录receipt路径、行数与SHA256，不再因SKIP发布空回执。默认Quest/JSSA身份与输出协议保持不变。
- 两次NFA失败smoke均未发布完成manifest：首次 `bat50kq7r` 因PYTHONPATH缺仓库根目录在导入阶段失败；修正后 `bmtn7yx4w` 真实生成但在验证处报 `(5,7,0)!=(1,7,0)`。读取 `GraphRunner.capture_all()` 后确认B1、`visible_len=8192` 的日程为 `(512,1024,2048,4096)+full-capacity`，应捕获5图而非1图；已改为按真实bucket日程计算capture期望，replay/fallback检查未放宽。
- 最新修正后 `tests/test_longbench_quality.py`、`test_official_longbench.py`、`test_frozen_region_runtime.py`、`test_multi_request_graph_runner.py` 普通与 `python -O` 均为 **69 passed、1 skipped、4 subtests passed**；相关Python `py_compile` 与 `git diff --check` PASS。质量harness SHA256=`b312e9947e5a9b1451e0ef36d3112a515b6ecee912598b04f58c3a08f46e00ef`，专项测试SHA256=`b16a266f689b26ad0872ba2fc3cc872bca3be01a605112ceadc7393084eb5c11`。
- 在本机实测双卡空闲后完成 `samsum:0` 四臂真实smoke，输入均9508 token、input IDs SHA=`df7dbfe0b6402dfbfec00fccbe6b2a48b75f3132cf1bd281a526d302fe49b6b3`，输出均为8 token且因换行 `stop_string` 停止，预测文本逐字一致：` Emilia is still angry with William.\n`。NFA Graph回执为逻辑/物理7步、capture5/replay7/fallback0/pool miss1；NFA eager为capture0/replay0/eager7；FlashInfer Graph为prefill36、Python decode append/wrapper72、capture1/replay7/fallback0；FlashInfer eager为prefill36、append/wrapper252、capture0/replay0。FlashInfer Graph完整SKIP已在当前源码下真实执行，日志明确 `[SKIP]`，回执人口与SHA保持不变，source receipt已刷新为当前harness哈希。
- 本机官方评分端首次因缺 `jieba` 在导入阶段失败、未写result；在同一本地依赖目录安装 `jieba=0.42.1`、`rouge=1.0.1` 后，未修改的 `benchmark.LongBench.eval --expect-count 1` 对四臂均严格评分通过，单样本ROUGE均80.0，`strict_validated=1`、cap31500、v2输入身份验证通过。该单样本分数只用于消费端验收，不作全量质量结论。
- 证据目录=`agent_doc/results/qwen3_lbv1_nfa_fi_quality_smoke_20261114/`。`audit_smoke.py` 重算四臂输入身份、预测/runcap绑定、回执人口与backend计数、官方result及当前源码哈希，状态PASS并生成27文件SHA256 manifest（排除锁文件）；audit SHA256=`3dec52bbaddbdbff0b7581ba5f76057b2e69447934211dda2d49504520162351`，审计脚本SHA256=`dd037f402c71ed7cbf7c3e4273a59437f7dd5d228aefa18ba2dd7071ca6c0202`，manifest SHA256=`036eaefe5ef37b25b2c54854ee4859b1ea3b67333fcc72f804f5323cdca79fa6`。四臂结束后双卡均1MiB且无compute process，全部保持pending-independent-verification。
- 全量功能质量采用eager，避免NFA逐请求独立cache下5次Graph capture的成本，也不把功能跑批与clean Graph性能口径混用；既有Graph性能结果不变。

### 2026-11-14：NFA/FlashInfer LongBench16 全量 eager 质量九路点火

- 五机均使用独立 `/tmp/qwen3-nfa-fi-quality-7fac274f-20261114` worktree，固定HEAD=`7fac274fda9df13e82caadd95f3cca5c8374a1a3`，质量harness SHA256=`b312e9947e5a9b1451e0ef36d3112a515b6ecee912598b04f58c3a08f46e00ef`，关键模块导入与 `py_compile` 全部PASS。远端内部镜像尚未包含新HEAD，改用本地 `main --not 482911be` 的最小git bundle部署，bundle SHA256=`211d58e6e36ccd5743cc20c60727072d2cc2a5636e03dfb2caa0525e3a72a568`；未修改远端主工作树。五机模型、LongBench数据、Python3.11、torch2.8.0+cu128、transformers4.56.2、FlashInfer0.5.2及磁盘均核验就位；远端缺pytest/jieba/rouge不影响生成，评分集中在本机已有严格评分环境执行。
- 点火前实测本机GPU0/1、pc40 GPU0/1、pc31 GPU0、pc32 GPU0、pc02 GPU0/2/3空闲；pc31 GPU1 PID3099703、pc32 GPU1 PID3072425、pc02 GPU1 PID968650均保持不动，pc14 SSH超时未派单。采用canonical合并器要求的标准 `source_order[rank::9]`：rank0/1=本机GPU0/1，rank2/3=pc40 GPU0/1，rank4=pc31 GPU0，rank5=pc32 GPU0，rank6/7/8=pc02 GPU0/2/3。每卡一条串行链，先NFA eager再FlashInfer eager，单卡同时只有一个生成进程，输出按method/rank完全隔离。
- 原始数据重算每rank人口为 **428/428/415/415/415/413/412/412/412**，合计3750；每臂完成后应为16任务、3750预测及3750逐请求回执。派单脚本=`/tmp/run_qwen3_nfa_fi_quality_eager_20261114.sh`，SHA256=`169261e80618f58a6119ec6dbc900e7308d740575e46a46fa13ab7c324dc6539`，包含HEAD/源码SHA校验、单rank flock、启动前逐卡compute-PID检查、eager/fallback/manifest人口终验，不支持隐式降级。
- 九条链均通过真实进程、显存/compute PID与模型日志三重验证；宿主Python PID分别为rank0/1=3991956/3991957、rank2/3=857599/857600、rank4=2966878、rank5=187519、rank6/7/8=318990/318994/318995。随后rank0–5已产生首批真实预测，例：rank0 `narrativeqa:0 in=31397 out=8`、rank1 `narrativeqa:1 in=31441 out=13`、rank2 `narrativeqa:2 in=8382 out=13`、rank4 `narrativeqa:4 in=31512 out=19`；rank6–8仍处于输入准备/模型启动日志阶段，继续按真实输出监督，不以PID冒充完成。
- 输出根目录=`~/ak_verify_results/qwen3_nfa_fi_longbench_quality_eager_20261114`，各机同路径按rank收割；日志与状态分别在其 `_logs/`、`_status/`。当前仅登记在跑，不登记AVG16或标任务完成；两臂全量闭集后仍需逐请求receipt/runcap/backend计数验收、九rank canonical合并、官方严格评分、SHA256 manifest与独立owner验收，全部保持 `pending-independent-verification`。

### 2026-11-14 11:17：九路首轮进度复核与闭集工具准备

- 五机九条链的真实 `ps`、逐卡显存/compute PID与最新逐请求日志均一致，未见failed状态或重复进程；pc31 GPU1、pc32 GPU1、pc02 GPU1的其他AI计算保持不动。NFA eager九rank预测人口为17/22/23/22/9/7/22/22/22，合计 **166/3750**；rank2已完成其22条narrativeqa并进入 `qasper:2`，其余rank继续推进，FlashInfer仍在各卡NFA之后等待接力，不提前点火。
- `_dispatch/audit_full.py` 已通过 `py_compile`，SHA256=`372d68c299d97142ed257cdc063bb026718470b85c8228377c9b0be19bbcbaf7`；该脚本只在两臂九rank闭集后重算人口、runcap/receipt、eager backend计数与跨臂输入身份，当前未对partial数据运行，不能登记审计PASS。
- 新增 `_dispatch/close_full.py`，SHA256=`afe55fc1356861031db09be84b0efd1218f77fa2ad7aeed4f764feb32f4ec620`，`py_compile` PASS。它强制先消费 `rank_audit.json.status=PASS`，再把每臂九rank的预测/runcap/receipts按原文件名扁平复制并校验SHA，逐任务调用既有canonical合并器，最后以 `--expect-count 150` 调用未修改的官方严格eval并重算macro AVG16；当前未执行，不产生任何全量分数。

### 2026-11-14 11:34：九路第二轮进度复核

- 五机九条链再次按真实宿主 `ps`、逐卡显存/compute PID、日志和预测文件人口复核，九条NFA eager链均在运行，未见failed状态、重复生成进程或FlashInfer提前接力；pc31 GPU1 PID3099703、pc32 GPU1 PID3072425、pc02 GPU1 PID968650的其他AI计算保持不动。
- 最新九rank预测人口为 **28/31/35/29/20/23/30/29/31，合计256/3750**。rank0/1/2/3/5/6/7/8已进入qasper，rank4继续narrativeqa；FlashInfer九rank均为0行并等待各卡NFA正常完成后串行接力。该快照仍是NFA第一臂partial人口，不运行闭集审计、不生成canonical结果或AVG16。
- 使用仓库外只读监控脚本 `/tmp/monitor_qwen3_nfa_fi_quality_20261114.py` 汇总各机进度；它只读预测、日志、状态和系统进程，不修改生产产物。后续失败必须先读真实日志再按同rank幂等恢复，不重复点火。当前脚本SHA256=`018ed8df90f0cbd2282fefbeacb50c26a8c5e057fbdb06f3bbd875fc617f997e`。
- 新增 `_dispatch/harvest_full.py`，SHA256=`9680f34e422378198c825a50006820191607bf3f9b0be9e97c6ebdc37a708ec1`，`py_compile` PASS。它要求本机及远端每rank整链done且两臂manifest齐备后才收割，经SSH/tar取回互斥rank目录、日志与完成状态，拒绝tar路径穿越和冲突覆盖，逐文件SHA校验后生成 `harvest_manifest.json`；当前未执行，不登记收割PASS。

### 2026-11-14 11:43：第三轮监督与 narrativeqa 完成分片早期审计

- 本轮先复核任务索引、TaskList和后台真实进程。Quest 256K `decode_estimate` 与512K `decode_sparse_attn` 的illegal-memory-access已由RAFT Top-K动态workspace修复，两个长度的新YaRN 1 warmup+3 repeats也已闭集并通过结构复核；本轮严格不重复测量。JSSA固定提交cu129 wheel仍阻塞，不换版本或fallback；其他AI的GLM AsyncTLS/NFA保持不动。
- 最新NFA eager九rank人口为 **31/34/39/33/24/27/36/33/42，合计299/3750**，九rank均进入qasper；FlashInfer仍等待串行接力。五机真实进程、显存/compute PID与日志一致，未见failed或重复进程。
- 新增 `_dispatch/audit_completed_task.py`，SHA256=`ac6bff72b01c05a08c4dbf45b9695a49050fc3ddd117c1b565f1984f4708f251`，`py_compile` PASS。对九台分片已完成的NFA `narrativeqa` 逐条核验标准分片、预测/runcap SHA与人口、receipt身份、真实停止原因、生成上下界、eager逻辑/物理计数和零Graph capture/replay；九rank共 **200/200** 均PASS，stop_reason全为eos，generated=1844、逻辑/物理decode均为1644。
- 早期审计汇总=`narrativeqa_nfa_partial_audit.json`，包含九rank预测/runcap/receipts SHA256；汇总文件SHA256=`28e36bb9aa81bbfb9dfb429358a34a390640298d7fa1d89395d2a314f9d5b06b`。该PASS仅限已完成的单任务分片，不是NFA全臂、FlashInfer、跨臂输入身份或canonical闭集，未执行官方评分或生成AVG16。

### 2026-11-14 12:36：第四轮监督与 qasper 完成分片早期审计

- 五机九条链再次按真实宿主 `ps`、逐卡显存/compute PID、最新逐请求日志和预测人口复核，九条NFA eager链均在运行，未见failed状态、重复生成进程或FlashInfer提前接力。rank0/1/2/3/6/7/8已进入hotpotqa，rank4/5继续multifieldqa_en；pc31 GPU1的其他AI计算保持不动，pc32 GPU1与pc02 GPU1本轮已观测为空闲，但不在跑批中途改变九分片或启动重复任务。
- 最新NFA eager九rank人口为 **66/64/65/67/51/54/65/69/68，合计569/3750**；FlashInfer九rank仍为0行并等待各卡NFA正常完成后串行接力。该快照只登记partial人口，不执行全臂审计、canonical合并或评分。
- 使用 `_dispatch/audit_completed_task.py` 对九路均已完成写盘的NFA `qasper` 分片逐条复核：标准 `source_order[rank::9]`、预测/runcap/receipts人口和SHA、source-index/input身份、生成上下界、真实停止原因、eager逻辑/物理计数及Graph capture/replay=0全部PASS。九rank合计 **200/200**，stop_reason全为eos，generated_tokens=2895，逻辑/物理decode均为2695。
- 五份机器级原始审计及其SHA256分别为：local=`bba12bb1aeae1daf928d43a9a43dd941373282a3bed3a341048cc2b9244ebe2e`、pc40=`72483c4cdf9b5fca77bc1dea3fa7b8a2b2fa5243d4849cefa373b7dd7aba2ab9`、pc31=`23001ef0f0dacb901565d94d5b8733c7647c3ae044440ebcc1974b924c238d5e`、pc32=`069c3ecd59cd509289b521f74b45c11cf47032a548742d4c619ae6fdbfd69dfd`、pc02=`3224f3413abe223ed8f68b8db2a229d60de0f6728baf924213511def175c9439`。九rank汇总=`qasper_nfa_partial_audit.json`，SHA256=`d4043339a13a094d6836f9564ee999fea88f6dcb22a5b0e8bda42fa2d8e65d1f`，包含逐rank预测/runcap/receipts SHA和上述五份来源哈希；仍保持 `pending-independent-verification`，不冒充NFA全臂、FlashInfer、跨臂身份或canonical闭集。

### 2026-11-14 14:28：第五轮监督与两项完成分片早期审计

- 本轮复核任务索引、TaskList和五机真实后台输出。Quest 256K/512K越界修复与新YaRN正式1 warmup+3 repeats均已闭集，不重复测量；JSSA固定cu129 wheel继续阻塞，不换版本或fallback，其他AI的GLM AsyncTLS/NFA不接管。
- 五机九条NFA eager链的宿主 `ps`、显存/compute PID、日志和预测人口一致，未见failed、重复进程或FlashInfer提前接力。最新九rank人口为 **142/135/162/152/105/108/172/161/171，合计1308/3750**：rank0/1在gov_report，rank2/3/7在qmsum，rank4在2wikimqa，rank5在musique，rank6/8在multi_news；FlashInfer仍等待各卡串行接力。
- 对九路均已完成写盘的 `multifieldqa_en` 与 `hotpotqa` 分片执行同一早期只读审计，标准分片、预测/runcap/receipts人口和SHA、source-index/input身份、真实停止原因、生成上下界、eager逻辑/物理计数和Graph capture/replay=0全部PASS。`multifieldqa_en` 共 **150/150**，stop_reason为eos=143、max_new_tokens=7，generated=3210、逻辑/物理decode=3060；`hotpotqa` 共 **200/200**，stop_reason为eos=196、max_new_tokens=4，generated=1078、逻辑/物理decode=878。max_new_tokens是允许的真实停止原因，不按异常处理。
- 汇总证据分别为 `multifieldqa_en_nfa_partial_audit.json`（SHA256=`6b363defaa91a11167a90fe2c45cb4573f08b34d366cd1e197920ed4e24af644`）和 `hotpotqa_nfa_partial_audit.json`（SHA256=`956aed77f7b18b898f64de0f8da9d595fffdc57eeaaf48de1f6234d53c02e512`），均含九rank逐文件SHA及五份机器级来源审计哈希。上述PASS仍只限已完成单任务分片，不是全臂、跨臂身份或canonical闭集，未生成AVG16。

### 2026-11-14 16:28–17:10：窄 far compact Top-K 故障、旧预测隔离与精确修复

- 第六轮监督发现旧 `7fac274f` NFA eager 链在多个短请求进入 far-only compact 路径时失败。最终旧NFA落盘人口为rank0–8=`191/200/299/187/177/181/412/412/412`，合计 **2471/3750**；rank0/1/3/5为failed，rank2/4为精确PID停车后的stopped，rank6/7/8虽有完整manifest也不能豁免数值问题。旧runner的EXIT trap曾把rank2/4停车写成伪done，已改回stopped并保留原始状态；旧FlashInfer rank6/7/8各412条为完整有效结果，NFA kernel故障不影响该臂。
- 真实日志定位到 `_l1_far_compact_packed_body()`：region2按 `prompt_length+max_new_tokens` 计算far前缀，生产 `K1=128、near_pages=32、K_F=min(96,KT)` 下，旧代码在 `FAR_WIDTH_PAD<K_F_PAD` 时仍执行 `tl.topk(packed,K_PAD)`。`FAR_WIDTH=1` 在Triton 3.4.0触发 `invalid axis -1`；宽度2/4/64虽可编译，但独立逐位oracle证实会静默错选。只把宽度padding到2的候选在K=4/96/128均逐位失败，未进入生产。
- 修复在 `sparse_attn/indexer/nfa_layer_batch_selector.py` 的共享 `_l1_far_compact_packed_body()` 中，对窄前缀把寄存器排序宽度扩到 `KT_PAD`，load mask仍严格限于原 `FAR_WIDTH/far_end/KT`，不读取poison后缀；宽前缀保持原compact路径，fused与prepare两条调用共享修复，无Torch fallback、无质量策略或expected变更。生产 `k_f=min(96,kt_cap)` 保证 `K_F_PAD<=KT_PAD`。
- `tests/ops/test_l1_topk_tiled_cuda.py` 新增窄前缀与极短生产形状oracle：覆盖 `far_width=1/2/4/16/21/64`、K=21/96/128、poison、ties、NaN/负NaN、±inf、±0、空far及非对齐长度，索引和顺序与full-width实现逐位一致。完整L1 Top-K文件 **10 tests PASS**；质量/Graph运行时普通和 `python -O` 均为 **69 passed、1 skipped、4 subtests passed**；`py_compile` 与 `git diff --check` PASS。
- 四条真实Qwen3修复smoke全部完成：原崩溃请求 `multi_news:108`（2119输入token、far_width=1）生成512 token、逻辑/物理decode均511；另有 `2wikimqa:55` width3、`multi_news:45` width5、`hotpotqa:127` width64。均使用eager、无dense/Torch fallback，输入身份与旧现场一致；这些只证明修复可达性，不是全量AVG16。
- 旧runcap按真实输入长度做几何可达性审计：rank0–8计划行中无法证明安全的行数分别为 `81/88/111/85/80/91/200/184/198`。receipt未持久化逐层 `skip_near`，故该审计只证明几何可达性，不能把任一旧NFA预测结构审计PASS当作数值安全证明；旧2471条NFA全部隔离，不以SKIP混入新canonical。五机冻结共504个文件条目，审计SHA256分别为pc33 `9ece6106d2504f92af79dcd522de047fa2353186b3da8d47ae293a10deaf9c52`、pc40 `34acdedee9378ad5ce0eb95e07b3a9360b53131a833bb3c66eaf062e1057a1ec`、pc31 `818462cec83e2a6327937ce1cb90c10cbf99f5524ea9fd3dba215c5214b2b3fa`、pc32 `daacc1e6e65aa2ed38e1916fe0c9da2f982de8057b1a5e477a72cfb01a9a5a1d`、pc02 `11f6446190f56dee979bcafc575746ee9196aa76cbe72fa7c98211f76213d2bb`。
- 修复仅提交selector与CUDA测试两个文件，rebase保护37份其他AI的GLM证据后正常推送，commit=`cbfd70da607eb5afed4343d6d055073f450e4162`，无force push、无任务文档或第三方工作树混入。

### 2026-11-14 18:33：修复版全量重跑、FlashInfer完整复用与九路健康复核

- 新输出根=`~/ak_verify_results/qwen3_nfa_fi_longbench_quality_eager_cbfd70da_20261114`。NFA九rank全部从 `/tmp/qwen3-nfa-fi-quality-cbfd70da-20261114` 的 `cbfd70da` 重跑；FlashInfer为保持同一臂producer一致，继续统一使用 `7fac274f`，不在同一臂内混用新旧代码。增量bundle SHA256=`7ad1726e8f0a8265c392947454b09f064430d053a2e3ac8f63fcb387731637ed`，五机worktree HEAD、harness SHA=`b312e994…`、导入与脚本语法均验收通过。
- 新runner=`/tmp/run_qwen3_nfa_fi_quality_eager_cbfd70da_20261114.sh`，SHA256=`823b5ed6e05209958db8df858fd0dde71bc6296f427b3165d693e81b8023a5f6`。只有正常退出、`CURRENT_METHOD=complete` 且两臂manifest同时存在才写done；继续逐rank检查HEAD/harness、16任务、预测/receipt人口、runcap complete与逐文件SHA。pc02 rank6/7/8的既有完整FlashInfer共 **1236条** 已逐文件复核并复制到新根，复用回执齐全，未重复生成；rank0–5仍待NFA完成后从同一 `7fac274f` worktree串行生成FlashInfer。
- 新闭集工具均 `py_compile PASS`：`_dispatch/audit_full.py` SHA256=`e946776c90dc587abe5268d954d6903b5664a8456057080c2f5d2e45e380927b`，分别核验两臂HEAD并检查FI复用回执；`close_full.py`=`d5d6fe0fe3be147242ec46a90fd5322c0bd358f480db5335e996bd62993a3a92`；`harvest_full.py`=`95c1b0d010bfb3e6aed59f4c0806c4987bd52b0b1ac99d2476c389580405b50d`。当前均未对partial人口执行闭集、合并或评分。
- 初次补点火时rank0、rank2–8已通过真实runner/Python、显存/compute app和首请求日志三重验证；本机rank1因启动列表中目录创建与后台重定向竞争而未启动，GPU1保持1MiB。随后单独顺序补点火，真实runner PID=437816、Python PID=437834，GPU1显存24952MiB且有唯一compute app，日志明确 `rank=1/9、head=cbfd70da、method=nfa`，已真实完成 `narrativeqa:1 in=31441 out=13` 与 `narrativeqa:10 in=31512 out=21`，补点火三重验证PASS。
- 18:33五机复核九条NFA eager链全部健康，无failed/done或重复生成进程。九rank NFA人口=`14/2/24/23/11/12/24/24/25`，合计 **159/3750**；FlashInfer为 `0/0/0/0/0/0/412/412/412`，合计1236。rank0/1/4/5在narrativeqa，rank2/3/6/7/8已进入qasper；这只是partial进度，不是闭集或AVG16。原始快照=`_dispatch/progress_20261114T183337.json`，SHA256=`73a93ac13baa66e4891c43880ad70d23caa07fea02d4b0f08e63c605b88dc24f`；监控脚本SHA256=`d87d35eed9f484d83f3dbb506542f6e0448c960d1f8aefa04041ca4bed130044`，全部保持pending-independent-verification。
- 随后九rank `narrativeqa` 均完成写盘，使用新 `_dispatch/audit_completed_task.py` 逐条复核标准分片、预测/runcap/receipt人口与SHA、producer HEAD/harness、source-index/input身份、真实停止原因、eager逻辑/物理计数和Graph capture/replay=0，**200/200 PASS**；stop_reason全为eos，generated=1844、逻辑/物理decode=1644。新脚本修正旧辅助脚本的 `multi_news max_new_tokens=64` 为真实512，并新增双臂分HEAD/harness核验，SHA256=`b5596d2c3772a75540780f7396653f8a478ee8427e31fe76be3500432d2cf306`。汇总=`_dispatch/narrativeqa_nfa_partial_audit.json`，SHA256=`00f41311ce663ae7a73cc6533ce4470756f9c5979b7540e0e9bc63fa47496855`；该PASS仍只限修复版单任务完成分片，不是全臂、FlashInfer、跨臂身份或canonical闭集，未生成AVG16。

### 2026-11-14 18:57：修复版第九轮进度复核

- 五机九条修复版NFA eager链再次按真实宿主 `ps`、逐卡显存/compute app、日志和预测人口复核，九条链均为running，未见failed/done、重复生成进程或FlashInfer提前接力；rank0–8的runner/Python PID保持为 `430045/430068`、`437816/437834`、`1495658/1495692`、`1495659/1495695`、`3381410/3381428`、`579071/579089`、`1309370/1309420`、`1309371/1309425`、`1309372/1309426`。
- 最新九rank NFA人口为 **47/45/52/52/37/44/56/53/59，合计445/3750**；FlashInfer仍为rank0–5各0、rank6–8各412，合计1236/3750。rank0/2/3/6/7/8已进入multifieldqa_en，rank1/5的qasper已写齐各自22条，rank4仍在qasper且当前仅15条，因此本轮不执行或宣称qasper九rank 200/200早期审计。
- 原始快照=`_dispatch/progress_20261114T185731.json`，JSON人口求和复核PASS，SHA256=`f2bb8a6c373988146072fe2869dc2b52417f663a3a234d4bf18c720da577c610`。本轮没有启动新GPU任务、重复测量闭集配置或运行全臂audit/close；两臂仍未完成canonical合并和官方评分，全部保持pending-independent-verification。

### 2026-11-14 19:09：修复版第十轮监督与 qasper 完成分片早期审计

- 五机九条修复版NFA eager链的宿主 `ps`、逐卡显存/compute app、日志和预测人口继续一致，未见failed/done、重复生成进程或FlashInfer提前接力。最新九rank NFA人口为 **55/53/61/62/44/50/63/61/65，合计514/3750**；FlashInfer仍为rank0–5各0、rank6–8各412，合计1236/3750。rank3/6/7/8已进入hotpotqa，其余rank继续qasper或multifieldqa_en。
- 九rank `qasper` 均已写齐，使用与narrativeqa相同的 `_dispatch/audit_completed_task.py` 逐条复核标准round-robin分片、预测/runcap/receipt人口和SHA、producer HEAD/harness、source-index/input身份、真实停止原因、生成上下界、eager逻辑/物理计数及Graph capture/replay=0，**200/200 PASS**；stop_reason全为eos，generated_tokens=2895、逻辑/物理submitted_decode_forwards=2695。首次远端调用误假设 `_dispatch` 脚本已部署到各机而报文件不存在，未触碰生产产物；改为通过SSH标准输入执行同一份本地固定脚本后五机全部PASS。
- 汇总证据=`_dispatch/qasper_nfa_partial_audit.json`，聚合人口与计数独立求和复核PASS，SHA256=`110c3b592b6d1460c991c77c8461c7b4bb82d6e80e968b1931f75f62e2c487fe`；进度快照=`_dispatch/progress_20261114T190945.json`，SHA256=`aa39763cb855b256e8883a36f2adb398c0eb10283403d8567e8f257b5e8a4edd`。该PASS仍只限九个完成的qasper分片，不是NFA全臂、FlashInfer、跨臂输入身份或canonical闭集，未生成AVG16，全部保持pending-independent-verification。

### 2026-11-14 20:59：修复版监督、三模型 LongBench v2 FullKV 与 RULER 性能入口核查

- 本轮按正确产物布局 `<method>/rank<N>/pred/*.jsonl` 重新统计，修复版 NFA 九rank人口为 **195/194/230/219/147/157/280/278/289，合计1989/3750**；FlashInfer仍为rank0–5各0、rank6–8各412，合计1236/3750。九条NFA链均有真实runner/Python进程且日志推进，未见failed/done、重复生成或rank0–5 FlashInfer提前接力；本轮未执行partial canonical合并或评分。
- 三条独立FullKV链继续健康推进：Qwen3-32B在pc31 GPU1为 **98/503**（宿主PID=3460664、compute PID=837982），GLM-4.7-Flash在pc32 GPU1为 **151/503**（宿主PID=669202、compute PID=2351837），Llama-3.1-8B在pc02 GPU1为 **350/503**（宿主PID=1508883、compute PID=2375748）。tqdm仅为处理进度，正式JSONL与complete runcap仍需结束后验收，不能把当前进度当作已落盘完整预测。
- 五机GPU逐一复核：pc33双卡为rank0/1，pc40双卡为rank2/3，pc31为rank4+Qwen32B，pc32为rank5+GLM，pc02为rank6/7/8+Llama；已核验空闲卡为0，本轮未启动任何竞争性能或精度任务，未接管其他AI的GLM AsyncTLS/NFA。
- 原始快照=`_dispatch/progress_20261114T205900.json`，九rank与FlashInfer人口求和复核PASS，SHA256=`38ae08085cde9e87d2714d789d6a657a22df837d57f08f0a167bbd151fb22abc`。远端监控脚本未常驻各机 `/tmp`，本轮通过SSH标准输入执行同一固定只读脚本，未把该部署差异误报为质量任务失败。
- 最新RULER MultiKey2四方法性能矩阵继续采用同一输入、固定33 token、相同warmup/repeats与 `T(33)-T(1)` 完整请求host-wall，不混入旧逐步CUDA Event结果。源码核查确认：官方Quest CUDA适配器当前B1 only；官方JSSA patched-vLLM当前 `max_num_seqs=1` 且固定cu129/radix门禁仍在；AsyncTLS只可采用K2=1024、K1=128×64、per-KV-group channel及上一步块集的论文语义复现；NFA需在现有B1两点入口上复用已验证B1/2/4 cache/scheduler能力。未用Torch/vendor fallback填补BS2/BS4，未宣称新性能结果，全部保持pending-independent-verification。

### 2026-11-14 21:35：NFA 真实 B1/2/4 两点入口与共享 batch 输出合同

- `benchmark_long_context_two_point.py` 已扩展到32K/64K/128K/256K/512K及B1/2/4。B>1将同一输入复制到各行，逐行B1 prefill进入同一固定cache arena，decode使用真实batched scheduler与对应Graph bucket；逐步不读取GPU token，request-wall停止后一次D2H。输出为batch-major逐行33 token，`generated_tokens=33×B` 与 `nfa_decode_steps=32` 分开记录，TPOT是整个batched decode step的host-wall且不除以B。FlashInfer在该入口B>1继续fail-closed，不以串行B1冒充batch。
- 共享 `timed_pair()` 现严格核验B1扁平和B>1 batch-major输出，拒绝混合布局、空输出、非整数token、参差行以及T(1)/T(N)行数不一致；保留B1既有字段兼容性，新增batch大小、输出布局、总token数、逐行token数和逐行EOS位置。新增B2/B4固定cache容量、scheduler max_batch、Graph bucket、pool复用、逐行清理、错误总人口与FlashInfer拒绝测试。
- 相邻Graph/cache/frozen回归首次为50 passed、1 failed：forced suffix测试在非阻塞 `prepare_step()` 后立即断言CPU `next_tokens`，偶发读到旧token。单独复跑通过，根因是测试把 `event.query()` 误当成D2H完成保证，而非生产forced suffix/EOS逻辑错误。只在测试检查CPU镜像前增加显式 `flush_ready_receipts()`，未把生产热路径改为逐步同步，也未改变预期token或EOS语义。
- 最终官方LongBench/RULER、JSSA接口、NFA两点、Graph/cache/frozen七组聚焦回归普通和 `python -O` 均为 **104 passed、1 skipped、9 subtests passed**；五文件 `py_compile` 与 `git diff --check` PASS。以上均为主机/mock及轻量CUDA调度回归，不是Qwen3-8B真实B2/B4 eager/Graph模型smoke或性能验收。
- 21:35本机双卡仍各有约23GB显存及compute app，对应rank0/1质量Python PID=430068/437834且宿主链仍在，未启动竞争性能任务。真实GPU B2/B4 smoke、AsyncTLS K2=1024两点入口和四方法48-cell矩阵仍未完成，不登记任何新TPOT，全部保持pending-independent-verification。

### 2026-11-14 22:03：官方四长度 RoPE/batch 边界与 AsyncTLS K2=1024 干净两点入口

- `long_context_official.py` 已扩展为32K/64K/128K/256K/512K，并显式增加 `--batch-size 1|2|4`。共享 `OfficialEngine` 与长上下文 `main()` 都在模型构造前调用统一batch门禁；Quest/JSSA当前只接受B1，B2/B4直接fail-closed，不用串行B1冒充batch。运行时和输出payload均记录真实 `batch_size`，schema升级为 `qwen-official-long-context-performance-v2`。
- 四长度RoPE不再无条件YaRN：32K保持原生配置且Quest走官方 `apply_rope_in_place`；64K/128K/256K/512K分别使用YaRN factor 2/4/8/16，Quest在这些长度使用HF position embeddings且只应用一次。测试逐方法覆盖五个长度，核验构造前config、CPU preflight、Quest rope mode、max-position容量和实际factor。
- 新增 `benchmark/efficiency/benchmark_asynctls_two_point.py`，只复用论文语义Graph runtime执行机制，明确标记“非官方TileLang、无真实异步offload”。预算在参数解析后及engine构造时双重固定为K2=1024、block64、K1=128，旧budget=512配置直接拒绝。输入、RoPE、固定33 token、warmup/repeats及 `T(33)-T(1)` host-wall公式与NFA/Quest/JSSA一致，TPOT不按batch归一化。
- AsyncTLS逐行prefill到同一固定cache arena，随后真实B1/2/4 batched Graph decode；首token和历史均保留在GPU，request-wall停止后一次D2H并输出batch-major行。无逐步CUDA Event、显存profiler或逐步host token读取。首个decode请求允许一次冷warmup+capture，之后T(1)/T(33)复用同一Graph；每次正式首replay前恢复 `M_(t-1)`，receipt分别核验逐行prefill调用、仅cold warmup/capture的Python decode调用、capture/replay/fallback和逐行token数，禁止把Graph replay冒充Python backend调用。
- 新增 `tests/test_asynctls_two_point.py` 并扩展官方RoPE/batch测试。首轮回归发现测试回执漏标冷capture，以及零decode步将 `[steps,batch]` 直接与首token按batch维拼接的形状错误；已分别修正测试身份和batch-major转置，未改变计时口径。最终八组相邻回归普通和 `python -O` 均为 **118 passed、1 skipped、19 subtests passed**，五文件 `py_compile` 与 `git diff --check` PASS；这些是主机/mock和轻量CUDA调度验收，不是AsyncTLS真实Qwen3 Graph smoke或性能数据。
- 22:03本机GPU0/1仍有约25.4/27.3GB显存和compute app，rank0/1质量Python PID=430068/437834及宿主shell均存活，未启动竞争GPU测量。真实NFA B2/B4 eager/Graph smoke、AsyncTLS实机smoke和四方法48-cell矩阵仍待空闲卡验收；本轮没有任何新TPOT，结果继续保持pending-independent-verification。

### 阻塞与下一步

1. Quest/NFA/FlashInfer 256K/512K正式3 repeats及LongBench16 clean性能均已闭集并通过结构复核，不重复测量；512K中位TPOT=83.8234/61.0523/40.1419ms，FlashInfer快于NFA的负排序保留，LongBench16 macro TPOT=11.3192/15.4917ms。
2. 持续监督修复版九rank，失败必须先读真实日志并精确定位；rank0–5的NFA正常完成后由同一runner串行接力FlashInfer，点火前继续核验同卡无其他compute进程，不改变九分片或重复生成rank6–8的1236条完整FI结果。
3. 两臂九rank全部有效完成后，依次运行新根 `_dispatch/harvest_full.py`、`audit_full.py`，只有 `rank_audit.json.status=PASS` 才能运行 `close_full.py` 做canonical合并和官方严格评分；最终重算16/16任务、3750/3750请求/臂、输入身份、真实停止、eager backend计数和逐文件SHA，当前不登记AVG16。
4. 取得同一 vLLM 提交的官方 cu129 预编译 wheel，完成 patched-vLLM editable 安装并运行官方 JSSA kernel tests；若官方运行时无法获得或550驱动不支持，如实记录阻塞，不退回Torch，也不换不匹配wheel。
5. 官方链收尾后清理旧Quest/JSSA vendor路由和文件，保留AsyncTLS；其他AI的GLM AsyncTLS/NFA保持不动。所有新结果落盘、官方评分、哈希manifest和不同owner独立验收前不宣称收官。
6. 官方Quest/JSSA四长度RoPE/batch边界与AsyncTLS K2=1024两点入口已完成主机验收；待本机质量链正常结束且双卡实测释放后，先做NFA B2/B4 eager/Graph和AsyncTLS B1/2/4 Graph真实模型smoke，核验逐行token、Graph capture/replay/fallback和receipt，再独占测量RULER MultiKey2四长度性能，不生成虚构TPOT。
