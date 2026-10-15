# ClusterKV 官方源码接入调研（2026-10-14 by deepseek）

## 结论先行

ClusterKV 官方是**端到端 LlamaAttention 整体替换**（非 mask/接口复用），依赖
**RAPIDS（pylibraft+rmm）+ 官方编译扩展（_clusterkv_knl）+ FlashInfer**，当前
主 python 三依赖全缺、官方自带 .venv 是空壳。完整接入是一个**独立大工程**，
需先装 RAPIDS 并编译官方 CUDA 扩展。但**官方默认 token_budget=1024 与我们
K2=1024 天然对齐**（对比 Quest/SparQ/FASA 预算口径无需换算），这是最大优点。

## 官方仓库形态（third_party/ClusterKV/）

- `clusterkv/clusterkv_models/ClusterKVAttention.py` — 完整 LlamaAttention 替换
  （含 q/k/v proj、RoPE、KV cache、prefill/decode 双路径、controller）
- `accuracy/cluster_attention.py` — 精度路径的 `forward_cluster`（绑定到
  LlamaAttention.forward）+ `build_cluster` + `cluster_attn_out`
- `accuracy/patch.py` — `enable_attention_eval`：`types.MethodType(forward_cluster,
  module)` 替换 forward，设 `module.cluster_cache/token_budget/chunk_size`
- `clusterkv/clusterkv_utils/clusterkv_controller.py` — controller 状态机
  （kv_seqlen/generated_len/sel_token_indices/offload）
- 依赖：`pylibraft.cluster`（KMeans/ivf_flat）、`rmm`、`clusterkv._clusterkv_knl`
  （search_indices，编译扩展）、FlashInfer（`BatchDecodeWithPagedKVCacheWrapper`）

## 算法语义（官方默认口径）

1. **prefill 建簇**：每层每 kv-head 对 prefill key 做 KMeans（`--nlist 400`、
   `--fit_iter 20`、`--dist cosine`，可选 balanced=ivf_flat），得到簇心
   `key_centroids [1,H,nlist,D]` 与 `cluster_key_indices/ptr/size`
2. **decode 选簇**（`cluster_attn_out`）：query 与簇心点积排序 → 取邻居簇 →
   按簇 size 累积到 `token_budget - sink` 选出所需簇数 → 同预算下所有 head 取
   相同簇数（`head_sel=truc`）
3. **decode 稀疏 attention**：只对选中簇的 KV 做 FlashInfer decode 稀疏注意力
4. **预算**：官方默认 `--token_budget 1024`、`--sink 16`、`--chunk_size 16`、
   `--nlist 400`、`--head_sel truc`、`--dist cosine`、`--balance False`

## 与我们主表口径的对应

| 项 | ClusterKV 官方默认 | 我们主表 |
|---|---|---|
| token budget | **1024** | **K2=1024（无需换算，天然对齐）** |
| sink | 16 | TLI sink_tokens=128（不同，但 sink 是方法内部细节） |
| chunk_size | 16 | —（ClusterKV 内部 KV 分页粒度，非预算） |
| nlist | 400 | —（簇数，方法参数） |
| dist | cosine | —（方法参数） |

## 接入路径评估

- **路径 A（正式，接我们 harness）**：主 python 装 `pylibraft-cu12 rmm-cu12`
  + 编译官方 `_clusterkv_knl`（C++/CUDA 源码）+ 写 `qwen3_clusterkv_patch.py`
  adapter（官方只支持 LlamaAttention；Qwen3Attention 需适配 controller 状态机、
  FlashInfer decode wrapper、cache 接口）。工作量：大（数小时级别，需环境就绪）。
- **路径 B（官方 eval 取数）**：补官方 .venv 依赖，跑官方 `accuracy/LongBench`
  /`pg19` 官方协议。但官方协议 ≠ 我们统一 harness（RULER 不在官方 eval 清单），
  数不能并入主表，仅作交叉验证。
- **预算对比意义**：ClusterKV 是 decode 稀疏（同 Quest/SparQ/FASA），预算 1024
  对齐 → 主表可比性无坑。ClusterKV 官方验证模型为 Llama-3.1，Qwen3 适配属
  我们协议层（同 SparQ/FASA 的 GQA 适配，语义不变）。

## 待办（建议独立任务）

1. 装 RAPIDS（pip install pylibraft-cu12 rmm-cu12，~GB 级下载）——先确认网络/镜像
2. 编译官方 `_clusterkv_knl` 扩展（看 `clusterkv/_clusterkv_knl*` 源码，setup 或
   pybind 构建）
3. 写 `qwen3_clusterkv_patch.py` adapter + pred 两入口 `clusterkv_vendor` 分支
   + treatment 名编码（token_budget/nlist/dist 等进名）
4. smoke（预算对齐：budget=1024 主表口径）+ RULER/LongBench 派单

## 风险

- RAPIDS（cu12）与 torch 2.8.0+cu128 的兼容性需验证
- `_clusterkv_knl` 编译扩展需要 CUDA toolkit + 官方 pybind 构建链
- FlashInfer 已可用（主 python OK），但需确认版本兼容官方 wrapper 接口
