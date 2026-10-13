# round13 任务监督：MInference vendor 通道打通（新 sparse-prefill 对比臂）

> 作者：deepseek · 2026-10-13 02:08 · 任务监督器（用户 2026-10-11 18:5X 指令，有实质进展才写）

## 1. 本轮具体动作

**打通 MInference（NeurIPS 2024 微软，动态稀疏 prefill）官方 vendor 通道**——这是 S-T023 自 Quest 全量收官后第一个实质推进的竞品对比臂，且**绕开了 github 大传输超时**（走 pypi.org 白名单）。

### 1.1 ClusterKV 硬阻塞再确认（非本轮动作，但为排除）
- pypi 无 clusterkv 包（JSON 404）→ ClusterKV 无 pip 通道；
- 全部 github 镜像代理（ghfast.top / ghproxy.net / gh.con.sh / gh-proxy.com）均不可达（FAIL）；
- 结论：ClusterKV 源码获取无解，B 轨维持登记阻塞（round11 结论不变）。

### 1.2 MInference 接入过程（障碍三连→全解）
| 步骤 | 结果 |
|---|---|
| 探测 | pypi `minference` 0.1.6.0 可达（sdist 含 CUDA 扩展 vertical_slash_index） |
| pip build ① | build isolation 临时环境缺 torch → `--no-build-isolation` 复用 quest venv torch |
| pip build ② | 缺 ninja/psutil（setup_requires 不自动装）→ 先 pip 装 |
| pip build ③ | nvcc 12.9 + torch 2.5.0 cu124 源码编译 CUDA 扩展 → **成功**（`minference/cuda.cpython-310-...so`） |
| vendor fork | 官方 0.1.6.0 全量 → `exp/vendor_baselines/minference_vendor/minference/` |
| 适配层 | quest venv transformers 4.45.2 无 `transformers.models.glm`（4.51+ 才有）→ 2 处 GLM try/except stub（仅 GLM-4 专属分支用到，Qwen 系不触发） |
| GPU smoke | `vertical_slash_sparse_attention`（CUDA 索引 + triton sparse attn fallback）假 q/k/v 输出 shape 对 + 全有限 + 末 token cos-sim vs dense 0.96-1.0 → **PASS** |

### 1.3 下一步（round14）
MInference 官方支持 Qwen2.5-7B/32B/72B（`configs/` 有对应 pattern JSON），**Qwen3-8B 不在 MODEL2PATH**——pattern 为逐层逐 head 的 `[vertical_and_slash, topk, slash, score]`（Qwen2.5-7B 是 28 层×28 头，与 Qwen3-8B 32 层×40 头不匹配）：
- **路径 A（sanity，最贴合 S-T023 阶段 1）**：官方 Qwen2.5-7B-Instruct-128k + 官方 pattern 跑 MInference → 对拍官方论文趋势（sparse prefill 近无损）。需确认 hf 模型下载可达。
- **路径 B（主表臂）**：qwen3_minference_patch（Qwen3Attention → MInference prefill forward 胶水 + 补 QK-Norm）+ pattern 来源需 `is_search` 在线搜（calibration 1-2h）或手写。

## 2. 状态盘点（对照 S-T023 竞品清单）

| 竞品 | 状态 |
|---|---|
| Quest | 全量收官（LBv1 8B 41.35/32B 38.08、LBv2 30.22、RULER 32/64/128K 42.85/26.25/19.33） |
| **MInference** | **通道打通（本轮）**，sanity/主表臂 round14 推进 |
| DeepSeek fullkv | 收官（AVG 40.55*，16K 口径 MLA dense 锚点） |
| ClusterKV | 硬阻塞（pypi 无包 + github 直连/镜像全超时） |
| MoBA | training-free sanity 暂缓（README 需训练 + flash-attn 缺） |
| FASA/SparQ | 未接入（同为 github 源码，受超时影响） |

## 3. GPU 资源

- 本地 GPU0/1 空（可用于 MInference smoke/sanity）；.187/.251 12 卡空（round11 确认，本轮未占用）。

## 4. 诚实声明

- MInference CUDA 扩展用本机 nvcc 12.9 编译（torch 为 cu124）；与官方 wheel 的编译配置差异已在产物标注口径，不影响算法语义。
- GLM stub 属适配层（非官方源码改动）：仅把 transformers 版本缺失导致的 import 崩降级为惰性可用，官方逻辑逐字节保留。
- 本轮无模型任务产出；具体 benchmark 数字待 round14 sanity 后补齐。

## 5. 下一步建议

1. 优先走路径 A（官方 Qwen2.5-7B + 官方 pattern）跑 MInference sanity，尽快补齐 sparse-prefill vendor 的论文趋势对拍。
2. 并行评估路径 B（Qwen3 适配）的 `is_search` pattern 校准成本，确定主表臂可行性。
