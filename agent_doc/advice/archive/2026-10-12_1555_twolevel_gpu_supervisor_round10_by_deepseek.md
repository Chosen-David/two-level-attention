# twolevel GPU 代码监督 round10（by deepseek）

- 时间：2026-10-12 15:5X
- 范围：round10 例行审查（GPU/进程/脚本 + ClusterKV 依赖阻塞排查）
- 结论：**在飞任务全正常无 bug；空闲 8 卡（.251×2/.24×4/.187×2）；ClusterKV 接入实质推进——网络阻塞解除（pylibraft/rmm/raft 均可获取），但发现 pylibraft 版本兼容障碍（官方 pin 24.10.0 无 cp311 wheel、新版 25.6/26.2 已移除 cluster API）+ 剩余前置（gcc-11/pybind/flashinfer/kernel build/qwen3 适配层）。**

## 1. GPU 占用表

| 机 | 卡 | 用途 | 状态 |
|---|---|---|---|
| 本地 | 0/1 | RULER quest 128K single_3/multikey_1（~50%） | 满 |
| .187 | 1/4/7 | RULER quest 128K multikey_2/3/multiquery | 满 |
| .187 | 3 | 32B quest qmsum 尾格（~75%） | 满 |
| .187 | 5/6 | E130（Opus 独占 TLI 消融 goff/g0.75） | 满 |
| .187 | 0/2 | — | **空闲** |
| .24 | 0-3 | RULER quest 128K multivalue/cwe/fwe/vt（~25%） | 满 |
| .24 | 4-7 | — | **空闲 4 卡** |
| .251 | 0/1 | —（128K single_1/2 已完成回收） | **空闲 2 卡** |

**空闲合计 8 卡**（.251×2 + .24×4 + .187×2）。

## 2. 在飞任务核实（脚本/参数正确，无失败无 bug）

- RULER quest 128K 11/11（pred_ruler.py quest_vendor --yarn factor 4.0，md5 bb3e1b9a 三机一致）推进正常。
- 32B quest qmsum 尾格（pred.py quest_vendor 035844 批）~75%，双跑无害。
- 本地 128K 2 格 + 8 格远程均正常，产物边跑边写 gen 目录 + 完成切 tli_gen。

## 3. ClusterKV 接入实质推进（本轮主要动作）

背景：ClusterKV 之前因 `pylibraft-cu11 需 pypi.nvidia.com 网络超时` 硬阻塞。本轮实测网络已恢复：

- ✅ **pylibraft-cu12 wheel 可下载**（26.10/25.6/26.2 均有 cp311 wheel，959KB）；
- ✅ **raft 源码可浅克隆**（github.com/rapidsai/raft → 3rdparty/raft）；
- ✅ **rmm-cu12 26.2.0 已可用**（quest venv 早有，librmm-cu12 26.2 亦在）。

**新发现 pylibraft 版本兼容障碍（真障碍，非网络）**：
- ClusterKV 官方 pin `pylibraft-cu11==24.10.0`（pyproject.toml，kernel raft branch-24.02）——**24.10 无 cp311 wheel**（仅 sdist，build 失败）；
- 新版 pylibraft（25.6/26.2）有 cp311 wheel，但 **cluster/neighbors API 已移除**（仅剩 common/config/random/sparse）；
- 当前系统 py3.11 无可用 pylibraft（既有 cluster API 又可装）→ ClusterKV 官方 accuracy 代码（`from pylibraft.cluster import KMeansParams`）无法 import。
- 可行出路：py3.10 venv + pylibraft-cu11 24.10（sdist 能否编通未验证），或 fork 官方代码把 KMeans 换成 sklearn 版（无 pylibraft 依赖）。

**ClusterKV 完整接入前置清单**（工作量 ~3-6h 独立工作包）：
1. pylibraft 24.10 版本适配（py3.10 venv 或 sklearn 替换）；
2. **gcc-11**（CMakeLists 硬编码 /usr/bin/gcc-11，四机全缺，需 apt 或软链）；
3. **pybind11 + flashinfer 源码**（kernel/CMakeLists 引用 3rdparty/pybind、3rdparty/flashinfer，现缺）；
4. clusterkv C++ kernel build（cmake 3.26+ ✓、CUDA 12.9 ✓ 已满足）；
5. **qwen3 适配层**（官方 patch.py 只 patch LlamaAttention/glm4，需写 qwen3 clusterkv patch）。

## 4. 空闲卡处理决策（本轮无新派单，理由如实）

- **MoBA**：README 明确"需 continue training，非 drop-in"，且 wrapper.py 顶层 `from flash_attn import flash_attn_func`（本地/远程全无 flash-attn，需编译）→ training-free sanity 开发量 + 编译前置，暂缓；
- **ClusterKV**：kernel build + 适配层未就绪（见 §3）；
- **主链**：RULER 128K quest 已 11/11 覆盖，无缺口可派。
- → 8 空卡保留，待 128K 完成后回收或作为 ClusterKV/MoBA 接入的派单资源。

## 5. 下一步

1. RULER 128K 全齐（~17:3X）→ rsync + 打分（--yarn-factor 4.0）对拍 e119 128K 锚点 → 主表拼接（Quest 三件套 + TLI + DeepSeek）。
2. ClusterKV：登记版本障碍（pylibraft 24.10 vs 新版 API），排期独立接入工作包（sklearn 替换 KMeans 是最短路径）。
3. 32B qmsum 尾格完成后 32B quest 收官确认。

— by deepseek（在飞任务全正常；ClusterKV 网络解阻 + pylibraft 版本障碍发现；空闲 8 卡暂留）
