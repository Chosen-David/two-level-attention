# round11 代码 bug 监督器：GPU 全空 + ClusterKV 阻塞根因全面修正

> 作者：deepseek · 2026-10-13 01:25 · 每 2h 轮询（用户 2026-10-11 指令 + 18:4X 无事不写规则）

## 1. GPU 占用表（全空）

| 机器 | GPU | 显存/利用率 | 任务 |
|---|---|---|---|
| 本地 | 0/1 | 1 MiB / 0% | 空 |
| .187（10.238.9.187） | 0-7 | 1 MiB / 0% | 空 |
| .251（10.238.139.251） | 0/1 | 1 MiB / 0% | 空 |

**12 卡全空，无任何 pred/score/ruler python 进程**（本地 + 远程 ps 确认）。.24（33.32.33.24）已删除，未查。

## 2. 在飞任务

- **无在飞任务**：S-T019/E116b、S-T022/E125、S-T023（Quest/LBv2/RULER 32/64/128K）均已收官，主表已更新（§3 128K=19.33 + §6 子空间消融）。
- **锁文件检查**：`e116b_ruler128k_formal_*.lock` / `e119_ruler128k_formal_*.lock`（10-11 时间戳）为**打分完成后的残留锁**，对应 result.json 均存在 → 非 bug、无害。
- 无任务失败、无卡死进程。

## 3. ClusterKV 阻塞根因全面修正（重要）

原 task-chain 登记："pylibraft 24.10.0 无 cp311 wheel → py3.11 无可用版本"。本轮实测**修正**：

1. **官方 .venv 是 py3.10.15，不是 py3.11**（原登记基于 py3.11 环境假设）。
2. **pylibraft-cu12 24.10.0 与 libraft-cu12 24.10.0 对任何 python 版本都无预编译 wheel**：
   - pypi 上 24.10.0 仅 sdist（`pip index versions` 显示 wheel 从 25.6.0 起才有）；
   - RAPIDS 官方 24.10 wheel 频道（pypi.anaconda.org/rapidsai-wheels-24.10）为空；
   - `--only-binary :all:` 安装报 "No matching distribution"。
3. 新版 25.6+/26.x 有 wheel 但**移除 cluster API**（`from pylibraft.cluster import KMeansParams, fit` 不可用）——原登记此点正确。
4. **3rdparty 三个子模块（raft/flashinfer/pybind11）gitlink 缺失**（浅 clone 副作用），`git submodule update --init` 无效果（.gitmodules 声明但 index 无 gitlink）。
5. **本机 github.com 大传输超时**（实测）：
   - `git clone` raft/flashinfer：`Failed to connect github.com port 443 after 129181 ms: Connection timed out`；
   - `curl` tarball（pybind11 v2.13.6）：124s 超时被杀；
   - 仅 `curl -sI https://github.com`（HEAD 无 body）返回 200。
   - 结论：**HTTPS 握手/小请求通，大文件/数据下载被限** → pylibraft sdist 构建卡在 PEP517 metadata（build backend 从 github 拉 raft 源码）根因即此。
6. 原生支持 glm4-9b / llama3 / llama3.1（accuracy/config/model2path.json），**Qwen3-8B 需适配层**（同 qwen3_quest_patch 模式）。

**ClusterKV 完整阻塞链**：无 wheel（需源码编译）→ 源码编译需 github 拉 raft（超时）→ 子模块 gitlink 缺失且无法拉取。**当前网络环境无法完成源码获取 = 硬阻塞**，非 pip 版本选择问题。

## 4. 派单情况

- 本轮**未能派 GPU 任务**：ClusterKV 接入（S-T023 唯一剩余竞品）因上述阻塞无法推进；待办池其余项（E118 sglang 集成 / E120 依赖 S-T009 / E127 Opus 职责）均不适用本轮。
- 已尝试动作：pylibraft 安装（sdist 构建 + RAPIDS wheel 源双路，均失败/无 wheel）→ 结论记录；子模块 git clone + tarball（均超时）→ 结论记录。

## 5. 下一步建议

1. **ClusterKV 源码获取需镜像/代理**：ghproxy 类镜像（github.com → 镜像站）或内网 GitHub 缓存，解决后即可按 README 走：raft build.sh → kernel build → flash-attn 2.6.3 → pip install -e . → qwen3 适配 → 空卡贴合验证 → 16 任务 AVG 派单（可占满 12 卡）。
2. **GPU 空置**：当前 12 卡无排队任务。建议用户确认优先方向：投入 ~半天完成 ClusterKV 接入，或临时改派其他实验（论文消融/图指标等）。
3. MoBA training-free 分支弱（README 明确需训练），不作为短期目标。

（无代码 fix；本报告为阻塞调查实质结论 + 状态盘点。）
