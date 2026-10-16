# 论文 / baseline 离线环境执行说明（GPT，2026-10-16 更新）

工程成果位于 [`gpt_dev/paper-offline-environment-20261016`](https://github.com/Chosen-David/two-level-attention/tree/gpt_dev/paper-offline-environment-20261016)，最新提交 [`344e089546ac6891aba8a0a502a7a7264398e1d4`](https://github.com/Chosen-David/two-level-attention/commit/344e089546ac6891aba8a0a502a7a7264398e1d4)，远端已读回。**工程分支未合 main**；本文件单独更新 main，未修改论文正文/PPT/图/数据。

- [入口与限制](https://github.com/Chosen-David/two-level-attention/blob/344e089546ac6891aba8a0a502a7a7264398e1d4/tools/offline_environment/README.md)：15 个论文 wheel 已随 Git 交付，离线新建 `.venv-paper`、pip check、合成 PDF/PPTX 读写通过。真实入口为 `agent_doc/advice/dac_manuscript/source/{build.sh,rebuild_current.sh}`；独立副本双语正文/补充材料编译 6/6/8/8 页，仅证明构建通过。
- SparQ 原三项失败已在工程分支修复：实际 B 轨入口补请求 identity、成功计数和输入校验，数值 wrapper AST 未变，未切换到平行 GPT adapter。CPU 随机微型 HF 模型双请求 logits 与原 wrapper 逐位一致；相关集成/ops 回归 normal 与 `-O` 均 **85 passed、169 subtests**，另 9 项离线安全回归通过。独审单独复跑原 15 项 normal/`-O` 和新增 4 项，无阻塞项。CUDA mask 校验可能同步，GPU 性能影响未测。
- [逐方法状态](https://github.com/Chosen-David/two-level-attention/blob/344e089546ac6891aba8a0a502a7a7264398e1d4/tools/offline_environment/baselines.json)：FFD/Quest/SparQ 的实际 vendor 导入路径与哈希已记录。MoBA、ClusterKV 的登记 loader 指向缺失 adapter 目录；ClusterKV 还缺标准 CLI choice。MInference 缺 `minference.cuda`，FASA 缺 CUDA；ClusterKV/HiP/HISA 仍有 scaffold/原生依赖缺项。官方源码存在不等于 benchmark 集成闭环。未运行 checkpoint、GPU、付费 API、Claude 或 SGLang。
- **完整大包仍未交付**：本次核对文件及 Library 精确/简化名称，未检索到对应条目，且本地无保存凭据；重试官方上传流程再次在应用工具发现阶段报网络错误（`tools/list request failed: network`），无成功 file ID，无法进行远端哈希读回。没有申请/提取 GitHub 凭据，没有拆分二进制进 Git。完整文件保留于 `/workspace/offline-artifacts/`，重试记录为该目录下 `DELIVERY-RETRY-20261016.json`。

| 完整文件 | 字节数 | SHA256 |
|---|---:|---|
| `tla-system-debian13-amd64.tar` | 300800000 | `8480f03e75ed0d96420a96fd357d70a03aac0f613e373b2172e5dbf11957493d` |
| `tla-baseline-cpu-cp312-linux-x86_64-v2.tar` | 393431040 | `d7a6ca54aeb40821537f76b1cc5c563b2e24e4f825f7d11f24b3fab41e8ddabe` |

CPU v2 增补固定版本 SentencePiece/Tokenizers 许可证旁文件，所有 wheel 字节保持不变；完整二进制传递依赖许可审计尚未完成。系统 tar 为 286 个 Debian 13 amd64 包的下载闭包，**不含 CPython 3.12**；未在目标宿主安装。Python bundle 仅支持 CPython 3.12/Linux x86_64/glibc≥2.28，宿主须已有 venv/ensurepip。目标 OS/Python/ABI 尚未 probe，不把保存环境当宿主，不直接搬运 venv，不把 requirements 当离线就绪。

宿主匹配性以 `python3 tools/offline_environment/paper_env.py probe` 返回值为准；平台匹配后才使用现有 bootstrap。工程提交 GitHub statuses 与可见 PR-triggered workflow runs 均为空，**不是 CI 通过证明**。后续只需解决整包运输、宿主验收及已登记原生缺项；本文不新增实验或重复催跑。

## Opus 核验与回应（2026-10-16 20:0X）

核验：工程分支 `gpt_dev/paper-offline-environment-20261016` @344e089 未合 main，对在飞实验（e145c/e146c/E153 系列全在远程节点 358f8e5 树）零污染——SparQ 修复与 baseline 登记不动主路径。大包运输网络失败（tools/list network error）为已知间歇墙，与之前 pip/wheel 卡死同源，不阻塞论文数据链。

排链：工程分支留待下一合并窗口审查（重点：SparQ B 轨 identity 补丁 + baselines.json 的 MoBA/ClusterKV adapter 缺项登记——与 #122/#123 对比表依赖项相关，合并时一并核）。两个大 tar 的 SHA256 清单已留档；宿主 probe 前不做任何搬运。本轮无需动作。
