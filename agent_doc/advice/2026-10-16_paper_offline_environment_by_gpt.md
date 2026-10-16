# 论文 / baseline 离线环境执行说明（GPT，2026-10-16）

工程成果已非force推到 [`gpt_dev/paper-offline-environment-20261016`](https://github.com/Chosen-David/two-level-attention/tree/gpt_dev/paper-offline-environment-20261016)，提交 `54e343aa896c8fee3ecaad29e3b4a8e8b16e5803`，已回读。已整合引言更新 `08ccb546…`；**工程分支未合main**，本文件单独发布main。

- [完整入口/限制](https://github.com/Chosen-David/two-level-attention/blob/54e343aa896c8fee3ecaad29e3b4a8e8b16e5803/tools/offline_environment/README.md)：15个论文wheel随Git交付，离线新建`.venv-paper`、pip check、合成PDF/PPTX读写通过。仅CPython3.12/Linux x86_64/glibc≥2.28；宿主尚未probe，保存环境不冒充宿主。
- 构建入口实际为`agent_doc/advice/dac_manuscript/source/{build.sh,rebuild_current.sh}`，已支持Python/TeX/字体配置并修复不存在的默认目标。最新双语正文/补充材料在独立副本编译6/6/8/8页，未改/发布正文、PPT、图或数据生成物。
- [逐方法清单](https://github.com/Chosen-David/two-level-attention/blob/54e343aa896c8fee3ecaad29e3b4a8e8b16e5803/tools/offline_environment/baselines.json)：复用现有vendor；ClusterKV/Quest/TileLang递归gitlinks已取得。FFD/Quest/SparQ CPU导入通过；FASA缺CUDA、MInference缺编译扩展、MoBA缺flash-attn。ClusterKV真实Python3.10.15与HiP/HISA仅scaffold，**不是完整环境**。ClusterKV未知许可不vendor；无权重/GPU/付费/Claude/SGLang。
- 9项离线安全回归normal/-O通过，含断网后源码续取/不覆盖改动。原baseline集成回归**12通过/3个SparQ接口失败**，未改算法掩盖。独立只读审查已做。GitHub返回该提交statuses空、PR触发workflow_runs空；仓库无workflow文件，**不是CI通过证明**。
- 大包尚未运输：Debian13 amd64系统包闭包286个、完整tar **300,800,000 B**；baseline CPU完整tar **393,318,400 B**。版本/哈希已锁、前者逐包与已认证APT索引核对，后者已离线重建；没有系统包安装/目标宿主验收。保存端在`/workspace/offline-artifacts/`，文件名与SHA256见[运输清单](https://github.com/Chosen-David/two-level-attention/blob/54e343aa896c8fee3ecaad29e3b4a8e8b16e5803/tools/offline_environment/large-artifacts.json)。不拆分绕Git限制；Release上传通道未建立，不能称全部离线交付完成。

宿主先建隔离worktree再运行`python3 tools/offline_environment/paper_env.py probe`。平台匹配时用`python3.12 tools/offline_environment/paper_env.py install`，再显式设置`PAPER_PYTHON="$PWD/.venv-paper/bin/python"`运行真实入口。系统包/原生CUDA环境待宿主probe与完整大包运输；不要直接搬运venv，也不要把requirements清单当就绪。
