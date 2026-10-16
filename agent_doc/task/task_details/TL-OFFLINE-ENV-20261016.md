# TL-OFFLINE-ENV-20261016 — 论文与已登记 baseline 离线环境

Task-ID: TL-OFFLINE-ENV-20261016
Date: 2026-10-16
Owner: GPT environment writer, branch `gpt_dev/paper-offline-environment-20261016`.

## Plan

用户明确授权在隔离副本补 `.venv-paper`、PDF/LaTeX 离线依赖及全部已登记
baseline 官方源码/环境，测试后直接非 force 推送自己的分支，不自动合 main。
不改人类 guide、模型算法、论文正文/PPT/图/数据，不调用 GPU、权重、付费、Claude、SGLang。
另一线程只改引言，本任务仅修改构建入口与字体路径配置代码。

先核 fresh main、指南映射、任务与既有源码/开发分支；复用已登记 SHA 与现有
vendor。baseline 源码、基础 venv、CPU 导入、真实模型执行、GPU 性能分开登记；
未知许可不发布，不用替代聚类或新增模型矩阵补缺。保存端与宿主环境区分，
Python wheel、系统包、CUDA 依赖分别处理，哈希锁与离线重建才算相应层完成。

## Progress

- 基线 main `88dcb39e3fd2e6d85126bd0610d36d0043783e9c`；项目无 AGENTS.md。
  `agent_doc/guide/README.md` 映射的远端 sglang 人类文件在本环境不可读，未登录远端；
  按本轮明确范围继续，指南零写入。读取技能/治理版本及文档哈希见
  `tools/offline_environment/evidence/{governance,skills}.json`。
- FFD/Quest/FASA/SparQ/MInference 已有 vendor，优先复用。旧清单 `/tmp` 实体不在
  本环境，按原固定对象重新获取；ClusterKV/Quest/TileLang 递归子模块已补齐。
  MInference 原 `d542c489…` 是 tag 对象，真实 commit 为 `d76b76e8…`，两者均记录。
- 实际创建 `.venv-paper`：15 wheels 随 Git 交付，SHA256/大小/版本与官方 PyPI
  release digest 对照，离线安装/pip check/PDF-PPTX 合成读写通过；新 venv 复放保留日志。
  目标限定 CPython3.12/Linux x86_64/glibc≥2.28，不把保存端 Debian13 当目标宿主。
- 系统闭包 286 个 Debian13 amd64 官方包已下载并逐个与已认证 APT 索引 SHA256
  对照；另存完整 tar、版本/源包/版权位置清单，不拆分、不把大包推普通 Git。
  只下载，未安装系统包。严格锁定下载器与浮动重解析 recipe 明确分开。
- 构建入口接受 Python/TeX/字体配置；修正已消失 evidence_register.tex 默认目标。
  隔离副本双语正文/补充材料编译 6/6/8/8 页，替代根路径也通过。未发布生成物，
  未作科学/全页视觉验收，未改正文或数据。
- baseline 分离 CPU venv：FFD 官方模块/Quest Qwen3 adapter/SparQ core 导入通过；
  FASA 导入时 CUDA allocation 阻塞、MInference 缺 `minference.cuda`、MoBA 缺
  flash-attn；HiP/HISA 是未装依赖 scaffold，不能称完成。ClusterKV 创建了真实
  Python3.10.15 scaffold，NVIDIA pylibraft24.10 wheel 索引403仍阻塞。
- 8 项离线安全回归 normal/-O 通过。原 baseline 集成测试 12通过/3个 SparQ 失败，
  normal/-O 一致；本轮未改该实现与测试，保留返回值/统计字段不匹配证据，不伪报全绿。
- 独立只读审查核对 wheel 哈希、pip check、shell语法及安装门禁；修正其指出的
  fresh-clone 测试副作用和来源推断表述，记录真实独审范围。

- 提交前fetch并merge引言线程新main `08ccb546…`，保留全部对方变更；针对新输入
  在另一隔离副本重复完整重建，证据见 `paper-build-current.json`。
- 两份 requirements.lock 被根目录旧 `*.lock` 规则忽略的问题已用精确例外修正，
  发布前从干净 checkout 验证实际交付文件，不能仅测工作树。

## 阻塞与下一步

1. 返回目标宿主 probe 后才能选择/重制 ABI 与系统包；当前无目标宿主安装证据。
2. 大型系统/CPU tar 已在保存环境产出并哈希，尚未上传 Release/传到宿主。
   `gh auth status` 失败；普通 Git 分支推送另行核验。不能把版本清单称完整离线交付。
3. ClusterKV 原项目许可未知，HiP FSL 条件许可/HISA作者入口未明确许可不新增vendor；
   CUDA工具链/编译扩展、模型接口/校准/训练条件仍见逐方法清单。
4. 全部真实 checkpoint 执行/GPU性能未做；不得使用 CPU导入证明这些项通过。
5. 当前工程包是可续工成果；原S-T023复现科学验收不标完成。

入口与完整边界：`tools/offline_environment/README.md`。
