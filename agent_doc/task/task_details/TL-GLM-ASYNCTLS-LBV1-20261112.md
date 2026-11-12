# [TL-GLM-ASYNCTLS-LBV1-20261112] GLM-4.7-Flash AsyncTLS LongBench v1 全量精度

Task-ID: TL-GLM-ASYNCTLS-LBV1-20261112
Date: 2026-11-12

## Plan

### 目标与验收

按用户优先级，先完成 GLM-4.7-Flash × AsyncTLS × LongBench v1 英文16任务全量精度，再处理其他旧口径 baseline，最后重测 NFA。最终 token 预算固定1024，Stage-1 固定128个64-token块；逐 KV group 校准32个 channel，INT4 量化-反量化，decode 使用上一步块集，完整 prompt 一次 dense prefill。

完成真实47层校准与 smoke、16/16任务3750/3750请求、确定性互斥分片、runcap 输入身份、官方逐样本评分、原始证据和 SHA256 manifest 后，只提交本任务文件并推送 `origin/main`。生产侧校验不冒充不同 owner 独立验收，所有新分数保持 `pending-independent-verification`。

### 实现方法

- 复用 `glm4_moe_lite_asynctls_patch.py` 的 GLM MLA 投影和共享 AsyncTLS helper；GLM 每个 KV group 一个 query head，不额外做 GQA 平均。
- 校准使用 qasper 前8条、每条最多4096 token、`dc=32`，输出 `[layer][kv_group][channel]`，绑定模型、数据、代码和校准文件 SHA256。
- 使用 pc02 的 transformers 5.12.1 环境；真实运行只用点火前实测空闲 GPU，不与既有计算进程混卡。
- 聚焦回归覆盖 dense/full-budget 对照、prefill 块集初始化、decode 上一步块集、多步推进、第二请求复位、稀疏形状、非法 channel 和 EOS list。

## Progress

### 进度与证据

- 2026-11-12：pc02 `/home/fluentllmenv/bin/python` + `~/.local/pylibs_tf512` 已确认 transformers=5.12.1，可导入 `Glm4MoeLiteAttention`；本机 transformers 4.56.2 不支持该模型，不作为 GLM 测试环境。
- 原生 `Glm4MoeLiteAttention.forward` 源码已核对，adapter 的 Q/K/V 投影、RoPE、cache update 和 eager 输出接口与之一致。
- 新增 `tests/test_glm4_moe_lite_asynctls.py`；pc02 临时隔离树实测 `6 passed in 9.87s`，覆盖 full-budget native dense 对照、`M_{t-1}`、多步推进、第二请求状态复位、稀疏形状、非法 channel 与 EOS 提前停止。
- 临时测试树为 `/tmp/glm-atls-unit-20261112`，不是正式性能或精度产物；adapter SHA256=`ae308b22ddedb0b66334bb2eafa828f6bbe45f9dd47f3f30c13aee03357e6fad`，calibrator SHA256=`d796993f4c05c14e133a9611be40ea3ab62cd39b57832252606020b9d3665213`。
- 点火前盘点 pc02 GPU0/1 被 PID1476356/1476358 占用约66/70GB，GPU2/3 各1MiB；GLM 权重与 qasper 数据存在。GPU2 选作校准候选，正式 smoke/分片前重新检查。
- 真实47层校准已完成：qasper 前8条/max4096/dc32，输出47×20×32，channel SHA256=`87b28849a4ec034c6c52df4a0b1d35ef58c04d88a8557327d967140c1dc8058a`；本地与 pc02 字节一致。
- qasper/samsum 真实首样本 smoke 均完成并落 complete runcap；samsum 生成142.50s，输出重复文本只说明链路可运行，不作精度结论。GLM AsyncTLS 走完整 prompt dense prefill + eager 逐 token decode，samsum 不落入非 AsyncTLS 的 `model.generate` 特例。
- LongBench v1 已支持 `source_order[rank::world_size]`：先对完整有序源行计算 identity，再切分；输出 `_id` 使用完整源行号，runcap 同时绑定完整源人口、选中人口、source/selected SHA256。默认 rank0/world1 的源行 ID 与旧口径一致。
- pc02 transformers 5.12.1 环境聚焦回归 `19 passed, 12 subtests passed`，覆盖 LongBench v1/v2 分片、runcap、防重复源 ID 及既有6项 GLM AsyncTLS 语义。真实双卡四样本 smoke 每片2请求，complete ID 分别为0/2与1/3，完整源 `sample_count=4`、每片 `actual_input_ids.count=2`，证明多请求与分片身份闭包。
- 新增 `run_longbench_glm_asynctls.sh`，固定16任务人口、K2=1024/block64/K1=128、channel 文件、NFA layer-skip=false、互斥输出目录和 complete-only SKIP；4样本 smoke 用全量 runner 重跑时被 source=200 人口门禁拒绝，属于预期 fail-closed，未冒充幂等通过。
- 实现提交 `a0b086850ba7d290fda47c1c27833e7057e774bb` 已推送 `origin/main`；只包含本任务9个文件，未带入其他会话的 NFA 文档或 sqlite。pc02 内部 origin 无新提交且 GitHub fetch 阻塞，改用本地 `git bundle`（基元 `482911be`）离线部署到 `/tmp/glm-atls-release-a0b08685`，HEAD、pred/runcap/adapter/calibrator/channel SHA 均与本地一致。
- 固定提交部署后复跑 GLM/LongBench pytest `12 passed`，runcap unittest `7 tests OK`，runner 与双分片链脚本 `bash -n` 通过。新增生产侧 `agent_doc/results/glm_asynctls_lb_v1_20261112/collect.py`，将校验完整源身份、双分片 selected IDs、treatment/channel/commit、结果 SHA、官方逐样本评分与文件 manifest；合成4样本双分片闭集、源顺序与官方 scorer 测试 PASS。
- 2026-11-12 22:26 +08:00 正式点火：点火前 pc02 GPU2/3 各1MiB、MemAvailable=154250324kB，其他远端机器均有既有计算进程，未混卡。world=2，rank0/GPU2 runner=4105983、pred=4106005；rank1/GPU3 runner=4105984、pred=4106006；输出 `~/ak_verify_results/glm_asynctls_lb_v1_20261112`。点火后两卡显存约59GB，日志均进入 lcc `selected 250/500`，随后持续推进；完成前没有全量分数。

### 阻塞与下一步

- 固定实现 commit/关键文件 SHA，重新盘点 GPU 与内存后按16任务×3750请求矩阵派正式分片；收割时恢复源顺序并用官方 `sample_score` 逐样本重算。
- 正式结果完成 raw/runcap/summary/SHA256 manifest 后只提交本任务文件并推送；本任务推送前不启动其他 baseline 或 NFA 跑批。
