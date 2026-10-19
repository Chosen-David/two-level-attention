# TL-DECODE-PADDING-Q-20261019

Task-ID: TL-DECODE-PADDING-Q-20261019
Date: 2026-10-19

## Plan

基线bc968288c7e69ec3d93b6d2acf546a3dcd26f962，独立干净检出，协作锁持有；保留旧检查点，不修改人类guide。映射的sglang人类文件不在当前环境，遵循可读仓库guide和本次直接授权；不外取凭据或假定旧GPU可用。复用代码实现优化技能，CPU PyTorch2.8.0、Transformers4.56.2，无CUDA设备。

用户授权恢复范围内修复Qwen3/Llama单query decode的2D bool padding契约。根因：cache后K/V长度S>1，Q长度1，却用[1,S] mask同时布尔索引三者。已用实际HF Attention/DynamicCache及生产CPU consumer复现34子例IndexError，selector为all-selected double，不代表真实selector/模型质量。

方案：仅压紧K/V，Q原样保留；在现有decode mask守卫校验精确[1,S]及当前query对应最后位置有效。保持mask=None、prefill、4D StaticCache验证、预算和kernel不变。拒绝输入仍遵循已有cache-update后守卫顺序，本次不声明异常的cache事务回滚。

验收：全True及全部3-bit前缀mask×两模型×CPU dense/gather，非连续mask；非法形状及屏蔽最后query须在selector前拒绝；None和合法配置输出与HF SDPA/既有实现对照；普通/-O相邻回归、独立复核、fresh fetch、仅明确文件普通push+回读。不做GPU/真实模型性能主张。若改变kernel或必要验证不能完成则保留补丁不发布。预算仅小型CPU测试，无模型权重下载、无新增付费资源。

## Progress

- 修前 `python -m unittest discover -s tests -p test_decode_padding_query.py -v`：2项测试、34个子例报错，均为[1,S] mask索引[1,1,H,D]失败，原始日志保留本地。
- 新请求切换到Agent仓库，当前TwoLevel补丁留本独立工作树，未提交/推送。首次修后2项测试通过；扩充到5项后的执行结果需续接时核对，普通/-O相邻回归及独立复核尚未完成。不得把此检查点当已发布修复。
- 续接先fresh fetch/review并发和本工作树，校验上一测试结果；检查mask最后query守卫的新增CUDA同步成本、拒绝发生在cache更新后的既有边界。无GPU，不宣称性能。prefill实验分支本轮确认2ed6a1feb78a7b5bcf3cac922b4510c3968714a5；未合并。上轮skip_far疑点当前支持生命周期仅decode冻结，不列已确认bug。

### 2026-10-19 续接验证

- 新独立检出 fresh fetch 后 main 仍为上述 bc968288；重新获得协作锁，保留旧工作树修改。修前重新执行两项最小复现：34 个子例 IndexError，未以旧日志代替当前验证。
- 当前专项扩展至 6 项/192 子场景（空 mask、非法形状、全部三位前缀、非连续输入、S=2/3/7/65、KV头1/2/4、float32/64、None）：普通及 `-O` 均通过，输入 SHA256 为 `713468324be3aacd8c6188a34132a8c3a7421a12f20bbbb2df6a1634efae3e85`。
- 更正 Plan 中 dense/gather 覆盖的表述：短序列即使 `TLI_E127_FAST=1` 仍是 dense 回退，不能算 gather 验证。独立复核另用压紧后 >16K 的 KV 和每 KV 头三个候选确认实际 torch gather，详见 results 下独立报告/可复跑脚本。
- 相邻 StaticCache 12 项、eager consumer 15 项，普通及 `-O` 均通过；合计 33 项测试方法（不把子场景数当方法数）。
- 修复只涉及两个 wrapper，selector、kernel、chunk2048、预算不变。显式 2D CUDA mask 新增标量读取同步，未实测成本，不宣称提速。真实 selector 增量缓存、GPU、实模质量/TTFT/显存仍待验证，异常后 cache 不保证回滚。
- 发布前再次 fetch，远端仍为 bc968288；主控复跑独立便携脚本 342 配置/1026 步通过，JSON 哈希与独立执行相同。人类 guide SHA256 不变；只提交本问题实现、测试、结果摘要和必要任务/advice，不包含旧工作树或其他仓库修改。
