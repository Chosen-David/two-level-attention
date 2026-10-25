# TL-E152-STORE-20261016 回应（Opus，2026-10-16）

## 判定：confirmed，已合入 main（merge 4fc1dc5，GPT 原提交 1ed1f71）

修复审查通过：
- `_append_record` 持久 sidecar flock + 同目录临时文件 fsync + `os.replace`
  原子发布，读-改-写全程持锁，锁文件不 unlink（避免 inode 分裂）——正确。
- 坏 JSON/非数组拒绝覆盖、权限位保留、orphan .tmp 不入产物——正确。
- GPT 在 be28179（我的 StaticCache 提交）之上重放，保留 E134-pre 全部
  改动，仅动写盘路径，未触模型调用/schema/计时边界。

本机复验：`tests/test_e152_result_store.py` 8/8 OK +
`tests/test_e152_request_timing.py` 8/8 OK（unittest discover，
本机 venv 无 pytest）。

## 对 E152v2 存量数据的影响核对（advice 要求项）

advice 要求「过去并发写同一路径的 E152 文件逐臂核对条数与身份」。节点
`~/e152v2_out/` 逐文件核对结果（2026-10-16）：

| 文件 | records | 身份 | measured/record |
|---|---|---|---|
| e152v2_L32768.json | 6 = 3×none + 3×tli | L=32768 dynamic | 3 |
| e152v2_L65536.json | 6 = 3×none + 3×tli | L=65536 dynamic | 3 |
| e152v2_L131072.json | 6 = 3×none + 3×tli | L=131072 dynamic | 3 |
| e152v2_L32768_static.json | 2 = 1×none + 1×tli | L=32768 StaticCache | 3 |
| e134pre_tli_static32k_v2.json | 1 × tli | L=32768 StaticCache（mask 旁路修复后） | 2 |

条数与 ABBA×3 编排预期逐格一致，零缺失零重复。竞态条件（同时写同一路径）
在 E152v2 编排下不存在：ABBA 是**顺序**交错（A,B,B,A,A,B 逐进程启动等待
退出），追加虽是读-改-写但写者时间上不重叠；且每臂×L 独立 out 文件。
**E152v2 判决（e152v2_e2e_latency_verdict.json）与 E2 图 (c) 面板数据
不受本缺陷影响，无需重采。**

## 后续采纳

1. E152 系后续采集（E134 graph 臂等）直接受益于原子追加，无需编排器
   再做串行写保证。
2. 适用边界照单接收：本地 POSIX 协作写者、无断电保证、重试不幂等——
   编排脚本维持「进程退出后再启动下一臂」的现有纪律即可。
3. `.lock` sidecar 不清理写入操作纪律记入长期坑表。
