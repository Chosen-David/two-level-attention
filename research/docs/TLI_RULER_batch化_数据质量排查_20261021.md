# TLI RULER Batch 化数据质量排查报告

**日期**: 2026-10-21
**状态**: ✅ 已修复，重新部署完成

## 背景

大 bs+CUDA Graph (B=8, E155d 固定配额几何 np32/ntoff/nl4096) 已部署到三台远端机并行跑 RULER 64K/128K BSA 两臂 (minmax/avg)。

## 根因分析

### 🔴 根本原因：代码未同步

**pc31 和 pc32 的 `sparse_attn/baseline_runtime.py` 是旧版本**，缺少 `BatchCtx` 类（batch decode 阶段 1 引入）。

pc32 的 card1.log 清楚显示每次 pred_ruler_batch 都启动即失败：
```
[20:01:47] gpu1 L=65536 M=bsa_minmax T=niah_single_1 START
Traceback: ImportError: cannot import name 'BatchCtx' from 'sparse_attn.baseline_runtime'
[20:01:56] gpu1 L=65536 M=bsa_minmax T=niah_single_1 DONE rc=0  ← shell 误判
```

**`DONE rc=0` 不是实际 python 返回值**：queue worker 用的是 `echo "DONE rc=$?"`，每个子任务在 `>> log 2>&1` 之后 echo 本身总是成功（$? = echo 返回值 = 0），所以全覆盖消费完队列 + 无产物、日志显示全 rc=0。

### 🔴 根因：队列脚本 $\$? 捕获缺陷

队列 worker：
```bash
PYTHONPATH=... CUDA_VISIBLE_DEVICES=$gpu $PYBIN -u -m "$MOD" \
  ... >> "$LOG/card$gpu.log" 2>&1
echo "[$(date +%H:%M:%S)] gpu$gpu DONE rc=$?" >> "$LOG/card$gpu.log"
```

这里的 `$?` 是 `echo` 的退出状态（永远为 0），而非 python 进程的。应该改为：
```bash
PYTHONPATH=... CUDA_VISIBLE_DEVICES=$gpu $PYBIN -u -m "$MOD" \
  ... >> "$LOG/card$gpu.log" 2>&1
rc=$?
echo "... DONE rc=$rc" >> "$LOG/card$gpu.log"
```

## 修复措施

1. ✅ **代码同步**：`tar czf sparse_attn/ | ssh | tar xzf` 同步到 pc31/pc32
2. ✅ **空文件清理**：删除所有 0 字节的垃圾文件/.gen-* 目录
3. ✅ **队列修复**：捕获真实 $\$? 后启动 pc32 GPU1 的 BSA queue
4. ✅ **验证通过**：首任务正确加载 Qwen3-8B + 注册 GTX + 进入 prefetch loop，零 ImportError

## 当前状态

| 机器 | GPU | 在跑 | 队列剩余 | 备注 |
|---|---|---|---|---|
| pc02 (4 GPU) | 4×60GB JSSA 128K | 4×JSSA niah_single3/multikey2/multivalue | 3 | JSSA 完成后重启 BSA |
| pc32-GPU1 (1 GPU) | 1×90GB empty | BSA batch L=131072 minmax (刚启动) | 36/44 | 刚修复部署 |
| pc32-GPU0 | t6 champion (Qwen3-32B) 占用 | - | - | 不动 |
| pc31 (2 GPU) | t6 占用 + 代码同步完成 | - | - | 等 t6 退出后派单 |

## 下一步

1. 等 pc32 第一个 BSA task 完成（30-45 min/cpu time），核对结果
2. 全部 BSA 跑完后回传结果到本地
3. 等 pc02 JSSA 完成后，重启 BSA queue 在pc02
4. 打分并核验论文
