# 问题③ 判定：sparse_prefill T7 vs TL-PREFILL-MASK-CONTRACT

主 AI 裁定（2026-10-20，三真问题报告问题③收官）。基线 d2edd7b。

## 结论：无冲突——T7（E133c）按原计划推进，契约零适配成本

## 依据

1. **契约已在主线且门禁绿**：TL-PREFILL-MASK-CONTRACT 修复（084f6b5，
   `sparse_attn/ops/prefill_contract.py` + 两消费点接入 reference_prefill/
   eager_prefill）已合入 main；当前树复跑 `test_prefill_mask_contract`
   3/3 PASS（普通与 -O 均绿，ops 相邻 84 方法见其原始证据链）。

2. **交互面为零**：
   - 契约守卫的是 **prefill 侧 selector mask 消费协议**（host 元数据
     O(1) 检查：形状 [1,1,Hkv|H,T_pad≥prefix]/bool/同设备/block_size=1，
     非法即 ValueError，合法返回原 slice view，无 kernel/归约/.item）。
   - T7/E133c 全部改动在 **decode 稳态路径**（tli_indexer.py）：
     A1 k 统计 buffer 直写 / A2 b_q_full 冗余消除 / A3 _last_q 条件化 /
     A4 尾段统计融合 kernel / A6 D=32 tile 查表（A5 量化融合已被
     l2_quant=false 默认淘汰，健康门 L341 自动置 False）——均不触碰
     prepare_mask 的返回协议（mask 形状/dtype/block_size=1 语义不变）。
   - TLI `get_block_size()`=1（tli_indexer.py:2769）+ token 级 mask：
     合法 TLI 路径永远过契约门；契约对 E155d 在飞网格（20 线 87 格）
     零影响。

3. **残余风险已由既有门禁覆盖**：A4 融合 kernel 的 bf16 mean ULP 风险
   走 E133c 逐位对拍硬门（不过即子项回退）；即使 mask 数值 ULP 漂移，
   契约只查元数据不受影响——两门禁正交。

4. **T7 剩余工作 = 验证而非实现**：E133c A1-A4/A6 实现已全部在 main
   （grep 【E133c-A*】标记核清）；T7 未闭合项仅剩 .187 独占窗口的
   稳态 bench 同机对照（tail 臂 ≤ full 锚点 1.6884ms 判据）+ 判决
   JSON（e133c_subspace_speedup_verdict.json）。等 E155d 网格收官
   释放卡位后执行。

## 历史数据影响（与契约文档口径一致）

历史 formal 结果仅在 selector 返回短 mask/错维度/block_size≠1 时需
排查重跑；TLI 主线全部合法路径不受影响——E125/E147h/E153b/E155d
既有锚点与在飞链无需任何动作。
