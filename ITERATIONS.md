# E127 迭代日志（ako4all 循环）

格式：每 iter = bench 运行 + 本日志追加 + git commit。基线 = 当前 main 的 TLI HF 路径（env 关）。

## Summary（随迭代更新）

| iter | 方向 | S=64K 全路径 GPU ms（median） | vs 基线 | 门禁 |
|------|------|------|------|------|
| baseline | 现状（无 env 开关） | 待测 | 1.00× | — |

## Iterations

### [baseline] 初始化 bench 与分阶段计时
- 结果：见 exp/bench/e127_stage_bench.py 输出
- 分析：待填
