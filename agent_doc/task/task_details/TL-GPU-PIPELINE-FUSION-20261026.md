# TL-GPU-PIPELINE-FUSION-20261026

用户要求：直接实现 GPU 融合算子，消除 copy/中间量，交付远端压测，继续更新 PR #7。
基线：2e21174；实现边界/开关/命令/证据见 `docs/tli_gpu_pipeline.md`。

已完成四类 Triton 融合与完整 batch 的 eager/Graph 公共路径接入。整行扫描
255 寄存器候选改为分块扫描，编译矩阵峰值 60；原候选源码/编译证据保留。
205 CPU tests + 205 subtests、95 interpreter checks、188 offline CUDA builds
通过；真实 CUDA、E127/batch gates、Nsight、模型 TPOT/吞吐和质量仍待远端。
本地无 GPU，未把 CPU/interpreter/ptxas 结果称为 GPU 提速，无独立 reviewer 验收。

任务保持未完成直至 GPU 与模型验收。`run_tli_gpu_pipeline` 提供失败即记录/返回
非零的远端入口，拒绝 CPU 和 interpreter。新路由默认关闭，QDQ 不支持并显式拒绝。
