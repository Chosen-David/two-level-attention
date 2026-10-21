# TL-BATCH-TREATMENT-SNAPSHOT-20261021

Task-ID: TL-BATCH-TREATMENT-SNAPSHOT-20261021
Date: 2026-10-21
Base: `0b44bfc942dd8730b76165a8bbbd50faf05dbf15`

## 问题与真实路径

`benchmark/RULER/pred_ruler_batch.py` 是 RULER batch>1 的真实生成入口。
TLI 的 D′ mask 与投影基属于会改变运行时行为的文件型 treatment 配置，
单样本入口已经要求在模型加载前解析一次 `TreatmentSnapshot`，并让 runtime
patch、输出名 `_h<hash10>` 与 v2 完成回执的 full manifest 共享该对象。

批量入口原实现破坏了这条闭包：模型加载完成后才为 `register_patch` 创建
快照，随后 `get_method_name_with_info(args, None)` 重新打开配置文件生成名字，
而 `build_yarn_receipt(..., treatment_manifest_json=None)` 又完全省略 manifest。
若文件在这些时点间被替换，可生成 runtime A、basename B、receipt 无身份的
混装；消费端为兼容 legacy 回执允许 manifest 缺失，不能据输出自身检出。
这不是 hash 碰撞，也不是数值容差问题。

修前入口级最小复现记录的事件顺序为 `tokenizer, model, snapshot`，并在
`events.index("snapshot") < events.index("model")` 处失败：

```text
python -m unittest -v tests.test_ruler_batch_treatment_snapshot
FAIL
AssertionError: 2 not less than 1
Ran 1 test in 0.113s -- FAILED (failures=1)
```

## 修复

- `benchmark/RULER/pred_ruler_batch.py:65-68` 引入与单样本入口同源的
  `get_treatment_manifest_json`。
- `benchmark/RULER/pred_ruler_batch.py:266-285` 在 tokenizer/model 加载前
  只为 TLI 冻结一次 snapshot。
- 同一 snapshot 传给 `register_patch`、`get_method_name_with_info` 和
  `get_treatment_manifest_json`；回执不再伪装成 legacy 无身份产物。
- none/FASA、selector、kernel、预算、chunk2048、sink/SWA、生成 token 与
  原子 generation 指针协议均未改。

`tests/test_ruler_batch_treatment_snapshot.py` 用标准库桩运行完整 `main()`，
验证快照先于模型加载，且 runtime、名字、回执接收同一对象。固定测试文件
SHA-256 为
`c09f39e559e6189d7ff3d774429b2eb4239edf635ed3c48760f48b10bb92b247`。

```text
python -m unittest -v tests.test_ruler_batch_treatment_snapshot
Ran 1 test in 0.007s -- OK

python -O -m unittest -v tests.test_ruler_batch_treatment_snapshot
Ran 1 test in 0.007s -- OK

python -m compileall -q benchmark/RULER/pred_ruler_batch.py \
  tests/test_ruler_batch_treatment_snapshot.py
git diff --check
# 均静默成功
```

环境为 Linux 6.18.55 x86_64、Python 3.12.14。当前环境未安装 torch、
transformers 或 tqdm，也没有可用 CUDA/GPU；GPU 架构、驱动、CUDA、Triton、
框架/kernel 配置无法登记。因此未执行真实 Qwen3、prefill→decode 交接、输出
质量对拍、速度、显存、kernel 数或 host 同步测量，不作性能/质量提升声明。

历史 batch TLI 产物若来自该旧入口，会缺少本应存在的 full treatment manifest；
在用于正式结论前应核对 producer SHA 与外部运行配置，必要时按修复后入口重跑。
当前没有证据证明配置在某次历史生成窗口内实际变化，因此不自动判定历史分数
失效。持续优化任务保持开放。
