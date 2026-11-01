# TL-SCHEDULER-D2H-STREAM-20261102

基线 main f5ed4cc3193765bcdc53a9b3046bce721a0b0f97，scheduler blob
439ed30736223a79cff939ed57c502ae56902abe。单Agent串行，无未合并依赖。

## 问题和修复

GraphRunner.step 在 replay/eager model 后调用 record_next_tokens。next_tokens
是 [batch] int64；先在生产流复制到持久双缓冲 device_buf，再将该缓冲复制到
pinned CPU。原实现只让 copy stream 等待生产流此前工作，没有切换当前流，
因此D2H仍在生产流上，完成事件却记录在copy stream。两流共同等待的前缀不
包含D2H；下轮 drain 仅同步该事件就读CPU token/EOS，缺少必要完成依赖。
源码可证明依赖缺口，但本环境没有GPU，未声称观测到真实设备竞争。

在 torch.cuda.stream(self._copy_stream) 内提交D2H并记录原事件。保留生产流
等待旧缓冲事件的复用保护、copy stream等待新D2D的依赖及CPU drain。复用持久
device_buf/CPU缓冲，无新增tensor/kernel/全设备同步，也不移除必要CPU等待。
与在生产流记录事件相比，本方案落实已有专用copy stream的意图；不是新的
overlap性能结论。scheduler每轮仍有argmax、boolean gather/scatter和EOS操作，
动态形状索引可能同步；本修复不声称解决这些开销或完整Graph模型正确性。

一手语义依据：PyTorch Stream wait_stream只建立依赖，stream context选择
提交流；event覆盖其记录流的此前工作。
- https://docs.pytorch.org/docs/stable/generated/torch.cuda.Stream_class.html
- https://docs.pytorch.org/docs/stable/generated/torch.cuda.stream_function.html
- https://docs.pytorch.org/docs/stable/notes/cuda

## 验证与边界

tests/test_scheduler_copy_stream.py 从实际record_next_tokens提取idx选择至
pending状态发布的完整传输段，使用独立FIFO流/事件快照执行器。三个host测试
覆盖事件后CPU数据、双缓冲与batch变化、不同生产流，共65次传输；同时检查
事件不要求后续无关生产流工作完成、退出context恢复原流。基线3失败，修后
普通及-O各3通过，py_compile通过。此测试不执行argmax/EOS前段，也不模拟
GPU吞吐，不能替代真实CUDA。原生CUDA用例加载完整scheduler，覆盖64轮token
回传、EOS和ID再用；因无Torch/CUDA跳过，待远端运行。

```bash
python tests/test_scheduler_copy_stream.py -v
python -O tests/test_scheduler_copy_stream.py -v
compute-sanitizer --tool memcheck python tests/test_scheduler_copy_stream.py CopyStreamCudaTest -v
```

修前：保存main scheduler.py到/tmp/baseline.py，设置
BASELINE_SOURCE=/tmp/baseline.py执行同一测试。GPU验收需补默认/非默认生产流、
真实GraphRunner replay与eager、不同batch/请求长度及EOS后新请求capture；
用profiler确认D2H及event在copy stream。性能需配对计时整个step，不能以CPU
模型或少改一行推断加速。PR10没有该scheduler；PR26/27保留同一未修blob，
分别处理请求行和驻留，未覆盖本缺口。其他未变分支证据复用，不覆盖他人分支。
