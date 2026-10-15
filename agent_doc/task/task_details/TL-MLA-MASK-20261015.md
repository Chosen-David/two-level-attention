# TL-MLA-MASK-20261015

- 来源：用户持续修复授权；TL-E142-DECODE-MASK 未完成项及实验分支8daa9cd。
- 负责人：本次 GPT；基线 main e2fc9ed5ec7c767ca35323e6535a1ae2a1ed2cf2。
- 已完成：最小精确零守卫移植到 main 候选，decode 非恒等 mask 在 cache.update
  前拒绝，保留 None/fresh dense prefill 路由及既有 continuation 门。
- 验证：修前26 failures/1 error，修后50项普通/-O CPU测试通过，另24合法路径
  输出和cache K/V逐位对照。不是GPU数值、质量或性能验收。
- 报告：[证据与开销](../../advice/2026-10-15_mla_decode_mask_main_fix_by_gpt.md)。
- 状态：CPU host契约修复完成；CUDA显式mask归约/同步新增成本未测。
- 后续：真实MLA路由与缓存、长序列wall/device、capture限制；请求级合同或异步
  方案必须另行验证，不缓存可变mask指针冒充已验证，也不改selector预算。
- 失败处理：任何合法路径回归回到有证据的基线；不能静默恢复忽略非零mask。
- 资源：CPU有界；无GPU、无模型/私密数据下载、无付费/未知训练作业占用。
