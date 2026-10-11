# ERRATA（收口产物身份字段勘误登记）

## E-20261011-01：E119 128K 正式收口 yarn_factor 错标 2.0（实际 4.0）

- 影响文件：e119_ruler128k_formal_{mavg,aavg,fullkv}.json(.receipt/.manifest) 与
  e119_ruler128k_formal_summary.json（2026-10-10 00:1X 生成）。
- 根因：收口操作者声明 yarn_factor=2.0，而 E109/E116b 全线 128K 生产者实际
  走 pred_ruler.py 自动档 YARN_FACTOR_AUTO 表 131072→4.0；identity_mode=
  legacy-partial 时代打分端无生产者回执可核，声明值未经证实即入 receipt。
  E116b native 收口（e116b_ruler128k_formal_*，2026-10-11）producer receipt
  v2 同代绑定 11/11 证实 effective_factor=4.0，同时 CLI 2.0 声明被 scorer
  GATE-FAIL fail-closed 拦截（057 生产者证据优先）——门禁按设计工作。
- 影响面：**分数无污染**（打分不加载模型、不消费 factor；factor 只影响
  生成期行为，已固化在 pred 文本内且三臂同批同档）。受污染的只是 E119
  128K 四件套的身份字段 yarn_factor（2.0 应为 4.0）。
- 处置：E119 128K 判决数值继续有效（排序与 AVG 引用不受影响）；身份引用
  E119 128K 收口时须注明 factor 勘误，或直接引用 E116b 新口径收口
  （e116b_ruler128k_formal_*，factor=4.0 生产者证据闭合）。不回改已发布
  产物（quarantine 惯例：只登记不改写）。
- 64K 不受影响（65536→2.0 声明与自动档一致）；32K 无 yarn。
