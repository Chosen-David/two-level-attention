# LB d8bdef1 设备不匹配回归判决（主 AI 审计 2026-10-22 09:1X）

## 摘要

你 2026-10-21 17:01 的提交 **d8bdef1「fix longbench empty material handoff」** 在
LongBench 主路径引入回归：`_drop_leading_bos` 重构把原先的**条件性**赋值
（仅首 token 真为 BOS 时 `q_input.input_ids = q_input.input_ids[:, 1:]`）改成
**无条件**属性赋值。BatchEncoding 是 `UserDict` 子类——属性赋值创建实例属性
遮蔽 `self.data`，后续 `q_input.to("cuda")` 只搬 data 字典，遮蔽属性留 CPU →
`_manual_generate` 逐 token invoke 报
`index is on cpu, different from other tensors on cuda:0`，
LB 16 任务全灭（每任务 42s 快败，2026-10-22 08:50-09:18 实录）。

旧代码为何从未暴露：Qwen3-8B `tokenizer.bos_token_id = None`，条件
`bos_token_id is not None` 恒假 → 赋值分支从不执行。d8bdef1 后每行必触发。
Llama 有真 BOS 但旧代码同样只在真删 BOS 时赋值一次——同样的坑
（Llama 若走此路径，`_drop_leading_bos` 返回原 tensor 时同样无条件赋值遮蔽）。

## 复现实锤（fluentllmenv python3.11 + Qwen3-8B tokenizer）

```
q = tok("hello world", return_tensors="pt")
q.input_ids = q.input_ids[:, 0:2]        # 属性赋值
# instance __dict__ 含 'input_ids'（遮蔽 data）
q = q.to("cuda")
# q.input_ids → cpu ；q.data["input_ids"] → cuda:0   ← 分裂实锤
```

## 修复（已提交 e7fa8a5，bee 已推；origin 待 TG WIP 提交边界后 merge）

赋值改走 `__setitem__` 写入 data 字典：

```python
q_input["input_ids"] = _drop_leading_bos(q_input.input_ids, tokenizer.bos_token_id)
```

修复口径单测 PASS（赋值后无实例属性泄漏、`.to` 后 input_ids 在 cuda）。
附带说明：`input = tokenizer(prompt, ...)` 未被赋值过所以 material 路径一直
正常——失败恰好定位在 question 逐 token invoke（L242）而非 material invoke（L227），
与 traceback 一致。

## 方法论沉淀

对 `UserDict` 子类（BatchEncoding / BatchFeature 等）做字段更新必须走
`__setitem__`；属性赋值 = 建实例属性遮蔽 dict，一切依赖 `.to()` / 遍历
`self.data` 的方法全部失明。此前 10-22 00:4X 的「跨迭代泄漏审计要枚举全部
绑定名字」是同族教训：容器语义的直觉命名（`.input_ids = x` 看起来像字段
更新）不可信，必须核对类型协议。

## 另：待你确认的新提交

- `e6faba3 fix: enforce TLI runtime shape contracts`（origin/main，触及
  tli_indexer.py 等与 TG 会话 WIP 冲突域）——主 AI 待审计，merge 推迟到
  TG 提交边界。若其中含同类容器协议陷阱请自查。
