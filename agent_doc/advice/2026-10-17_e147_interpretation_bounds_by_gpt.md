# E147a/c：配额算术与解释边界

核对基线：`c99315c06bb358adde333801d2ac7b382a1fa830`，2026-10-17 12:11 UTC轮。仅核已入库汇总、静态公式和身份缺口，不改 raw、不请求新增实验，也不改当前默认配置。此前 E145d/E146ca 方向建议已有回复与修正，保持原件，不重复或删除。

1. **E147a 描述修正。** 两份 formal JSON 各11任务、每格100例，重算均值56.822727与58.3，差−1.477273分；2升7降2平。“唯fwe +12”遗漏 multivalue +2.25。far词元252→572为2.27倍。K1=128→64同时把近/远页16/112改为8/56、词元640/252改为320/572；这些汇总不能单独证明“页级多样性崩塌”或瓶颈已定位。
2. **区分两个 E147c 比较。** K128/β.0625的名义页8/120、词元320/572。对旧K128/β.125，页与词元分配都变；对K64/β.125，近页和两区名义词元配额相同，far候选页56→120。因此不能笼统说所有β相关比较都混杂；但候选页效应仍需实际执行/样本身份支持。现有quick两任务报告CWE75.9（对旧K128 +7.1、对K64 +17），MK2=73（分别0、+5），不证明全任务或因果收益，也不宜将+7.1称统计“显著”。
3. **候选文字算术。** quick的β=.03125、K1=128、γ=.625、B64、Kmid892，依已写公式得到Pn4/Pf124、kn160/kf732；不是256/636。256/636对应文中另一项β=.0625、γ=.5。这里只纠正已存在的假设配置计算，不推荐执行或外推恢复斜率。
4. **身份与噪声边界。** fresh树中E147c仍只有quick，无formal/predictions/manifest/receipt；不能把该快报当正式完整验收。E147a formal及sidecar已在，但原始预测、逐任务生产回执与解析后的treatment manifest未随当前树取得；`effective_config_sha256`明确排除per-arm treatment_manifest，同hash不证明处理臂相同。E147b两任务两次汇总分数相同，仅说明这些记录未观察到分数变动，不证明输出逐位相同或全局无噪声；跨机历史FullKV不是已校准的≤3分噪声带。

建议将上述 verdict 机制语句标为待验证解释，保留真实分数与配置事实；若身份材料已有宿主归档，只需补稳定路径/内容hash绑定，不必重采。

## 固定证据

- `exp/trace/results/e147a_r32k_formal.json`、`e146c_r32k_formal.json`：逐任务分数与n；E147a sidecar只按已取得范围解读。
- `exp/trace/results/e147_improvement_verdict.json`：conclusion及配置解释。
- `exp/trace/results/e147c_quick_verdict.json`，blob `aeadf99216d25925b6f2d435a27e8921b2c64f98`：results、verdict、next_candidates。
- `exp/trace/results/e147b_rep_verdict.json`：两次汇总与跨机限制。
- `sparse_attn/indexer/tli_indexer.py` 名义页/词元预算公式；`benchmark/RULER/yarn_receipt.py:986` 配置指纹收录边界。

本条按上述固定提交定位，实际作者复算与独立审查分开完成；不将角色一致意见作为实验机制证据。
