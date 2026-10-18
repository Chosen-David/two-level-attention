# E147d β=.03125 全量 verdict：基线归属纠错

稳定ID：`TL-E147D-VERDICT-BASELINES-20261018`。仅建议修正汇总解释，不改原始分数，不新增实验。已查 fresh main `e51ab224b24824a01cdb15c53ec85ad4382cb5db` 的 advice/results 与既有 E147 回复；旧建议处理名义配额配对及 quick 标签，未覆盖本条新增 full verdict 转录错误。

`exp/trace/results/e147d_beta03125_verdict.json` 的 `judgment[1]` 与 `caveats[1]` 应按三份 formal 的 `scores` 更正（差值单位为任务分数点）：

| 项目 | β=.03125（E147d） | β=.0625（E147c） | β=.125（E146c） | E147d−E147c / E147d−E146c |
|---|---:|---:|---:|---:|
| CWE | 76.1 | 75.9 | 68.8 | +0.2 / +7.3 |
| MK3 (`niah_multikey_3`) | 7 | 9 | 7 | −2 / 0 |
| `niah_single_3` | 90 | 95 | 97 | −5 / −7 |
| MK2 (`niah_multikey_2`) | 76 | 73 | 73 | +3 / +3（原值正确） |

CWE 的68.8属于β=.125，不能标成β=.0625；+7.3也只能归于前者。MK3 caveat的“.0625=7/.125=0”与formal不符。single_3的“−10”不对应这两项基线，结构段自身列出的95/97则正确。

三份formal各含11任务、每任务100样本；算术AVG分别58.074545、59.290909、58.300000。因此E147d相对两基线−1.216364/−0.225455，以及原四舍五入的−1.22/−0.23仍成立。保留该描述性排序；这些汇总没有提供统计“甜点/峰值确认”或“噪声带内/远超噪声”的充分证据，任务分数同分也不证明逐预测一致或排除机器效应。

三条formal配置键均为**名义K=1024、K1=128**，不是K512；本次不验证实际执行预算，不据此推出K512或E2E结论。论文不新增该结果，现有public160清单不纳入本建议。

来源固定于上述source commit；以下路径均在 `exp/trace/results/`，Git blob为SHA-1，内容另列SHA-256：

| JSON文件 | Git blob | SHA-256 |
|---|---|---|
| e147d_beta03125_verdict.json | `6d035fda8928312a516c56dfbe562fc6be488bf8` | `52b5a431b96a8edbc9b1fdf3ca663ca3abfb8c3cc09d12f77939a0ca92b21114` |
| e147d_r32k_formal.json | `58d057b11a22990efe756258d3e76909611d8225` | `713154bd5f40172a5363b9a1e50df9076c689ebc11d27532d2af710d16b15153` |
| e147c_r32k_formal.json | `ee4cbe16dc7fb92713f7c28eb1fcb01c9ff4622b` | `3eb3780fc50d540764f976a851969be752d92378dc3b158715751e3156063754` |
| e146c_r32k_formal.json | `d8ec2cddd361416d3f6fb1750d6c45a3aaa91f20` | `fef62458e98ce567cbb2ac8fb4063f4a0eca4e876319655626212384c4842c59` |

来源最后变更提交：verdict及E147d formal为`4959fb1a6f078915f58bc19dfd67399c7c6e058a`；E147c formal为`12c17c7f554701c295a1db908940a02ef1d4dd42`；E146c formal为`d9181d76dcceaf3750ab4d69f0a439a2f73e004a`。本条仅纠错建议，不声称原verdict已经修复。
