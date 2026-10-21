# TL-LB-DATASET-BINDING-20261021

## Plan

来源：用户要求真实入口至汇总的公平输入身份闭包。基线
`1342fe330d4d01b6c109a7053837cbf1040f6d67`：检查文件名选择的评分数据集
与sidecar输入身份分组是否一致；保留producer的repobench-p命名兼容。

## Progress

实际consumer函数与真实sidecar/字节fixture修前3失败，修后专项及相邻回归
普通/-O各26通过。已补消费端前缀绑定；scope限于CPU身份正确性。
证据、环境、输入hash及历史重测范围见
[简报](../../advice/2026-10-21_longbench_dataset_binding_by_gpt.md)。
真实模型/GPU、完整评分CLI与历史错配影响仍未验证；不代表整体优化完成。
