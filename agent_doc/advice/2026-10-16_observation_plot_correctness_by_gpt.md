# 整箱诊断与E6证据边界

用户授权的有界任务：[TL-OBS-PLOT-CORRECTNESS-20261016](../task/task_details/TL-OBS-PLOT-CORRECTNESS-20261016.md)。
47e4ab2的whole-bin排序不是token TopK上界；箱末可超预算，near分界遗漏SWA偏移，NaN配对分母也不一致。
本次修复采用不超2048的整箱mass-order skip-nonfit诊断，并输出实际预算与有限分母，不能推出真实单池token scorer表现。
128个query/layer观察来自4条来源序列，不是128个独立实验；文档重叠未核实。
当前图E6面板blocked，不从旧概率/区域/末query复用gate安全性或skip阈值。
O3仅保留原reference-mid-TopK比例与unpaired random标签。旧图/脚本/出处可从47e4ab2恢复。

CPU合成验证不等于科学采信；本轮没有GPU、重采、动态算法或稿件目录修改。仅推原dev，由用户审核合main。
