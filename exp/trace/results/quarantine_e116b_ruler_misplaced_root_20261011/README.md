# 隔离：E116b RULER 32K/64K 首轮打分错位产物（2026-10-11 20:0X）

原因：打分 root /tmp/e116b_ruler/{arm}/ 下同时挂 L32768+L65536 双档 symlink，
scorer 单 cell 门禁拒收双 cell 输入——mavg/aavg 四格拒收未产出；
fullkv 两格产出的 JSON 文件名与实际 cell 内容错位
（e116b_ruler32768k_formal_fullkv.json 实际 cell=L65536/none）。
分数本身无污染（每格内数据正确），但产物名↔内容绑定断裂，按隔离惯例
rename 入本目录不删除。重打产物以单档隔离 root 生成（E119 64K 教训）。
