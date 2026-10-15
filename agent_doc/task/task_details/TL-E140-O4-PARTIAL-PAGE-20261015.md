# TL-E140-O4-PARTIAL-PAGE-20261015

用户授权：仅修 O4 有效末页与诊断身份；CPU 正反例、独审后推原 dev，由用户审核合 main。零 GPU、零重采、零模型下载，不扩大 O1/O3/MLA/SparQ 范围。

main 558fa558 的 O4 `n_mid//64` 丢弃部分末页。修为 ceil 页数、有效 token 均值/min/max、候选裁界；保留既有 near/far 配额且不 refill。空区/零预算返回空集；非法分区/负预算显式拒绝，普通与 -O 一致。默认 dry-run；未来输出 v3，标注 valid-partial-pages-v1，旧 v2 不覆写。

q512 mid=[128,385)，near=[320,385)：K1=64 257/组，K1=128 193/组（near65+far128）。新增 CPU 测试覆盖长度1/63/64/65/257、正负页统计、单页预算高分尾页、空near/far、边界和执行门禁。完整结果在最终提交验证记录。

O4 v2 共128条query全部部分末页；末页共30272 group-token，两TLI臂均选0，oracle选21785。raw字节保留。面板(d)标历史诊断/不可采信；仅在原图外加警示，不重算图数值，O1不因此失效。manifest保留此前资产身份并绑定main新底图、raw与TSV。默认renderer拒绝，显式诊断opt-in仍加警示。O3比例仍为相对reference-mid-TopK mass，不是全注意力质量。

未执行真实采集；修复代码不代表新实验证据或科学结论成立。

验收：独立审查无阻塞；新增7项normal/-O各通过。连同既有O1/observation，
`python [-O] -m unittest discover -s tests -p 'test_e140_o*.py' -v` 各15项通过；
系统python资产回归`test_e140_asset_provenance.py` normal/-O各6项通过。
CPU环境torch 2.14.1+cpu，OMP/OPENBLAS线程1，CUDA_VISIBLE_DEVICES空。
固定seed140、长度64/256/1024/4096 × K1=8/64/128，共12组整页选择与main旧实现逐位一致。
旧q512实测256/192，新257/193；旧末token全False。限1页预算高分部分页与整页竞争测试通过。
O4 v2 raw SHA256：`4a9970ecfd802e0b295f0627bac2ee306402b684a1abdec9c37678e1e1ab97e9`。
未发现`.github` CI配置；无托管CI通过声明。SVG保留原节点空白导致diff-check尾随空白提示；
其余源文件diff-check通过，SVG原图子树与PNG原图像素/PDF原文内容保留已由测试验证。
