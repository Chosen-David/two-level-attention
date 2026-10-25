# TL-E151C-LAYER-20261019：配额消费必须绑定有效层号

基线 `58358135a887031b294f55d67482d1aa43644226`。fresh fetch后相对上次
fa187e9新增19个文稿/实验文件，生产源码无变化；继续旧检查点明确未完的层号契约。
实验分支e138-ablation=`2ed6a1feb78a7b5bcf3cac922b4510c3968714a5`仅核对身份，不合并。

## 问题与改动

`sparse_attn/indexer/tli_indexer.py` 基线2329–2331、2360–2362使用
`quota[self.layer_idx or 0]`。prefill激活配额时，None/False被当作第0层，
负层号按Python负索引读表尾，True当第1层；未覆盖的正层号抛IndexError，
但此前已进入mask工作。表内容合法不代表当前层身份合法。

新1856–1867行在compute_mask读取/修改score前统一校验：层号必须为Python int，
且在[0,len(quota))内；错误统一ValueError并带层号/表长信息。
两处far预算分支复用已解析值，保留原min(quota,K2_mid)裁剪。零配额用
`is not None`判断，不能误当关闭。无配额或_e151c_on=False时不要求层号。
层号仍由register_patch后注入，所以不在构造期假设固定36层或要求表长等于模型层数。

真实链路：patch.py注入layer_idx→reference/eager prefill置旗标→prepare_mask
构建索引/分数→compute_mask守卫→原near/far选集→attention。
本修复发生在mask消费点，**没有承诺在模型patch或索引缓存构建前拒绝**；
空/dense短序列未消费配额仍沿原路径。非法请求触发后prefill的finally会清旗标。
完整模型表长预检、跨层配额文件快照一致性仍为后续事项。

## 验证与复现

按实现优化技能保留修前失败→小补丁→相邻回归，不更改算法或预算来制造提速。
结果目录 `agent_doc/results/e151c-layer-20261019/`：

- 修前新5测试：6个失败子例、6个错误子例；错误为旧IndexError/TypeError/KeyError，
  测试要求明确ValueError。不是12个独立bug。固定git基线重放得到相同结果。
- 修后E151c专项21测试，普通及-O通过：首/末有效层、零/不同配额、非法层、
  decode不依赖prefill层号、reference/candidate异常后旗标清理。
- 固定seed15120+S，S=512/577、layer0/1/2、quota开/关、D′开/关共24场景，
  新旧真实prepare_mask选集逐位一致；普通及-O各执行一次。GQA=2、FP32、CPU。
- 相邻E139 13项、paper runtime 5项普通及-O均通过，每种模式共39项测试。
  本轮人工源码复查及可执行对照，未使用独立Agent；本记录不代替GPU验收。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python agent_doc/results/e151c-layer-20261019/verify.py --baseline-fail
# 上条预期退出1；去掉--baseline-fail执行24场景差分
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_e151c*.py' -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_e139*.py' -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p test_paper_runtime.py -v
# 修后上述各项另以python -O运行，不能用assert作生产输入守卫。
```

环境：Python3.12.14、Torch2.8.0+cpu、Linux6.18.55 x86_64，Xeon Platinum8573C，
9个可见CPU；OMP/MKL各1线程，CUDA=False，无GPU驱动/架构/kernel编译数据。
parity.json保留输入张量hash及基线源码hash；SHA256SUMS绑定本次实现/测试/原始日志。
没有测吞吐、TTFT、显存、kernel数或真实模型质量，没有性能提升主张。
单次守卫O(1)，无新增张量归约或host/device同步；这属于代码性质，不是实测速度结论。

## 历史影响和后续

启用quota且layer_idx缺失/负值/bool的历史实验可能用错层预算，应核对实际层号与
原配额表后重测；正常已注入且表覆盖的Python int层号不因本补丁自动失效。
旧测量未改。完整GPU路由、真实模型结果、候选内L2、紧凑索引和fused attention/
overlap仍未验收。当前环境无GPU，保留这些门禁，不以CPU差分冒充GPU结果。

## 回应（wangyuanshuo02，2026-10-19 02:1X）

采纳。独立核验与在飞影响评估：

1. **代码核验**（dd696aa 直接 diff 复查）：守卫集中在 compute_mask 入口（L1857-1867），`type(layer) is not int` 拒绝 bool/None/负值，两处 far 预算分支（e64_partition 与 minmax/avg 臂）复用 `oracle_far_budget` 已解析值，`min(quota, K2_mid)` 裁剪语义保留。判真 bug 成立——`self.layer_idx or 0` 确会把 None/False 静默当第 0 层、负层号走 Python 负索引读表尾，无任何报错。
2. **在飞零污染**：E147h/E147j 全部在飞臂（128K/LB EN 等）不激活 `_e151c_quota`（标准 α/β/γ 配额路径，`_e151c_quota is None` 时守卫零介入）；.187 冻结树（tmp_e146b_tree_187）与 wys 在跑进程（模块已于启动时加载）均不受 main 树新提交影响。判无需停车。
3. **历史影响挂账**：E151c 判决（#275）本已有 routing-receipt 复查待办（4496349 轮披露 oracle 配额可能在 reference 路径静默丢失）。本守卫 adds 第二维度：若历史 oracle 臂 layer_idx 注入失败，实际用的是第 0 层配额表项。两项复查合并执行——核对 e151c 结果目录回执的实际派发路径 + 层号消费记录后决定判决是否降级披露。E134-pre 历史数字保留（GPT 亦未证明错）。
4. **边界声明认可**：CPU 差分不冒充 GPU 验收、完整模型表长预检为后续事项——如实入账。

