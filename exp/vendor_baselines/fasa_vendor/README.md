# fasa_vendor —— FASA 官方源码的 Qwen3 适配层（B 轨 vendor）

TL-FASA-VENDOR-PUBLISH-141（GPT 审计 B 批）：本目录是 FASA
（ICLR 2026 frequency-aware sparse attention）vendor 复用的全套入库
源码——修复前 `exp/` 整体被 .gitignore 忽略，`benchmark/LongBench/pred.py`
与 `benchmark/RULER/pred_ruler.py` 的 `method=fasa_vendor` 分支
`from qwen3_fasa_patch import enable_fasa` 在干净检出上直接
ModuleNotFoundError。本目录随 141 修复 force-add 入库（`__pycache__`/
`*.pyc` 与校准产物 `dimension_rankings/*.pkl` 不入库，见下）。

## 上游来源与版本锁定（如实记录）

- 上游仓库：https://github.com/wangyifei0047/FASA-ICLR2026
  （ICLR 2026 论文 "FASA: Frequency-aware Sparse Attention"）。
- 本仓留档副本：`third_party/FASA/`（该目录同样被 .gitignore 忽略、
  未入库，本地树使用）。
- **上游 commit/SHA：未记录。** `third_party/FASA/` 无 `.git` 元数据
  （非独立 clone，是裸文件拷贝），vendored 时刻的上游 commit 无法
  追溯，如实标注。可复核的替代证据：
  `fasa_utils.py` 与 `third_party/FASA/monkey_patch/frequency/utils.py`
  **逐字节一致**（filecmp.cmp shallow=False 实证，2026-10-13 B 批
  修复时核验），即官方 `utils.py`（`constructe_adaptive_selection` /
  `core_module_with_padding_old`）零改动复用。
- `qwen3_identify_fre.py` 是官方 `identify_important_fre.py` 的
  Qwen3/transformers 4.56 移植（官方仅支持 Llama/Qwen2 flash 类，
  头注释有逐条差异说明，统计语义逐字节保持官方
  `partial_frequency_against_full_frequency`）。
- `convert_agreement.py` 为官方缺失的 res_dict → agreement.pkl
  聚合转换脚本（官方仓库未发布转换脚本，语义见文件头注释）。
- `qwen3_fasa_patch.py` 仅接口胶水（Qwen3 单类统一 Attention 的
  绑定 + decode 剪枝前手动 repeat_kv），不重写任何选择逻辑。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `fasa_utils.py` | 官方 `utils.py` 逐字节一致（核心选择/剪枝逻辑） |
| `qwen3_fasa_patch.py` | Qwen3Attention 绑定 FASA 剪枝 forward（胶水层） |
| `qwen3_identify_fre.py` | Phase 1 离线校准（Qwen3 移植版，产出 res_dict pkl） |
| `convert_agreement.py` | Phase 1.5 res_dict → `<dataset>_agreement.pkl` 聚合转换 |
| `smoke_qwen3_fasa.py` | 冒烟脚本（单层对拍） |

派单/评测 shell 脚本（`dispatch_lb_fasa_2card.sh` / `eval_lb_fasa.sh` /
`run_ruler_fasa.sh`）是本机操作脚本，不入库。

## 校准产物（dimension_rankings/*.pkl）不入库

`dimension_rankings/<model>/<dataset>_agreement.pkl` 是 Phase 1 校准
产物（`qasper_1/2/8.pkl` 中间产物共数百 MB），不入库。干净检出上跑
`method=fasa_vendor` 时入口会 fail-closed 提示「先跑
qwen3_identify_fre.py + convert_agreement.py」；treatment 名附加
agreement 内容 SHA256 前 8 位（TL-FASA-CALIBRATION-IDENTITY-142，
见 `sparse_attn/info.py`），与产物内容一一绑定。

## import 冒烟（141 验收口径）

```bash
# 干净检出（git archive HEAD | tar -x -C /tmp/clean_repo）后：
PYTHONPATH=/tmp/clean_repo/exp/vendor_baselines/fasa_vendor \
    python3 -c "import qwen3_fasa_patch; print('ok')"
```
