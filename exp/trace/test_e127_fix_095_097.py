#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E127 GPT 审计 095/097 修复门禁套件（GPT 审计 2026-10-12_0527，主 AI
核验 confirmed 两项；修复 agent：Opus-fix）。

095（P1）deferred error flag 生产入口未消费：
  eager_decoding 把三类契约违规（空 mask 行=1 / 选中数超 TLI_E127_W_UPPER=2
  / cu_seqlens_k 契约破坏=3）延迟到 GPU 端 _ERR_FLAG，设计声明由
  eager_decoding_check_errors() 生成结束一次读回显式 raise、整次生成作废
  ——但修复前三条生产入口（pred.py / pred_ruler.py / llm_eval.py）均未
  调用，触发时零输出/截断注意力照常生成并写盘 JSONL。本套件：
    - guard 语义：进入清旧 flag（防跨请求继承）/ 退出校验 raise；
    - code 1/2/3 注入 → 对应文案 raise；raise 后 flag 已清零；
    - 合法请求不被旧 flag 误杀；guard 窗口内自身违规不被 enter-clear 吞；
    - 三入口级注入（真实 get_pred / pred_ruler.main / llm_eval.main 代码
      路径 + stub 模型在 attention 位注入异常 mask）：raise 必须先于任何
      完成外观产物（record / JSONL 行 / generation 指针 / 结果 txt）。

097（P2）PER_QH grid 冗余 program：
  修复前 _eager_attn_kernel 固定 grid=(R*G,)，per_qh 口径（m0 行数=H）
  j 维完全未用——Qwen3 H=32/G=4 时 128 program 对 32 行各写 4 次（冗余
  计算 + 同址非原子 store 竞争；同输入派生同值故 allclose 能过，数值错误
  inconclusive，E127 正式配置 per_q_head=False 不受影响）。修复后 per_qh
  grid=(R,) 一行一 program。本套件：
    - program 数断言（_attn_grid_size 纯函数 + 真实 launch grid 实测狙击：
      修复前 (R*G,) → 修复后 (R,)，共享口径前后一致）；
    - 修复前后逐位一致：ref-dump 模式在【修复前】代码上生成输出锚点
      （/tmp/e127_097_bitref.pt，可用 E127_097_REF 覆盖），默认模式逐位
      对拍——per_qh=False 正式路径与 per_qh=True 修复分支均须
      torch.equal（修复只消除冗余 writer，不改数值路径）；
    - full/empty mask 调用后 check_errors 均 raise（flag 语义不变）。
  H%Hkv≠0 非整组：GQA 契约外——dense einops '(h g)' 直接 raise，Triton
  路径 kvh=r//G 越界读（修复前后同行为，out-of-contract 不入逐位矩阵，
  如实登记为边界；生产 GQA 恒 H%Hkv==0）。

用法（仓根）：
  # 修复前（红证 + 逐位锚点生成）：
  CUDA_VISIBLE_DEVICES=1 python3 exp/trace/test_e127_fix_095_097.py ref-dump
  CUDA_VISIBLE_DEVICES=1 python3 exp/trace/test_e127_fix_095_097.py   # 期望 095/097 红
  # 修复后（绿证）：
  CUDA_VISIBLE_DEVICES=1 python3 exp/trace/test_e127_fix_095_097.py   # 期望全绿
  python3 -O 同命令双跑（-O 不影响 check() 模式断言）。
"""
import glob
import json
import os
import shutil
import sys
import tempfile
import types
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch  # noqa: E402

PASS, FAIL = [], []
REF_PATH = os.environ.get("E127_097_REF", "/tmp/e127_097_bitref.pt")


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"[PASS] {name} {detail}")
    else:
        FAIL.append(name)
        print(f"[FAIL] {name} {detail}")


# ---------------------------------------------------------------------------
# 095 注入机：通过真实 eager_decoding_attn（TLI_E127_FAST=1）置 flag，
# 覆盖三类契约违规。张量全局缓存（64K bf16 ×2 ≈ 256MB，避免逐调用重分配）。
# ---------------------------------------------------------------------------
_INJ = {}


def _inject(dev, code):
    """code=1 空 mask 行 / 2 全 True（counts 超 W_upper）/ 3 cu 契约破坏。"""
    from sparse_attn.ops.eager_decoding import eager_decoding_attn
    if "t" not in _INJ:
        S = 65536 - 13
        _INJ["t"] = dict(
            S=S,
            k=torch.randn(1, S, 8, 128, device=dev, dtype=torch.bfloat16),
            v=torch.randn(1, S, 8, 128, device=dev, dtype=torch.bfloat16),
            q=torch.randn(1, 1, 32, 128, device=dev, dtype=torch.bfloat16),
        )
    t = _INJ["t"]
    S = t["S"]
    if code == 1:
        m = torch.zeros(1, 1, 8, S, dtype=torch.bool, device=dev)
    elif code == 2:
        m = torch.ones(1, 1, 8, S, dtype=torch.bool, device=dev)
    else:
        m = torch.rand(1, 1, 8, S, device=dev) > 0.98
        m[..., -128:] = True
    cu0 = 5 if code == 3 else 0
    cu = torch.tensor([cu0, S], dtype=torch.int32, device=dev)
    eager_decoding_attn(t["q"], t["k"], t["v"], m, 1, cu)


# ---------------------------------------------------------------------------
# 095 guard 语义
# ---------------------------------------------------------------------------
def test_guard_semantics(dev):
    try:
        from sparse_attn.ops.eager_decoding import (
            eager_decoding_request_guard, eager_decoding_check_errors)
    except ImportError:
        check("095 guard: 模块提供 eager_decoding_request_guard（修复锚）",
              False, "修复前版本无该 API（红证：deferred flag 无请求级消费）")
        return
    os.environ["TLI_E127_FAST"] = "1"
    try:
        msgs = {}
        for code in (1, 2, 3):
            raised = None
            try:
                with eager_decoding_request_guard():
                    _inject(dev, code)
            except RuntimeError as e:
                raised = str(e)
            msgs[code] = raised
        check("095 guard: code1 注入 → 退出 raise「全 False 行」",
              msgs.get(1) is not None and "全 False 行" in msgs[1],
              f"raised={msgs.get(1)!r}")
        check("095 guard: code2 注入 → 退出 raise「TLI_E127_W_UPPER」",
              msgs.get(2) is not None and "TLI_E127_W_UPPER" in msgs[2],
              f"raised={msgs.get(2)!r}")
        check("095 guard: code3 注入 → 退出 raise「cu_seqlens_k 契约破坏」",
              msgs.get(3) is not None and "cu_seqlens_k 契约破坏" in msgs[3],
              f"raised={msgs.get(3)!r}")
        # raise（clear=True）后 flag 已清零：再次 check 不 raise
        cleared = True
        try:
            eager_decoding_check_errors(dev)
        except RuntimeError:
            cleared = False
        check("095 guard: raise 后 flag 已清零（再次 check 不 raise）", cleared)
        # 合法请求不继承旧 flag：先注入置位 → 新 guard 进入即清 → 空段退出不误杀
        _inject(dev, 1)
        stale_ok = True
        try:
            with eager_decoding_request_guard():
                pass
        except RuntimeError:
            stale_ok = False
        check("095 guard: 进入清旧 flag——合法请求不被旧 flag 误杀", stale_ok)
        # guard 窗口内自身违规不被 enter-clear 吞掉（fail-closed 不失效）
        cur_ok = False
        try:
            with eager_decoding_request_guard():
                _inject(dev, 2)
        except RuntimeError as e:
            cur_ok = "TLI_E127_W_UPPER" in str(e)
        check("095 guard: 窗口内自身违规仍 fail-closed raise", cur_ok)
    finally:
        os.environ.pop("TLI_E127_FAST", None)


# ---------------------------------------------------------------------------
# 095 静态接线锚
# ---------------------------------------------------------------------------
def test_entry_wiring():
    ents = ["benchmark/LongBench/pred.py",
            "benchmark/RULER/pred_ruler.py",
            "benchmark/RULER/llm_eval.py"]
    missing = []
    for rel in ents:
        src = open(os.path.join(REPO, rel), encoding="utf-8").read()
        if "eager_decoding_request_guard" not in src:
            missing.append(rel)
    check("095: 三生产入口均接线 eager_decoding_request_guard（静态源锚）",
          not missing, f"missing={missing}")


# ---------------------------------------------------------------------------
# 095 入口级注入：真实 get_pred 代码路径 + stub 模型
# ---------------------------------------------------------------------------
def test_entry_pred_longbench(dev):
    import benchmark.LongBench.pred as pred_mod

    class _Enc:
        def __init__(self, ids):
            self.input_ids = ids

        def __len__(self):
            return 1

        def to(self, *a, **k):
            return self

    class _Tok:
        bos_token_id = None
        eos_token_id = None

        def __call__(self, text, truncation=False, return_tensors="pt"):
            return _Enc(torch.tensor([[101, 102, 103]], device=dev))

        def decode(self, ids, skip_special_tokens=True):
            return "pred"

    class _Model:
        """stub 模型：每次 forward 在 attention 位注入 code=1（真实
        eager_decoding_attn 机置 flag，模拟异常 mask 的稀疏层）。"""

        def __call__(self, input_ids=None, past_key_values=None,
                     use_cache=True, **kw):
            _inject(dev, 1)
            return SimpleNamespace(
                logits=torch.zeros(1, 1, 8, device=dev),
                past_key_values=object())

    row = {"context": "hello world", "answers": ["a"],
           "all_classes": None, "length": 2}
    os.environ["TLI_E127_FAST"] = "1"
    raised = None
    try:
        pred_mod.get_pred(
            model=_Model(), tokenizer=_Tok(), data=[row],
            max_length=128000, max_gen=2, prompt_format="{context}",
            dataset="trec", device=dev, model_name="Qwen3-8B", data_fp="fp")
    except RuntimeError as e:
        raised = str(e)
    finally:
        os.environ.pop("TLI_E127_FAST", None)
    check("095 入口 pred.py：注入 → 生成段 raise 先于 record 构造/落盘",
          raised is not None and "全 False 行" in raised,
          f"raised={raised!r}（红口径：修复前 record 照常 append、无 raise）")


def test_entry_pred_ruler(dev):
    import benchmark.RULER.pred_ruler as pr
    from benchmark.RULER.yarn_receipt import generation_pointer_path

    tmp = tempfile.mkdtemp(prefix="e095_ruler_")
    data_root = os.path.join(tmp, "data")
    os.makedirs(os.path.join(data_root, "16384"))
    with open(os.path.join(data_root, "16384", "niah_single_1.jsonl"),
              "w", encoding="utf-8") as f:
        f.write(json.dumps({"input": "x", "outputs": ["y"],
                            "length": 1, "index": 0}) + "\n")
    out_root = os.path.join(tmp, "out")

    class _Tok:
        eos_token_id = None

        def encode(self, *a, **k):
            return []

        def __call__(self, text, truncation=False, return_tensors="pt"):
            return SimpleNamespace(
                input_ids=torch.tensor([[101, 102]], device=dev))

        def decode(self, ids, skip_special_tokens=True):
            return "pred"

    class _Model:
        def eval(self):
            return self

        def __call__(self, input_ids=None, past_key_values=None,
                     use_cache=True, **kw):
            _inject(dev, 1)
            return SimpleNamespace(
                logits=torch.zeros(1, 1, 4, device=dev),
                past_key_values=object())

    saved = (pr.AutoConfig, pr.AutoModelForCausalLM, pr.AutoTokenizer,
             pr.register_patch)
    argv = sys.argv
    os.environ["TLI_E127_FAST"] = "1"
    try:
        pr.AutoConfig = SimpleNamespace(
            from_pretrained=lambda *a, **k: SimpleNamespace())
        pr.AutoModelForCausalLM = SimpleNamespace(
            from_pretrained=lambda *a, **k: _Model())
        pr.AutoTokenizer = SimpleNamespace(
            from_pretrained=lambda *a, **k: _Tok())
        pr.register_patch = lambda model, args, snapshot: 0
        sys.argv = ["pred_ruler", "--model_path", tmp, "--task",
                   "niah_single_1", "--context_length", "16384",
                   "--data-root", data_root, "--output-dir", out_root,
                   "--t", "t095", "--max-num", "1", "--method", "none"]
        raised = None
        try:
            pr.main()
        except RuntimeError as e:
            raised = str(e)
    finally:
        (pr.AutoConfig, pr.AutoModelForCausalLM, pr.AutoTokenizer,
         pr.register_patch) = saved
        sys.argv = argv
        os.environ.pop("TLI_E127_FAST", None)

    out_dir = os.path.join(out_root, "L16384", "pred")
    out_path = os.path.join(out_dir, "niah_single_1-none-t095.jsonl")
    jsonl_top = glob.glob(os.path.join(out_dir, "*.jsonl"))
    ptr_exists = os.path.exists(generation_pointer_path(out_path))
    check("095 入口 pred_ruler：注入 → raise 且无完成外观产物（best-file+指针）",
          raised is not None and "全 False 行" in raised
          and not jsonl_top and not ptr_exists,
          f"raised={raised!r} jsonl={jsonl_top} ptr={ptr_exists}")
    shutil.rmtree(tmp, ignore_errors=True)


def test_entry_llm_eval(dev):
    import benchmark.RULER.llm_eval as le

    tmp = tempfile.mkdtemp(prefix="e095_lleval_")

    class _Batch(dict):
        def to(self, *a, **k):
            return self

    class _Tok:
        eos_token = "<eos>"

        def apply_chat_template(self, msgs, tokenize=False):
            return "x"

        def __call__(self, texts, return_tensors="pt", padding=True):
            return _Batch(input_ids=torch.tensor([[1]]))

        def batch_decode(self, outs):
            return ["ok"]

    class _Model:
        def eval(self):
            return self

        def generate(self, **kw):
            return None

    mod_names = ("lm_eval", "lm_eval.utils", "lm_eval.models",
                 "lm_eval.models.huggingface", "lm_eval.tasks")
    saved_mods = {k: sys.modules.get(k) for k in mod_names}
    saved_attrs = (le.AutoTokenizer, le.AutoModelForCausalLM, le.register_patch)
    argv = sys.argv

    def fake_simple_evaluate(*a, **k):
        # 模拟 lm_eval 生成段：attention 位注入 code=1（真实 flag 机）
        _inject(dev, 1)
        return {"results": {}}

    pkg = types.ModuleType("lm_eval")
    pkg.simple_evaluate = fake_simple_evaluate
    utils_mod = types.ModuleType("lm_eval.utils")
    utils_mod.make_table = lambda d: "table"
    utils_mod.pattern_match = lambda pats, all_tasks: []
    models_mod = types.ModuleType("lm_eval.models")
    hf_mod = types.ModuleType("lm_eval.models.huggingface")
    hf_mod.HFLM = lambda **kw: object()
    models_mod.huggingface = hf_mod
    tasks_mod = types.ModuleType("lm_eval.tasks")
    tasks_mod.TaskManager = lambda **kw: SimpleNamespace(all_tasks=[])
    pkg.utils = utils_mod
    pkg.models = models_mod
    pkg.tasks = tasks_mod

    os.environ["TLI_E127_FAST"] = "1"
    try:
        sys.modules.update({n: m for n, m in zip(
            mod_names, (pkg, utils_mod, models_mod, hf_mod, tasks_mod))})
        le.AutoTokenizer = SimpleNamespace(
            from_pretrained=lambda *a, **k: _Tok())
        le.AutoModelForCausalLM = SimpleNamespace(
            from_pretrained=lambda *a, **k: _Model())
        le.register_patch = lambda model, args, snapshot: 0
        sys.argv = ["llm_eval", "--model", tmp, "--model_name", "Qwen3-8B",
                    "--tasks", "niah_single_1", "--output_dir", tmp,
                    "--method", "none", "--max_length", "16384"]
        raised = None
        try:
            le.main()
        except RuntimeError as e:
            raised = str(e)
    finally:
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        (le.AutoTokenizer, le.AutoModelForCausalLM,
         le.register_patch) = saved_attrs
        sys.argv = argv
        os.environ.pop("TLI_E127_FAST", None)

    produced = glob.glob(os.path.join(tmp, "**", "*.txt"), recursive=True)
    check("095 入口 llm_eval：注入 → raise 且 output_dir 无任何结果文件",
          raised is not None and "全 False 行" in raised and not produced,
          f"raised={raised!r} produced={produced}")
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 097 逐位矩阵（ref-dump 修复前锚点 / 默认模式逐位对拍 + grid 实测）
# ---------------------------------------------------------------------------
def _run_matrix(dev, grids=None):
    """跑 097 配置矩阵，返回 (outputs, flag_raised_list)。

    grids 传入 dict 时记录每次 kernel B 的真实 launch grid（triton
    JITFunction.run kwarg 狙击）——修复前 per_qh 应为 (R*G,)、修复后
    (R,)；共享口径恒 (R*G,)。
    """
    from sparse_attn.ops.eager_decoding import (
        eager_decoding_attn, eager_decoding_check_errors)
    outs, flags = {}, []
    ed = None
    orig_run = None
    if grids is not None:
        import sparse_attn.ops.eager_decoding as _ed
        ed = _ed
        orig_run = _ed._eager_attn_kernel.run

        def _spy(*a, **k):
            grids.setdefault("last", []).append(k.get("grid"))
            return orig_run(*a, **k)

        ed._eager_attn_kernel.run = _spy
    os.environ["TLI_E127_FAST"] = "1"
    try:
        idx = 0
        for (G, Hkv) in [(1, 8), (2, 8), (4, 8), (8, 8), (4, 2), (8, 4)]:
            H, D = Hkv * G, 128
            for per_qh in (False, True):
                rows = H if per_qh else Hkv
                for S in (16384, 65536 - 13):
                    for kind in ("rand", "full", "empty"):
                        key = f"G{G}_Hkv{Hkv}_pqh{int(per_qh)}_S{S}_{kind}"
                        torch.manual_seed(20260000 + idx)
                        idx += 1
                        k = torch.randn(1, S, Hkv, D, device=dev,
                                        dtype=torch.bfloat16)
                        v = torch.randn(1, S, Hkv, D, device=dev,
                                        dtype=torch.bfloat16)
                        q = torch.randn(1, 1, H, D, device=dev,
                                        dtype=torch.bfloat16)
                        if kind == "full":
                            m = torch.ones(1, 1, rows, S, dtype=torch.bool,
                                           device=dev)
                        else:
                            m = torch.rand(1, 1, rows, S, device=dev) > 0.98
                            m[..., -128:] = True   # sink 强制区：行非空
                            if kind == "empty":
                                m[..., 0, :] = False   # row0 全 False → code1
                        cu = torch.tensor([0, S], dtype=torch.int32,
                                          device=dev)
                        if grids is not None:
                            grids["last"] = []
                        o = eager_decoding_attn(q, k, v, m, 1, cu)
                        outs[key] = o.detach().cpu().clone()
                        if grids is not None and grids["last"]:
                            grids[key] = grids["last"][-1]
                        if kind in ("full", "empty"):
                            raised = False
                            try:
                                eager_decoding_check_errors(dev)
                            except RuntimeError:
                                raised = True   # raise 即清 flag（clear=True）
                            flags.append(raised)
        try:
            eager_decoding_check_errors(dev)   # 清 rand 档残余（不应 raise）
        except RuntimeError:
            pass
    finally:
        if orig_run is not None:
            ed._eager_attn_kernel.run = orig_run
        os.environ.pop("TLI_E127_FAST", None)
    return outs, flags


def ref_dump(dev):
    """修复前运行：生成逐位锚点 + 记录修复前 per_qh grid（(R*G,) 红证）。"""
    grids = {}
    outs, flags = _run_matrix(dev, grids)
    grids.pop("last", None)
    torch.save({"outs": outs, "grids": grids,
                "flag_raised": bool(flags) and all(flags)}, REF_PATH)
    per_qh_sample = {k: v for k, v in grids.items()
                     if "_pqh1_" in k and k.startswith("G4_Hkv8")}
    print(f"[ref-dump] {len(outs)} 配置锚点 -> {REF_PATH}")
    print(f"[ref-dump] flag_raised={all(flags)} ({len(flags)} full/empty 调用)")
    print(f"[ref-dump] 修复前 per_qh launch grid（应为 (R*G,) 冗余）: "
          f"{per_qh_sample}")


def test_097_grid_and_bitwise(dev):
    # program 数纯函数（修复锚；修复前 ImportError → 红）
    try:
        from sparse_attn.ops.eager_decoding import _attn_grid_size as gs
        ok = (gs(32, 4, True) == 32      # per_qh：Qwen3 H=32 → 32（修复前 128）
              and gs(8, 4, False) == 32   # shared：Hkv=8×G=4 → 32（不变）
              and gs(32, 8, True) == 32
              and gs(4, 8, False) == 32
              and gs(8, 1, False) == 8
              and gs(16, 2, True) == 16)
        check("097: _attn_grid_size program 数（per_qh 一行一 program）", ok)
    except ImportError:
        check("097: _attn_grid_size program 数（per_qh 一行一 program）",
              False, "修复前版本无该 helper（红证：per_qh 冗余 program 未修）")

    if not os.path.exists(REF_PATH):
        check("097: 逐位锚点存在", False,
              f"{REF_PATH} 缺失——先在【修复前】代码跑 ref-dump 生成锚点")
        return
    ref = torch.load(REF_PATH)
    grids = {}
    outs, flags = _run_matrix(dev, grids)
    grids.pop("last", None)
    # grid 实测：per_qh 修复前 (R*G,) → 修复后 (R,)；shared 前后一致
    grid_changed, grid_same = True, True
    for key, g in grids.items():
        G = int(key.split("_")[0][1:])
        Hkv = int(key.split("_")[1][3:])
        per_qh = "_pqh1_" in key
        rg = ref["grids"].get(key)
        if per_qh and G == 1:
            continue   # G=1 时 H==Hkv，mask 形状不可分 per_qh/shared（退化为 shared）
        if per_qh:
            R = Hkv * G
            if not (g == (R,) and rg == (R * G,)):
                grid_changed = False
                print(f"  per_qh grid 异常: {key} now={g} ref={rg}")
        else:
            R = Hkv
            if not (g == (R * G,) and rg == (R * G,)):
                grid_same = False
                print(f"  shared grid 异常: {key} now={g} ref={rg}")
    check("097: per_qh launch grid 实测 (R*G,)→(R,)（G=4 时 128→32）",
          grid_changed)
    check("097: shared launch grid 前后一致（(R*G,) 不变）", grid_same)
    mism = [k for k, t in ref["outs"].items()
            if not torch.equal(t, outs.get(k))]
    check("097: 修复前后输出逐位一致（per_qh 双路径 × full/empty mask 全矩阵）",
          (not mism) and len(outs) == len(ref["outs"]),
          f"n={len(outs)}/{len(ref['outs'])} mismatch={mism}")
    check("097: full/empty mask 调用后 check_errors 均 raise（flag 语义不变）",
          bool(flags) and all(flags))


if __name__ == "__main__":
    if "ref-dump" in sys.argv[1:]:
        if not torch.cuda.is_available():
            sys.exit("[ref-dump] 需要 CUDA（逐位锚点须与门禁同 GPU 设备）")
        ref_dump(torch.device("cuda:0"))
        sys.exit(0)
    if not torch.cuda.is_available():
        test_entry_wiring()
        print("\n[SKIP] 无 CUDA——仅静态接线锚，其余门禁需 GPU")
        sys.exit(1 if FAIL else 0)
    dev = torch.device("cuda:0")
    test_entry_wiring()
    test_guard_semantics(dev)
    test_entry_pred_longbench(dev)
    test_entry_pred_ruler(dev)
    test_entry_llm_eval(dev)
    test_097_grid_and_bitwise(dev)
    print(f"\n=== {len(PASS)} PASS / {len(FAIL)} FAIL ===")
    for f in FAIL:
        print(f"FAILED: {f}")
    sys.exit(1 if FAIL else 0)
