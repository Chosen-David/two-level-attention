#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E121 kimi3 清单（2026-10-08）修复验收套件——开源净化后活代码子集。

覆盖修复与验收点（红绿证据：主树未修态实测红、worktree 修复态绿，标注在各
测试 docstring 的【红】【绿】行）：

  E121-A1 (R01)  pred.py BOS 感知切片：Qwen3 系 tokenizer 不加 BOS，
                 原无条件 [:, 1:] 会切掉 question 真实首 token。
                 —— 三态行为测试（FakeModel/FakeTokenizer 全进程内 mock，
                    不依赖 GPU/真实权重）：
                    态1 加 BOS：切 BOS 后 1 个 q-stage 调用（主树同绿，回归位）
                    态2 不加 BOS：不切，1 个 q-stage 调用【红：主树切成 0 个】
                    态3 声明 bos_token_id 但实际未加：不切【红：主树切成 0 个】
  E121-A3 (B04)  pred.py 首 token EOS 判停：原循环漏判首 token。
                    正例 首 token=EOS → 0 次额外生成调用【红：主树 +3 次】
                    负例 首 token≠EOS → 照常生成 max_gen-1 步（两侧同绿）
  E121-B2 (F2)   qwen3/llama3 patch decode 分支入口硬守卫：
                    use_cache=False / chunked prefill(q_len>1) / batch>1 /
                    4D float mask 一律显式 RuntimeError【红：主树分别
                    AttributeError / 静默进 decode / 静默错算】；prefill
                    分支回归位两侧同绿
  E121-B3 (F3)   register_patch 计数 + 零匹配显式拒绝：未支持架构跑
                    method≠none 不再静默 dense【红：主树静默返回】
  E121-B6 (F10)  metrics.add_k_delta 极差改减法【红：主树 (max+min).abs()】
  E121-B7 (F12)  eager_decoding 全 False 行防线【红：主树 NaN 静默返回】

【开源净化 2026-10-23】以下验收项随实验分支退役删除（消费的接口已删）：
  A4（info.py a{α}_b{β}_g{γ}/BD 渲染锚点——αβγ 比例 CLI + kmeans/
      layer_skip 可读段退役；hash 锚点断言由 test_e155e_vendor_identity.py
      T1 承接）、A2 三项（kmeans 建簇 far_hi/_taskmd_regions + CCLUSTER_CFG）、
  A5/B1（αβγ 比例几何期望值）、B4（tli_sigma/sigma_select）、B5（proj_basis）、
  C1-C3（外部 sglang 仓静态契约——开源面不依赖 sglang 仓，SG 侧验收由
  sglang 仓自持）。A2/A5/B1 守护的 near/far 分区行为由
  exp/bench/test_e127_gates.py + exp/bench/test_tli_batch_gates.py 四配置
  矩阵承接。

依赖注入：sparse_attn / benchmark.LongBench 以 __path__ 注入（不污染进程外
环境）；transformers/datasets 走真实安装（本套件只用到 pred.py 的
get_pred 函数与 FakeModel，不加载权重）。

用法：python3 test_e121_kimi3_fixes.py   （two-level-attention/ 下）
  红态复跑（对主树）：TLI_E121_ROOT=<主树路径> python3 test_e121_kimi3_fixes.py
"""
import os
import sys
import types

sys.dont_write_bytecode = True

import torch

torch.set_num_threads(1)

# 根路径：默认本文件所在树（worktree 检出即测 worktree 代码）；TLI_E121_ROOT
# 可指主树做红态复跑（只读，不写主树）。
ROOT = os.environ.get("TLI_E121_ROOT", os.path.dirname(os.path.abspath(__file__)))

# 状态显式化：全部用显式条件检查（非 assert——python -O 会剥 assert，
# 门禁退化为恒过；本套件 -O 双跑是验收硬指标）。
RESULTS = []   # (name, status, detail)


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    RESULTS.append((name, status, detail))
    print(f"{status}  {name}" + (f"  -- {detail}" if detail else ""))


def _require(cond, msg):
    """-O 安全断言：显式 raise，不依赖 assert。"""
    if not cond:
        raise RuntimeError(msg)


def _register_pkg(name, path):
    """命名空间包注入（sparse_attn 无顶层 __init__.py 的既有模式）。"""
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    sys.modules[name] = mod
    return mod


_register_pkg("sparse_attn", os.path.join(ROOT, "sparse_attn"))
_register_pkg("benchmark", os.path.join(ROOT, "benchmark"))
_register_pkg("benchmark.LongBench", os.path.join(ROOT, "benchmark", "LongBench"))

import importlib  # noqa: E402

IDX_PKG = importlib.import_module("sparse_attn.indexer")
INFO_MOD = importlib.import_module("sparse_attn.info")

# 【开源净化 2026-10-23】sglang 仓 import（TLIIndexerSG/TLIProfile，A2-SG/
# C1-C3 消费）随 SG 侧验收退役删除——开源面不依赖外部 sglang 仓。

# pred.py：真实模块导入（transformers/datasets 用本机安装；get_pred 只在
# FakeModel 上跑，不触 GPU/权重）。导入耗时 ~10-30s 属正常。
PRED_MOD = importlib.import_module("benchmark.LongBench.pred")


# ================================================================ 公共构造
def make_args(**kw):
    a = types.SimpleNamespace(
        tia_block_size=64,
        tia_level1_topk=128,
        tia_level2_topk=1024,
        tia_level2_cmp_ratio=4,
        tia_enable_async_topk=False,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# 【开源净化 2026-10-23】gen_kq/run_mask/CCLUSTER_CFG（kmeans/ccluster
# 构造臂专用的合成 KV 工具）随 A2/A5/B1/B4/B5 删除——保留函数不再消费。


# ================================================================ Fake 模型/分词器
class _TokOut:
    """tokenizer(...) 返回体：input_ids [1,L] + .to(device) 链式（设备无关）。

    【开源净化 2026-10-23 夹具适配】d8bdef1（LB-DEV-BS 修复）后 pred.py 对
    q_input 走 __setitem__（BatchEncoding=UserDict 字典语义）重写
    input_ids——桩补 __getitem__/__setitem__ 模拟 UserDict，恢复三态测试。
    """

    def __init__(self, ids, offsets=None):
        self.input_ids = torch.tensor([ids])
        self.offset_mapping = offsets

    def to(self, *a, **kw):
        return self

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        setattr(self, key, val)


class FakeTokenizer:
    """字符→token id 确定性映射：id = 3 + ord(c)%40（∈[3,42]，不撞 BOS=1/EOS=2）。

    adds_bos=True 时 __call__ 附加 BOS（模拟 Llama 系）；bos_token_id=None
    模拟 Qwen3 系（不加 BOS）。
    """

    def __init__(self, bos_token_id=None, eos_token_id=2, adds_bos=False):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.adds_bos = adds_bos

    def __call__(self, text, truncation=False, return_tensors="pt",
                 add_special_tokens=True, return_offsets_mapping=False):
        ids = [3 + (ord(c) % 40) for c in text]
        offsets = [(i, i+1) for i in range(len(text))]
        if ids and add_special_tokens and self.adds_bos and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
            offsets = [(0, 0)] + offsets
        return _TokOut(ids, offsets if return_offsets_mapping else None)

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            if skip_special_tokens and i == self.eos_token_id:
                continue
            out.append(chr(60 + i))
        return "".join(out)

    def encode(self, text, add_special_tokens=False):
        return [3 + (ord(c) % 40) for c in text]


class _ModelOut:
    def __init__(self, logits, past):
        self.logits = logits
        self.past_key_values = past


class FakeModel:
    """按 script 逐调用发射 argmax token；记录每次收到的 input_ids。"""

    VOCAB = 16

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self._past = object()

    def __call__(self, input_ids=None, past_key_values=None, use_cache=True, **kw):
        self.calls.append(input_ids.clone()
                          if isinstance(input_ids, torch.Tensor) else input_ids)
        step = len(self.calls) - 1
        tok = self.script[min(step, len(self.script) - 1)]
        L = int(input_ids.shape[-1])
        logits = torch.zeros(1, L, self.VOCAB)
        logits[0, -1, tok] = 10.0
        return _ModelOut(logits, self._past)


def run_get_pred(tok, model, max_gen=4, max_length=64):
    """跑 pred.get_pred 一行数据（dataset='zzz' 不进任何特殊分支）。"""
    data = [{"context": "abcdefgh",
             "answers": ["ans"], "all_classes": None, "length": 8}]
    preds = PRED_MOD.get_pred(
        model, tok, data, max_length=max_length, max_gen=max_gen,
        prompt_format="{context}", dataset="zzz", device="cpu",
        model_name="zzz-model", data_fp="fp",
    )
    return preds


# ================================================================ E121-A1 (R01)
def a1_bos_aware_slice():
    """R01：BOS 感知切片三态。

    【开源净化 2026-10-23 口径适配】d8bdef1（benchmark batch 修复）后
    _manual_generate 走 material/question 两段形态：短 prompt 的 material
    段为空、question 段逐 token invoke（seed 首 token + 其余逐个）。
    断言改为「模型实际收到的 token 序列」——R01 红绿判别不变：态1 若
    BOS 未被切，BOS(1) 必出现在模型输入首；态2/3 若被误切，首 token
    (20) 必然丢失。

    【红】无 R01 修复（无条件 [:, 1:]）：态2/态3 首 token 被切 → 序列
        起始于 21（丢 20）；态1 BOS 保留 → 序列起始于 1。
    【绿】修复态：态1 切 BOS（[20..27]）；态2/态3 不切（同 [20..27]）。
    """
    name = "A1(R01) canonical BOS/首 token 保留三态"
    try:
        want = [3 + (ord(c) % 40) for c in "abcdefgh"]   # [20..27]

        def fed_tokens(mdl):
            return [int(c.reshape(-1)[0]) for c in mdl.calls]

        # ---- 态1：tokenizer 加 BOS（Llama 系）→ 切掉 BOS，question 段喂真实 token
        tok = FakeTokenizer(bos_token_id=1, eos_token_id=2, adds_bos=True)
        mdl = FakeModel(script=[5] * 12)
        run_get_pred(tok, mdl)
        toks = fed_tokens(mdl)
        q_seg = toks[:len(want)+1]
        _require(q_seg == [1]+want,
                 f"态1 加 BOS：完整输入应为 {[1]+want}，实际 {q_seg}"
                 f"——canonical token 流不得丢失 BOS 或首 token")
        _require(toks[0] == 1 and toks[1] == want[0],
                 f"态1 首两次调用应为 BOS 与真实首 token，实际 {toks[:2]}")

        # ---- 态2：tokenizer 不加 BOS（Qwen3 系）→ 不切
        # 【红绿判别位】无条件切片口径下首 token(20) 被切 → 序列起始于 21
        tok = FakeTokenizer(bos_token_id=None, eos_token_id=2, adds_bos=False)
        mdl = FakeModel(script=[5] * 12)
        run_get_pred(tok, mdl)
        q_seg = fed_tokens(mdl)[:len(want)]
        _require(q_seg == want,
                 f"态2 不加 BOS：question 段应 {want}（首 token 必须进模型），"
                 f"实际 {q_seg}——R01 缺陷复现（question 首 token 被切）")

        # ---- 态3：声明 bos_token_id=1 但实际未加 BOS → 首 token 非 BOS 不切
        # 【红绿判别位】按「声明了 bos」切片口径下首 token(20) 被切
        tok = FakeTokenizer(bos_token_id=1, eos_token_id=2, adds_bos=False)
        mdl = FakeModel(script=[5] * 12)
        run_get_pred(tok, mdl)
        q_seg = fed_tokens(mdl)[:len(want)]
        _require(q_seg == want,
                 f"态3 声明 BOS 未加：question 段应 {want}（首 token 非 BOS "
                 f"不切），实际 {q_seg}")
        report(name, True, "三态全过：canonical 完整输入保留实际 BOS 与首 token")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-A3 (B04)
def a3_first_token_eos():
    """B04：首生成 token 即 EOS 必须停（原循环漏判首 token）。

    【开源净化 2026-10-23 口径适配】逐 token 形态下首生成 token 取自
    最后一次 question call（第 8 个，输入 idx 7）的输出——EOS 应放在
    script[7]；判停后 decode 阶段 0 次调用。

    【红】主树：首 token=EOS 后继续生成 max_gen-1=3 次 decode → FAIL。
    【绿】修复态：decode 0 次、pred 为空串；负例照常 3 次 decode。
    """
    name = "A3(B04) 首 token EOS 判停（正例停/负例照常生成）"
    try:
        tok = FakeTokenizer(bos_token_id=None, eos_token_id=2, adds_bos=False)
        # 正例：question 段 8 calls（idx 0..7），首生成 token 取自 idx 7
        # 输出 → script[7]=EOS(2) → 判停，decode 0 次
        mdl = FakeModel(script=[5, 5, 5, 5, 5, 5, 5, 2, 9, 9, 9])
        preds = run_get_pred(tok, mdl, max_gen=4)
        n_q = 8                                   # "abcdefgh" 逐 token
        _require(len(mdl.calls) == n_q,
                 f"首 token=EOS 应立即停（question {n_q} 次后 0 次 decode），"
                 f"实际 {len(mdl.calls)} 次——B04 缺陷复现（首 token 不停）")
        _require(preds[0]["pred"] == "",
                 f"首 token=EOS 的 pred 应为空串（skip_special_tokens），"
                 f"实际 {preds[0]['pred']!r}")

        # 负例：首 token 非 EOS → 照常生成 max_gen-1=3 步 decode（回归位）
        mdl = FakeModel(script=[5] * 11)
        preds = run_get_pred(tok, mdl, max_gen=4)
        _require(len(mdl.calls) == n_q + 3,
                 f"首 token≠EOS 应生成 max_gen-1=3 步（共 {n_q + 3} 次调用），"
                 f"实际 {len(mdl.calls)}")
        _require(len(preds[0]["pred"]) == 4,
                 f"非 EOS 应产出 4 个可见 token，实际 {preds[0]['pred']!r}")
        report(name, True, f"EOS 首停 decode 0 次（共 {n_q} 调用）；"
                           f"非 EOS 3 次 decode（共 {n_q + 3}）")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# 【开源净化 2026-10-23】A4/A2×3/A5/B1（D1 段）+ B4/B5（D2 段）+
# C1-C3（D3 段）随实验分支退役删除——详见文件头净化标记。
# ================================================================ E121-B2 (F2)
def b2_patch_branch_guards():
    """F2：qwen3/llama3 patch decode 分支入口硬守卫（三走廊外形态 fail-closed）。

    【红】主树：① past=None → AttributeError；② chunked prefill（q_len=2
        带 past）→ 静默进 decode 分支（stub 侧证）；③ 4D float mask →
        静默错算；④ llama3 同构。四例均非预期 RuntimeError → FAIL。
    【绿】修复态：三形态 + llama3 全部显式 RuntimeError；prefill 分支
        （cache 序==q_len）走 dense stub 正常返回（回归位两侧同绿）。
    """
    name = "B2(F2) patch decode 入口守卫（use_cache/chunk/batch/4D mask）"
    try:
        import torch.nn as nn
        QW3_MOD = importlib.import_module("sparse_attn.patches.qwen3_attn_patch")
        LL3_MOD = importlib.import_module("sparse_attn.patches.llama3_attn_patch")

        class _DecodeEntered(Exception):
            pass

        class StubIndexer:
            def __init__(self):
                self.calls = []

            def clear(self):
                self.calls.append("clear")

            def prepare_mask(self, *a, **kw):
                self.calls.append("decode")
                raise _DecodeEntered("decode branch entered")

        class FakeCache:
            """KV cache 桩：get_seq_length 决定分支（≠q_len → decode）；
            update 返回预置 k/v（真实语义=返回全历史），两 knob 解耦——
            ⑤ 用 kv_len=1 使 2D bool mask 宽度与 q/k 序长一致可走通
            unpad（生产 unpad_tensor 是 x[mask] 布尔索引，要求宽度相等）。"""
            def __init__(self, seqlen, kv_len=None, hkv=2, hd=4):
                self._s = seqlen
                n = kv_len if kv_len is not None else seqlen
                torch.manual_seed(seqlen * 10 + n)
                # HF cache 约定：update 返回 [B, H, S, D]（head-major）
                self.k = torch.randn(1, hkv, n, hd)
                self.v = torch.randn(1, hkv, n, hd)

            def get_seq_length(self, layer_idx=0):
                return self._s

            def get_max_cache_shape(self):
                # b9988a0 静态缓存判别协议：DynamicCache 语义返回 -1
                # （桩保持动态路径，与测试原分支意图一致）
                return -1

            def update(self, k, v, layer_idx, cache_kwargs=None):
                return self.k, self.v

        def mk_self(hu=4, hkv=2, hd=4, hidden=16):
            return types.SimpleNamespace(
                q_proj=nn.Linear(hidden, hu * hd), q_norm=nn.Identity(),
                k_proj=nn.Linear(hidden, hkv * hd), k_norm=nn.Identity(),
                v_proj=nn.Linear(hidden, hkv * hd),
                o_proj=nn.Linear(hidden, hidden), head_dim=hd,
                config=types.SimpleNamespace(_attn_implementation="eager"),
                scaling=0.5, layer_idx=0, sliding_window=128,
                training=False, attention_dropout=0.0,
                indexer=StubIndexer(),
            )

        def call(fwd, fake_self, q_len, past, attn_mask=None, seq_len=None):
            sl = seq_len or q_len
            hs = torch.randn(1, q_len, 16)
            cos = torch.ones(1, sl, 4)
            sin = torch.zeros(1, sl, 4)
            cache_pos = torch.arange(sl)
            return fwd(fake_self, hs, (cos, sin), attn_mask, past, cache_pos)

        def expect_runtime_error(fn, needle, tag):
            try:
                fn()
            except RuntimeError as e:
                _require(needle in str(e),
                         f"{tag}: RuntimeError 应含 {needle!r}，实际 {str(e)!r}")
            except _DecodeEntered:
                raise RuntimeError(
                    f"{tag}: 静默进了 decode 分支（F2 缺陷路径——守卫缺失）")
            except Exception as e:
                raise RuntimeError(
                    f"{tag}: 预期 RuntimeError（含 {needle!r}），实际 "
                    f"{type(e).__name__}: {e}")
            else:
                raise RuntimeError(f"{tag}: 未抛异常（F2 缺陷：静默通过）")

        # ① use_cache=False（past=None，训练/PPL 形态）
        expect_runtime_error(
            lambda: call(QW3_MOD.qwen3_attn_forward, mk_self(), 2, None),
            "use_cache", "①use_cache=False")
        # ② chunked prefill：past 非空（序 3）+ q_len=2 → 主树静默进 decode
        expect_runtime_error(
            lambda: call(QW3_MOD.qwen3_attn_forward, mk_self(), 2, FakeCache(3)),
            "q_len", "②chunked_prefill")
        # ③ 4D float mask（HF 实际形态）+ decode（序 5 ≠ q_len 1）
        expect_runtime_error(
            lambda: call(QW3_MOD.qwen3_attn_forward, mk_self(), 1, FakeCache(5),
                         attn_mask=torch.zeros(1, 1, 1, 5, dtype=torch.float32)),
            "2D bool", "③4D_float_mask")
        # ④ llama3 同构（use_cache=False）
        expect_runtime_error(
            lambda: call(LL3_MOD.llama3_attn_forward, mk_self(), 2, None),
            "use_cache", "④llama3_use_cache=False")
        # ⑤ 2D bool mask（合法形态）decode 不被守卫拦截：mask 宽度须与
        # q/k 序长一致（unpad_tensor 是 x[mask] 布尔索引）→ cache 桩
        # kv_len=1、mask [1,1] 全 True，走通 unpad + stub mask + eager_decoding
        fake = mk_self()
        # stub prepare_mask 返回 (mask, block_size)：mask 须 4 维 [1,1,Hkv,tk]
        # （eager_decoding squeeze(0) 后 mask[0] 才是 2 维 [Hkv, tk]）
        def _pm(*a, **kw):
            fake.indexer.calls.append("decode")
            tk = a[2].shape[1]
            return torch.ones(1, 1, 2, tk, dtype=torch.bool), 1
        fake.indexer.prepare_mask = _pm
        call(QW3_MOD.qwen3_attn_forward, fake, 1, FakeCache(5, kv_len=1),
             attn_mask=torch.ones(1, 1, dtype=torch.bool))
        _require("decode" in fake.indexer.calls,
                 "⑤2D bool mask 合法 decode 路径被守卫误拦（过拦——回归）")
        # ⑥ prefill 分支回归：cache 序 == q_len → dense stub 正常返回
        fake = mk_self()
        _orig_eager = QW3_MOD.eager_attention_forward
        QW3_MOD.eager_attention_forward = (
            lambda self, q, k, v, am, dropout=0.0, scaling=1.0, **kw:
            (torch.zeros(1, 2, 16), None))
        try:
            out, _ = call(QW3_MOD.qwen3_attn_forward, fake, 2, FakeCache(2))
            _require(tuple(out.shape) == (1, 2, 16),
                     f"⑥prefill 输出形状应 (1,2,16)，实际 {tuple(out.shape)}")
            # 【开源净化 2026-10-23】E85f observe_prefill_q 已随静态 pair
            # 采集退役删除——prefill 分支仅剩 clear()（dense 路径不变）。
            _require(fake.indexer.calls == ["clear"],
                     f"⑥prefill 应仅 clear（sparse-prefill env 关），"
                     f"实际 {fake.indexer.calls}")
        finally:
            QW3_MOD.eager_attention_forward = _orig_eager
        report(name, True, "四守卫全 raise；2D bool/prefill 回归位不误拦")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-B3 (F3)
def b3_register_patch_count():
    """F3：register_patch 计数返回 + method≠none 零匹配显式拒绝。

    【红】主树：① 未支持架构零匹配 → 静默返回（dense 被标稀疏）；
        ② 返回值 None（无计数 API）→ 两例 FAIL。
    【绿】修复态：零匹配 RuntimeError；Qwen3 匹配返回计数 1；
        method=none/未知方法回归不变。
    【开源净化 2026-10-23】GLM patch 支持已退役——零匹配文案不再含 GLM
    提示，断言收紧为 patched=0（原「GLM 系」用例改名为通用未支持架构）。
    """
    name = "B3(F3) register_patch 计数 + 零匹配显式拒绝（未支持架构不再静默 dense）"
    try:
        import torch.nn as nn
        PATCH_MOD = importlib.import_module("sparse_attn.patches.patch")

        class Holder:
            """named_modules 鸭子类型容器（绕开 nn.Module 注册机制）。"""
            def __init__(self, mods):
                self._mods = list(mods)

            def named_modules(self):
                yield from self._mods

        class GlmFake(nn.Module):
            """任意未支持架构桩（原 GLM 用例——净化后通用化）。"""
            pass

        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

        class FakeQ3(Qwen3Attention):
            pass

        # ① 未支持架构零匹配 → 显式拒绝（红：主树静默返回 None）
        glm_model = Holder([("glm_attn", GlmFake())])
        args_tli = make_args()   # method 需显式给（make_args 默认无 method）
        args_tli.method = "tli"
        raised = False
        try:
            PATCH_MOD.register_patch(glm_model, args_tli)
        except RuntimeError as e:
            raised = True
            _require("patched=0" in str(e),
                     f"①零匹配拒绝信息应含 patched=0，实际 {str(e)!r}")
        _require(raised, "①零匹配应 RuntimeError（主树缺陷态：静默返回，"
                         "dense 结果被标稀疏方法——F3 复现）")
        # ② 计数 API：Qwen3 匹配 1 个 → 返回 1（红：主树返回 None）
        q3 = FakeQ3.__new__(FakeQ3)
        q3_model = Holder([("attn0", q3)])
        n = PATCH_MOD.register_patch(q3_model, args_tli)
        _require(n == 1, f"②Qwen3 匹配应返回计数 1，实际 {n!r}")
        _require(getattr(q3, "indexer", None) is not None,
                 "②Qwen3 模块应被挂 indexer")
        # ③ method=none → 不 patch、返回 0、不 raise（回归位）
        n = PATCH_MOD.register_patch(q3_model, make_args(method="none"))
        _require(n == 0, f"③method=none 应返回 0，实际 {n!r}")
        # ④ 未知 method → ValueError（回归位）
        raised = False
        try:
            PATCH_MOD.register_patch(q3_model, make_args(method="nope"))
        except ValueError:
            raised = True
        _require(raised, "④未知 method 应 ValueError")
        report(name, True, "零匹配拒 + 计数 API + none/未知回归全过")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-B6 (F10)
def b6_k_delta_range():
    """F10：add_k_delta 极差 = k_max − k_min（原 (k_max+k_min).abs() 无几何意义）。

    【红】主树：块 max=1、min=−2 → delta=|1+(−2)|=1（错误）。
    【绿】修复态：极差 3。死代码防御修复（无生产者）。
    """
    name = "B6(F10) add_k_delta 极差减法（红 |max+min|=1 / 绿 max−min=3）"
    try:
        METRICS_MOD = importlib.import_module("sparse_attn.metrics")
        m = METRICS_MOD.Metrics()
        # 单块 64 token：max=1.0, min=−2.0 → 极差 3.0
        k = torch.full((1, 64, 1, 1), -2.0)
        k[0, 0, 0, 0] = 1.0
        m.add_k_delta(k, 64)
        got = float(m.get_k_delta_max())
        _require(abs(got - 3.0) < 1e-6,
                 f"极差应 3.0 = max−min（主树缺陷态 |max+min|=1.0），实际 {got}")
        report(name, True, f"k_delta={got}（极差口径）")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ E121-B7 (F12)
def b7_all_false_row_guard():
    """F12：eager_decoding_attn 全 False 行防线（消费端 fail-closed）。

    【红】主树：全 False mask → softmax NaN 静默返回。
    【绿】修复态：显式 RuntimeError。
    """
    name = "B7(F12) eager_decoding 全 False 行防线（红 NaN 静默 / 绿 raise）"
    try:
        EAGER_MOD = importlib.import_module("sparse_attn.ops.eager_decoding")
        torch.manual_seed(7)
        # mask 契约：4 维 [1, x, Hkv, tk]（eager_decoding squeeze(0) 后
        # mask[0] 才是 2 维 [Hkv, tk]，与生产 prepare_mask 输出同构）
        q = torch.randn(1, 1, 2, 8)
        k = torch.randn(1, 4, 2, 8)
        v = torch.randn(1, 4, 2, 8)
        mask = torch.zeros(1, 1, 2, 4, dtype=torch.bool)   # 全 False
        raised = False
        try:
            EAGER_MOD.eager_decoding_attn(
                q, k, v, mask, 1, torch.tensor([0, 4]), softmax_scale=0.5)
        except RuntimeError as e:
            raised = True
            _require("全 False" in str(e),
                     f"RuntimeError 应含「全 False」，实际 {str(e)!r}")
        _require(raised,
                 "全 False 行应 RuntimeError（主树缺陷态：NaN 静默返回——"
                 "F12 复现）")
        # 回归位：合法 mask（含 True）正常返回有限值
        mask_ok = torch.zeros(1, 1, 2, 4, dtype=torch.bool)
        mask_ok[0, 0, 0, 2] = True
        mask_ok[0, 0, 1, 3] = True
        o = EAGER_MOD.eager_decoding_attn(
            q, k, v, mask_ok, 1, torch.tensor([0, 4]), softmax_scale=0.5)
        _require(bool(torch.isfinite(o).all()),
                 "合法 mask 输出应有限（守卫不应误拦正常路径）")
        report(name, True, "全 False raise；合法 mask 回归有限输出")
    except RuntimeError as e:
        report(name, False, str(e))
    except Exception as e:
        report(name, False, f"异常: {type(e).__name__}: {e}")


# ================================================================ main
if __name__ == "__main__":
    a1_bos_aware_slice()
    a3_first_token_eos()
    b2_patch_branch_guards()
    b3_register_patch_count()
    b6_k_delta_range()
    b7_all_false_row_guard()
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    _require(n_pass + n_fail == len(RESULTS) == 6, "门禁不允许漏项")
    print(f"\n===== E121 kimi3 活代码子集验收：{n_pass}/{n_pass + n_fail} PASS"
          f"（-O 安全：全部显式检查非 assert）=====")
    sys.exit(1 if n_fail else 0)
