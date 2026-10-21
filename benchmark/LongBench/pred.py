# Modified from: https://github.com/mit-han-lab/Quest/blob/main/evaluation/LongBench/pred.py

import os
import re
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse

from benchmark.LongBench.lbv2_choice import extract_choice_official
from benchmark.LongBench.context_capability_gate import (
    enforce_context_capability_gate,
)
# 096（TL-DSV2-CAP-PROVENANCE-096，GPT 审计 2026-10-12_0527_08819a7）：
# run 级 cap provenance manifest——全 method（含 none/fullkv）把实验输入
# cap 绑进结果产物身份，与 081 treatment snapshot 是两个独立层（cap 不进
# info.py treatment manifest，见 run_cap_manifest.py 模块头注释）。
from benchmark.LongBench.run_cap_manifest import (
    build_lbv1_input_identity,
    build_run_cap_core,
    compute_result_binding,
    finalize_run_cap_manifest,
    gate_run_cap_identity,
    input_ids_sha256,
    resolve_producer_identity,
    runcap_gate_lock,   # 103（TL-E096-CAP-GATE-TOCTOU-103）：gate→发布锁
    write_run_cap_manifest,
)
from sparse_attn.baseline_runtime import baseline_request, load_ffd, baseline_preflight
from sparse_attn.arguments import add_sparse_attn_args
from sparse_attn.patches import register_patch
from sparse_attn.metrics import get_metrics
from sparse_attn.info import get_method_name_with_info, \
    truncate_output_name_keep_hash, get_treatment_manifest_json, \
    gate_output_treatment_identity, resolve_treatment_snapshot
# 142（TL-FASA-CALIBRATION-IDENTITY）：FASA agreement 源/目标身份的
# 唯一 staging 口径（与 RULER 入口、treatment 名三方同源）。
from sparse_attn.info import stage_fasa_agreement
# 149（TL-MINFERENCE-PATTERN-IDENTITY）：MInference pattern 内容身份
# 的入口冻结（与 RULER 入口、treatment 名三方同源）。
from sparse_attn.info import freeze_minference_pattern_sha8
# 【095 修复（GPT 审计 2026-10-12_0527）】TLI_E127_FAST=1 的 gather 路径把
# 三类契约违规延迟到 GPU 端错误标志，须由本入口在每样本生成段边界消费
# ——guard 退出校验 raise 必须先于 record 构造/append 与任何落盘。
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "llama2-7b-chat-4k",
            "longchat-v1.5-7b-32k",
            "xgen-7b-8k",
            "internlm-7b-8k",
            "chatglm2-6b",
            "chatglm2-6b-32k",
            "chatglm3-6b-32k",
            "vicuna-v1.5-7b-16k",
            "Mistral-7B-Instruct-v0.3",
            "Meta-Llama-3.1-8B-Instruct",
            'Qwen3-8B',
            'Qwen3-14B',
            'Qwen3-32B',
            'DeepSeek-V2-Lite-Chat',
        ],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="If model has been downloaded locally",
    )
    parser.add_argument("--e", type=int, default=0, help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, help="task name", required=True)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--t", type=str, default="benchmark launch time")
    parser.add_argument(
        "--algo-config-path",
        type=str,
        help="Algorithm config path",
    )
    parser.add_argument('--pred_postfix', type=str, default="")
    # 稀疏 prefill（用户 2026-10-08 指令，MoBA/NSA/DSA 口径）：设
    # TLI_SPARSE_PREFILL=1 环境变量（须在模型加载/patch 挂载前；patch 层
    # 逐 forward 读 env，此处 parse 后立即设置即可）。默认不传 = 0 =
    # prefill 走原 dense 分支逐位不变（在跑链零扰动）。
    parser.add_argument(
        '--tli-sparse-prefill', action='store_true',
        help='prefill 也走 indexer 稀疏选择（chunk 共享选择，MoBA 口径）',
    )
    # E138 消融①（prefill-only 稀疏）：decode 显式回退 dense——分离 prefill
    # 稀疏 KV 污染与 decode 稀疏选择两因素（对照 --tli-sparse-prefill 单开
    # = 消融② prefill+decode 双稀疏）。
    parser.add_argument(
        '--tli-decode-dense', action='store_true',
        help='decode 回退 dense 原生前向（须与 --tli-sparse-prefill 同用；'
             'E138 消融① prefill-only 稀疏臂）',
    )

    parser.add_argument('--paper-capture',choices=('off','plan','timing'),default='off',
                        help='candidate Qwen manual-path telemetry; plan exits before model load')
    add_sparse_attn_args(parser)

    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "Llama-3.1" in model_name:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|> {prompt} <|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    elif "Qwen3" in model_name:
        prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    elif "GLM-4" in model_name:
        message = [{"role": "user", "content": prompt},]
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    elif "DeepSeek-V2" in model_name:
        # DeepSeek-V2-Lite-Chat（MLA）：LongBench 官方 pred.py 对 DeepSeek
        # 用 apply_chat_template 原生模板（S-T023 MLA 矩阵 fullkv 锚点）。
        message = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
    return prompt


def _split_prompt_question(prompt, dataset):
    """Split the material prefix from the final question-stage suffix.

    The manual generation path sends at most the last 100 characters through
    the question phase, unless a later task-specific anchor narrows that
    suffix.  Clamp the fallback boundary to zero: Python's ``-1`` slice would
    otherwise route only the final character for every unanchored prompt
    shorter than 100 characters.
    """
    if dataset in ("qasper", "hotpotqa"):
        anchor_pos = prompt.rfind("Question:")
    elif dataset == "lbv2":
        anchor_pos = prompt.rfind("Choices:")
    elif dataset in ("multifieldqa_en", "gov_report"):
        anchor_pos = prompt.rfind("Now,")
    elif dataset == "triviaqa":
        anchor_pos = prompt.rfind("Answer the question")
    elif dataset == "narrativeqa":
        anchor_pos = prompt.rfind("Do not provide")
    else:
        anchor_pos = -1
    split_at = max(0, len(prompt) - 100, anchor_pos)
    return prompt[:split_at], prompt[split_at:]


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def extract_choice_letter(text):
    """LongBench-v2 四选一解析（E117b 统一到 lbv2_choice.py 官方口径）：
    仅 'The correct answer is (X)' / 'The correct answer is X' 两条大小写
    敏感格式，无匹配返回 None。历史兜底（IGNORECASE 首个独立字母）会把
    冠词 a 判成选项 A，已删除——pred_choice 落盘值与正式评分口径一致。
    """
    return extract_choice_official(text)


def _manual_generate(model, material_ids, question_ids, max_gen, eos_token_id, capture=None):
    """Existing LongBench greedy loop, instrumented only when capture is enabled."""
    if type(max_gen) is not int or max_gen<1:raise ValueError('max_gen must be positive')
    if capture is not None:
        capture.generation_policy=dict(max_new_tokens=max_gen,eos_token_id=eos_token_id,decoding='greedy')
    def invoke(phase,**kwargs):
        return model(**kwargs) if capture is None else capture.call(phase,model,**kwargs)
    with torch.no_grad():
        output=invoke('material',input_ids=material_ids,past_key_values=None,use_cache=True)
        cache=output.past_key_values
        for token in question_ids[0]:
            output=invoke('question',input_ids=token.unsqueeze(0).unsqueeze(0),past_key_values=cache,use_cache=True)
            cache=output.past_key_values
        token=output.logits[:,-1,:].argmax(dim=-1).unsqueeze(1)
        generated=[token.item()]
        if capture is not None:capture.token_ready(generated[-1])
        if eos_token_id is None or generated[-1]!=eos_token_id:
            for _ in range(max_gen-1):
                output=invoke('decode',input_ids=token,past_key_values=cache,use_cache=True)
                cache=output.past_key_values
                token=output.logits[:,-1,:].argmax(dim=-1).unsqueeze(1)
                generated.append(token.item())
                if capture is not None:capture.token_ready(generated[-1])
                if generated[-1]==eos_token_id:break
        if capture is not None:capture.generation_complete()
        return generated


def _paper_capture_request(mode,model,material,question,dataset,model_name):
    from contextlib import nullcontext
    if mode=='off':return nullcontext(),None
    if mode!='timing':raise ValueError('plan is entrypoint-only; no request executed')
    if dataset=='samsum' or 'qwen3' not in model_name.lower():
        raise ValueError('paper capture unsupported for generate/other-model paths')
    from exp.paper.receipt import QwenRequestCapture
    capture=QwenRequestCapture(model,material,question)
    return capture,capture


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
    data_fp="",
    run_stats=None,  # 096：每样本实际输入 token 数 / 截断标记（run-cap manifest）
    paper_capture_mode="off",
):
    from sparse_attn.baseline_runtime import baseline_request
    preds = []

    # DeepSeek-V2 兼容修复（S-T023 2026-10-12）：官方 modeling_deepseek.py
    # 是旧 transformers 时代代码，与 4.56.2 新 DynamicCache（self.layers 结构）
    # 不兼容，且 DeepseekV2Attention.forward 强制 assert attention_mask is not
    # None。手动 prefill/decode 循环在 DeepSeek 上会触发 legacy cache 转换陷阱
    # （见 elif IS_DSV2 分支注释）→ 对 DeepSeek 改走 model.generate 路径。
    # 对 Qwen3 等模型保持原手动循环协议，结果零变化。
    IS_DSV2 = "DeepSeek" in model_name

    # HACK: sample 10 examples for debugging
    # data = np.random.choice(data, 10)

    for row_idx, json_obj in enumerate(tqdm(data)):
        torch.cuda.empty_cache()
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        # 096（TL-DSV2-CAP-PROVENANCE-096）：记录本样本是否发生中部截断
        # （截断判据 = 模板前 tokenized_prompt 超过本次 max_length）——
        # run-cap manifest 的 per-sample truncated 标记来源。
        if len(tokenized_prompt) > max_length:
            truncated_flag = True
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        else:
            truncated_flag = False
        # TL-DSV2-DOUBLE-BOS-094（GPT 审计 2026-10-12_0330）：记录本样本是否
        # 套了 chat template——DeepSeek 模板首部即 {{ bos_token }}，渲染文本
        # 已以 BOS 开头，后续 tokenizer 编码不得再自动补 BOS（见下方
        # add_special_tokens=False 分支注释）。
        chat_template_applied = False
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
            chat_template_applied = True

        # Split the prompt and question (simulate decoding in the question
        # stage).  The helper also handles empty/short prompts without Python's
        # negative-index fallback silently turning the question into one char.
        prompt, question = _split_prompt_question(prompt, dataset)

        if "chatglm3" in model_name:
            # input = prompt.to(device)
            input = prompt.to("cuda")
        else:
            # TL-DSV2-DOUBLE-BOS-094 修复（GPT 审计 2026-10-12_0330，031e6b8）：
            # DeepSeek chat template 渲染文本首部已是 BOS（模板首部
            # {{ bos_token }}），tokenizer_config add_bos_token=true 下默认
            # tokenizer(prompt, ...) 会再自动补一个 → [BOS,BOS,"User:",...]，
            # 偏离官方单 BOS prompt。对 DeepSeek 模板渲染路径显式
            # add_special_tokens=False（与官方 apply_chat_template(tokenize=True)
            # 内部口径一致）；非模板路径（trec 等 5 任务）仍走默认调用保持单
            # BOS；Qwen3 等其他模型（模板为无 BOS 的 f-string）逐位不变。
            if chat_template_applied and IS_DSV2:
                input = tokenizer(
                    prompt, truncation=False, return_tensors="pt",
                    add_special_tokens=False,
                )
            else:
                # input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                input = tokenizer(prompt, truncation=False, return_tensors="pt")
            q_input = tokenizer(question, truncation=False, return_tensors="pt")
            # R01 修复（kimi3 清单 2026-10-08）：原无条件 [:, 1:] 去首 token
            # 假定 tokenizer 加 BOS，但 Qwen3 系不加 → question 被切掉真实
            # 首 token。仅当首 token 确为 BOS 且非 None 时才切。
            if (
                tokenizer.bos_token_id is not None
                and int(q_input.input_ids[0, 0].item()) == tokenizer.bos_token_id
            ):
                q_input.input_ids = q_input.input_ids[:, 1:]

            # TL-LBV1-INPUT-IDENTITY-20261020：在 CPU BatchEncoding 上冻结
            # 模板、截断与 question split 后**实际送模**的 token ID。先 hash
            # 再搬 CUDA，避免为 provenance 新增 device→host 同步；上面的 BOS
            # 判定也因此不再在 GPU tensor 上调用 .item()。哈希口径由
            # run_cap_manifest.input_ids_sha256 固定为 signed-int64 little-endian。
            if run_stats is not None:
                realized_ids = (
                    input.input_ids[0].tolist()
                    + q_input.input_ids[0].tolist()
                )
                run_stats.setdefault("input_ids_sha256", []).append(
                    input_ids_sha256(realized_ids))
            input = input.to("cuda")
            q_input = q_input.to("cuda")

        context_length = input.input_ids.shape[-1] + q_input.input_ids.shape[-1]
        # 096（TL-DSV2-CAP-PROVENANCE-096）：每样本实际输入 token 数（chat
        # 模板后、question 拼接后的终值）进 run-cap manifest——修复前每行
        # JSONL 只记数据集原字段 length，本次 max_length/实际输入/是否截断
        # 均不落盘（096 审计 P1）。
        if run_stats is not None:
            run_stats.setdefault("input_tokens", []).append(
                int(context_length))
            run_stats.setdefault("truncated", []).append(
                bool(truncated_flag))

        # 【095 修复（GPT 审计 2026-10-12_0527，2026-10-12）】请求级
        # fail-closed guard：TLI_E127_FAST=1 的 gather 路径把三类契约违规
        # （空 mask 行 / 选中数超 TLI_E127_W_UPPER / cu_seqlens_k 破坏）
        # 延迟到 GPU 端错误标志（decode 零同步的配套），由 guard 在本样本
        # 生成段边界一次校验 raise——必须先于 record 构造/append（get_pred
        # 内）与任何落盘（out_path 写入在 get_pred 返回后，raise 传播即
        # 整次任务输出作废）。进入时清旧 flag：上一样本未消费的违规不
        # 误杀本样本（guard 语义/并发限制见其 docstring）。
        capture_context,capture = _paper_capture_request(
            paper_capture_mode,model,input.input_ids,q_input.input_ids,dataset,model_name)
        with baseline_request(model) as baseline_receipt, capture_context, eager_decoding_request_guard():
            if (
                dataset == "samsum"
            ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                # B03 修复（GPT 审查 2026-10-08）：原版 model.generate(**input) 缺两件事——
                # ① q_input 切出的 question 没接回，只喂了 prefix；② output 赋值后
                # generated_content 从未赋值，下方共用 decode 必 UnboundLocalError。
                # 修法：拼接完整输入（与 else 分支同协议），generate 后取新增 token。
                full_ids = torch.cat([input.input_ids, q_input.input_ids], dim=-1)
                output = model.generate(
                    input_ids=full_ids,
                    attention_mask=torch.ones_like(full_ids),
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=full_ids.shape[-1] + 1,
                    eos_token_id=[
                        tokenizer.eos_token_id,
                        tokenizer.encode("\n", add_special_tokens=False)[-1],
                    ],
                )[0]
                generated_content = output[full_ids.shape[-1]:].tolist()
            elif IS_DSV2:
                # S-T023 DeepSeek fullkv 修复（2026-10-12）：官方 modeling_deepseek.py
                # 是旧 transformers 时代代码，手动 prefill/decode 循环在 4.56.2 新
                # DynamicCache（self.layers 结构）下会触发 legacy cache 转换陷阱——
                # prefill 返回 to_legacy_cache 的 tuple，decode 再 from_legacy_cache
                # 转回，转换后 get_seq_length 返回 0 → _prepare_4d_causal_attention_mask
                # 生成 (1,1,1,1) mask，与 attention 层期望 (1,1,1,kv_seq_len) 不符 →
                # ValueError。改用 model.generate 路径（与 samsum 分支同协议：full_ids
                # 一次 prefill + 贪心生成），cache_position/cache 全权交给 transformers
                # 管理，生成语义与手动 argmax 循环严格等价。
                full_ids = torch.cat([input.input_ids, q_input.input_ids], dim=-1)
                output = model.generate(
                    input_ids=full_ids,
                    attention_mask=torch.ones_like(full_ids),
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )[0]
                generated_content = output[full_ids.shape[-1]:].tolist()
            else:
                generated_content = _manual_generate(
                    model,input.input_ids,q_input.input_ids,max_gen,tokenizer.eos_token_id,capture)

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        # pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)

        # print("Score: ", flush=True)
        # score_info.print_min_sum_single_query()

        # Record budget
        # avg_budget = budget_info.get_total_avg_budget()
        # avg_score = score_info.get_total_avg_score()
        # budget_info.reset()
        # score_info.reset()
        # avg_B0 = budget_info.get_total_avg_budget_B0()

        metrics = get_metrics()
        budget = None if baseline_receipt else metrics.get_select_tokens()
        metrics.clear()

        record = {
            "pred": pred,
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
            "budget": budget,
            "score_sum": None,
            # "B0": avg_B0,
            # "B1": avg_budget,
        }
        if baseline_receipt:
            record['baseline_receipt'] = baseline_receipt
        if capture is not None:
            if capture.receipt is None:raise ValueError('request capture incomplete')
            record['paper_request_capture']=capture.receipt
        if dataset == "lbv2":
            # LongBench-v2：额外落盘解析字母与样本 _id（打分口径 accuracy = pred_choice == answer）
            record["pred_choice"] = extract_choice_letter(pred)
            record["_id"] = json_obj["_id"]
        else:
            # E116a（GPT TL-LBV1-SAMPLE-GATE-024）：v1 记录补稳定 sample_id，
            # 绑定 任务名 + 数据指纹 + 源行 index——评分端据此做 fail-closed
            # 完整性门禁（无重复/与 manifest 集合相等），缺行混行不再静默通过。
            record["_id"] = f"{dataset}:{data_fp}:{row_idx}"
        preds.append(record)
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def _patch_deepseek_dynamic_cache_compat():
    """DeepSeek-V2 trust_remote_code modeling（modeling_deepseek.py）依赖
    transformers 旧版 DynamicCache API：get_usable_length / seen_tokens /
    get_max_length。transformers 4.56.2 已移除三者 → monkey-patch 补齐
    兼容 shim（语义对齐旧版，仅当方法缺失时注入，不影响其他模型）。
    """
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "get_usable_length"):

        def _get_usable_length(self, new_seq_length=None, layer_idx=0):
            # 旧版语义：返回该层已缓存 KV 数量（层未创建返回全局已见数）。
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = _get_usable_length

    if not hasattr(DynamicCache, "get_max_length"):

        def _get_max_length(self):
            # 旧版 DynamicCache 语义：无固定上限 → None（不触发 mask 裁剪）。
            return None

        DynamicCache.get_max_length = _get_max_length

    if not hasattr(DynamicCache, "seen_tokens"):

        @property
        def _seen_tokens(self):
            # 4.56.2 无全局 _seen_tokens 计数；无裁剪连续生成时
            # layer0 长度 == 全局已见 token 数。
            return self.get_seq_length(0)

        DynamicCache.seen_tokens = _seen_tokens


def load_model_and_tokenizer(path, model_name, device, args, snapshot=None):
    baseline_preflight(args)
    if args.method == 'ffd_vendor':
        model = load_ffd(path, args)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        return model, tokenizer
    # DeepSeek-V2 trust_remote_code 需要旧版 DynamicCache API 兼容
    if "DeepSeek-V2" in model_name:
        _patch_deepseek_dynamic_cache_compat()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = model.eval()

    # 【B3 修复（kimi3 清单 F3，2026-10-08）】patch 计数显式校验：
    # method≠none 而 0 匹配（如 GLM-4 等未支持架构）时 fail-closed——
    # 防止静默跑 dense 却把输出文件打上稀疏方法标签。
    # 081（TL-E121-OUTPUT-SNAPSHOT-081）：snapshot 透传给每个 attention
    # 层的 TLIIndexer（共享同一冻结对象，不按层重开文件）。
    if args.method == "quest_vendor":
        # B 轨 vendor（官方源码复用）：不走 indexer_type_dict（A 轨
        # quest_indexer 是语义复现），改为把官方 Quest forward 原样绑定
        # Qwen3Attention——quest_attention_official.py 与官方仓库逐字节
        # 一致（md5 78e87fb8...），qwen3_quest_patch.py 只做接口胶水。
        import sys
        vendor_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "exp", "vendor_baselines", "quest_vendor",
        )
        sys.path.insert(0, vendor_dir)
        from qwen3_quest_patch import enable_quest
        n_patched = enable_quest(
            model,
            token_budget=getattr(args, "quest_token_budget", 1024),
            chunk_size=getattr(args, "quest_chunk_size", 16),
        )
    elif args.method == "minference_vendor":
        # B 轨 vendor（官方源码复用）：MInference（微软 NeurIPS 2024）动态稀疏
        # prefill + decode dense。qwen3_minference_patch.py 只做接口胶水（显式
        # q_len>1 判 prefill sparse / decode dense），核心算法 vertical+slash
        # topk 索引与 sparse attention 逐字节来自官方 minference_forward.py。
        # config_path 由 is_search 在线搜出后复用（pred 正式跑 is_search=False）。
        import sys
        vendor_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "exp", "vendor_baselines", "minference_vendor",
        )
        sys.path.insert(0, vendor_dir)
        from qwen3_minference_patch import enable_minference
        cfg_path = getattr(args, "minference_config_path", None)
        if not cfg_path:
            raise RuntimeError(
                "method=minference_vendor 必须传 --minference_config_path "
                "（is_search 在线搜出的 best_pattern JSON）"
            )
        # 149（TL-MINFERENCE-PATTERN-IDENTITY，GPT 审计 8b56bbd + C+ 批
        # 并发残余）：入口加载期单次冻结 pattern 完整内容身份快照（sha8
        # + raw bytes + 解析后 JSON）——treatment 名尾段 _cfg<sha8> 消费
        # 冻结值；adapter（enable_minference）同样消费冻结对象构建
        # best_pattern，不再二次 open 可变 config_path（旧实现窗口内
        # 源被替换则 name 绑 A、runtime 读 B，与 142 同构）。
        freeze_minference_pattern_sha8(args)
        frozen_snap = getattr(args, "_minference_pattern_frozen", None)
        n_patched = enable_minference(
            model,
            is_search=False,
            config_path=cfg_path,
            starting_layer=getattr(args, "minference_starting_layer", -1),
            frozen_pattern=(frozen_snap.pattern
                            if frozen_snap is not None else None),
        )
    elif args.method == "fasa_vendor":
        # B 轨 vendor（官方源码复用）：FASA（ICLR 2026 frequency-aware
        # sparse attention）两阶段：Phase1 离线校准（qwen3_identify_fre.py）
        # + convert_agreement.py 产出 agreement.pkl；Phase2 在线剪枝
        # （qwen3_fasa_patch.py 的 enable_fasa + 官方 core_module 纯 top-k）。
        # 与 pred_ruler.py fasa_vendor 分支同构。
        import sys
        repo = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", ".."
        )
        vendor_dir = os.path.join(repo, "exp", "vendor_baselines", "fasa_vendor")
        sys.path.insert(0, vendor_dir)
        from qwen3_fasa_patch import enable_fasa
        fasa_dataset = getattr(args, "fasa_dataset", "qasper")
        # 142（TL-FASA-CALIBRATION-IDENTITY，GPT 审计 B 批 + C+ 批并发
        # 残余 + sha8 前缀碰撞残余 P2）：--fasa_agreement_dir 接线经
        # sparse_attn/info.py 的 142 三函数（两入口与 treatment 名三方
        # 同源）；物化目标是内容寻址进程私有目录
        # dimension_rankings_staged/<full_sha256>/（共享固定路径下并发
        # 首 stage 不同内容会后者覆盖前者、先到进程的官方 loader 读到
        # 后者内容——错绑；sha8 前缀碰撞档同构，故目录段用完整 SHA256，
        # sha8 只作名字显示标识）。stage 返回 staged 根，adapter（enable_
        # fasa）调官方 constructe_adaptive_selection 前把 CWD 切到该
        # 根（try/finally 恢复），官方相对路径 dimension_rankings/ 即解
        # 析到内容寻址副本；旧仓库根 dimension_rankings/ 不再被读写。
        staged_root = stage_fasa_agreement(args, repo)
        n_patched = enable_fasa(
            model,
            agreement_dir=staged_root,
            model_name=args.model, dataset_name=fasa_dataset,
            budget=getattr(args, "fasa_budget", 1024),
            threshold_ratio=getattr(args, "fasa_threshold_ratio", 0.8),
            min_k=getattr(args, "fasa_min_k", 16),
            max_k=getattr(args, "fasa_max_k", 16),
            mode=getattr(args, "fasa_mode", "topk"))
    elif args.method == "sparq_vendor":
        # B 轨 vendor（官方源码复用）：SparQ（graphcore bandwidth-efficient
        # sparse attention）纯 PyTorch 无 CUDA kernel 无离线校准。qwen3_
        # sparq_patch.py 只做接口胶水（GQA repeat_kv 展开 + V_mean dtype/
        # device 迁移 + DynamicCache 4.56 全量 slice 适配），核心 sparq_attn
        # 来自 sparq_attention_official.py（与官方 sparq.py 逐字节一致，
        # md5=522a085c）。主表口径（用户 2026-10-14 拍板）：k 固定 1024；
        # r 随序列长度由 ratio=context_length/(2*k) 反推。LongBench 任务
        # 序列长度 0.5k~32k，context_length 默认取模型 cap（31500）。
        import sys
        repo = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", ".."
        )
        vendor_dir = os.path.join(repo, "exp", "vendor_baselines", "sparq_vendor")
        sys.path.insert(0, vendor_dir)
        from qwen3_sparq_patch import enable_sparq
        # --sparq_context_length 默认 None → 回退到 LongBench 协议 cap
        # （model2maxlen.json，Qwen3-8B=31500）：ratio=31500/(2*1024)=15.4
        # → r=round(128/15.4)=8，与用户拍板口径（S=32768→r=8）一致。
        # 不用 model.config.max_position_embeddings（Qwen3-8B=40960 → r=6，
        # 偏离口径，高估序列上界）。生效 ctx 存 _sparq_ctx_effective 供
        # treatment 名编码——r 随序列长度变，不同 ctx = 不同 treatment，
        # 防跨档位同名互覆。
        _sparq_cl = getattr(args, "sparq_context_length", None)
        if _sparq_cl is None:
            try:
                import json as _json
                _cap_file = os.path.join(
                    getattr(args, "config_path", ""), "model2maxlen.json")
                with open(_cap_file) as _f:
                    _sparq_ctx = int(_json.load(_f)[args.model])
            except (OSError, KeyError, TypeError, ValueError):
                _sparq_ctx = 32768  # 兜底：用户口径 S=32768 → r=8
        else:
            _sparq_ctx = _sparq_cl
        setattr(args, "_sparq_ctx_effective", _sparq_ctx)
        n_patched = enable_sparq(
            model,
            k=getattr(args, "sparq_k", 1024),
            context_length=_sparq_ctx,
        )
    elif args.method == "moba_vendor":
        # MoBA（Mixture of Block Attention）B 轨官方源码复用——官方语义：
        # prefill 块级 gate 稀疏（chunk-mean key gate → per-query top-k 选块，
        # 尾 chunk 总是选）+ decode dense（官方 wrapper.py:54-82）。qwen3_
        # moba_patch.py 仅接口胶水（GQA 展开 + 形状搬运），核心
        # moba_attn_varlen 来自 moba_attention_official.py（=
        # third_party/MoBA/moba/moba_efficient.py 逐字节一致，md5=da636efc）。
        # 主表预算口径对齐 K2=1024：topk 默认 None → 换算
        # max(2, ceil(budget/chunk))。官方默认 chunk=4096 → topk=2 → 预算 8192
        # （块粒度下限，官方 efficient 内部 topk-1，topk=1 退化为 dense）；
        # 严格 1024 须改 chunk（如 256→topk=4），论文标注 budget-normalized 消融。
        import sys
        repo = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", ".."
        )
        vendor_dir = os.path.join(repo, "exp", "vendor_baselines", "moba_vendor")
        sys.path.insert(0, vendor_dir)
        from qwen3_moba_patch import enable_moba
        n_patched = enable_moba(
            model,
            chunk_size=getattr(args, "moba_chunk_size", 4096),
            topk=getattr(args, "moba_topk", None),
            budget=getattr(args, "moba_budget", 1024),
        )
    elif args.method == "asynctls_vendor":
        # AsyncTLS（arXiv:2604.07815，人大+美团）C 轨语义复现——官方尚无开源
        # 仓库（用户 2026-10-17 授权「无源码则读论文给出近似实现」），三级流水：
        # Stage-1 块粗筛（B=64 top-kb=128 块符号感知 GEMM 打分）+ Stage-2 channel
        # 细筛（Double-Sparsity 校准 top-dc=32 channel + INT4 key + 组内平均
        # softmax）+ Stage-3 全维 KV sparse attn（kt=token_budget）。
        # qwen3_asynctls_patch.py 全为语义复现（_baseline_identity 已标注非官方）。
        # 校准文件（--asynctls_channel_file）必填：identify_channel.py 离线产出。
        import sys
        repo = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", ".."
        )
        vendor_dir = os.path.join(repo, "exp", "vendor_baselines", "asynctls_vendor")
        sys.path.insert(0, vendor_dir)
        from qwen3_asynctls_patch import enable_asynctls
        _atls_chan = getattr(args, "asynctls_channel_file", None)
        if not _atls_chan or not os.path.isfile(_atls_chan):
            raise RuntimeError(
                "method=asynctls_vendor 需要 --asynctls_channel_file 指向校准产物"
                "（exp/vendor_baselines/asynctls_vendor/identify_channel.py 生成）")
        n_patched = enable_asynctls(
            model,
            channel_file=_atls_chan,
            token_budget=getattr(args, "asynctls_token_budget", 1024),
            block_size=getattr(args, "asynctls_block_size", 64),
            top_kb=getattr(args, "asynctls_top_kb", 128),
        )
    elif args.method == "hisa_vendor":
        # HISA（COLM 2026，arXiv:2603.28458v4）B 轨官方 kernel 语义 torch 等价——
        # 两级 indexer：Stage-1 块均值池化 + Q·BlockedK 线性打分 + 首/末块 +inf
        # 保护 + top-block；Stage-2 块内全分辨率打分 + top-token。Qwen3 适配：
        # GQA 组内平均替代 DSA per-head weights、bf16 均值替代 fp8（identity 标注）。
        import sys
        repo = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", ".."
        )
        vendor_dir = os.path.join(repo, "exp", "vendor_baselines", "hisa_vendor")
        sys.path.insert(0, vendor_dir)
        from qwen3_hisa_patch import enable_hisa
        n_patched = enable_hisa(
            model,
            hisa_budget=getattr(args, "hisa_budget", 1024),
            k_block_size=getattr(args, "hisa_k_block_size", 128),
            block_topk=getattr(args, "hisa_block_topk", 64),
        )
    elif args.method == "jssa_vendor":
        # JSSA（arXiv:2610.19026v1）B 轨官方 vLLM patch 语义 torch 等价——
        # decode 单级 token 稀疏，校准基 M=TopEig_r(G_Qm/tr+G_K/tr) 投影打分
        # top-budget/head + 精确注意力，prefill dense。qwen3_jssa_patch.py
        # 按官方语义纯 torch 等价（identity 标注）。basis_path 必填。
        import sys
        repo = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", ".."
        )
        vendor_dir = os.path.join(repo, "exp", "vendor_baselines", "jssa_vendor")
        sys.path.insert(0, vendor_dir)
        from qwen3_jssa_patch import enable_jssa
        n_patched = enable_jssa(
            model,
            basis_path=getattr(args, "jssa_basis_path", None),
            budget=getattr(args, "jssa_budget", 1024),
            rank=getattr(args, "jssa_rank", 32),
        )
    elif args.method == "clusterkv_vendor":
        # ClusterKV（KMeans 聚类稀疏）B 轨官方源码复用——官方语义：
        # prefill dense；decode 后期 pylibraft KMeans 建簇（build_cluster）→
        # query×centroids 最近邻选簇到 token_budget−sink → sel_key_indices →
        # +sink 偏移 → gather + 精确 matmul 细筛（cluster_attn_out 官方逐字节
        # 一致）。qwen3_clusterkv_patch.py 仅接口胶水；官方 cluster_attention.py
        # 从 third_party/ClusterKV/accuracy 直接 import（md5 冻结）；官方
        # `_clusterkv_knl.search_indices` 编译内核不可用（网络阻塞），由 vendor
        # 包 clusterkv/_clusterkv_knl.py 纯 torch 等价实现（与官方
        # KMeansSearchIndicesKernel 语义一致），官方文件零改动。
        import sys
        repo = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", ".."
        )
        vendor_dir = os.path.join(
            repo, "exp", "vendor_baselines", "clusterkv_vendor")
        sys.path.insert(0, vendor_dir)  # vendor clusterkv/ stub 必须最前
        ckv_root = os.path.join(repo, "third_party", "ClusterKV")
        if ckv_root not in sys.path:
            sys.path.append(ckv_root)  # 官方树放 vendor 之后
        from qwen3_clusterkv_patch import enable_clusterkv
        n_patched = enable_clusterkv(
            model,
            token_budget=getattr(args, "clusterkv_token_budget", 1024),
            sink=getattr(args, "clusterkv_sink", 16),
            nlist=getattr(args, "clusterkv_nlist", 400),
            fit_iter=getattr(args, "clusterkv_fit_iter", 20),
            head_sel=getattr(args, "clusterkv_head_sel", "truc"),
            balance=getattr(args, "clusterkv_balance", False),
            dist_t=getattr(args, "clusterkv_dist_t", "cosine"),
            gqa_policy=getattr(args, "clusterkv_gqa_policy", None),
            topk_stat=getattr(args, "clusterkv_topk_stat", False),
        )
    else:
        n_patched = register_patch(model, args, snapshot)
    if args.method != "none" and n_patched == 0:
        raise RuntimeError(
            f"method={args.method!r} 但成功挂载 0 个 "
            f"attention 模块，模型实际将以 dense 运行——拒绝产出被误标为"
            f"稀疏方法的结果。"
        )

    return model, tokenizer


if __name__ == "__main__":
    args = parse_args()
    if args.paper_capture=='plan':
        print(json.dumps({'schema':'longbench-paper-capture-plan-v1','gpu_started':False,
            'supported':'Qwen3 manual greedy path only','unsupported':['samsum generate','DeepSeek generate','unknown attention forwards'],
            'support_observation':'separate pass required; not implemented by timing hooks','execution':'disabled'}))
        raise SystemExit(0)
    if args.paper_capture=='timing' and (not args.model or 'qwen3' not in args.model.lower() or args.task=='samsum'):
        raise ValueError('paper timing supports only Qwen3 manual path; rejected before model loading')
    seed_everything(42)
    # 稀疏 prefill 门控：在模型加载/patch 挂载前设置环境变量
    # （qwen3_attn_patch 逐 forward 读取；默认不设 = dense 逐位不变）
    if args.tli_sparse_prefill:
        os.environ["TLI_SPARSE_PREFILL"] = "1"
    # E138 消融①：TLI_DECODE_DENSE=1（qwen3_attn_patch decode 分支逐
    # forward 读 env）。单独开（无稀疏 prefill）无意义且与产出标注冲突，
    # 拒绝产出被误标的结果。
    if args.tli_decode_dense:
        if not args.tli_sparse_prefill:
            raise SystemExit(
                "[GATE-FAIL] E138: --tli-decode-dense 须与 --tli-sparse-prefill "
                "同用（prefill-only 稀疏消融臂）；单独开 = 全 dense，"
                "method=tli 标注失真，fail closed")
        os.environ["TLI_DECODE_DENSE"] = "1"

    config_path = args.config_path
    model2maxlen = json.load(open(f"{config_path}/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define your model
    model_name = args.model
    if args.model_path is None:
        model2path_list = json.load(open(f"{config_path}/model2path.json", "r"))
        model2path = model2path_list[model_name]
    else:
        model2path = args.model_path
    # 081（TL-E121-OUTPUT-SNAPSHOT-081）：tli 臂在模型加载/patch（文件
    # 被任何 attention 层消费）之前冻结一次解析的 treatment snapshot——
    # D′ 掩码 / 投影基每个文件只 open 一次，后续 method_name（472）、
    # sidecar 写门（501）全部消费同一冻结对象。修复前：每层各读一次 +
    # 生成后再读 → 生成窗口内文件被替换时 runtime A / 文件名 B /
    # sidecar C 三方混装可达。非 tli 臂零改动（snapshot=None）。
    if args.method == "tli":
        snapshot = resolve_treatment_snapshot(args)
    else:
        snapshot = None
    # TL-DSV2-CONTEXT-CAP-093 修复（GPT 审计 2026-10-12_0330，031e6b8）：
    # 加载 tokenizer/model 之前核对实验 cap 与模型上下文能力（注册表 +
    # 本地快照活值），三方冲突打印明细不静默挑选，cap 超过
    # min(max_position_embeddings, 模型卡声明) 时 fail-closed。未登记模型
    # （Qwen3 等既有链）门禁跳过，行为零变化。max_length 读取从
    # load_model_and_tokenizer 之后提前到此处（同一 dict 查找，仅求值时序
    # 变化，语义不变）。
    max_length = model2maxlen[model_name]
    # 096（TL-DSV2-CAP-PROVENANCE-096）：接收并消费 gate 返回值——修复前
    # 该调用不接收返回值，含 experiment_input_cap / 注册表 protocol cap 的
    # 字典被丢弃（096 审计 P1 原话）。cap_gate_info 由 run-cap manifest
    # 消费（cap 绑进结果产物身份），None（未登记模型）按 unregistered 折叠。
    cap_gate_info = enforce_context_capability_gate(
        model_name, model2path, max_length, config_path
    )
    model, tokenizer = load_model_and_tokenizer(
        model2path, model_name, device, args, snapshot
    )

    datasets = [args.task]
    dataset2prompt = json.load(open(f"{config_path}/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(f"{config_path}/dataset2maxlen.json", "r"))

    # 并发派单修复（2026-10-13）：多卡同时启动写同一 output-dir 时，多个进程
    # 同过 `not os.path.exists` 检查后都执行 makedirs → 竞态 FileExistsError。
    # 改为 exist_ok=True 幂等创建。
    if args.e:
        dir = f"{args.output_dir}/pred_e{args.pred_postfix}"
    else:
        dir = f"{args.output_dir}/pred{args.pred_postfix}"
    os.makedirs(dir, exist_ok=True)

    local_data_set = args.dataset_path
    has_local_data_set = os.path.exists(local_data_set)
    for dataset in datasets:
        if dataset == "lbv2":
            # LongBench-v2（全量三件套之一）：503 题四选一 MCQ，单个 data.json
            # （JSON 数组，非 jsonl）。--dataset-path 可直接给 data.json 文件路径，
            # 也可给目录（依次找 lbv2.json / data.json）。
            # 断点续跑/SKIP 与 v1 同口径：由链脚本按输出 jsonl 行数（全集 503）判定。
            v2_path = local_data_set
            if os.path.isdir(v2_path):
                for cand in ("lbv2.json", "data.json"):
                    if os.path.exists(os.path.join(v2_path, cand)):
                        v2_path = os.path.join(v2_path, cand)
                        break
            if not os.path.isfile(v2_path):
                raise FileNotFoundError(
                    f"lbv2 需要 --dataset-path 指向 LongBench-v2 data.json（或含它的目录），当前路径不存在: {v2_path}"
                )
            with open(v2_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # 归一化到 v1 管线字段约定：answers 列表 + all_classes（打分走 eval.py 的 lbv2 scorer）
            data = []
            for obj in raw:
                obj = dict(obj)
                obj["answers"] = [obj["answer"]]
                obj["all_classes"] = ["A", "B", "C", "D"]
                data.append(obj)
            print(f"lbv2: loaded {len(data)} examples from {v2_path}", flush=True)
        elif args.e:
            if not has_local_data_set:
                data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
            else:
                data = load_dataset(
                    "json", data_files=f"{local_data_set}/{dataset}_e.jsonl", split="train"
                )
        else:
            if not has_local_data_set:
                data = load_dataset("THUDM/LongBench", dataset, split="test")
            else:
                data = load_dataset(
                    "json", data_files=f"{local_data_set}/{dataset}.jsonl", split="train"
                )

        # Create output.jsonl file name
        dataset_prefix = dataset.split("-")[0]

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        # TL-LBV1-INPUT-IDENTITY-20261020：LBv1 v2 数据身份覆盖有序完整源行，
        # 并把 prompt template 单独绑定进 run identity。旧 E116a 仅哈希
        # answers+length，会让 context/question 改写后仍沿用同一 _id/run-cap。
        # lbv2 有官方 _id，继续沿用其独立口径。
        if dataset == "lbv2":
            data_fp = ""
            input_identity = None
        else:
            input_identity = build_lbv1_input_identity(data, prompt_format)
            data_fp = input_identity["source_fingerprint"]

        # 096（TL-DSV2-CAP-PROVENANCE-096）：method_name/out_fn/out_path 从
        # get_pred 之后上移到此处——run-cap 写门的早期 [CAP-CLASH] 检测在
        # 烧 GPU 之前完成（同 task/method/时间戳不同 cap 直接拒，不必等
        # 整轮生成跑完才发现口径冲突）。method_name 只依赖 args/snapshot
        # （与 get_pred 无数据依赖），提前计算逐位不变。
        # 081：method_name 从冻结 snapshot 派生（tli 臂）——生成窗口内
        # 文件被替换时文件名仍等于 runtime/sidecar 的内容身份。
        method_name = get_method_name_with_info(args, snapshot)
        out_fn = f"{dataset_prefix}-{method_name}-{args.t}"
        # avoid too long file name
        out_fn = out_fn.replace(" ", "")
        out_fn = out_fn.replace("'", "")
        if len(out_fn) > 230:
            # 076（TL-E121-OUTPUT-ID）：截断不得从尾部吃掉 method_name 的
            # 身份 hash 尾段（_h<hash10> 被截后不同 treatment 可再次同名
            # 互覆）——先截可读中段再保 hash；非 tli 名（无 hash 段）回退
            # 旧口径。截断重建后统一补一次空格/撇号清洗（dataset_prefix
            # 走重建路径时不经过上面的 replace，与未截断路径口径一致）。
            # 081 + kimi3 0316 追加修复 1：上限 245 → 230——out_fn 落盘
            # 要拼 ".jsonl"（6）且写门会创建 sidecar ".tli_manifest.json"
            # （18）+ ".runcap.json"（13，096），245 下 sidecar 实名超
            # ext4 255 单文件名上限 → open 抛 OSError 36 裸 traceback
            # （破坏 076 fail-closed 统一口径）；230 = 255 − 6 − 18 − 1，
            # 与 truncate_output_name_keep_hash 缺省 limit 同源。
            out_fn = truncate_output_name_keep_hash(
                dataset_prefix, method_name, args.t)
            out_fn = out_fn.replace(" ", "").replace("'", "")
        out_path = f"{dir}/{out_fn}.jsonl"

        # 096：run 级 cap provenance manifest（全 method 含 none/fullkv——
        # 修复前 fullkv 无任何 provenance 旁挂，16K 与 31.5K 产物可同路径
        # 互覆或被当同一 treatment，产物自身无法证明处理口径，096 审计 P1）。
        # 写门 fail-closed：同路径已存在不同 cap 的 manifest → [CAP-CLASH]
        # 拒绝运行（093 修复链的落盘闭包）。
        runcap_core = build_run_cap_core(
            model_name=model_name,
            model_path=model2path,
            method=args.method,
            dataset=dataset,
            dataset_fingerprint=data_fp,
            run_timestamp=args.t,
            effective_cap=max_length,
            cap_gate_info=cap_gate_info,
            max_new_tokens=max_gen,
            producer=resolve_producer_identity(__file__),
            input_identity=input_identity,
        )
        # 103（TL-E096-CAP-GATE-TOCTOU-103，GPT 审计 2026-10-12_0739）：
        # gate 与发布间的跨进程独占锁（fcntl.flock，锁文件 = 输出目录下
        # .runcap_gate.lock）——修复前两进程可在 sidecar 尚不存在时同时
        # 过 gate，随后交错覆盖数据与 manifest（TOCTOU）。锁持有窗口 =
        # gate → generating manifest → 数据写 → complete manifest（下方
        # 整块）；gate 在锁内重新执行即 double-check（后到者锁内重读身份，
        # 先到者已写出的 manifest 触发 [CAP-CLASH] 或同 cap 幂等放行）。
        # try/finally 语义与 095 guard 的 raise 传播兼容（异常必释放）。
        with runcap_gate_lock(out_path):
            # 105（run_cap backfill 绑定，GPT TL-E096-BACKFILL-BINDING-105）：
            # 本 run 的 stdout 身份绑定行——backfill 工具按
            # [CONTEXT-CAP-GATE] model= 行分块后，块内以本行绑定
            # task/时间戳/输出 basename（修复前日志与产物无任何可绑定
            # 线索，跨 run 日志可把 partial 产物洗白成另一模型的 complete）。
            print(f"[CONTEXT-CAP-GATE] run-bind task={dataset} t={args.t} "
                  f"output={os.path.basename(out_path)}", flush=True)
            runcap_sidecar = gate_run_cap_identity(out_path, runcap_core)
            # 096：生成期采集每样本实际输入 token 数 / 截断标记（get_pred 内
            # 逐样本 append，见 run_stats 参数）。
            run_stats = {}
            preds = get_pred(
                model,
                tokenizer,
                data,
                max_length,
                max_gen,
                prompt_format,
                dataset,
                device,
                model_name,
                data_fp=data_fp,
                run_stats=run_stats,
                paper_capture_mode=args.paper_capture,
            )
            # 096：method_name/out_fn/out_path 计算块已上移至 get_pred 之前
            # （run-cap 写门早期 CAP-CLASH 检测，
            # TL-DSV2-CAP-PROVENANCE-096）。

            # 076（TL-E121-OUTPUT-ID）fail-closed 写入门：tli 输出目标已存在
            # 时，须以 sidecar manifest 字节级证明同 treatment 才允许重写
            # （同配置重跑幂等 = SKIP 语义；异配置/无记录 → 拒绝运行，不覆
            # 盖）。先写 sidecar 再写数据：中断只留孤儿 sidecar，下次目标
            # 不存在 → 门自然放行。非 tli 方法可读段已编码其全部旋钮
            # （quest/twia/tia/none），维持旧覆盖语义（076 审计范围 = tli
            # treatment 矩阵）。
            # 079（TL-E121-OUTPUT-ID-079）：本门消费的 manifest 已升级为
            # resolved manifest（tia_enable_async_topk 生效值 + tli_proj_basis/
            # tli_layer_skip_path 的 realpath+内容 sha256+shape/n_skip，文件
            # 缺失/损坏 fail closed）——「同路径、内容已变」不再同 manifest，
            # 写门必拒；与 method hash/RULER receipt 共用 info.py 单一事实源。
            # 081：manifest 从冻结 snapshot 派生（与 472 行 method_name、
            # runtime 各层消费的 skip/basis 同一对象）——删除生成后重读
            # 路径的调用，生成窗口内的文件替换不再产生 name/sidecar 混装。
            if args.method == "tli":
                manifest_json = get_treatment_manifest_json(args, snapshot)
                sidecar = gate_output_treatment_identity(out_path, manifest_json)
                with open(sidecar, "w", encoding="utf-8") as f:
                    f.write(manifest_json)

            # 096（TL-DSV2-CAP-PROVENANCE-096）：先落 generating manifest 再写
            # 数据（076 tli sidecar 同款崩溃语义：数据写之前 manifest 已在盘上，
            # 中断窗口留 manifest 可证 cap，不留无主数据；status=generating 告知
            # 消费侧结果 sha 未绑定）。写门 + 两阶段均在 run_cap_manifest.py。
            write_run_cap_manifest(runcap_sidecar, finalize_run_cap_manifest(
                runcap_core, status="generating", n_samples=len(preds),
                input_tokens=run_stats.get("input_tokens"),
                truncated_flags=run_stats.get("truncated"),
                result_file=os.path.basename(out_path),
                result_sha256=None, result_lines=None,
                input_ids_sha256=run_stats.get("input_ids_sha256")))

            with open(out_path, "w", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write("\n")

            # 096：生成完成后回填结果文件 sha256 + 行数（status generating →
            # complete；原子替换，读侧永不见半文件）。
            runcap_sha, runcap_lines = compute_result_binding(out_path)
            write_run_cap_manifest(runcap_sidecar, finalize_run_cap_manifest(
                runcap_core, status="complete", n_samples=len(preds),
                input_tokens=run_stats.get("input_tokens"),
                truncated_flags=run_stats.get("truncated"),
                result_file=os.path.basename(out_path),
                result_sha256=runcap_sha, result_lines=runcap_lines,
                input_ids_sha256=run_stats.get("input_ids_sha256")))

        budgets = [pred["budget"] for pred in preds if pred["budget"] is not None]
        ave_budget = sum(budgets) / len(budgets) if len(budgets) > 0 else None
        print(f"ave_budget: {ave_budget}")
