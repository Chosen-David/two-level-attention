import os
import torch
import json
import hashlib
from datetime import datetime
from argparse import ArgumentParser
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sparse_attn.arguments import add_sparse_attn_args
from sparse_attn.patches import register_patch
from sparse_attn.metrics import get_metrics
# 【095 修复（GPT 审计 2026-10-12_0527，2026-10-12）】deferred error flag
# 的请求级消费点——lm_eval 生成段（simple_evaluate）边界接线，guard 退出
# 校验 raise 必须先于 save_results 任何结果落盘。
from sparse_attn.ops.eager_decoding import eager_decoding_request_guard
from sparse_attn.info import (
    get_method_name_with_info,
    resolve_treatment_snapshot,
    TREATMENT_MANIFEST_SIDECAR_SUFFIX,
)

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def build_result_filename(method_name, max_length, tasks, timestamp):
    """【106】输出文件名身份 = method + cap + 任务集摘要 + 时间戳。

    修复前文件名只含 method_name + 秒级时间戳——同 method 同秒先后用
    16384 与 31500 跑，路径完全相同可互相覆盖（096 审计同型缺口在
    RULER 评估入口的残留；GPT 审计 2026-10-12_0739
    TL-E096-RULER-CAP-IDENTITY-106，P2）。修复后 cap 或任务集任一不同
    则文件名必不同；结果身份由 save_results 的 receipt 同代绑定
    （result sha/行数/cap 回执），原子落盘防半文件。
    """
    tasks_sha8 = hashlib.sha256(
        json.dumps(sorted(tasks), ensure_ascii=False,
                   sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return (f"{method_name}_L{max_length}_T{tasks_sha8}_{timestamp}")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen3-8B") # meta-llama/Meta-Llama-3.1-8B-Instruct
    parser.add_argument('--model', type=str, default="hf_models/downloads/Qwen/Qwen3-8B") # meta-llama/Meta-Llama-3.1-8B-Instruct
    parser.add_argument('--lm_eval_batch_size', type=int, default=1)
    parser.add_argument('--tasks', nargs='+',
        default=[
            "niah_single_1", "niah_single_2",
            "niah_multikey_1", "niah_multikey_2",
            "niah_multiquery", "niah_multivalue",
            # "ruler_qa_squad", "ruler_qa_hotpot",
        ],
        help='Tasks to evaluate on LM Eval.'
    )
    # "arc_easy", "arc_challenge", "boolq", "hellaswag", "lambada_openai", "piqa", "siqa", "winogrande",
    parser.add_argument('--max_length', type=int, default=1024 * 64)
    parser.add_argument('--output_dir', type=str, default="./exp/results_ruler", help='Directory to save evaluation results')

    add_sparse_attn_args(parser)
    
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    return args

@torch.no_grad()
def main():
    args = parse_args()
    if args.method == "tli":
        raise RuntimeError(
            "llm_eval TLI cache path has not been migrated; use "
            "benchmark/RULER/pred_ruler.py with PagedRegionCache")
    print("model: {}".format(args.model))

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()

    snapshot = resolve_treatment_snapshot(args) if args.method == "tli" else None
    method_name = get_method_name_with_info(args, snapshot)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = build_result_filename(
        method_name, args.max_length, args.tasks, timestamp)

    prompts = [
        [
            {
                "role": "user", "content": (
                    "Please help me implement the python code that print \"Hello World!\"."
                )
            },
        ],
    ]
    formatted_prompts = [tokenizer.apply_chat_template(messages, tokenize=False) for messages in prompts]

    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=16, do_sample=True)
    results = tokenizer.batch_decode(outputs)
    print(results)

    register_patch(model, args, snapshot)

    import lm_eval
    from lm_eval import utils as lm_eval_utils
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, max_batch_size=args.lm_eval_batch_size, max_length=args.max_length)

    task_manager = TaskManager(
        include_path="./benchmark/RULER/lm_eval_tasks", 
        include_defaults=False,
        metadata={"max_seq_lengths": [8192, 8192 * 2, 8192 * 4], "pretrained": args.model}
    )
    task_names = lm_eval_utils.pattern_match(args.tasks, task_manager.all_tasks)

    print("task_names: {}".format(task_names))
    # 【095 修复】TLI_E127_FAST=1 的 gather 路径契约违规延迟到 GPU 端错误
    # 标志，由 guard 在生成段（simple_evaluate 全程）边界一次校验 raise
    # ——异常传播后 save_results 不可达，无任何完成外观结果产物落盘。
    with eager_decoding_request_guard():
        result_dict = lm_eval.simple_evaluate(
            hflm, tasks=task_names,
            batch_size=args.lm_eval_batch_size,
            max_batch_size=args.lm_eval_batch_size,
            task_manager=task_manager,
            confirm_run_unsafe_code=True,
        )
    result_table = lm_eval_utils.make_table(result_dict)
    
    # 获取稀疏注意力指标
    metrics = {
        "mean_select_tokens": get_metrics().get_select_tokens(),
        "k_delta_max": get_metrics().get_k_delta_max(),
        "k_delta_mean": get_metrics().get_k_delta_mean()
    }
    
    # 打印结果
    print(args.model)
    print(result_table)
    print("mean select tokens:", metrics["mean_select_tokens"])
    
    # 保存结果到文件
    save_results(args, result_filename, result_dict, result_table, metrics,
                 snapshot, method_name)

def save_results(args, filename, result_dict, result_table, metrics,
                 snapshot=None, method_name=None):
    """保存评估结果到文件

    【106】结果落盘三件套（全部原子 tmp+os.replace——读侧永不见半文件）：
      ① 结果 txt（文件名含 cap/任务集身份，见 main 内 106 注记）；
      ② 结果 receipt（{txt}.receipt.json）：result sha256 + 行数/字节数 +
         cap 回执（max_length 原值）+ 任务集清单与摘要 + 081 treatment
         manifest sha——结果身份可独立复核，不靠文件名单点承载；
      ③ 082 treatment manifest sidecar（tli 臂）。
    """

    # 保存文本格式的结果表格
    txt_path = os.path.join(args.output_dir, f"{filename}.txt")
    body = (
        f"Model: {args.model}\n"
        f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Tasks: {', '.join(args.tasks)}\n"
        f"Batch Size: {args.lm_eval_batch_size}\n"
        f"Max Length: {args.max_length}\n"
        + "\n" + "="*80 + "\n\n"
        + "Evaluation Results:\n"
        + str(result_table)
        + "\n\n" + "="*80 + "\n\n"
        + "Sparse Attention Metrics:\n"
        + f"Mean Select Tokens: {metrics['mean_select_tokens']}\n"
    )
    # 106：原子落盘（tmp + os.replace）——崩溃/并发窗口不留半文件被当
    # 完成产物消费。
    tmp = txt_path + f".tmp-{os.getpid()}"
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(body)
    os.replace(tmp, txt_path)
    print(f"Text results saved to: {txt_path}")

    # 106：结果 receipt——result sha/行数/cap 回执同代绑定（先落 txt 再算
    # sha，receipt 与最终文件字节级一致）。
    txt_bytes = open(txt_path, "rb").read()
    receipt = {
        "schema": "ruler-eval-receipt-v1",
        "created_utc": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_name": args.model_name,
        "model": args.model,
        "method_name": method_name,
        "cap": args.max_length,          # 106：cap 回执原值进身份绑定
        "tasks": sorted(args.tasks),
        "batch_size": args.lm_eval_batch_size,
        "result_file": os.path.basename(txt_path),
        "result_sha256": hashlib.sha256(txt_bytes).hexdigest(),
        "result_lines": txt_bytes.count(b"\n"),
        "result_bytes": len(txt_bytes),
        "treatment_manifest_sha256": None,
    }

    # 082：tli 臂结果旁挂 canonical treatment manifest sidecar（与
    # LongBench 写门/sidecar 同一 manifest 对象、同一后缀语义）——
    # 结果身份不靠短哈希单点承载，可独立复核。106：原子落盘 + sha 进
    # receipt。
    if snapshot is not None:
        sidecar_path = txt_path + TREATMENT_MANIFEST_SIDECAR_SUFFIX
        tmp = sidecar_path + f".tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(snapshot.manifest_json)
        os.replace(tmp, sidecar_path)
        receipt["treatment_manifest_sha256"] = hashlib.sha256(
            snapshot.manifest_json.encode("utf-8")).hexdigest()
        print(f"Treatment manifest saved to: {sidecar_path}")

    receipt_path = txt_path + ".receipt.json"
    tmp = receipt_path + f".tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=4, sort_keys=True)
    os.replace(tmp, receipt_path)
    print(f"Result receipt saved to: {receipt_path}")
    

if __name__ == '__main__':
    main()
