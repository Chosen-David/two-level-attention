# res_dict → <dataset>_agreement.pkl 聚合转换（deepseek 2026-10-13）
# 官方 identify_important_fre.py 只 dump res_dict（字典嵌套），Phase 2 的
# constructe_adaptive_selection 消费 <dataset>_agreement.pkl 形状
# [budget_num, layer_num, frequency_group_num, head_num]。官方仓库未发布
# 转换脚本，这里补上（语义：每 (budget, freq_group, layer) 上所有样本的
# top-k 交集比例取平均，budget 维度顺序 = sparsity_list）。
import argparse
import os

import dill
import numpy as np
import torch

SPARSITY_LIST = [128, 256, 512, 1024, 2048, 4096]  # 必须与 Phase 1 一致
BUDGET_DICT = {128: 0, 256: 1, 512: 2, 1024: 3, 2048: 4, 4096: 5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen3-8B")
    parser.add_argument("--dataset_name", type=str, default="qasper")
    parser.add_argument("--evaluate_num", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=36,
                        help="Qwen3-8B = 36，其它模型按 config 调整")
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--out_root", type=str, default="dimension_rankings")
    args = parser.parse_args()

    src = os.path.join(args.out_root, args.model_name,
                       f"{args.dataset_name}_{args.evaluate_num}.pkl")
    with open(src, "rb") as f:
        res_dict = dill.load(f)

    freq_group_num = args.head_dim // 2
    budget_num = len(SPARSITY_LIST)
    layer_num = args.num_layers
    head_num = args.num_heads

    agreement = np.zeros((budget_num, layer_num, freq_group_num, head_num),
                         dtype=np.float32)
    for b_idx, sparsity in enumerate(SPARSITY_LIST):
        if sparsity not in res_dict:
            print(f"[convert] 警告：budget {sparsity} 无数据")
            continue
        for fg_idx in range(freq_group_num):
            if fg_idx not in res_dict[sparsity]:
                continue
            for layer_idx in range(layer_num):
                samples = res_dict[sparsity][fg_idx][layer_idx]
                if not samples:
                    continue
                arr = np.asarray(samples, dtype=np.float32)  # [n_samples, head_num]
                agreement[b_idx, layer_idx, fg_idx] = arr.mean(axis=0)

    dst = os.path.join(args.out_root, args.model_name,
                       f"{args.dataset_name}_agreement.pkl")
    # 官方 utils.py 消费口径：agreement 为 torch.Tensor（adaptive_selection_*
    # 内调用 .numpy()）。官方识别脚本 dump 的是 res_dict（list 聚合），
    # 转换时须输出 torch.Tensor。
    agreement_t = torch.from_numpy(agreement)
    with open(dst, "wb") as f:
        dill.dump(agreement_t, f)
    print(f"[convert] {src} -> {dst} shape={tuple(agreement_t.shape)} "
          f"dtype={agreement_t.dtype}")


if __name__ == "__main__":
    main()
