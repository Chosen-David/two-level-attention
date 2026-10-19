# Vendor baseline LaTeX 片段（#31，2026-10-19 deepseek 轨）

可直接粘贴进 `sglang/paper/TLI_paper_en.tex` 的 Experiments 部分（建议插在
`tab:longbench` 之后、`tab:betacross` 之前，作为独立统一批次表；或在
`\subsection{End-to-End Quality}` 内新增小节）。口径警示见蓝本：vendor 批为
独立统一批次（同批 FullKV 作内参照），不与 PSI 主表（PSI 实验批）混同。

## 表 A：RULER 32K 逐任务（vendor 统一批，AVG11，n=100）

- 列顺序对齐蓝本：FullKV / TLI-mavg / JSSA / AsyncTLS / FASA / Quest。
- TLI-mavg 列用 e116b 正式批 59.65（**PSI 主臂数字归 Opus 确认**，如需以论文
  PSI@1024=60.78 替换，只改该列 AVG 与标题即可）。
- JSSA=59.13（e119_ruler32768_formal_jssa_vendor，2026-10-18 21:45 发布）；
  AsyncTLS=50.09（e119_ruler32768_formal_asynctls_vendor_final）；
  FASA=18.64（e119_ruler32768_formal_fasa_vendor_10182330，2026-10-19 08:23 发布，
  完整重跑批 formal 口径）；Quest=42.85（e119_ruler32k_formal_quest_vendor）。

```latex
\begin{table}[h]\centering
\small\setlength{\tabcolsep}{4pt}
\caption{RULER 32K per-task breakdown over the unified vendor batch (string\_match\_all $\times$100, n=100 per task; FullKV and TLI-mavg run in the same batch as internal references; JSSA = official subspace method at budget 1024/rank 32; AsyncTLS = block channel coarse selection at kt=512/b64/kb128; FASA = offline-calibrated page selection at budget 1024/qasper/t0.8; Quest = the lower-bound family reference at budget 1024. The mavg column is the two-level indexer's same-batch reference (the paper's PSI main-table arm number to be filled by the method track).}
\label{tab:ruler32kvendor}
\begin{tabular}{lcccccc}
\toprule
 & FullKV & TLI-mavg & JSSA & AsyncTLS & FASA & Quest \\
\midrule
niah\_single\_1 & 100.00 & 100.00 & 100.00 & 94.00 & 10.00 & 41.00 \\
niah\_single\_2 & 100.00 & 100.00 & 100.00 & 89.00 & 73.00 & 96.00 \\
niah\_single\_3 & 100.00 & 98.00 & 100.00 & 58.00 & 10.00 & 36.00 \\
niah\_multikey\_1 & 99.00 & 98.00 & 99.00 & 95.00 & 50.00 & 87.00 \\
niah\_multikey\_2 & 76.00 & 84.00 & 76.00 & 43.00 & 2.00 & 44.00 \\
niah\_multikey\_3 & 4.00 & 8.00 & 2.00 & 0.00 & 0.00 & 0.00 \\
niah\_multiquery & 24.50 & 24.50 & 24.50 & 38.25 & 19.75 & 20.00 \\
niah\_multivalue & 25.00 & 28.50 & 25.75 & 47.50 & 23.50 & 32.00 \\
cwe & 83.50 & 71.30 & 85.50 & 30.20 & 8.40 & 44.60 \\
fwe & 7.00 & 11.67 & 2.67 & 15.67 & 3.00 & 18.00 \\
vt & 34.20 & 32.20 & 35.00 & 40.40 & 5.40 & 52.80 \\
\addlinespace
AVG11 & 59.38 & 59.65 & 59.13 & 50.09 & 18.64 & 42.85 \\
\bottomrule
\end{tabular}
\end{table}
```

## 表 B：LongBench 13 任务（vendor 统一批，AVG13）

- 来源：FullKV=`fullkv/pred/result.json`；JSSA=`pred_jssa_vendor_qwen3_8b_k1024_r32/result.json`
  （2026-10-19 03:36，16/16 strict runcap）；AsyncTLS=`pred_asynctls/`；Quest=`pred_qv_1024_16/`；
  FASA=`exp/results_longbench_fasa/Qwen3-8B/pred_fasa_1024/pred/`。

```latex
\begin{table}[h]\centering
\small\setlength{\tabcolsep}{4pt}
\caption{LongBench 13-subset table over the unified vendor batch (official per-task metric averaged; same-batch FullKV as internal reference; budget protocols as in Table~\ref{tab:ruler32kvendor}, with JSSA/AsyncTLS at their LongBench budgets 1024; the 16-task averages are FullKV 49.39 / JSSA 49.28 / AsyncTLS 44.97 / Quest 41.35 / FASA 33.31).}
\label{tab:longbenchvendor}
\begin{tabular}{lccccc}
\toprule
 & FullKV & JSSA & AsyncTLS & Quest & FASA \\
\midrule
hotpotqa & 59.07 & 60.20 & 49.03 & 45.09 & 12.58 \\
2wikimqa & 43.29 & 42.34 & 38.31 & 30.78 & 24.73 \\
musique & 35.44 & 34.25 & 20.99 & 23.12 & 4.15 \\
passage\_retrieval\_en & 100.00 & 100.00 & 95.50 & 88.25 & 60.12 \\
qasper & 47.47 & 48.16 & 45.41 & 36.43 & 34.85 \\
multifieldqa\_en & 53.19 & 53.83 & 50.74 & 47.86 & 37.90 \\
gov\_report & 33.36 & 33.58 & 32.38 & 32.32 & 28.35 \\
qmsum & 23.92 & 24.24 & 23.37 & 23.20 & 18.80 \\
multi\_news & 24.78 & 24.64 & 24.65 & 25.46 & 25.33 \\
narrativeqa & 27.59 & 26.47 & 17.85 & 18.85 & 2.53 \\
triviaqa & 91.11 & 89.82 & 81.91 & 72.73 & 71.36 \\
lcc & 68.10 & 68.37 & 67.13 & 60.23 & 61.26 \\
repobench & 64.74 & 65.54 & 57.39 & 53.63 & 44.84 \\
\addlinespace
AVG13 & 51.70 & 51.65 & 46.51 & 42.92 & 32.83 \\
\bottomrule
\end{tabular}
\end{table}
```

## 表 C：decode 端到端时延（eager torch 语义等价实现，非官方 kernel）

- 脚本 `exp/bench/vendor_e2e_latency.py`；.187 H20 143GB 单卡；合成 needle prompt、
  max\_gen=128、warmup1+repeats2-3；预算对齐主表（JSSA k1024 r32 / AsyncTLS
  kt512 b64 kb128 / FASA qasper b1024 / Quest 1024）。
- **发现**：JSSA 固定预算 1024 稀疏 decode tpot 随 L 增长最缓（32$\to$128K 1.85$\times$
  对 dense 2.57$\times$），128K 反超 dense（61.69 $<$ 66.94，$-8\%$）——与论文
  prefill 长上下文加速故事同构；Quest eager 全维 token gather 开销最大（128K 621ms）。

```latex
\begin{table}[h]\centering
\small\setlength{\tabcolsep}{4pt}
\caption{Decode end-to-end latency per token (ms, mean of measured repeats; eager torch semantic equivalents of the official kernels, not official CUDA implementations; the dense row is the same-batch internal reference; the budget protocols match the main table).}
\label{tab:perfvendor}
\begin{tabular}{lcccc}
\toprule
 & 32K tpot & 64K tpot & 128K tpot & 128K wall \\
\midrule
dense & 26.05 & 39.62 & 66.94 & 84026.6 \\
JSSA & 33.30 & 39.68 & \textbf{61.69} & 83297.2 \\
FASA & 75.64 & 133.00 & 265.91 & 108597.7 \\
AsyncTLS & 105.29 & 123.42 & 160.25 & 95778.8 \\
Quest & 177.04 & 317.38 & 621.37 & 154381.5 \\
\bottomrule
\end{tabular}
\end{table}
```

## 待 Opus/用户确认
1. TLI-mavg 列最终值（e116b 59.65 vs 记忆 59.99）与 PSI 主臂是否替换本表 mavg 列。
2. RULER 64K（AsyncTLS 34.11 / FASA 1.72）与 128K（在飞）是否并入本表或其扩展。
3. 性能表是否放 End-to-End Speed 小节（\S4.4 现文本主要讲 TLI 自身 kernel），
   vendor 表更合适应为独立小节。
