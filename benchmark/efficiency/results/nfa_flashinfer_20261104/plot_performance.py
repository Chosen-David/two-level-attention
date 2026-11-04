#!/usr/bin/env python3
"""从正式 ABBA JSON 生成 NFA / FlashInfer 论文性能图。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESULT_ROOT = Path(__file__).resolve().parent
E2E_SOURCES = tuple(RESULT_ROOT / 'e2e' / name for name in (
    'ruler32k.json',
    'ruler64k.json',
    'ruler128k_b1_graph.json',
    'ruler128k_b1_eager.json',
    'ruler128k_b2_b4.json',
))
LENGTHS = (32768, 65536, 131072)
BATCHES = (1, 2, 4)
EXECUTIONS = ('eager', 'graph')
COLORS = {
    'nfa': '#81D4FA',
    'nfa_edge': '#0B5E8E',
    'flashinfer': '#CFCECE',
    'flashinfer_edge': '#5C6466',
    'l1': '#CDEBFA',
    'l2': '#7FB3F0',
    'attention': '#3775BA',
}
FONT_PATH = Path('/home/wangyuanshuo02/texlive/texmf-local/fonts/opentype/urwbase35/NimbusRoman-Regular.otf')
if FONT_PATH.is_file():
    font_manager.fontManager.addfont(FONT_PATH)
    FONT_FAMILY = font_manager.FontProperties(fname=FONT_PATH).get_name()
else:
    FONT_FAMILY = 'DejaVu Serif'
plt.rcParams.update({
    'font.family': [FONT_FAMILY],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.linewidth': 0.7,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
    'hatch.linewidth': 0.35,
})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def source_receipt(path, data):
    return {
        'path': str(Path(path).resolve()),
        'sha256': sha256(path),
        'source_head': data.get('source', {}).get('head', data.get('repo_head')),
        'source_diff_sha256': data.get('source', {}).get(
            'diff_sha256', data.get('repo_diff_sha256')),
    }


def load_e2e(allow_partial=False):
    records = {}
    receipts = []
    for path in E2E_SOURCES:
        if not Path(path).is_file():
            if allow_partial:
                continue
            raise FileNotFoundError(f'正式 e2e 结果尚未完成: {path}')
        data = read_json(path)
        receipts.append(source_receipt(path, data))
        args = data['arguments']
        length = args['context_length']
        for row in data['comparisons']:
            key = (length, row['batch_size'], row['execution'])
            if key in records:
                raise ValueError(f'e2e 配置重复: {key}')
            config = f"B{row['batch_size']}/{row['execution']}"
            hygiene = data['gpu_hygiene']['configs'][config]['status']
            if hygiene != 'clean':
                raise ValueError(f'{path} {config} 不是 clean: {hygiene}')
            if not row['nfa_repeat_token_identity'] or not row['flashinfer_repeat_token_identity']:
                raise ValueError(f'{path} {config} ABBA repeat identity 未通过')
            arm_receipts = {}
            for arm_name, arm_receipt in data['runs'][config].items():
                original_arm_path = Path(arm_receipt['json'])
                archived_arm_path = (
                    RESULT_ROOT / 'e2e' / 'arms' / Path(path).stem
                    / original_arm_path.name)
                arm_path = (archived_arm_path if archived_arm_path.is_file()
                            else original_arm_path)
                arm = read_json(arm_path)
                if arm['decode_steps'] != 511 or arm['warmup_steps_excluded_from_steady'] != 3:
                    raise ValueError(f'{arm_path} decode/warmup 口径不符')
                if (len(arm['generated_token_ids']) != row['batch_size']
                        or any(len(tokens) != 512 for tokens in arm['generated_token_ids'])):
                    raise ValueError(f'{arm_path} 未完整生成每行 512 tokens')
                if arm['input']['sha256'] != data['input']['sha256']:
                    raise ValueError(f'{arm_path} 与汇总输入 SHA 不一致')
                arm_receipts[arm_name] = {
                    'path': str(Path(arm_path).resolve()), 'sha256': sha256(arm_path)}
            nfa_ms = row['nfa_steady_tpot_ms']
            fi_ms = row['flashinfer_steady_tpot_ms']
            nfa_median = statistics.median(nfa_ms)
            fi_median = statistics.median(fi_ms)
            if not math.isclose(nfa_median, row['nfa_steady_tpot_median_ms'], rel_tol=1e-12):
                raise ValueError(f'{path} {config} NFA median 与 repeats 不一致')
            records[key] = {
                'context_length': length,
                'batch_size': row['batch_size'],
                'execution': row['execution'],
                'nfa_tpot_ms': nfa_median,
                'flashinfer_tpot_ms': fi_median,
                'nfa_repeat_tpot_ms': nfa_ms,
                'flashinfer_repeat_tpot_ms': fi_ms,
                'nfa_throughput_tokens_per_second': 1000.0 / nfa_median,
                'flashinfer_throughput_tokens_per_second': 1000.0 / fi_median,
                'flashinfer_over_nfa_tpot': fi_median / nfa_median,
                'nfa_flashinfer_token_identity': row['nfa_flashinfer_token_identity'],
                'nfa_flashinfer_first_divergence': row['nfa_flashinfer_first_divergence'],
                'nfa_skip_policy_counts': row.get('nfa_skip_policy_counts'),
                'nfa_skip_policy_layers': row.get('nfa_skip_policy_layers'),
                'gpu_hygiene': hygiene,
                'input': data['input'],
                'source': receipts[-1],
                'arm_receipts': arm_receipts,
            }
    expected = {(length, batch, execution) for length in LENGTHS
                for batch in BATCHES for execution in EXECUTIONS}
    missing = sorted(expected - records.keys())
    if missing and not allow_partial:
        raise ValueError(f'e2e 配置不完整: {missing}')
    return [records[key] for key in sorted(records)], receipts, missing


def load_operators(root, allow_partial=False):
    root = Path(root)
    records = []
    receipts = []
    for length in (32768, 65536, 98304, 131072):
        for execution in EXECUTIONS:
            arms = {}
            for method, abba in (('nfa', 'a1'), ('flashinfer', 'b1'),
                                 ('flashinfer', 'b2'), ('nfa', 'a2')):
                path = root / f'niah_multikey_1_s0_{length}_{execution}_{abba}.json'
                if not path.is_file():
                    if allow_partial:
                        continue
                    raise FileNotFoundError(f'纯算子结果尚未完成: {path}')
                data = read_json(path)
                if data.get('schema') != 'nfa-pure-kernel-stages-v1':
                    raise ValueError(f'{path} schema 不符')
                if data.get('gpu_hygiene') != 'clean':
                    raise ValueError(f'{path} 不是 clean')
                if data['method'] != method or data['execution'] != execution:
                    raise ValueError(f'{path} arm 身份不符')
                if data['warmup_steps'] != 3 or data['repeat'] != 5:
                    raise ValueError(f'{path} warmup/repeat 口径不符')
                arms[abba] = data
                receipts.append(source_receipt(path, data))
            if len(arms) != 4:
                continue
            input_sha = {arm['input']['sha256'] for arm in arms.values()}
            if len(input_sha) != 1:
                raise ValueError(f'{length} {execution} ABBA 输入 SHA 不一致')
            stages = {}
            for method, names, abbas in (
                    ('nfa', ('l1_ms', 'l2_ms', 'main_attention_ms'), ('a1', 'a2')),
                    ('flashinfer', ('flashinfer_dense_ms',), ('b1', 'b2'))):
                for name in names:
                    arm_medians = [arms[abba]['summary'][name]['median_ms'] for abba in abbas]
                    stages[name] = statistics.median(arm_medians)
                    stages[f'{name}_arm_medians_ms'] = arm_medians
            nfa_total = stages['l1_ms'] + stages['l2_ms'] + stages['main_attention_ms']
            if nfa_total <= 0 or stages['flashinfer_dense_ms'] <= 0:
                raise ValueError(f'{length} {execution} 纯算子计时出现非正数')
            source_identities = {
                (arm['repo_head'], arm['repo_diff_sha256'])
                for arm in arms.values()}
            if len(source_identities) != 1:
                raise ValueError(f'{length} {execution} ABBA 源码 receipt 不一致')
            row = {
                'context_length': length,
                'execution': execution,
                **stages,
                'nfa_operator_total_ms': nfa_total,
                'flashinfer_over_nfa_operator': stages['flashinfer_dense_ms'] / nfa_total,
                'nfa_skip': arms['a1']['metadata']['skip'],
                'input': arms['a1']['input'],
                'gpu_hygiene': 'clean',
                'source_paths': [arms[abba]['output'] if 'output' in arms[abba]
                                 else str(root / f'niah_multikey_1_s0_{length}_{execution}_{abba}.json')
                                 for abba in ('a1', 'b1', 'b2', 'a2')],
            }
            for abba in ('a1', 'a2'):
                for repeat in arms[abba]['repeats']:
                    values = repeat['stages']
                    expected_total = values['l1_ms'] + values['l2_ms'] + values['main_attention_ms']
                    if not math.isclose(values['nfa_operator_total_ms'], expected_total,
                                        rel_tol=1e-9, abs_tol=1e-9):
                        raise ValueError(f'{length} {execution} {abba} NFA 总耗时不是阶段和')
            records.append(row)
    return records, receipts


def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#E4E9EC', linewidth=0.6, linestyle=(0, (2, 2)))
    ax.set_axisbelow(True)
    ax.tick_params(width=0.7, length=3)


def save_figure(fig, stem, output_dir):
    output_dir = Path(output_dir)
    for suffix in ('png', 'pdf'):
        path = output_dir / f'{stem}.{suffix}'
        fig.savefig(path, bbox_inches='tight', pad_inches=0.04,
                    dpi=300 if suffix == 'png' else None)
    plt.close(fig)
    print(f'saved {stem}.png/.pdf')


def e2e_figure(rows, execution, metric, output_dir, index):
    selected = [row for row in rows if row['execution'] == execution]
    if not selected:
        return None
    positions = {(length, batch): i for i, (length, batch) in enumerate(
        (pair for length in LENGTHS for pair in ((length, batch) for batch in BATCHES)))}
    selected.sort(key=lambda row: positions[(row['context_length'], row['batch_size'])])
    labels = [f"{row['context_length'] // 1024}K · B{row['batch_size']}" for row in selected]
    x = np.arange(len(selected), dtype=float)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 3.55), constrained_layout=True)
    style_axis(ax)
    if metric == 'tpot':
        nfa_key, fi_key = 'nfa_tpot_ms', 'flashinfer_tpot_ms'
        nfa_repeat = [row['nfa_repeat_tpot_ms'] for row in selected]
        fi_repeat = [row['flashinfer_repeat_tpot_ms'] for row in selected]
        ylabel = 'Steady decode TPOT (ms/token)'
        stem = f'E2E-{index:02d}_{execution}_tpot'
    else:
        nfa_key = 'nfa_throughput_tokens_per_second'
        fi_key = 'flashinfer_throughput_tokens_per_second'
        nfa_repeat = [[1000.0 / value for value in row['nfa_repeat_tpot_ms']] for row in selected]
        fi_repeat = [[1000.0 / value for value in row['flashinfer_repeat_tpot_ms']] for row in selected]
        ylabel = 'Per-request decode throughput (tokens/s)'
        stem = f'E2E-{index:02d}_{execution}_throughput'
    nfa_values = np.array([row[nfa_key] for row in selected])
    fi_values = np.array([row[fi_key] for row in selected])
    nfa_range = np.array([(nfa_values - np.min(nfa_repeat, axis=1)),
                          (np.max(nfa_repeat, axis=1) - nfa_values)])
    fi_range = np.array([(fi_values - np.min(fi_repeat, axis=1)),
                         (np.max(fi_repeat, axis=1) - fi_values)])
    bars_nfa = ax.bar(x - width / 2, nfa_values, width, color=COLORS['nfa'],
                      edgecolor=COLORS['nfa_edge'], linewidth=0.8, hatch='///',
                      yerr=nfa_range, capsize=2.5, error_kw={'linewidth': 0.7})
    bars_fi = ax.bar(x + width / 2, fi_values, width, color=COLORS['flashinfer'],
                     edgecolor=COLORS['flashinfer_edge'], linewidth=0.8,
                     yerr=fi_range, capsize=2.5, error_kw={'linewidth': 0.7})
    ax.bar_label(bars_nfa, fmt='%.1f', padding=3, fontsize=6.8)
    ax.bar_label(bars_fi, fmt='%.1f', padding=3, fontsize=6.8)
    ax.set_xticks(x, labels)
    ax.set_xlabel('Native RULER context length · batch size')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(np.max(nfa_repeat), np.max(fi_repeat)) * 1.24)
    handles = [
        Patch(facecolor=COLORS['nfa'], edgecolor=COLORS['nfa_edge'], hatch='///', label='NFA Sparse'),
        Patch(facecolor=COLORS['flashinfer'], edgecolor=COLORS['flashinfer_edge'], label='FlashInfer Dense'),
        Line2D([0], [0], color='#4B555B', marker='|', markersize=7,
               linewidth=0.7, linestyle='None', label='ABBA repeat range'),
    ]
    ax.legend(handles=handles, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.01),
              frameon=False, columnspacing=1.3, handlelength=1.6)
    save_figure(fig, stem, output_dir)
    return {
        'execution': execution,
        'metric': metric,
        'configurations': [
            {'context_length': row['context_length'], 'batch_size': row['batch_size'],
             'nfa': row[nfa_key], 'flashinfer': row[fi_key],
             'nfa_repeats': row['nfa_repeat_tpot_ms'] if metric == 'tpot' else nfa_repeat[i],
             'flashinfer_repeats': row['flashinfer_repeat_tpot_ms'] if metric == 'tpot' else fi_repeat[i]}
            for i, row in enumerate(selected)],
    }


def operator_figure(rows, execution, output_dir, index):
    selected = sorted((row for row in rows if row['execution'] == execution),
                      key=lambda row: row['context_length'])
    if not selected:
        return None
    x = np.arange(len(selected), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 3.65), constrained_layout=True)
    style_axis(ax)
    l1 = np.array([row['l1_ms'] for row in selected])
    l2 = np.array([row['l2_ms'] for row in selected])
    attention = np.array([row['main_attention_ms'] for row in selected])
    total = l1 + l2 + attention
    fi = np.array([row['flashinfer_dense_ms'] for row in selected])
    left = x - width / 2
    ax.bar(left, l1, width, color=COLORS['l1'], edgecolor=COLORS['nfa_edge'], linewidth=0.75)
    ax.bar(left, l2, width, bottom=l1, color=COLORS['l2'],
           edgecolor=COLORS['nfa_edge'], linewidth=0.75)
    ax.bar(left, attention, width, bottom=l1 + l2, color=COLORS['attention'],
           edgecolor=COLORS['nfa_edge'], linewidth=0.75, hatch='///')
    fi_bars = ax.bar(x + width / 2, fi, width, color=COLORS['flashinfer'],
                     edgecolor=COLORS['flashinfer_edge'], linewidth=0.8)
    for i, value in enumerate(total):
        ax.text(left[i], value * 1.018, f'{value:.2f}', ha='center', va='bottom', fontsize=7)
    ax.bar_label(fi_bars, fmt='%.2f', padding=3, fontsize=7)
    ax.set_xticks(x, [f"{row['context_length'] // 1024}K" for row in selected])
    ax.set_xlabel('Input context length')
    ax.set_ylabel('Pure GPU kernel time (ms/decode step)')
    ax.set_ylim(0, max(total.max(), fi.max()) * 1.23)
    handles = [
        Patch(facecolor=COLORS['l1'], edgecolor=COLORS['nfa_edge'], label='NFA L1'),
        Patch(facecolor=COLORS['l2'], edgecolor=COLORS['nfa_edge'], label='NFA L2'),
        Patch(facecolor=COLORS['attention'], edgecolor=COLORS['nfa_edge'], hatch='///',
              label='NFA main Attention'),
        Patch(facecolor=COLORS['flashinfer'], edgecolor=COLORS['flashinfer_edge'],
              label='FlashInfer Dense'),
    ]
    ax.legend(handles=handles, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.01),
              frameon=False, columnspacing=1.1, handlelength=1.5)
    stem = f'OP-{index:02d}_{execution}_stage_breakdown'
    save_figure(fig, stem, output_dir)
    return {
        'execution': execution,
        'nfa_total_definition': 'sum of the two-arm medians of L1, L2 and main Attention',
        'configurations': selected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('e2e', 'operators', 'all'), default='all')
    parser.add_argument('--kernel-root', default=RESULT_ROOT / 'operators' / 'natural')
    parser.add_argument('--output-dir', default=RESULT_ROOT / 'figures')
    parser.add_argument('--allow-partial', action='store_true',
                        help='仅用于明确隔离的预览；正式交付不得使用')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        'schema': 'nfa-flashinfer-performance-figures-v1',
        'partial_preview': args.allow_partial,
        'font_family': FONT_FAMILY,
        'color_semantics': {
            'nfa_sparse': 'blue fill with diagonal hatch',
            'flashinfer_dense': 'neutral gray fill',
            'error_bars': 'range of two measured ABBA repeats',
        },
        'correctness_note': 'Cross-method token identity is reported, never required or hidden; '
                            'NFA is a sparse approximation of FlashInfer Dense.',
        'skip_note': 'Single-sided near/far skip restricts the L1/L2 candidate range; '
                     'the middle quota can be refilled from the other region, so visible_lens '
                     'need not decrease. Both-skip uses the protected-only path.',
        'figures': {},
    }
    if args.mode in ('e2e', 'all'):
        rows, receipts, missing = load_e2e(args.allow_partial)
        stats['e2e'] = {
            'scope': 'complete decode; 512 generated tokens; first 3 steps excluded; '
                     'steady TPOT median of two ABBA measured repeats; prefill excluded',
            'throughput_scope': 'per-request tokens/s = 1000 / TPOT; not batch-aggregated',
            'missing_configurations': [list(item) for item in missing],
            'sources': receipts,
            'records': rows,
        }
        index = 1
        for execution in EXECUTIONS:
            for metric in ('tpot', 'throughput'):
                result = e2e_figure(rows, execution, metric, output_dir, index)
                if result is not None:
                    stats['figures'][f'E2E-{index:02d}'] = result
                index += 1
    if args.mode in ('operators', 'all'):
        rows, receipts = load_operators(args.kernel_root, args.allow_partial)
        stats['operators'] = {
            'scope': 'pure CUDA device kernels; 3 warmup + 5 profiled repeats per arm; '
                     'ABBA arm medians; no host or other model kernels',
            'sources': receipts,
            'records': rows,
        }
        for index, execution in enumerate(EXECUTIONS, start=1):
            result = operator_figure(rows, execution, output_dir, index)
            if result is not None:
                stats['figures'][f'OP-{index:02d}'] = result
    with (output_dir / 'stats.json').open('w', encoding='utf-8') as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print(f'saved stats -> {output_dir / "stats.json"}')


if __name__ == '__main__':
    main()
