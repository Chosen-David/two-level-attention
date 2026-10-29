"""Compact reproducible result artifacts; detailed cache traces stay in raw files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_tsv(path,columns,rows):
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream,delimiter='\t');writer.writerow(columns);writer.writerows(rows)


def export(outdir,plot=False):
    outdir=Path(outdir);summary=json.loads((outdir/'summary.json').read_text())
    validation=json.loads((outdir/'validation.json').read_text())
    assert summary['status']=='complete' and validation['status']=='complete-validated'
    assert summary['identity']==validation['identity']
    tasks=summary['protocol']['datasets'];arms=list(summary['arms'])
    score_rows=[]
    for arm,a in summary['arms'].items():
        comparison=summary['comparisons'].get(arm+'-vs-full',{}).get('avg5')
        score_rows.append([arm]+[a['per_task'][ds] for ds in tasks]+[a['avg5'],
            comparison['delta_pp'] if comparison else 0,
            comparison['ci95_pp'][0] if comparison else '',comparison['ci95_pp'][1] if comparison else ''])
    write_tsv(outdir/'scores.tsv',['arm']+tasks+['AVG5','delta_vs_full_pp','ci95_low_pp','ci95_high_pp'],score_rows)
    paired=[]
    for name,c in summary['comparisons'].items():
        for ds,r in c['per_task'].items():paired.append([name,ds,r['n'],r['delta_pp'],*r['ci95_pp'],r['degraded'],r['improved']])
        r=c['avg5'];paired.append([name,'AVG5',summary['completed_requests'],r['delta_pp'],*r['ci95_pp'],'',''])
    write_tsv(outdir/'paired_differences.tsv',['comparison','scope','n','delta_pp','ci95_low_pp','ci95_high_pp','degraded','improved'],paired)
    activation=[]
    for arm,a in summary['arms'].items():
        activation.append([arm,*[a['gate_counts'][s] for s in ['keep','near','far','both']],
            a['near_active_requests'],a['far_active_requests'],a['frozen_kv_ratio_mean'],
            a['frozen_attention_token_ratio_mean'],a['near_only_layer_kv_ratio_mean'],
            a['near_only_layer_attention_ratio_mean'],a['exact_full_generation'],a['far_decisions_differ_from_far_only']])
    write_tsv(outdir/'activation_storage.tsv',['arm','keep_layers','near_only_layers','far_only_layers','both_layers',
        'near_active_requests','far_active_requests','frozen_resident_KV_ratio','frozen_attention_token_ratio',
        'near_only_layer_KV_ratio','near_only_layer_attention_token_ratio','exact_full_generations','changed_far_decisions_vs_far_only'],activation)
    task_activation=[]
    for arm,a in summary['arms'].items():
        for ds,r in a['per_task_activation'].items():
            task_activation.append([arm,ds,r['n'],r['near_active'],r['far_active'],r['exact_full_generation']])
    write_tsv(outdir/'per_task_activation.tsv',['arm','dataset','n','near_active_requests','far_active_requests','exact_full_generation'],task_activation)
    raw_index=[];row_scores=[]
    with (outdir/'predictions.jsonl').open('w') as stream:
        for path in sorted(outdir.glob('requests/*/*.json')):
            record=json.loads(path.read_text());assert record['status']=='complete'
            assert record['identity']==summary['identity']
            raw_index.append(dict(path=str(path.relative_to(outdir)),sha256=digest(path),bytes=path.stat().st_size,
                dataset=record['dataset'],row_index=record['row_index'],input_sha256=record['input']['input_sha256']))
            for arm,a in record['arms'].items():
                row_scores.append([record['dataset'],record['row_index'],arm,a['score']*100,
                    (a['score']-record['arms']['full']['score'])*100,record['input']['input_tokens'],
                    a['generated_tokens'],a['gate_counts']['near'],a['gate_counts']['far'],a['gate_counts']['both']])
                stream.write(json.dumps(dict(dataset=record['dataset'],row_index=record['row_index'],arm=arm,
                    input_sha256=record['input']['input_sha256'],prediction=a['prediction'],answers=record['answers'],
                    score=a['score'],generated_ids=a['generated_ids'],gate_counts=a['gate_counts'],
                    raw_path=str(path.relative_to(outdir))),ensure_ascii=False)+'\n')
    write_tsv(outdir/'per_request_scores.tsv',['dataset','row_index','arm','score','delta_vs_full_pp','input_tokens',
        'generated_tokens','near_only_layers','far_only_layers','both_layers'],row_scores)
    (outdir/'raw_index.json').write_text(json.dumps(dict(root=str(outdir.resolve()),identity=summary['identity'],
        records=raw_index,note='Detailed cache traces are retained locally and indexed by hash; compact predictions are versioned.'),indent=2)+'\n')
    report_path=outdir/'REPORT.md'
    report=report_path.read_text()
    marker='<!-- frozen-summary:end -->'
    if marker in report:report=report.split(marker,1)[1].lstrip()
    explanation=[f"本轮已完成：{len(tasks)} 个海选任务各 {summary['protocol']['samples_per_task']} 条、{summary['completed_requests']} 条配对请求、{len(row_scores)} 组生成；这不是每任务 200/500 条的全量海选。",'']
    for threshold in ['0.1','0.15']:
        arm='dual_'+threshold;a=summary['arms'][arm];c=summary['comparisons'][arm+'-vs-full']['avg5']
        explanation.append(f"- 双向 skip，阈值 {threshold}：AVG5 {a['avg5']:.4f}，相对 Full {c['delta_pp']:+.4f} pp；95% bootstrap CI [{c['ci95_pp'][0]:+.4f}, {c['ci95_pp'][1]:+.4f}]。Near 在 {a['near_active_requests']}/{summary['completed_requests']} 条请求中触发；层状态为 Near-only {a['gate_counts']['near']}、Far-only {a['gate_counts']['far']}、双 skip {a['gate_counts']['both']}。")
    explanation+=['','平均分没有下降也不能声称逐任务无损或总体等价；各任务和逐条变化必须分别审阅。CI包含0不证明等价。',
        'Near-only 的 KV 暂存继续向 Far 迁移，所有 Near-only 层的驻留 KV 比例为 1；其收益仅体现在 attention 读取量。表中的 KV/attention 比例是节点容量与 key 数量口径，不是整体 GPU 显存或实测速率。',
        ('Near/Far gate独立比较各自mass；同阈值下加入Near gate不会改变Far gate判定。' if summary['protocol'].get('schema')=='frozen-near-far-screen-v3-independent-gates' else '历史联合预算策略会因Near门改变Far判定，不能作为独立双门的证据。'),
        ('结果复用真实 TLI tail32 + L1/L2 选择器；full 代表 TLI 双 gate 关闭，非 Dense FullKV。当前为 eager 精度参考适配器，不作速度或总显存结论。' if summary['protocol'].get('attention','').startswith('tli') else '结果来自独立 eager 分区缓存，不包含生产 TLI L1/L2 选择器。')+'已完成本地可执行复核，未表示独立第三方审阅或论文主表验收完成。','',marker,'']
    hopping=[]
    for threshold in ['0.1','0.15']:
        name='dual_'+threshold
        for ds in ['musique','hotpotqa']:
            pair=summary['comparisons'][name+'-vs-full']['per_task'][ds]
            active=summary['arms'][name]['per_task_activation'][ds]['near_active']
            hopping.append(f"多跳单独核查，{ds} ε={threshold}：n={pair['n']}，均分变化 {pair['delta_pp']:+.4f} pp，逐条下降 {pair['degraded']}、改善 {pair['improved']}，Near实际触发 {active} 条。")
    explanation[-2:-2]=hopping+['小样本不推翻此前大样本中可能掉点的观察。','']
    report_path.write_text('\n'.join(explanation)+report)
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
            'axes.spines.top':False,'axes.spines.right':False})
        modes=[a for a in arms if a!='full'];labels=[a.replace('_',' ') for a in modes]
        fig,(ax,bx)=plt.subplots(1,2,figsize=(8,3.3))
        y=np.arange(len(modes))
        for i,arm in enumerate(modes):
            c=summary['comparisons'][arm+'-vs-full']['avg5'];lo,hi=c['ci95_pp'];mean=c['delta_pp']
            color='#007eb5' if arm.startswith('far_') else '#ff6540'
            ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=3,linewidth=1.3)
        ax.axvline(0,color='#777777',linewidth=.8,linestyle=':')
        ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('AVG5 change vs full (pp)')
        ax.set_title('(a) Paired quality, 95% bootstrap CI',loc='left',fontsize=10,weight='bold')
        kv=[100*(1-summary['arms'][a]['frozen_kv_ratio_mean']) for a in modes]
        work=[100*(1-summary['arms'][a]['frozen_attention_token_ratio_mean']) for a in modes]
        bx.bar(y-.17,kv,.34,label='resident KV',color='#6c767e')
        bx.bar(y+.17,work,.34,label='attention tokens',color='#00abc8')
        bx.set_xticks(y,[a.replace('_','\n') for a in modes]);bx.set_ylabel('reduction (%)')
        bx.set_title('(b) Cache capacity and work proxy',loc='left',fontsize=10,weight='bold')
        bx.legend(frameon=False,fontsize=8);bx.set_ylim(0,max(work+kv)*1.3)
        bx.grid(axis='y',linewidth=.5,color='#dddddd');bx.set_axisbelow(True)
        fig.tight_layout(w_pad=2.5)
        for ext in ['png','pdf','svg']:fig.savefig(outdir/f'quality_tradeoff.{ext}',dpi=300,bbox_inches='tight')
    return dict(requests=len(raw_index),predictions=len(row_scores),identity=summary['identity'])


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('outdir');ap.add_argument('--plot',action='store_true');args=ap.parse_args()
    print(json.dumps(export(args.outdir,args.plot),indent=2))
