#!/usr/bin/env python3
"""Recompute descriptive evidence from frozen stage22.38 artifacts, not new runs."""
import datetime as dt
from pathlib import Path

import numpy as np
from stage22_38_common import ROOT, read, digest, write_json, atomic_text


def main():
    report = read(ROOT/'reports/stage22_38_diagnostic.json')
    assert report['completed'] and report['queue_state']=='SUCCESS'
    run = Path(report['run_dir'])
    m = read(run/'metrics.json')
    assert all(m['checks'].values()) and m['sample_count']==1648
    assert not m['test_split_accessed'] and not m['updates_model_or_memory']
    assert digest(run/'samples.npz')==m['samples_sha256']
    config = __import__('yaml').safe_load((run/'resolved_config.yaml').read_text())
    channels = config['official_args']['d_var']
    with np.load(run/'samples.npz',allow_pickle=False) as samples:
        count = samples['complete_count']
        for route,value in m['routes'].items():
            np.testing.assert_allclose(samples[f'{route}__sse'].sum()/count.sum(),value['mse'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(samples[f'{route}__sae'].sum()/count.sum(),value['mae'],rtol=0,atol=1e-12)
        assert np.array_equal(samples['sample_index'],np.arange(1648))
    summary = dict(stage='22.38',completed=True,run_dir=str(run),
        conclusion='NO_SUPPORT_FOR_QUERY_CONTAMINATION_AS_PRIMARY_BOTTLENECK',
        interpretation='三种填补替换的主要预测效应来自历史数值直达；不是新模型收益，也不证明原填补错误。',
        scope='Traffic/seed2027/1648 validation windows/frozen checkpoint/three specified fills',
        source_report_sha256=digest(ROOT/'reports/stage22_38_diagnostic.json'),
        original=dict(mse=m['mse'],mae=m['mae']),routes=m['routes'],
        max_absolute_query_mse_change_percent=max(abs(m['routes'][f'query_{f}']['mse_gain_percent'])
            for f in ('zero','prefix_mean','ffill')),
        max_absolute_retrieval_mse_change_percent=max(abs(m['routes'][f'retrieval_{f}']['mse_gain_percent'])
            for f in ('zero','prefix_mean','ffill')),
        horizon_mse={r:[v[f'sse_h{i+1}']/(24*channels) for i in range(4)] for r,v in m['routes'].items()},
        groups=m['groups'],checks=m['checks'],
        limitations=['单数据集单seed机制诊断，不能推广至全部数据与训练方式。',
            '替代填补对固定checkpoint属于分布外干预；退化不等于重训必然无效。',
            '全部最长缺失段落在6–20组，不能证明跨缺失尺度差异。',
            '重叠窗口并非独立统计重复；尾部缺失组还与时间段混杂。',
            '归一化后的查询漂移不是预测瓶颈证据；当前不区分编码稳定和融合分支贡献弱。'],
        next_step='22.39：审计历史/查询/原型融合分量尺度与衰减距离实现；先定位后选择单一最小改进。')
    lines=['# 步骤22.38结果与机制结论','',
        '已完成：2/2任务SUCCESS，退出0，无OOM；9项CPU、9项队列及全部GPU门禁通过。',
        f'正式1648窗口；计算耗时{m["elapsed_seconds"]:.1f}秒；峰值预留显存{m["peak_gpu_reserved_mb"]:.0f}MiB。',
        '', '## 核心结论','',summary['interpretation'],
        '当前证据不支持优先围绕“查询受填补污染”增加编码或注意力模块。',
        '查询向量平均约2%的相对漂移，并未产生相应规模的预测变化。',
        '', '| 填补方式 | 完整路径MSE | 查询+检索MSE | 历史直达MSE | 纯检索MSE |',
        '| --- | ---: | ---: | ---: | ---: |']
    for fill,label in (('zero','标准化零'),('prefix_mean','前缀观测均值'),('ffill','前向填充')):
        rows=[m['routes'][f'{p}_{fill}']['mse'] for p in ('full','query','direct','retrieval')]
        lines.append('| '+label+' | '+' | '.join(f'{v:.9f}' for v in rows)+' |')
    lines+=['',f'原始基线MSE={m["mse"]:.9f}，MAE={m["mae"]:.9f}；与历史基线整体MSE差约1.5e-9。',
        f'查询路径MSE最大相对变化仅{summary["max_absolute_query_mse_change_percent"]:.6f}%；',
        f'纯检索最大相对变化仅{summary["max_absolute_retrieval_mse_change_percent"]:.6f}%，不作为有意义的提升。',
        '', '## 尾部缺失分组','',
        '| 末端缺失长度 | 窗口数 | 原始MSE | 仅历史直达零填补MSE | 仅历史直达均值MSE | 仅历史直达前填MSE |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,value in m['groups']['trailing_gap'].items():
        rows=[value['routes'][p]['mse'] for p in ('original','direct_zero','direct_prefix_mean','direct_ffill')]
        lines.append('| '+name+' | '+str(value['sample_count'])+' | '+' | '.join(f'{v:.9f}' for v in rows)+' |')
    lines+=['','## 边界与下一步','']+['- '+v for v in summary['limitations']]
    lines+=['', '分预测段原始sse_h为24步SSE均值；本摘要JSON已除以24×862转换为horizon_mse。',
        '', summary['next_step'],
        '优先核实融合各分量实际作用；不直接用硬替换、检索门控或新注意力堆叠解释当前结果。',
        '若发现基线实现问题，应单独记录修复与重新对照，不能当作创新模块收益。',
        '', f'原始运行：{run}', '完整13路径及表示漂移见stage22_38_diagnostic.json；逐窗资产为samples.npz。']
    write_json(ROOT/'reports/stage22_38_mechanism_summary.json',summary)
    atomic_text(ROOT/'reports/stage22_38_mechanism_summary.md','\n'.join(lines)+'\n')
    archive=ROOT/'reports/archive'/('stage22_38__'+dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    for name in ('stage22_38_diagnostic.json','stage22_38_diagnostic.md',
                 'stage22_38_mechanism_summary.json','stage22_38_mechanism_summary.md',
                 'latest_results.csv','latest_results.md'):
        atomic_text(archive/name,(ROOT/'reports'/name).read_text())
    print(f'SUMMARY_PASS recomputed_routes={len(m["routes"])} archive={archive}')


if __name__=='__main__': main()
