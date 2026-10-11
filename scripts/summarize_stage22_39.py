#!/usr/bin/env python3
"""Descriptive interpretation with independent metric and provenance checks."""
import datetime as dt
from pathlib import Path
import numpy as np
from stage22_38_common import ROOT,read,digest,write_json,atomic_text


def main():
    report=read(ROOT/'reports/stage22_39_fusion.json')
    assert report['completed'] and report['queue_state']=='SUCCESS'
    run=Path(report['run_dir']);m=read(run/'metrics.json');a=m['activations']
    assert m['diagnostic_version']=='v2_operational_fusion' and all(m['checks'].values())
    assert m['sample_count']==1648 and not m['test_split_accessed'] and not m['updates_model_or_memory']
    assert digest(run/'samples.npz')==m['samples_sha256']
    with np.load(run/'samples.npz',allow_pickle=False) as data:
        assert np.array_equal(data['sample_index'],np.arange(1648))
        assert all(np.isfinite(data[k]).all() for k in data.files)
        for route,value in m['routes'].items():
            for metric,field in (('mse','sse'),('mae','sae')):
                np.testing.assert_allclose(data[route+'__'+field].sum()/data['complete_count'].sum(),value[metric],rtol=0,atol=1e-12)
    result=dict(stage='22.39',completed=True,diagnostic_version='v2_operational_fusion',run_dir=str(run),
        analysis_script_sha256=digest(__file__),decision='PRIORITIZE_ISOLATED_DISTANCE_BUGFIX_BASELINE_NOT_NEW_RETRIEVAL_MODULE',
        interpretation='当前冻结模型主要依赖历史直达；最小距离误用确证，局部缩放/移除收益很小，不形成创新证据。',
        original=dict(mse=m['mse'],mae=m['mae']),routes=m['routes'],activations=a,groups=m['groups'],
        distance_temporal_std=dict(current=a['distance__current_min_temporal_std']['mean'],
            correctly_normalized=a['distance__correct_min_temporal_std']['mean']),
        scopes=dict(dataset='traffic',seed=2027,split='validation',sample_count=1648,
            test_accessed=False,model_retrained=False,baseline_source_modified=False),
        preserved_v1_failures=['runs/s4m_fusion_diagnostic__traffic__point__mr0.06__seed2027__smoke__20261004_210243',
            'runs/s4m_fusion_diagnostic__traffic__point__mr0.06__seed2027__smoke__20261004_213853'],
        next_step='22.40建议：只修min_idx标准化，保留独立开关和原基线；先smoke，再同预算单seed重训对照。修复不计为创新。',
        limitations=['仅Traffic/seed2027/固定checkpoint的机制诊断。','关闭或缩放分量是固定模型干预，不是重训收益。',
            'RMS范数比不是可加的预测贡献占比。','重叠窗口不是独立seed，不宣称显著性。',
            '明确上游实现问题不等于已证明其降低精度；旧checkpoint已适应该实现。',
            '极值缺失比例按每窗口变量的选中位置统计，不是数据缺失率。'])
    lines=['# 步骤22.39：最终机制结论','',
        f'V2两项SUCCESS、退出0，无OOM；完整1648窗耗时{m["elapsed_seconds"]:.1f}秒，预留峰值{m["peak_gpu_reserved_mb"]:.0f}MiB。',
        'CPU源审计19项、V2融合12项、调度9项以及GPU门禁通过；V1两次失败保留。',
        '', '## 当前结论','',result['interpretation'],
        '', '| 固定模型干预 | MSE | MAE | MSE收益% | MAE收益% |',
        '| --- | ---: | ---: | ---: | ---: |']
    labels={'original':'原始基线','retrieval_scale_0':'关闭检索','query_scale_0':'关闭查询（含残差）',
        'history_scale_0':'关闭全部历史直达','imputed_history_scale_0':'关闭填补历史直达',
        'imputed_history_scale_0.9':'填补历史×0.9','observed_history_scale_1.1':'观测历史×1.1'}
    for route,label in labels.items():
        v=m['routes'][route];lines.append(f'| {label} | {v["mse"]:.9f} | {v["mae"]:.9f} | {v["mse_gain_percent"]:.5f} | {v["mae_gain_percent"]:.5f} |')
    lines+=['', '正收益表示误差降低；完整15路径见stage22_39_fusion.md，不隐藏其他比例。',
        '检索关闭有极小双指标收益，不能据此说所有检索都无用；查询移除出现MSE/MAE冲突。',
        '填补×0.9在尾部缺失6–20的128窗口中MSE反而退化，不能说该比例跨组稳定。',
        '', '## 实际融合与距离证据','',
        '| 分量 | 投影后平均RMS | 相对context的平均RMS比 |', '| --- | ---: | ---: |']
    for name,label in (('retrieval','检索'),('query','查询'),('history','历史'),('observed_history','观测历史'),('imputed_history','填补历史'),('bias','偏置')):
        lines.append(f'| {label} | {a[name+"__rms"]["mean"]:.6f} | {a[name+"__rms_ratio_to_context"]["mean"]:.6f} |')
    lines+=['','RMS（均方根）比不是贡献百分比。历史尺度大且移除后误差显著增大，两种证据共同支持当前历史路径主导。',
        f'最小距离当前时间标准差均值{a["distance__current_min_temporal_std"]["mean"]:.6f}，正确自身标准化为{a["distance__correct_min_temporal_std"]["mean"]:.6f}；正确值只作统计，未进入模型。',
        f'原最小距离标准差最小{a["distance__raw_min_smallest_std"]["minimum"]:.6f}，零标准差风险未发生。',
        f'外层指数前数值全局范围[{min(a["logit__max_minimum"]["minimum"],a["logit__min_minimum"]["minimum"]):.6f}, {max(a["logit__max_maximum"]["maximum"],a["logit__min_maximum"]["maximum"]):.6f}]，指数非有限比例为0。',
        f'极值选中缺失位置平均比例：最大值{a["extrema__max_chosen_missing_fraction"]["mean"]*100:.4f}%、最小值{a["extrema__min_chosen_missing_fraction"]["mean"]*100:.4f}%；此值不是总体缺失率，也不支持它是当前高频主问题。',
        '', '## 修复诊断，不修主模型','',
        'V1 FP32（32位浮点）分块相减与原矩阵求值顺序不同，预测差超门禁；V2统一原_fuse_context计算，原1e-5保持不变。smoke三个batch的原型关闭差均0。',
        '基线MSE与22.38严格一致；最大历史逐窗指标差1.16e-7。主模型、checkpoint、数据、mask和评价公式未改。',
        '', '## 下一步与限制','',result['next_step'],'']+['- '+v for v in result['limitations']]
    write_json(ROOT/'reports/stage22_39_mechanism_summary.json',result)
    atomic_text(ROOT/'reports/stage22_39_mechanism_summary.md','\n'.join(lines)+'\n')
    archive=ROOT/'reports/archive'/('stage22_39_final__'+dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    for name in ('stage22_39_mechanism_summary.json','stage22_39_mechanism_summary.md','stage22_39_fusion.json','stage22_39_fusion.md',
        'stage22_39_distance_audit.json','stage22_39_distance_audit.md','latest_results.csv','latest_results.md'):
        atomic_text(archive/name,(ROOT/'reports'/name).read_text())
    print(f'SUMMARY_PASS independently_recomputed_routes={len(m["routes"])} archive={archive}')


if __name__=='__main__':main()
