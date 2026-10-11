#!/usr/bin/env python3
"""Render existing CPU evidence, without recomputing/selecting a model."""
import csv
import io
import sys
import subprocess
from pathlib import Path
from stage22_34_factorial import ROOT, read, atomic_text, write_json, digest


def main():
    result=read(ROOT/'reports/stage22_35_diagnosis.json')
    assert result['status']=='SUCCESS' and result['uses_test_split'] is False
    rows=[]
    lines=['# 步骤22.35：PG95退化诊断结果','',
           'CPU诊断成功；不读取test、不训练、不改主模型。现有证据尚未识别单个模块的因果作用。','',
           '## 1. 训练与误差定位','',
           '六训练均最佳epoch1、epoch4早停、2992优化步，没有少训或多训造成的预算差异。各方法每epoch计算量仍不同。',
           '训练日志Train Loss包含各自损失，不能直接视为同一定义的预测MSE；选模观测目标MSE也不等于完整目标MSE/MAE。','',
           '| seed | PG95+CNM相对S4M+CNM MSE变化 | MAE变化 | MSE更差窗口 | MAE更差窗口 |',
           '| --- | ---: | ---: | ---: | ---: |']
    for r in result['records']:
        v=r['error_strata']['multiscale']['all']
        lines.append(f'| {r["seed"]} | {v["mse_change_percent"]:+.4f}% | {v["mae_change_percent"]:+.4f}% | {100*v["mse_worse_window_fraction"]:.2f}% | {100*v["mae_worse_window_fraction"]:.2f}% |')
        for route,groups in r['error_strata'].items():
            for group,value in groups.items(): rows.append(dict(seed=r['seed'],route=route,group=group,**value))
    lines += ['', '正变化表示误差升高。seed2028四个时间段及三档历史缺失率的多尺度MSE/MAE均退化；最后412窗口全部双指标更差，不是仅几个极端点。seed2027主要风险在第三时间段，但缺失率与时间段有关，不能将分组关联当作原因。',
              '', '## 2. 能确认的机制限制','',
              '- 当前源码同seed构造时，三个seed均有124/314个公共状态条目不同（包括linear1、mask_trans、out_proj及部分S4参数）。新增MIPP构造在这些层之前消耗随机数；CPU控制流检查不等于读取历史初始权重。',
              '- 原方法级三seed比较并未因此作废，但不能把单seed的差异都归到写入门/不变性损失。训练额外视图也使用共享随机状态，下一次严格消融应隔离初始化、数据顺序和辅助掩码随机流。',
              '- 三seed PG95常规门平均放行99.6912%/99.2835%/99.5345%；不是强筛选。暖启动的放行率约80.47%，不能与常规写入混合表述。',
              '- 加权额外loss均值约1.29e-5/3.10e-6/2.30e-6；这些是全程平均，不是epoch1指标。数值小不能证明梯度小，也不能直接推出应加大权重。','',
              '| seed | S4M原型数 | PG95原型数 | S4M中心化有效秩 | PG95中心化有效秩 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for r in result['records']:
        a=r['sources']['s4m']['prototype_geometry']; b=r['sources']['pg95']['prototype_geometry']
        lines.append(f'| {r["seed"]} | {a["count"]} | {b["count"]} | {a["centered_effective_rank"]:.3f} | {b["centered_effective_rank"]:.3f} |')
    lines += ['', '有效秩是中心化原型矩阵奇异值能量分布的熵指数，最大值受原型数量限制，不能直接跨数量解释为相同程度的坍缩。seed2028中表现更好的S4M也只有6个原型，不能将“小原型库”单独认定为根因。',
              '', '## 3. 训练信息假设','',
              '历史code.diff与当前代码均显示辅助视图由`_build_mipp_pair(seq_true_x)`生成，而不是原观测掩码的子集。它可能重新暴露主视图被人工隐藏的位置。该流程依赖完整训练历史可用这一假设；如果论文目标是只有不完整训练输入，必须单独修订并验证，不能直接声称满足该设定。这一限制不是本轮退化的已证实原因。',
              '', '## 4. 下一最短路径（未启动）','',
              '步骤22.36建议先做匹配初始化和独立随机流的原方法机制对照，不新增复杂模块、不直接调大损失或改门阈值。先在CPU证明公共初始权重逐项一致、数据顺序一致、辅助掩码不会推进主干随机流；随后在新experiment ID下以原完整预算重训一对S4M/PG95（seed2027），同CRN-v1 validation评价。它是归因控制而非创新收益，也不是重新跑到满意为止。',
              '若该对照仍显示PG95不稳定，再做严格单因素子机制干预；此时才决定保留或替换写入门。涉及训练随机协议修订，应在启动前明确记录并确认；本步没有启动训练，也没有切换主模型。',
              '', '## 5. 复核边界','',
              '三seed全部保留；逐窗口时间重叠、分组不是独立验证集，没有显著性检验或二区结论。没有逐通道完整预测、逐epoch门统计、独立梯度或子模块消融，不能宣称已锁定唯一根因。','',
              f'正式诊断run：`{result["run_dir"]}`。完整结构化依据：`reports/stage22_35_diagnosis.json`；所有分组：`reports/stage22_35_error_strata.csv`。','']
    content='\n'.join(lines)
    atomic_text(ROOT/'reports/stage22_35_diagnosis.md',content)
    atomic_text(Path(result['run_dir'])/'diagnosis.md',content)
    stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_text(ROOT/'reports/stage22_35_error_strata.csv',stream.getvalue())
    atomic_text(Path(result['run_dir'])/'error_strata.csv',stream.getvalue())
    # Additional provenance for the executed CPU diagnosis; preserve its source snapshot.
    run=Path(result['run_dir'])
    evidence=[]
    for cmd in (['git','rev-parse','HEAD'],['git','status','--short'],['git','diff','--stat']):
        p=subprocess.run(cmd,cwd=ROOT/'baselines/S4M',capture_output=True,text=True)
        evidence.append(f'{cmd} exit={p.returncode}\n{p.stdout}')
    atomic_text(run/'git_state.txt','\n'.join(evidence))
    diff=subprocess.run(['git','diff','--binary'],cwd=ROOT/'baselines/S4M',capture_output=True,text=True,check=True).stdout
    atomic_text(run/'code.diff',diff)
    write_json(run/'report_render_manifest.json',dict(renderer_sha256=digest(Path(__file__)),
        evidence_captured_after_diagnostic=True,model_code_modified_during_diagnostic=False))
    subprocess.run([sys.executable,str(ROOT/'scripts/aggregate_results.py')],cwd=ROOT,check=True)
    print('Rendered evidence report and all 48 strata rows; global results updated.')


if __name__=='__main__': main()
