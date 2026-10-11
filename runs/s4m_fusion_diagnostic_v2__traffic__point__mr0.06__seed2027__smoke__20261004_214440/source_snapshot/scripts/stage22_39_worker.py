#!/usr/bin/env python3
"""Frozen fusion contribution probes; never trains or constructs a test loader."""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from stage22_38_common import ROOT, read, write_json, digest
from stage22_38_worker import tensor_hash, memory_hash, errors, input_groups

BRANCH_SCALES = {'retrieval':[0.,.9,1.1],'query':[0.,.9,1.1],'history':[0.,.9,1.1],
                 'observed_history':[.9,1.1],'imputed_history':[0.,.9,1.1]}


def decompose(retrieved, query, raw, mask, weight, bias):
    h = query.shape[-1]
    assert retrieved.shape == query.shape and raw.shape == mask.shape
    assert weight.shape == (2*h+raw.shape[-1],h)
    values = dict(retrieval=retrieved@weight[:h],query_projection=query@weight[h:2*h],
        query_residual=query,history=raw@weight[2*h:],
        observed_history=(raw*mask)@weight[2*h:],imputed_history=(raw*(1-mask))@weight[2*h:],
        bias=bias.view(1,1,-1).expand_as(query))
    values['query'] = values['query_projection']+values['query_residual']
    return values


def intervene(context, component, scale):
    return context if scale == 1 else context+(scale-1)*component


def operational_intervene(base,mask,branch,scale,fuse):
    """Evaluate equivalent branch scaling through the original fused GEMM."""
    if scale==1:return base['context']
    r,u,z=base['retrieved'],base['query'],base['raw']
    if branch=='retrieval':r=r*scale
    elif branch=='query':u=u*scale
    elif branch=='history':z=z*scale
    elif branch=='observed_history':z=torch.where(mask.bool(),z*scale,z)
    elif branch=='imputed_history':z=torch.where(mask.bool(),z,z*scale)
    else:raise ValueError(branch)
    b,l,h=u.shape
    return fuse(r.reshape(-1,h),u.reshape(-1,h),z,b,l,h)


def route_name(branch,scale): return f'{branch}_scale_{scale:g}'


def run(config,out):
    sys.path[:0] = [str(ROOT/p) for p in ('baselines/S4M/s4m','baselines/S4M/s4m/model',
        'baselines/S4M/s4m/model/S4/s4','baselines/S4M/s4m/model/S4/s4/extensions/kernels')]
    from model.S4M import Model
    from data_provider.data_loader import Dataset_Custom4
    from torch.utils.data import DataLoader
    from utils.paired_eval_rng import paired_eval_rng
    args=SimpleNamespace(**config['official_args']);device=torch.device('cuda:0')
    args.device=device;args.use_gpu=True;args.gpu=0
    assert torch.cuda.is_available() and args.use_mipp is False
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    environment=dict(python=sys.version,pytorch=torch.__version__,cuda_runtime=torch.version.cuda,
        conda_environment='s4m_missing_forecasting',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_name=torch.cuda.get_device_name(device))
    print('ENVIRONMENT '+json.dumps(environment),flush=True)
    model=Model(args).float()
    checkpoint=torch.load(config['source']['checkpoint'],map_location='cpu')
    model.load_state_dict(checkpoint['model'],strict=True)
    memory=model.mem_net.memory
    for key in ('memory','repre','num_dic'):setattr(memory,key,checkpoint[key])
    del checkpoint
    model=model.to(device);memory.repre=memory.repre.to(device)
    model.eval();model.is_training=0;model.mem_net.is_training=0;model.set_prototype_enabled(True)
    assert not any(getattr(model,key,False) for key in ('use_mrra','use_mcpr'))
    dataset=Dataset_Custom4(root_path=args.root_path,args=args,flag='val',size=[args.seq_len,args.label_len,args.pred_len],
        features=args.features,data_path=args.data_path,target=args.target,timeenc=1 if args.embed=='timeF' else 0,
        freq=args.freq,mask=args.mask)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,drop_last=True,num_workers=args.num_workers)
    assert len(loader)*args.batch_size==1648
    with np.load(config['source']['cache'],allow_pickle=False) as reference:
        cache={k:reference[k].copy() for k in ('sample_index','complete_count','complete_sse_on','complete_sae_on','history_mask_packed')}
    assert config['diagnostic']['branch_scales']==BRANCH_SCALES
    assert config['diagnostic']['fusion_evaluation_order']=='operational_original_fuse_v2'
    routes=('original',)+tuple(route_name(b,s) for b,scales in BRANCH_SCALES.items() for s in scales)
    arrays={k:[] for k in ('sample_index','complete_count','missing_rate','longest_gap','trailing_gap')}
    for route in routes:
        for key in ('sse','sae','prediction_difference_mse'):arrays[f'{route}__{key}']=[]
    stats={};captured={};original_fuse=model.mem_net._fuse_context
    def capture(r,u,z,b,l,h):
        c=original_fuse(r,u,z,b,l,h)
        captured.update(retrieved=r.reshape(b,l,h),query=u.reshape(b,l,h),raw=z,context=c)
        return c
    model.mem_net._fuse_context=capture
    def add(name,value):stats.setdefault(name,[]).append(value.detach().double().cpu().numpy())
    max_recomposition=max_cache_difference=max_off_difference=0.
    warmup_changes=[];freeze_hash=freeze_memory=None
    started=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.no_grad():
            for batch_index,batch in enumerate(loader):
                if 0<config['diagnostic']['max_eval_batches']<=batch_index:break
                x,mask=[batch[i].float().to(device) for i in (0,1)]
                # Hidden history truth (batch[4]) is never inspected or transferred.
                target=batch[5].float().to(device)[:,-args.pred_len:,:]
                extrema=[batch[i].float().to(device) for i in (6,7,8,9)]
                original_args=(x,mask,*extrema)
                if batch_index==0:
                    before={k:tensor_hash(v) for k,v in model.state_dict().items()};before_memory=memory_hash(model)
                    with paired_eval_rng(True,args.seed,0,device):model(*original_args)
                    warmup_changes=[k for k,v in model.state_dict().items() if tensor_hash(v)!=before[k]]
                    assert all(k.endswith('.C') or k.endswith('.l_kernel') for k in warmup_changes)
                    assert memory_hash(model)==before_memory
                    freeze_hash=tensor_hash(model.state_dict());freeze_memory=memory_hash(model)
                    print(f'FROZEN_MODEL parameters={sum(p.numel() for p in model.parameters())} prototypes={len(memory.repre)} warmup_changes={warmup_changes}',flush=True)
                with paired_eval_rng(True,args.seed,batch_index,device):prediction=model(*original_args)[:,-args.pred_len:,:]
                base=dict(captured);c=base['context'];b,l,h=base['query'].shape
                assert torch.equal(base['raw'][mask.bool()],x[mask.bool()])
                values=decompose(base['retrieved'],base['query'],base['raw'],mask,model.mem_net.pred_w,model.mem_net.pred_b)
                reconstructed=values['retrieval']+values['query']+values['history']+values['bias']
                relative=(reconstructed.double()-c.double()).norm()/c.double().norm().clamp_min(1e-12)
                max_recomposition=max(max_recomposition,float(relative))
                assert float(relative)<=config['diagnostic']['decomposition_relative_atol']
                assert torch.allclose(values['observed_history']+values['imputed_history'],values['history'],rtol=1e-5,atol=1e-4)
                projected_mask=model.mask_trans(mask)
                def forecast(context):
                    with paired_eval_rng(True,args.seed,batch_index,device):
                        return model._forecast_from_context(context,projected_mask,observation_mask=mask)[:,-args.pred_len:,:]
                assert torch.equal(forecast(intervene(c,values['history'],1.)),prediction),'CODE_ERROR identity forecast'
                count=args.pred_len*args.d_var;index=np.arange(batch_index*b,(batch_index+1)*b)
                np.testing.assert_array_equal(cache['sample_index'][index],index)
                np.testing.assert_array_equal(cache['complete_count'][index],np.full(b,count))
                np.testing.assert_array_equal(cache['history_mask_packed'][index],np.packbits(mask[:,:,0].byte().cpu().numpy(),axis=1))
                sse,sae=errors(prediction,target)
                for current,key in ((sse,'complete_sse_on'),(sae,'complete_sae_on')):
                    delta=float(np.max(np.abs(current.cpu().numpy()-cache[key][index])/count))
                    max_cache_difference=max(max_cache_difference,delta)
                    assert delta<=config['diagnostic']['baseline_metric_atol'],'CODE_ERROR historical baseline mismatch'
                missing,longest,trailing=input_groups(mask)
                for key,value in (('sample_index',index),('complete_count',np.full(b,count)),('missing_rate',missing),('longest_gap',longest),('trailing_gap',trailing)):
                    arrays[key].append(value.cpu().numpy() if torch.is_tensor(value) else value)
                rms_c=c.double().square().mean((1,2)).sqrt().clamp_min(1e-12)
                for name,value in dict(values,context=c,input_query=base['query'],input_retrieval=base['retrieved'],input_history=base['raw']).items():
                    rms=value.double().square().mean((1,2)).sqrt();add(name+'__rms',rms)
                    add(name+'__l2',value.double().flatten(1).norm(dim=1))
                    if value.shape==c.shape:
                        add(name+'__rms_ratio_to_context',rms/rms_c)
                        add(name+'__alignment_to_context',(value.double()*c.double()).sum((1,2))/c.double().square().sum((1,2)).clamp_min(1e-12))
                main=[values[k].double().flatten(1) for k in ('retrieval','query','history')]
                add('fusion__cancellation_ratio',sum(main).norm(dim=1)/sum(v.norm(dim=1) for v in main).clamp_min(1e-12))
                for i,j,label in ((0,1,'retrieval_query'),(0,2,'retrieval_history'),(1,2,'query_history')):
                    dot=(main[i]*main[j]).sum(1);add('fusion__'+label+'_dot',dot)
                    add('fusion__'+label+'_cosine',dot/(main[i].norm(dim=1)*main[j].norm(dim=1)).clamp_min(1e-12))
                # Observe the current buggy distances/logits; corrected distance is not fed to the model.
                dmax,dmin,_,_=extrema
                normalized_max=(dmax-dmax.mean(1,keepdim=True))/dmax.std(1,keepdim=True)
                current_min=(normalized_max-dmin.mean(1,keepdim=True))/dmin.std(1,keepdim=True)
                correct_min=(dmin-dmin.mean(1,keepdim=True))/dmin.std(1,keepdim=True)
                add('distance__current_min_temporal_std',current_min.std(1).mean(1));add('distance__correct_min_temporal_std',correct_min.std(1).mean(1))
                add('distance__current_min_mean',current_min.mean((1,2)));add('distance__raw_min_smallest_std',dmin.std(1).min(1).values)
                add('distance__current_vs_correct_mse',(current_min.double()-correct_min.double()).square().mean((1,2)))
                for name,logit in (('max',-model.decay1(normalized_max)),('min',-model.decay2(current_min))):
                    assert bool(torch.isfinite(logit).all())
                    add('logit__'+name+'_minimum',logit.flatten(1).min(1).values);add('logit__'+name+'_maximum',logit.flatten(1).max(1).values)
                    add('logit__'+name+'_exp_nonfinite_fraction',(~torch.isfinite(logit.exp())).double().mean((1,2)))
                for name,indices in (('max',x.argmax(1)),('min',x.argmin(1))):
                    extrema_observed=mask.gather(1,indices.unsqueeze(1)).squeeze(1)
                    add('extrema__'+name+'_chosen_missing_fraction',(1-extrema_observed).mean(1))
                def record(name,output):
                    assert bool(torch.isfinite(output).all()),'NAN_OR_DIVERGENCE'
                    a,e=errors(output,target)
                    for key,v in (('sse',a),('sae',e),('prediction_difference_mse',(output.double()-prediction.double()).square().mean((1,2)))):
                        arrays[f'{name}__{key}'].append(v.cpu().numpy())
                record('original',prediction)
                for branch,scales in BRANCH_SCALES.items():
                    for scale in scales:
                        record(route_name(branch,scale),forecast(operational_intervene(base,mask,branch,scale,original_fuse)))
                if config['experiment']['smoke_test']:
                    model.set_prototype_enabled(False)
                    try:
                        with paired_eval_rng(True,args.seed,batch_index,device):off=model(*original_args)[:,-args.pred_len:,:]
                    finally:model.set_prototype_enabled(True)
                    off_capture=dict(captured)
                    direct_off_context=original_fuse(torch.zeros_like(base['retrieved']).reshape(-1,h),
                        base['query'].reshape(-1,h),base['raw'],b,l,h)
                    direct_off_prediction=forecast(direct_off_context)
                    subtractive_off_context=intervene(c,values['retrieval'],0.)
                    subtractive_delta=float((off-forecast(subtractive_off_context)).abs().max())
                    delta=float((off-forecast(operational_intervene(base,mask,'retrieval',0.,original_fuse))).abs().max())
                    diagnostic=dict(batch_index=batch_index,
                        gate_prediction_atol=config['diagnostic']['prototype_off_prediction_atol'],
                        operational_vs_subtractive_prediction_max_abs=subtractive_delta,
                        operational_v2_gate_prediction_max_abs=delta,
                        operational_vs_direct_zero_fusion_prediction_max_abs=float((off-direct_off_prediction).abs().max()),
                        subtractive_vs_direct_zero_context_max_abs=float((subtractive_off_context-direct_off_context).abs().max()),
                        off_vs_base_query_max_abs=float((off_capture['query']-base['query']).abs().max()),
                        off_vs_base_history_max_abs=float((off_capture['raw']-base['raw']).abs().max()),
                        off_retrieved_max_abs=float(off_capture['retrieved'].abs().max()),
                        unchanged_gate=True,scope='v2_original_fusion_evaluation_order_no_tolerance_change')
                    write_json(out/'prototype_off_check.json',diagnostic)
                    print('PROTOTYPE_OFF_CHECK '+json.dumps(diagnostic),flush=True)
                    max_off_difference=max(max_off_difference,delta)
                    assert delta<=config['diagnostic']['prototype_off_prediction_atol'],'CODE_ERROR prototype-off mismatch'
                print(f'FUSION_DIAGNOSTIC batch={batch_index+1}/{3 if config["experiment"]["smoke_test"] else len(loader)} windows={(batch_index+1)*b}',flush=True)
        assert tensor_hash(model.state_dict())==freeze_hash and memory_hash(model)==freeze_memory
    finally:model.mem_net._fuse_context=original_fuse;model.set_prototype_enabled(True)
    arrays.update({k:np.concatenate(v) for k,v in stats.items()});arrays={k:np.concatenate(v) if isinstance(v,list) else v for k,v in arrays.items()}
    expected=48 if config['experiment']['smoke_test'] else 1648
    assert len(arrays['sample_index'])==expected and all(np.isfinite(v).all() for v in arrays.values())
    tmp=out/'samples.tmp.npz';np.savez_compressed(tmp,**arrays);tmp.replace(out/'samples.npz')
    summaries={r:dict(mse=float(arrays[r+'__sse'].sum()/arrays['complete_count'].sum()),mae=float(arrays[r+'__sae'].sum()/arrays['complete_count'].sum()),
        prediction_difference_mse=float(arrays[r+'__prediction_difference_mse'].mean())) for r in routes}
    for r,value in summaries.items():
        for metric in ('mse','mae'):value[metric+'_gain_percent']=(1-value[metric]/summaries['original'][metric])*100
    groups={}
    for name,chosen in (('0',arrays['trailing_gap']==0),('1_to_5',(arrays['trailing_gap']>=1)&(arrays['trailing_gap']<=5)),('6_to_20',(arrays['trailing_gap']>=6)&(arrays['trailing_gap']<=20))):
        if chosen.any():groups[name]=dict(sample_count=int(chosen.sum()),routes={r:dict(mse=float(arrays[r+'__sse'][chosen].sum()/arrays['complete_count'][chosen].sum()),mae=float(arrays[r+'__sae'][chosen].sum()/arrays['complete_count'][chosen].sum())) for r in routes})
    metrics=dict(stage='22.39',diagnostic_version='v2_operational_fusion',evaluation_split='validation',test_split_accessed=False,uses_test_split=False,updates_model_or_memory=False,
        scope='frozen_fusion_contribution_intervention_not_trained_model',sample_count=expected,mse=summaries['original']['mse'],mae=summaries['original']['mae'],
        parameter_count=sum(p.numel() for p in model.parameters()),checkpoint=config['source']['checkpoint'],routes=summaries,groups=groups,
        activations={k:dict(mean=float(v.mean()),minimum=float(v.min()),maximum=float(v.max()),p50=float(np.median(v))) for k,v in arrays.items() if k in stats},
        checks=dict(observed_values_preserved=True,historical_baseline_paired=True,identity_forecast=True,decomposition_valid=True,
            frozen_weights=True,frozen_memory=True,finite=True),
        prototype_off_check_scope='passed_in_this_smoke' if config['experiment']['smoke_test'] else 'verified_by_dependency_smoke',
        baseline_max_window_metric_difference=max_cache_difference,decomposition_max_relative_difference=max_recomposition,prototype_off_max_prediction_difference=max_off_difference,
        warmup_state_changes=warmup_changes,frozen_weights_sha256=freeze_hash,frozen_memory_sha256=freeze_memory,
        source_checkpoint_sha256=digest(config['source']['checkpoint']),samples_sha256=digest(out/'samples.npz'),artifact_status='PASS',environment=environment,
        peak_gpu_reserved_mb=torch.cuda.max_memory_reserved(device)/1024**2,peak_gpu_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2,elapsed_seconds=time.monotonic()-started)
    write_json(out/'metrics.json',metrics)
    print('FUSION_DIAGNOSTIC_COMPLETE '+json.dumps({k:metrics[k] for k in ('sample_count','mse','mae','elapsed_seconds','peak_gpu_reserved_mb')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--run-dir',required=True)
    a=parser.parse_args();run(yaml.safe_load(Path(a.config).read_text()),Path(a.run_dir))
