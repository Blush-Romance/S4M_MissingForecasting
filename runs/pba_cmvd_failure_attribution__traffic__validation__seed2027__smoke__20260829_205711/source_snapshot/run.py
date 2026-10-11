import argparse
# from experiments.exp_long_term_forecasting2 import Exp_Long_Term_Forecast as Exp_Long_Term_Forecast2
from experiments.exp_pretrain1 import Exp_Long_Term_Forecast as Exp_pretrain1
# from experiments.exp_transformer import Exp_Long_Term_Forecast as exp_transformer
# from experiments.exp_BiaTCGNet import Exp_Long_Term_Forecast as exp_BiaTCGNet
# from experiments.exp_brits import Exp_Long_Term_Forecast as Exp_brits
# from experiments.exp_grafiti import Exp_Long_Term_Forecast as Exp_grafiti
# from experiments.exp_cru import Exp_Long_Term_Forecast as Exp_cru
import random
import numpy as np
import wandb
import torch, gc
import os
import math


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int,  default=1, help='status')  # required=True,
    parser.add_argument('--seed', type=int,  default=2023)  # required=True,
    parser.add_argument('--plot', type=int,  default=0)  # required=True,

    parser.add_argument('--model_id', type=str,default='test', help='model id') # required=True, 
    parser.add_argument('--model', type=str,  default='grafiti',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer,S_Mamba ]')#required=True,

    # data loader
    parser.add_argument('--data', type=str, default='classify', help='dataset type')  #required=True,
    parser.add_argument('--test', type=str, default='test', help='test function type')  #required=True,

    parser.add_argument('--root_path', type=str, default='/home/project/S4M/data/ETT-small/chunk_missing/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh15_3.csv', help='data csv file')
    parser.add_argument('--true_data_path', type=str, default='', help='path to the complete reference csv')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='/home/project/S4M/checkpoints/', help='location of model checkpoints')
    parser.add_argument('--results_root', type=str, default='./results', help='structured result directory')
    parser.add_argument('--test_results_root', type=str, default='./test_results', help='plot result directory')
    parser.add_argument('--metrics_path', type=str, default='', help='optional JSON metrics output path')
    parser.add_argument(
        '--post_train_evaluation',
        type=str,
        choices=['test', 'validation'],
        default='test',
        help=(
            'split evaluated after training; validation preserves the historical '
            'test split for validation-gated external studies'
        ),
    )
    parser.add_argument('--gate_h_diagnostic', action='store_true', help='run paired prototype-on/off evaluation only')
    parser.add_argument('--gate_h_checkpoint', type=str, default='', help='checkpoint used by Gate H paired evaluation')
    parser.add_argument('--gate_h_metrics_path', type=str, default='', help='Gate H summary JSON path')
    parser.add_argument('--gate_h_samples_path', type=str, default='', help='Gate H per-sample NPZ path')
    parser.add_argument('--gate_h_delta', type=float, default=0.0, help='absolute loss margin used only for HRR reporting')
    parser.add_argument('--hcup_diagnostic', action='store_true', help='run the target-causal historical counterfactual utility probe')
    parser.add_argument('--hcup_checkpoint', type=str, default='', help='frozen S4M checkpoint used by the HCUP diagnostic')
    parser.add_argument('--hcup_metrics_path', type=str, default='', help='HCUP summary JSON path')
    parser.add_argument('--hcup_samples_path', type=str, default='', help='HCUP per-sample NPZ path')
    parser.add_argument('--hcup_split', type=str, choices=['val', 'test'], default='val', help='chronological split evaluated by HCUP')
    parser.add_argument('--hcup_probe_lag', type=int, default=96, help='number of windows between the historical probe and current forecast origin')
    parser.add_argument('--hcup_probe_horizon', type=int, default=96, help='number of already-observed leading probe predictions used for utility')
    parser.add_argument('--hcup_threshold', type=float, default=0.0, help='validation-locked normalized historical utility threshold')
    parser.add_argument('--stage12_validation_diagnostic', action='store_true', help='cache chronological validation errors and input masks from a frozen checkpoint')
    parser.add_argument('--stage12_checkpoint', type=str, default='', help='frozen PG95 or MRBA checkpoint used by the stage12 diagnostic')
    parser.add_argument('--stage12_metrics_path', type=str, default='', help='stage12 validation summary JSON path')
    parser.add_argument('--stage12_samples_path', type=str, default='', help='stage12 per-sample NPZ path')
    parser.add_argument('--stage22_validation_diagnostic', action='store_true', help='cache Electricity validation errors and equal-count mask response from a frozen checkpoint')
    parser.add_argument('--stage22_checkpoint', type=str, default='', help='frozen S4M or PG95 checkpoint used by the stage22 diagnostic')
    parser.add_argument('--stage22_metrics_path', type=str, default='', help='stage22 validation summary JSON path')
    parser.add_argument('--stage22_samples_path', type=str, default='', help='stage22 per-sample NPZ path')
    parser.add_argument('--stage22_nested_target_rate', type=float, default=0.15, help='fixed additional missing rate for the stage22 response probe')
    parser.add_argument('--stage22_nested_block_length', type=int, default=5, help='fixed equal-count causal block length for stage22')
    parser.add_argument('--stage22_nested_seed', type=int, default=0, help='deterministic equal-count mask seed for stage22')
    parser.add_argument('--stage22_internal_features', action='store_true', help='cache label-free MRRA and retrieval features during stage22 validation diagnosis')
    parser.add_argument('--stage22_mcpr_attribution', action='store_true', help='cache exact frozen-backbone MCPR residual attribution during stage22 validation diagnosis')
    parser.add_argument('--stage22_inference_timing', action='store_true', help='measure frozen model-only forward latency on chronological Electricity validation')
    parser.add_argument('--stage22_timing_checkpoint', type=str, default='', help='frozen checkpoint used by stage22 inference timing')
    parser.add_argument('--stage22_timing_metrics_path', type=str, default='', help='stage22 inference timing JSON output path')
    parser.add_argument('--stage22_timing_warmup_batches', type=int, default=10, help='initial validation batches excluded from each timing repeat')
    parser.add_argument('--stage22_timing_repeats', type=int, default=2, help='number of complete chronological validation timing repeats')
    parser.add_argument('--stage22_ard_information_diagnostic', action='store_true', help='cache frozen ARD sample decisions and label-free structured features on validation')
    parser.add_argument('--stage22_ard_diagnostic_checkpoint', type=str, default='', help='frozen ARD checkpoint used by the stage22.20 diagnostic')
    parser.add_argument('--stage22_ard_diagnostic_metrics_path', type=str, default='', help='stage22.20 diagnostic summary JSON output path')
    parser.add_argument('--stage22_ard_diagnostic_samples_path', type=str, default='', help='stage22.20 per-sample NPZ output path')
    parser.add_argument('--stage22_27_bf_cmvd_diagnostic', action='store_true', help='attribute frozen BF-CMVD-v2 teacher/student gaps on validation')
    parser.add_argument('--stage22_27_checkpoint', type=str, default='', help='frozen BF-CMVD-v2 checkpoint used by stage22.27')
    parser.add_argument('--stage22_27_metrics_path', type=str, default='', help='stage22.27 diagnostic summary JSON output path')
    parser.add_argument('--stage22_27_samples_path', type=str, default='', help='stage22.27 compact token diagnostic NPZ output path')
    parser.add_argument('--stage22_27_target_rate', type=float, default=0.15, help='frozen equal-count nested removal rate for stage22.27')
    parser.add_argument('--stage22_27_block_lengths', type=int, nargs='+', default=[5, 20], help='frozen teacher block lengths for stage22.27')
    parser.add_argument('--stage22_27_nested_seed', type=int, default=0, help='frozen teacher nested seed for stage22.27')
    parser.add_argument('--stage22_27_original_weight', type=float, default=0.5, help='frozen original teacher weight for stage22.27')
    parser.add_argument('--stage22_27_l5_weight', type=float, default=0.25, help='frozen L5 teacher weight for stage22.27')
    parser.add_argument('--stage22_27_l20_weight', type=float, default=0.25, help='frozen L20 teacher weight for stage22.27')
    parser.add_argument('--stage22_31_pba_diagnostic', action='store_true', help='attribute frozen PBA-CMVD tail risk, spillover, and teacher-direction gaps on validation')
    parser.add_argument('--stage22_31_checkpoint', type=str, default='', help='frozen PBA-CMVD delta checkpoint used by stage22.31')
    parser.add_argument('--stage22_31_backbone_checkpoint', type=str, default='', help='frozen PG95 source checkpoint used by stage22.31')
    parser.add_argument('--stage22_31_metrics_path', type=str, default='', help='stage22.31 diagnostic summary JSON output path')
    parser.add_argument('--stage22_31_samples_path', type=str, default='', help='stage22.31 compact token diagnostic NPZ output path')
    parser.add_argument('--stage22_31_target_rate', type=float, default=0.15, help='frozen equal-count nested removal rate for stage22.31')
    parser.add_argument('--stage22_31_block_lengths', type=int, nargs='+', default=[5, 20], help='frozen teacher block lengths for stage22.31')
    parser.add_argument('--stage22_31_nested_seed', type=int, default=0, help='frozen teacher nested seed for stage22.31')
    parser.add_argument('--stage22_31_original_weight', type=float, default=0.5, help='frozen original teacher weight for stage22.31')
    parser.add_argument('--stage22_31_l5_weight', type=float, default=0.25, help='frozen L5 teacher weight for stage22.31')
    parser.add_argument('--stage22_31_l20_weight', type=float, default=0.25, help='frozen L20 teacher weight for stage22.31')
    parser.add_argument('--stage22_29_dmrb_diagnostic', action='store_true', help='attribute frozen DMRB branch supervision and composition failures on validation')
    parser.add_argument('--stage22_29_checkpoint', type=str, default='', help='frozen DMRB checkpoint used by stage22.29')
    parser.add_argument('--stage22_29_metrics_path', type=str, default='', help='stage22.29 diagnostic summary JSON output path')
    parser.add_argument('--stage22_29_samples_path', type=str, default='', help='stage22.29 compact token diagnostic NPZ output path')
    parser.add_argument('--stage22_29_target_rate', type=float, default=0.15, help='frozen equal-count nested removal rate for stage22.29')
    parser.add_argument('--stage22_29_block_lengths', type=int, nargs='+', default=[5, 20], help='frozen teacher block lengths for stage22.29')
    parser.add_argument('--stage22_29_nested_seed', type=int, default=0, help='frozen teacher nested seed for stage22.29')
    parser.add_argument('--stage22_29_original_weight', type=float, default=0.5, help='frozen original teacher weight for stage22.29')
    parser.add_argument('--stage22_29_l5_weight', type=float, default=0.25, help='frozen L5 teacher weight for stage22.29')
    parser.add_argument('--stage22_29_l20_weight', type=float, default=0.25, help='frozen L20 teacher weight for stage22.29')
    parser.add_argument('--stage14_uncertainty_diagnostic', action='store_true', help='run frozen validation-only nested-mask uncertainty diagnosis')
    parser.add_argument('--stage14_checkpoint', type=str, default='', help='frozen PG95 checkpoint used by the stage14 diagnostic')
    parser.add_argument('--stage14_metrics_path', type=str, default='', help='stage14 validation summary JSON path')
    parser.add_argument('--stage14_samples_path', type=str, default='', help='stage14 per-sample NPZ path')
    parser.add_argument('--stage14_nested_target_rate', type=float, default=0.15, help='target additional block-missing rate for stage14')
    parser.add_argument('--stage14_nested_block_length', type=int, default=5, help='causal nested-mask block length for stage14')
    parser.add_argument('--stage14_nested_seed', type=int, default=0, help='deterministic nested-mask seed for stage14')
    parser.add_argument('--stage15_nested_ensemble_test', action='store_true', help='evaluate the fixed causal nested-mask equal ensemble on test')
    parser.add_argument('--stage15_checkpoint', type=str, default='', help='frozen PG95 checkpoint used by the stage15 test')
    parser.add_argument('--stage15_metrics_path', type=str, default='', help='stage15 test summary JSON path')
    parser.add_argument('--stage15_samples_path', type=str, default='', help='stage15 per-sample test NPZ path')
    parser.add_argument('--stage15_nested_target_rate', type=float, default=0.15, help='fixed additional missing rate for stage15')
    parser.add_argument('--stage15_nested_block_length', type=int, default=5, help='fixed causal nested block length for stage15')
    parser.add_argument('--stage15_nested_seed', type=int, default=0, help='deterministic nested-mask seed for stage15')
    parser.add_argument('--stage15_ensemble_weight', type=float, default=0.5, help='fixed nested-view weight for stage15')
    parser.add_argument('--stage18_equal_rate_multiscale_validation', action='store_true', help='evaluate equal-count L1/L5/L20 and fixed multiscale CNM-SE on validation')
    parser.add_argument('--stage18_checkpoint', type=str, default='', help='frozen PG95 checkpoint used by the stage18 validation')
    parser.add_argument('--stage18_metrics_path', type=str, default='', help='stage18 validation summary JSON path')
    parser.add_argument('--stage18_samples_path', type=str, default='', help='stage18 per-sample validation NPZ path')
    parser.add_argument('--stage18_target_rate', type=float, default=0.15, help='per-sample target additional missing rate for stage18')
    parser.add_argument('--stage18_block_lengths', type=int, nargs='+', default=[1, 5, 20], help='frozen equal-count causal block lengths for stage18')
    parser.add_argument('--stage18_nested_seed', type=int, default=0, help='deterministic equal-count mask seed for stage18')
    parser.add_argument('--stage18_original_weight', type=float, default=0.5, help='frozen original-view weight for stage18 multiscale fusion')
    parser.add_argument('--stage18_l5_weight', type=float, default=0.25, help='frozen L5-view weight for stage18 multiscale fusion')
    parser.add_argument('--stage18_l20_weight', type=float, default=0.25, help='frozen L20-view weight for stage18 multiscale fusion')
    parser.add_argument('--stage18_timing_warmup_batches', type=int, default=10, help='number of initial validation batches excluded from stage18 timing')
    parser.add_argument('--stage19_equal_rate_multiscale_test', action='store_true', help='evaluate the frozen equal-count multiscale CNM-SE on an external test split')
    parser.add_argument('--stage19_checkpoint', type=str, default='', help='frozen external-dataset checkpoint used by stage19')
    parser.add_argument('--stage19_metrics_path', type=str, default='', help='stage19 external-test summary JSON path')
    parser.add_argument('--stage19_samples_path', type=str, default='', help='stage19 per-sample external-test NPZ path')
    parser.add_argument('--stage19_target_rate', type=float, default=0.15, help='per-sample target additional missing rate for stage19')
    parser.add_argument('--stage19_block_lengths', type=int, nargs='+', default=[1, 5, 20], help='frozen equal-count causal block lengths for stage19')
    parser.add_argument('--stage19_nested_seed', type=int, default=0, help='deterministic equal-count mask seed for stage19')
    parser.add_argument('--stage19_original_weight', type=float, default=0.5, help='frozen original-view weight for stage19 multiscale fusion')
    parser.add_argument('--stage19_l5_weight', type=float, default=0.25, help='frozen L5-view weight for stage19 multiscale fusion')
    parser.add_argument('--stage19_l20_weight', type=float, default=0.25, help='frozen L20-view weight for stage19 multiscale fusion')
    parser.add_argument('--stage19_timing_warmup_batches', type=int, default=10, help='number of initial external-test batches excluded from stage19 timing')
    parser.add_argument('--stage21_equal_rate_multiscale_evaluation', action='store_true', help='evaluate the frozen equal-count multiscale CNM-SE on Electricity validation or test')
    parser.add_argument('--stage21_split', type=str, choices=['val', 'test'], default='val', help='Step-21 evaluation split; test is permitted only by the external scientific gate')
    parser.add_argument('--stage21_checkpoint', type=str, default='', help='frozen Electricity PG95 checkpoint used by stage21')
    parser.add_argument('--stage21_metrics_path', type=str, default='', help='stage21 summary JSON path')
    parser.add_argument('--stage21_samples_path', type=str, default='', help='stage21 per-sample NPZ path')
    parser.add_argument('--stage21_target_rate', type=float, default=0.15, help='per-sample target additional missing rate for stage21')
    parser.add_argument('--stage21_block_lengths', type=int, nargs='+', default=[1, 5, 20], help='frozen equal-count causal block lengths for stage21')
    parser.add_argument('--stage21_nested_seed', type=int, default=0, help='deterministic equal-count mask seed for stage21')
    parser.add_argument('--stage21_original_weight', type=float, default=0.5, help='frozen original-view weight for stage21 multiscale fusion')
    parser.add_argument('--stage21_l5_weight', type=float, default=0.25, help='frozen L5-view weight for stage21 multiscale fusion')
    parser.add_argument('--stage21_l20_weight', type=float, default=0.25, help='frozen L20-view weight for stage21 multiscale fusion')
    parser.add_argument('--stage21_timing_warmup_batches', type=int, default=10, help='number of initial stage21 batches excluded from timing')
    parser.add_argument('--stage22_24_batch_parallel_comparison', action='store_true', help='compare serial and batch-parallel three-view CNM-SE without changing predictions')
    parser.add_argument('--stage22_25_cuda_stream_comparison', action='store_true', help='compare serial CNM-SE with three same-shape concurrent CUDA streams')
    parser.add_argument('--mask',action='store_true', help='whether use mask')  # required=True
    parser.add_argument('--check',action='store_true', help='whether use mask')  # required=True
    parser.add_argument('--reverse_mask',action='store_true', help='whether use reverse mask')  # required=True
    parser.add_argument('--grud',action='store_true', help='whether use mask')  # required=True
    parser.add_argument('--s4_pred',action='store_true', help='whether use mask')  # required=True
    parser.add_argument('--s4_pred_inner',action='store_true', help='whether use mask')  # required=True
    parser.add_argument('--bidirectional',action='store_true', help='whether use bidirectional')  # required=True
    parser.add_argument('--pos_emb',action='store_true', help='whether use mask')  # required=True
    parser.add_argument('--pad', type=str,  default=0.0, help='how to fill nan')  # required=True
    parser.add_argument('--drop', action='store_true', help='whether to drop nan')  # 
    parser.add_argument('--analysis', action='store_true', help='whether to drop nan') 
    parser.add_argument('--mean_type', type=str, default='emperical', help='dataset type')

    parser.add_argument('--fillna', type=str, default='mean', help='dataset type')
    parser.add_argument('--encoder_type', type=int, default=1, help='dataset type')
 
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=10, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=862, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=862, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=862, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--classification', type=int, default=0, help='whether to classify')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--embed_type', type=int,default=2)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--loss_weight1', type=float, default=1.0, help='loss weight1')
    parser.add_argument('--loss_weight2', type=float, default=1.0, help='loss weight2')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='sgd', help='optimizer identity')
    parser.add_argument('--max_train_batches', type=int, default=-1, help='smoke-only train batch cap; -1 disables the cap')
    parser.add_argument('--max_eval_batches', type=int, default=-1, help='smoke-only validation/test batch cap; -1 disables the cap')
    parser.add_argument('--wandb_mode', type=str, choices=['disabled', 'offline', 'online'], default='disabled')
    parser.add_argument('--deterministic_cluster', action='store_true', help='use deterministic best-of-N prototype clustering')
    parser.add_argument('--use_mipp', action='store_true', help='enable mask-intervention prototype purification')
    parser.add_argument('--use_mrra', action='store_true', help='enable mask-responsive residual adaptation')
    parser.add_argument('--mrra_version', type=str, choices=['v1', 'v2_normalized_bounded', 'v3_gradient_isolated_two_timescale'], default='v1', help='MRRA architecture version')
    parser.add_argument('--mrra_lr_multiplier', type=float, default=1.0, help='MRRA optimizer learning-rate multiplier')
    parser.add_argument('--mrra_rank', type=int, default=32, help='low-rank MRRA width')
    parser.add_argument('--mrra_kernel_size', type=int, default=5, help='causal MRRA temporal kernel size')
    parser.add_argument('--mrra_reconstruction_weight', type=float, default=0.1, help='missing-history reconstruction loss weight')
    parser.add_argument('--use_radr', action='store_true', help='enable rank-adaptive dual-risk forecasting objective')
    parser.add_argument('--use_mcpr', action='store_true', help='enable frozen-backbone mask-conditioned Pareto residual adaptation')
    parser.add_argument('--mcpr_objective', type=str, choices=['mse_only', 'pareto_common_descent', 'tail_trust_mse', 'analytic_decision_distill', 'benefit_filtered_multiview_distill', 'token_benefit_filtered_multiview_distill', 'dual_scale_residual_basis_distill'], default='mse_only', help='MCPR adapter optimization rule')
    parser.add_argument('--mcpr_rank', type=int, default=32, help='low-rank MCPR width')
    parser.add_argument('--mcpr_kernel_size', type=int, default=5, help='causal MCPR temporal kernel size')
    parser.add_argument('--mcpr_learning_rate', type=float, default=0.001, help='MCPR-only optimizer learning rate')
    parser.add_argument('--mcpr_backbone_checkpoint', type=str, default='', help='frozen PG95 source checkpoint for MCPR adaptation')
    parser.add_argument('--mcpr_delta_checkpoint', action='store_true', help='store MCPR-only parameters plus a hashed frozen source reference')
    parser.add_argument('--use_bf_cmvd', action='store_true', help='enable benefit-filtered causal multiview distillation for MCPR')
    parser.add_argument('--bf_cmvd_target_rate', type=float, default=0.15, help='training-only equal-count nested removal rate')
    parser.add_argument('--bf_cmvd_block_lengths', type=int, nargs='+', default=[5, 20], help='training-only teacher block lengths')
    parser.add_argument('--bf_cmvd_nested_seed', type=int, default=0, help='base seed for epoch-varying teacher masks')
    parser.add_argument('--bf_cmvd_original_weight', type=float, default=0.5, help='frozen original teacher weight')
    parser.add_argument('--bf_cmvd_l5_weight', type=float, default=0.25, help='frozen L5 teacher weight')
    parser.add_argument('--bf_cmvd_l20_weight', type=float, default=0.25, help='frozen L20 teacher weight')
    parser.add_argument('--use_dmrb', action='store_true', help='replace the single MCPR head with dual L5/L20 residual bases')
    parser.add_argument('--dmrb_l5_enabled', action='store_true', help='enable the DMRB L5 residual basis')
    parser.add_argument('--dmrb_l20_enabled', action='store_true', help='enable the DMRB L20 residual basis')
    parser.add_argument('--use_pba_cmvd', action='store_true', help='amortize the frozen causal multiview teacher into the nonlinear S4 tail')
    parser.add_argument('--pba_backbone_checkpoint', type=str, default='', help='frozen PG95 source checkpoint for PBA-CMVD')
    parser.add_argument('--pba_tail_layers', type=int, default=2, help='number of final S4 layers opened by PBA-CMVD')
    parser.add_argument('--pba_delta_checkpoint', action='store_true', help='store only the changed PBA tail plus a hashed source reference')
    parser.add_argument('--pba_target_rate', type=float, default=0.15, help='training-only equal-count nested removal rate for PBA-CMVD')
    parser.add_argument('--pba_block_lengths', type=int, nargs='+', default=[5, 20], help='frozen teacher block lengths for PBA-CMVD')
    parser.add_argument('--pba_nested_seed', type=int, default=0, help='base seed for epoch-varying PBA-CMVD teacher masks')
    parser.add_argument('--pba_original_weight', type=float, default=0.5, help='frozen original-view PBA-CMVD teacher weight')
    parser.add_argument('--pba_l5_weight', type=float, default=0.25, help='frozen L5 PBA-CMVD teacher weight')
    parser.add_argument('--pba_l20_weight', type=float, default=0.25, help='frozen L20 PBA-CMVD teacher weight')
    parser.add_argument('--use_ttrr', action='store_true', help='enable tail-aware trust-region scaling inside MCPR')
    parser.add_argument('--ttrr_tail_fraction', type=float, default=0.1, help='fraction of largest positive candidate-vs-backbone MSE harms')
    parser.add_argument('--ttrr_tail_weight', type=float, default=1.0, help='weight of the TTRR positive-tail non-inferiority penalty')
    parser.add_argument('--use_ard', action='store_true', help='enable analytic residual decision distillation inside MCPR')
    parser.add_argument('--ard_hidden_dim', type=int, default=16, help='ARD label-free feature hidden width')
    parser.add_argument('--ard_distill_warmup_epochs', type=int, default=1, help='raw-residual-only epochs before ARD distillation')
    parser.add_argument('--ard_energy_epsilon', type=float, default=1e-8, help='minimum observed raw residual energy for ARD teacher validity')
    parser.add_argument('--ard_action_epsilon', type=float, default=1e-6, help='analytic alpha boundary used for OFF/FULL actions')
    parser.add_argument('--ard_action_weight', type=float, default=1.0, help='ARD action cross-entropy weight')
    parser.add_argument('--ard_scale_weight', type=float, default=1.0, help='ARD continuous scale distillation weight')
    parser.add_argument('--use_content_mask_factorization', action='store_true', help='factor content and mask statistics inside MIPP')
    parser.add_argument('--use_mask_intervention_pair', action='store_true', help='construct a paired historical mask view during training')
    parser.add_argument('--use_stability_write_gate', action='store_true', help='write only mask-stable content candidates to prototype memory')
    parser.add_argument('--mipp_inv_weight', type=float, default=0.05)
    parser.add_argument('--mipp_var_weight', type=float, default=0.01)
    parser.add_argument('--mipp_cov_weight', type=float, default=0.001)
    parser.add_argument('--mipp_dec_weight', type=float, default=0.01)
    parser.add_argument('--mipp_mask_weight', type=float, default=0.01)
    parser.add_argument('--mipp_variance_target', type=float, default=1.0)
    parser.add_argument('--mipp_stability_threshold', type=float, default=0.80)
    parser.add_argument(
        '--mipp_write_gate_mode',
        type=str,
        choices=[
            'absolute', 'budget_topk', 'stability_diverse_rank',
            'mask_regime_balanced_rank',
        ],
        default='absolute',
        help='absolute, budget top-k, content-diverse rank, or mask-regime balanced rank admission',
    )
    parser.add_argument(
        '--mipp_write_budget',
        type=int,
        default=50,
        help='candidate budget used by budgeted write modes; must match the existing memory M budget',
    )
    parser.add_argument('--mipp_pair_rate_min', type=float, default=0.15)
    parser.add_argument('--mipp_pair_rate_max', type=float, default=0.50)
    parser.add_argument('--mipp_pair_block_lengths', type=int, nargs='+', default=[1, 5, 20])
    parser.add_argument(
        '--use_cfr',
        action='store_true',
        help='enable training-only counterfactual regret regularization for prototype reads',
    )
    # FUSR (Forecast-Utility-Aware Selective Retrieval) is strictly opt-in.
    # Keeping the master switch false preserves the original S4M retrieval
    # path and avoids instantiating or optimizing router parameters.
    parser.add_argument('--use_fusr', action='store_true', help='enable forecast-utility-aware selective prototype retrieval')
    parser.add_argument('--use_utility_read_gate', action='store_true', help='apply the learned FUSR utility gate to retrieved prototypes')
    parser.add_argument('--use_fusr_spectral_uncertainty', action='store_true', help='include history-only spectral uncertainty in router features')
    parser.add_argument('--fusr_hidden_dim', type=int, default=32, help='hidden width of the lightweight utility router')
    parser.add_argument('--fusr_util_weight', type=float, default=0.10, help='weight of the FUSR utility regression objective')
    parser.add_argument('--fusr_sign_weight', type=float, default=0.50, help='weight of the beneficial/harmful sign objective')
    parser.add_argument('--fusr_gate_temperature', type=float, default=0.25, help='temperature used by the differentiable training gate')
    parser.add_argument('--fusr_uncertainty_penalty', type=float, default=1.0, help='uncertainty penalty applied to predicted retrieval utility')
    parser.add_argument('--fusr_delta', type=float, default=0.0, help='utility decision margin selected without test targets')
    parser.add_argument('--fusr_utility_clip', type=float, default=1.0, help='symmetric clip applied to normalized utility targets')
    parser.add_argument('--fusr_neutral_margin', type=float, default=0.02, help='absolute utility margin treated as neutral')
    parser.add_argument('--fusr_eval_paired_diagnostics', action='store_true', help='run paired on/off diagnostics during evaluation only')
    parser.add_argument(
        '--fusr_objective_version',
        type=str,
        choices=['v1_joint', 'v2_frozen_balanced_rank'],
        default='v1_joint',
        help='FUSR training protocol; v2 fits only the router on a frozen backbone',
    )
    parser.add_argument(
        '--fusr_feature_set',
        type=str,
        choices=['base8', 'intervention14'],
        default='base8',
        help='target-free router feature contract',
    )
    parser.add_argument(
        '--fusr_gate_score',
        type=str,
        choices=['utility_lcb', 'sign_logit'],
        default='utility_lcb',
        help='router score used for the retrieval decision',
    )
    parser.add_argument(
        '--fusr_gate_prior_mode',
        type=str,
        choices=['learned', 'none'],
        default='learned',
        help='whether the FUSR gate includes the legacy trainable prior',
    )
    parser.add_argument('--fusr_router_only', action='store_true', help='freeze the S4M backbone and fit only the FUSR router')
    parser.add_argument('--fusr_backbone_checkpoint', type=str, default='', help='frozen baseline checkpoint used by FUSR V2')
    parser.add_argument('--fusr_router_epochs', type=int, default=100, help='maximum offline router-training epochs')
    parser.add_argument('--fusr_router_patience', type=int, default=10, help='offline router early-stopping patience')
    parser.add_argument('--fusr_router_batch_size', type=int, default=512, help='offline router cache batch size')
    parser.add_argument('--fusr_router_lr', type=float, default=1e-3, help='offline router learning rate')
    parser.add_argument('--fusr_router_weight_decay', type=float, default=1e-4, help='offline router weight decay')
    parser.add_argument('--fusr_rank_weight', type=float, default=0.5, help='pairwise beneficial-versus-harmful ranking loss weight')
    parser.add_argument('--fusr_fit_fraction', type=float, default=0.6, help='chronological validation-cache fraction used to fit the router')
    parser.add_argument('--fusr_dev_fraction', type=float, default=0.2, help='chronological validation-cache fraction used for router selection')
    parser.add_argument('--fusr_embargo_windows', type=int, default=1, help='number of windows embargoed between cache partitions')
    parser.add_argument('--fusr_cache_split', type=str, choices=['val'], default='val', help='source split for frozen on/off cache construction')
    parser.add_argument('--fusr_cache_max_batches', type=int, default=-1, help='smoke-only cache batch cap; -1 uses the complete split')
    parser.add_argument(
        '--fusr_calibration_mode',
        type=str,
        choices=['validation_min_mse'],
        default='validation_min_mse',
        help='validation-only threshold calibration rule',
    )
    parser.add_argument(
        '--fusr_calibration_tie_break',
        type=str,
        choices=['prefer_on'],
        default='prefer_on',
        help='deterministic threshold tie-break rule',
    )

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--cpu', action='store_true', help='force CPU execution for smoke testing')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')
    parser.add_argument('--d_state', type=int, default=32, help='parameter of Mamba Block')
    parser.add_argument('--ssm', type=str, default='s4', help='ssm type')
    parser.add_argument('--num_dG', type=float, default=1)
    parser.add_argument('--num_pattn', type=int, default=20)    

    
    # Mermory Module
    parser.add_argument('--short_len', type=int, default=50)    
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--W', type=int, default=6)
    parser.add_argument('--en_conv_hidden_size', type=int, default=256)    
    parser.add_argument('--en_rnn_hidden_sizes', type=int, nargs='+', default=[20,32])    
    parser.add_argument('--output_keep_prob', type=float, default=0.9)    
    parser.add_argument('--input_keep_prob', type=float, default=0.9)  
    parser.add_argument('--weight', type=float, default=1.0)      
    parser.add_argument('--individual', type=int, default=1)
    parser.add_argument('--memnet', type=int, default=13)
    parser.add_argument('--mem_type', type=int, default=2)
    parser.add_argument('--shuffle', type=int, default=1)


    #diffusion
    parser.add_argument('--layers', type=int, default=2)    
    parser.add_argument('--channels', type=int, default=64)    
    parser.add_argument('--nheads', type=int, default=8)    
    parser.add_argument('--diffusion_embedding_dim', type=int, default=128)    
    parser.add_argument('--beta_start', type=float, default=0.0001)    
    parser.add_argument('--beta_end', type=float, default=0.5)    
    parser.add_argument('--num_steps', type=int, default=50)    
    parser.add_argument('--schedule', type=str, default='quad')    
    parser.add_argument('--is_linear', type=bool, default=True)    

    parser.add_argument('--is_unconditional', type=int, default=0)    
    parser.add_argument('--timeemb', type=int, default=128)    
    parser.add_argument('--featureemb', type=int, default=16)    
    parser.add_argument('--target_strategy', type=str, default="test")    
    parser.add_argument('--num_sample_features', type=int, default=64)    
    parser.add_argument("--nsample", type=int, default=10)

    #Brits
    parser.add_argument("--impute_weight", type=float, default=0.3)
    parser.add_argument("--label_weight", type=float, default=1)
    parser.add_argument("--impute_path", type=str, default='./')


    #BiaTCGNet
    parser.add_argument("--gcn_true", type=bool, default=True)
    parser.add_argument("--buildA_true", type=bool, default=True)
    parser.add_argument("--gcn_depth", type=int, default=2)
    parser.add_argument("--num_nodes", type=int, default=207)
    parser.add_argument('--kernel_set', type=list, default=[2,3,6,7], help='kernel set')
    parser.add_argument("--subgraph_size", type=int, default=5)
    parser.add_argument("--node_dim", type=int, default=3)
    # parser.add_argument("--node_number", type=int, default=3)
    parser.add_argument("--conv_channels", type=int, default=8)
    parser.add_argument("--skip_channels", type=int, default=8)
    parser.add_argument("--seq_channels", type=int, default=8)
    parser.add_argument("--residual_channels", type=int, default=8)
    parser.add_argument("--end_channels", type=int, default=8)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--in_dim", type=int, default=1)
    parser.add_argument("--tanhalpha", type=float, default=3)
    parser.add_argument("--propalpha", type=float, default=0.05)
    parser.add_argument("--layer_norm_affline", type=bool, default=True)


    # memory bank
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--topK", type=int, default=10)
    parser.add_argument("--topM", type=int, default=100)
    parser.add_argument("--max_k", type=int, default=1000)
    parser.add_argument("--x_in", type=int, default=32)
    parser.add_argument("--thres1", type=float, default=0.6)
    parser.add_argument("--thres2", type=float, default=0.3)
    parser.add_argument("--M", type=int, default=30)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--memory_size", type=int, default=256)
    parser.add_argument("--per_mem_size", type=int, default=50)

    parser.add_argument('--pretrain', action='store_true', help='whether to pretrain') 
    parser.add_argument('--no_renew', action='store_true', help='whether to pretrain') 
    parser.add_argument("--tau", type=int, default=32)
    parser.add_argument('--remask', type=float,default=0.0, help='whether to remask') 
    parser.add_argument("--hidden_v", type=int, default=32)
    parser.add_argument("--hidden_t", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--mem_repeat", type=int, default=10)
    parser.add_argument('--kernel', type=int, nargs='+', default=[5,5])    
    parser.add_argument('--moving_avgs', type=int, nargs='+', default=[25,20,15,10,5,5])   
    parser.add_argument('--tucker_kernels', type=int, nargs='+', default=[10,5,5])   


    #CRU
    parser.add_argument('--task', type=str, default='extrapolation')   
    parser.add_argument('--ts', type=float, default=1, help="Scaling factor of timestamps for numerical stability.")
    parser.add_argument('--hidden-units', type=int, default=24, help="Hidden units of encoder and decoder.")
    parser.add_argument('--num-basis', type=int, default=15, help="Number of basis matrices to use in transition model for locally-linear transitions. K in paper")
    parser.add_argument('--bandwidth', type=int, default=3, help="Bandwidth for basis matrices A_k. b in paper")
    parser.add_argument('--enc-var-activation', type=str, default='elup1', help="Variance activation function in encoder. Possible values elup1, exp, relu, square")
    parser.add_argument('--dec-var-activation', type=str, default='elup1', help="Variance activation function in decoder. Possible values elup1, exp, relu, square")
    parser.add_argument('--trans_net_hidden_activation', type=str, default='tanh', help="Activation function for transition net.")
    parser.add_argument('--trans_net_hidden_units', type=list, default=[], help="Hidden units of transition net.")
    parser.add_argument('--trans_var_activation', type=str, default='elup1', help="Activation function for transition net.")
    parser.add_argument('--learn_trans_covar', type=bool, default=True, help="If to learn transition covariance.")
    parser.add_argument('--learn_initial_state_covar', action='store_true', help="If to learn the initial state covariance.")
    parser.add_argument('--initial_state_covar', type=int, default=1, help="Value of initial state covariance.")
    parser.add_argument('--trans_covar', type=float, default=0.1, help="Value of initial transition covariance.")
    parser.add_argument('--t_sensitive_trans_net',  action='store_true', help="If to provide the time gap as additional input to the transition net. Used for RKN-Delta_t model in paper")
    parser.add_argument('--f-cru',  action='store_true', help="If to use fast transitions based on eigendecomposition of the state transitions (f-CRU variant).")
    parser.add_argument('--rkn',  action='store_true', help="If to use discrete state transitions (RKN baseline).")
    parser.add_argument('--orthogonal', type=bool, default=True, help="If to use orthogonal basis matrices in the f-CRU variant.")


    parser.add_argument('--merge_size',type=int,default = 2)
    parser.add_argument('--dp_rank', type=int,default = 8)
    parser.add_argument('--use_statistic',action='store_true', default=False)
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--task_name', type=str,default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    args = parser.parse_args()
    mipp_subswitches = (
        args.use_content_mask_factorization,
        args.use_mask_intervention_pair,
        args.use_stability_write_gate,
        args.mipp_write_gate_mode != 'absolute',
    )
    if any(mipp_subswitches) and not args.use_mipp:
        parser.error('MIPP sub-switches require --use_mipp')
    if not 0.0 <= args.mipp_pair_rate_min <= args.mipp_pair_rate_max < 1.0:
        parser.error('MIPP paired mask rates must satisfy 0 <= min <= max < 1')
    if any(length < 1 or length > args.seq_len for length in args.mipp_pair_block_lengths):
        parser.error('MIPP paired block lengths must be within [1, seq_len]')
    if args.mipp_write_budget < 1:
        parser.error('MIPP write budget must be positive')
    if args.mipp_write_gate_mode in (
        'budget_topk', 'stability_diverse_rank', 'mask_regime_balanced_rank'
    ):
        if not args.use_stability_write_gate:
            parser.error('MIPP budgeted write modes require --use_stability_write_gate')
        if args.mipp_write_budget != args.M:
            parser.error('MIPP budgeted write modes must reuse the existing M write budget')
    if args.use_cfr and not args.use_mipp:
        parser.error('CFR requires the PG95/MIPP prototype path')
    if args.use_cfr and args.use_fusr:
        parser.error('CFR and FUSR are mutually exclusive training interventions')
    if args.fusr_hidden_dim < 1:
        parser.error('FUSR hidden dimension must be positive')
    if args.fusr_util_weight < 0.0 or args.fusr_sign_weight < 0.0:
        parser.error('FUSR loss weights must be non-negative')
    if args.fusr_gate_temperature <= 0.0:
        parser.error('FUSR gate temperature must be positive')
    if args.fusr_utility_clip <= 0.0:
        parser.error('FUSR utility target clip must be positive')
    if args.fusr_uncertainty_penalty < 0.0:
        parser.error('FUSR uncertainty penalty must be non-negative')
    if args.fusr_neutral_margin < 0.0:
        parser.error('FUSR neutral margin must be non-negative')
    fusr_subswitches = (
        args.use_utility_read_gate,
        args.use_fusr_spectral_uncertainty,
        args.fusr_eval_paired_diagnostics,
    )
    if any(fusr_subswitches) and not args.use_fusr:
        parser.error('FUSR sub-switches require --use_fusr')
    if args.hcup_diagnostic:
        if args.is_training != 0:
            parser.error('HCUP diagnostic requires --is_training 0')
        if not args.hcup_checkpoint or not os.path.isfile(args.hcup_checkpoint):
            parser.error('HCUP diagnostic requires an existing --hcup_checkpoint')
        if not args.hcup_metrics_path or not args.hcup_samples_path:
            parser.error('HCUP diagnostic requires metrics and samples paths')
        if args.hcup_probe_lag <= 0:
            parser.error('HCUP probe lag must be positive')
        if args.hcup_probe_horizon <= 0:
            parser.error('HCUP probe horizon must be positive')
        if args.hcup_probe_horizon > args.hcup_probe_lag:
            parser.error('HCUP probe horizon cannot exceed probe lag')
        if args.hcup_probe_horizon > args.pred_len:
            parser.error('HCUP probe horizon cannot exceed pred_len')
        if not math.isfinite(args.hcup_threshold):
            parser.error('HCUP threshold must be finite')
    if args.stage12_validation_diagnostic:
        if args.is_training != 0:
            parser.error('stage12 validation diagnostic requires --is_training 0')
        if not args.stage12_checkpoint or not os.path.isfile(args.stage12_checkpoint):
            parser.error('stage12 validation diagnostic requires an existing checkpoint')
        if not args.stage12_metrics_path or not args.stage12_samples_path:
            parser.error('stage12 validation diagnostic requires metrics and samples paths')
    if args.stage22_validation_diagnostic:
        if args.is_training != 0:
            parser.error('stage22 validation diagnostic requires --is_training 0')
        if not args.stage22_checkpoint or not os.path.isfile(args.stage22_checkpoint):
            parser.error('stage22 validation diagnostic requires an existing checkpoint')
        if not args.stage22_metrics_path or not args.stage22_samples_path:
            parser.error('stage22 validation diagnostic requires metrics and samples paths')
        if not 0.0 < args.stage22_nested_target_rate < 1.0:
            parser.error('stage22 nested target rate must lie in (0, 1)')
        if not 1 <= args.stage22_nested_block_length <= args.seq_len:
            parser.error('stage22 nested block length must lie in [1, seq_len]')
        if args.stage22_internal_features and not args.use_mrra:
            parser.error('stage22 internal features require --use_mrra')
        if args.stage22_mcpr_attribution and not args.use_mcpr:
            parser.error('stage22 MCPR attribution requires --use_mcpr')
        if args.stage22_mcpr_attribution and args.stage22_internal_features:
            parser.error('stage22 MCPR attribution cannot be combined with MRRA internal features')
    elif args.stage22_mcpr_attribution:
        parser.error('stage22 MCPR attribution requires stage22 validation diagnostic mode')
    if args.stage22_inference_timing:
        if args.is_training != 0:
            parser.error('stage22 inference timing requires --is_training 0')
        if not args.stage22_timing_checkpoint or not os.path.isfile(
            args.stage22_timing_checkpoint
        ):
            parser.error('stage22 inference timing requires an existing checkpoint')
        if not args.stage22_timing_metrics_path:
            parser.error('stage22 inference timing requires a metrics path')
        if args.stage22_timing_warmup_batches < 0:
            parser.error('stage22 timing warmup batches must be non-negative')
        if args.stage22_timing_repeats < 1:
            parser.error('stage22 timing repeats must be positive')
        if (
            args.max_eval_batches > 0
            and args.stage22_timing_warmup_batches >= args.max_eval_batches
        ):
            parser.error('stage22 timing warmup must leave a measured batch')
    if args.stage22_ard_information_diagnostic:
        if args.is_training != 0:
            parser.error('stage22 ARD information diagnostic requires --is_training 0')
        if not args.use_mcpr or not args.use_ard:
            parser.error('stage22 ARD information diagnostic requires MCPR and ARD')
        if args.mcpr_objective != 'analytic_decision_distill':
            parser.error('stage22 ARD information diagnostic requires the analytic objective')
        if not args.stage22_ard_diagnostic_checkpoint or not os.path.isfile(
            args.stage22_ard_diagnostic_checkpoint
        ):
            parser.error('stage22 ARD information diagnostic requires an existing checkpoint')
        if not args.stage22_ard_diagnostic_metrics_path:
            parser.error('stage22 ARD information diagnostic requires a metrics path')
        if not args.stage22_ard_diagnostic_samples_path:
            parser.error('stage22 ARD information diagnostic requires a samples path')
    if args.stage22_27_bf_cmvd_diagnostic:
        if args.is_training != 0:
            parser.error('stage22.27 BF-CMVD diagnostic requires --is_training 0')
        if not args.use_mipp or not args.use_mcpr or not args.use_bf_cmvd:
            parser.error('stage22.27 requires the PG95 BF-CMVD-v2 model identity')
        if args.mcpr_objective != 'token_benefit_filtered_multiview_distill':
            parser.error('stage22.27 requires the token-filter BF-CMVD-v2 objective')
        if not args.stage22_27_checkpoint or not os.path.isfile(
            args.stage22_27_checkpoint
        ):
            parser.error('stage22.27 requires an existing BF-CMVD-v2 checkpoint')
        if not args.stage22_27_metrics_path or not args.stage22_27_samples_path:
            parser.error('stage22.27 requires metrics and samples paths')
        if not math.isclose(
            args.stage22_27_target_rate, 0.15, rel_tol=0.0, abs_tol=1e-12
        ):
            parser.error('stage22.27 target rate is frozen at 0.15')
        if args.stage22_27_block_lengths != [5, 20]:
            parser.error('stage22.27 teacher block lengths are frozen at 5 20')
        if args.stage22_27_nested_seed != args.seed:
            parser.error('stage22.27 nested seed must equal the evaluation seed')
        if (
            args.stage22_27_original_weight != 0.5
            or args.stage22_27_l5_weight != 0.25
            or args.stage22_27_l20_weight != 0.25
        ):
            parser.error('stage22.27 teacher weights are frozen at 0.5/0.25/0.25')
    if args.stage22_29_dmrb_diagnostic:
        if args.is_training != 0:
            parser.error('stage22.29 DMRB diagnostic requires --is_training 0')
        if not (
            args.use_mipp and args.use_mcpr and args.use_bf_cmvd
            and args.use_dmrb and args.dmrb_l5_enabled and args.dmrb_l20_enabled
        ):
            parser.error('stage22.29 requires the two-branch PG95 DMRB identity')
        if args.mcpr_objective != 'dual_scale_residual_basis_distill':
            parser.error('stage22.29 requires the DMRB scale-factored objective')
        if not args.stage22_29_checkpoint or not os.path.isfile(
            args.stage22_29_checkpoint
        ):
            parser.error('stage22.29 requires an existing DMRB checkpoint')
        if not args.stage22_29_metrics_path or not args.stage22_29_samples_path:
            parser.error('stage22.29 requires metrics and samples paths')
        if not math.isclose(
            args.stage22_29_target_rate, 0.15, rel_tol=0.0, abs_tol=1e-12
        ):
            parser.error('stage22.29 target rate is frozen at 0.15')
        if args.stage22_29_block_lengths != [5, 20]:
            parser.error('stage22.29 teacher block lengths are frozen at 5 20')
        if args.stage22_29_nested_seed != args.seed:
            parser.error('stage22.29 nested seed must equal the evaluation seed')
        if (
            args.stage22_29_original_weight != 0.5
            or args.stage22_29_l5_weight != 0.25
            or args.stage22_29_l20_weight != 0.25
        ):
            parser.error('stage22.29 teacher weights are frozen at 0.5/0.25/0.25')
    if args.stage14_uncertainty_diagnostic:
        if args.is_training != 0:
            parser.error('stage14 uncertainty diagnostic requires --is_training 0')
        if not args.stage14_checkpoint or not os.path.isfile(args.stage14_checkpoint):
            parser.error('stage14 uncertainty diagnostic requires an existing checkpoint')
        if not args.stage14_metrics_path or not args.stage14_samples_path:
            parser.error('stage14 uncertainty diagnostic requires metrics and samples paths')
        if not 0.0 < args.stage14_nested_target_rate < 1.0:
            parser.error('stage14 nested target rate must lie in (0, 1)')
        if not 1 <= args.stage14_nested_block_length <= args.seq_len:
            parser.error('stage14 nested block length must lie in [1, seq_len]')
    if args.stage15_nested_ensemble_test:
        if args.is_training != 0:
            parser.error('stage15 nested ensemble test requires --is_training 0')
        if not args.stage15_checkpoint or not os.path.isfile(args.stage15_checkpoint):
            parser.error('stage15 nested ensemble test requires an existing checkpoint')
        if not args.stage15_metrics_path or not args.stage15_samples_path:
            parser.error('stage15 nested ensemble test requires metrics and samples paths')
        if not 0.0 < args.stage15_nested_target_rate < 1.0:
            parser.error('stage15 nested target rate must lie in (0, 1)')
        if not 1 <= args.stage15_nested_block_length <= args.seq_len:
            parser.error('stage15 nested block length must lie in [1, seq_len]')
        if args.stage15_ensemble_weight != 0.5:
            parser.error('stage15 ensemble weight is frozen at 0.5')
    if args.stage18_equal_rate_multiscale_validation:
        if args.is_training != 0:
            parser.error('stage18 validation requires --is_training 0')
        if not args.stage18_checkpoint or not os.path.isfile(args.stage18_checkpoint):
            parser.error('stage18 validation requires an existing checkpoint')
        if not args.stage18_metrics_path or not args.stage18_samples_path:
            parser.error('stage18 validation requires metrics and samples paths')
        if not 0.0 < args.stage18_target_rate < 1.0:
            parser.error('stage18 target rate must lie in (0, 1)')
        if args.stage18_block_lengths != [1, 5, 20]:
            parser.error('stage18 block lengths are frozen at 1 5 20')
        if (
            args.stage18_original_weight != 0.5
            or args.stage18_l5_weight != 0.25
            or args.stage18_l20_weight != 0.25
        ):
            parser.error('stage18 multiscale weights are frozen at 0.5/0.25/0.25')
        if args.stage18_timing_warmup_batches < 0:
            parser.error('stage18 timing warmup batches must be non-negative')
    if args.stage19_equal_rate_multiscale_test:
        if args.is_training != 0:
            parser.error('stage19 external test requires --is_training 0')
        if not args.stage19_checkpoint or not os.path.isfile(args.stage19_checkpoint):
            parser.error('stage19 external test requires an existing checkpoint')
        if not args.stage19_metrics_path or not args.stage19_samples_path:
            parser.error('stage19 external test requires metrics and samples paths')
        if not 0.0 < args.stage19_target_rate < 1.0:
            parser.error('stage19 target rate must lie in (0, 1)')
        if args.stage19_block_lengths != [1, 5, 20]:
            parser.error('stage19 block lengths are frozen at 1 5 20')
        if (
            args.stage19_original_weight != 0.5
            or args.stage19_l5_weight != 0.25
            or args.stage19_l20_weight != 0.25
        ):
            parser.error('stage19 multiscale weights are frozen at 0.5/0.25/0.25')
        if args.stage19_timing_warmup_batches < 0:
            parser.error('stage19 timing warmup batches must be non-negative')
    if (
        args.stage21_equal_rate_multiscale_evaluation
        or args.stage22_24_batch_parallel_comparison
        or args.stage22_25_cuda_stream_comparison
    ):
        if args.is_training != 0:
            parser.error('frozen multiscale evaluation requires --is_training 0')
        if not args.stage21_checkpoint or not os.path.isfile(args.stage21_checkpoint):
            parser.error('frozen multiscale evaluation requires an existing checkpoint')
        if not args.stage21_metrics_path or not args.stage21_samples_path:
            parser.error('frozen multiscale evaluation requires metrics and samples paths')
        if not 0.0 < args.stage21_target_rate < 1.0:
            parser.error('stage21 target rate must lie in (0, 1)')
        if args.stage21_block_lengths != [1, 5, 20]:
            parser.error('stage21 block lengths are frozen at 1 5 20')
        if (
            args.stage21_original_weight != 0.5
            or args.stage21_l5_weight != 0.25
            or args.stage21_l20_weight != 0.25
        ):
            parser.error('stage21 multiscale weights are frozen at 0.5/0.25/0.25')
        if args.stage21_timing_warmup_batches < 0:
            parser.error('stage21 timing warmup batches must be non-negative')
    if (
        int(args.hcup_diagnostic)
        + int(args.gate_h_diagnostic)
        + int(args.stage12_validation_diagnostic)
        + int(args.stage22_validation_diagnostic)
        + int(args.stage22_inference_timing)
        + int(args.stage22_ard_information_diagnostic)
        + int(args.stage22_27_bf_cmvd_diagnostic)
        + int(args.stage22_29_dmrb_diagnostic)
        + int(args.stage14_uncertainty_diagnostic)
        + int(args.stage15_nested_ensemble_test)
        + int(args.stage18_equal_rate_multiscale_validation)
        + int(args.stage19_equal_rate_multiscale_test)
        + int(args.stage21_equal_rate_multiscale_evaluation)
        + int(args.stage22_24_batch_parallel_comparison)
        + int(args.stage22_25_cuda_stream_comparison)
        > 1
    ):
        parser.error('only one frozen diagnostic mode can be active')
    if args.fusr_router_only and args.fusr_objective_version != 'v2_frozen_balanced_rank':
        parser.error('--fusr_router_only is only valid with FUSR objective v2_frozen_balanced_rank')
    if args.mrra_rank < 1:
        parser.error('MRRA rank must be positive')
    if args.mrra_kernel_size < 1 or args.mrra_kernel_size > args.seq_len:
        parser.error('MRRA kernel size must lie in [1, seq_len]')
    if not math.isfinite(args.mrra_reconstruction_weight) or args.mrra_reconstruction_weight < 0.0:
        parser.error('MRRA reconstruction weight must be finite and non-negative')
    if not math.isfinite(args.mrra_lr_multiplier) or args.mrra_lr_multiplier <= 0.0:
        parser.error('MRRA learning-rate multiplier must be finite and positive')
    if (
        args.mrra_version == 'v3_gradient_isolated_two_timescale'
        and not math.isclose(args.mrra_lr_multiplier, 0.1, rel_tol=0.0, abs_tol=1e-12)
    ):
        parser.error('MRRA-v3 requires --mrra_lr_multiplier 0.1')
    if args.mcpr_rank < 1:
        parser.error('MCPR rank must be positive')
    if args.mcpr_kernel_size < 1 or args.mcpr_kernel_size > args.seq_len:
        parser.error('MCPR kernel size must lie in [1, seq_len]')
    if not math.isfinite(args.mcpr_learning_rate) or args.mcpr_learning_rate <= 0.0:
        parser.error('MCPR learning rate must be finite and positive')
    if not 0.0 < args.ttrr_tail_fraction <= 1.0:
        parser.error('TTRR tail fraction must lie in (0, 1]')
    if not math.isfinite(args.ttrr_tail_weight) or args.ttrr_tail_weight <= 0.0:
        parser.error('TTRR tail weight must be finite and positive')
    if args.use_ttrr and not args.use_mcpr:
        parser.error('TTRR requires --use_mcpr')
    if args.use_ttrr and args.mcpr_objective != 'tail_trust_mse':
        parser.error('TTRR requires --mcpr_objective tail_trust_mse')
    if args.mcpr_objective == 'tail_trust_mse' and not args.use_ttrr:
        parser.error('MCPR tail_trust_mse requires --use_ttrr')
    if args.use_ard and not args.use_mcpr:
        parser.error('ARD requires --use_mcpr')
    if args.use_ard and args.use_ttrr:
        parser.error('ARD and TTRR are mutually exclusive')
    if args.use_ard and args.mcpr_objective != 'analytic_decision_distill':
        parser.error('ARD requires --mcpr_objective analytic_decision_distill')
    if args.mcpr_objective == 'analytic_decision_distill' and not args.use_ard:
        parser.error('MCPR analytic_decision_distill requires --use_ard')
    if args.use_bf_cmvd and not args.use_mcpr:
        parser.error('BF-CMVD requires --use_mcpr')
    bf_cmvd_objectives = {
        'benefit_filtered_multiview_distill',
        'token_benefit_filtered_multiview_distill',
        'dual_scale_residual_basis_distill',
    }
    if args.use_bf_cmvd and args.mcpr_objective not in bf_cmvd_objectives:
        parser.error('BF-CMVD requires a benefit-filtered multiview objective')
    if args.mcpr_objective in bf_cmvd_objectives and not args.use_bf_cmvd:
        parser.error('benefit-filtered multiview distillation requires --use_bf_cmvd')
    if args.use_bf_cmvd:
        if args.use_ttrr or args.use_ard:
            parser.error('BF-CMVD cannot be combined with TTRR or ARD')
        if not math.isclose(args.bf_cmvd_target_rate, 0.15, rel_tol=0.0, abs_tol=1e-12):
            parser.error('BF-CMVD target rate is frozen at 0.15')
        if args.bf_cmvd_block_lengths != [5, 20]:
            parser.error('BF-CMVD block lengths are frozen at 5 20')
        if args.bf_cmvd_nested_seed != args.seed:
            parser.error('BF-CMVD nested seed must equal the experiment seed')
        if (
            args.bf_cmvd_original_weight != 0.5
            or args.bf_cmvd_l5_weight != 0.25
            or args.bf_cmvd_l20_weight != 0.25
        ):
            parser.error('BF-CMVD teacher weights are frozen at 0.5/0.25/0.25')
    if args.use_dmrb:
        if not args.use_mcpr or not args.use_bf_cmvd:
            parser.error('DMRB requires --use_mcpr and --use_bf_cmvd')
        if args.mcpr_objective != 'dual_scale_residual_basis_distill':
            parser.error('DMRB requires --mcpr_objective dual_scale_residual_basis_distill')
        if not (args.dmrb_l5_enabled or args.dmrb_l20_enabled):
            parser.error('DMRB requires at least one enabled scale basis')
        if args.use_ttrr or args.use_ard:
            parser.error('DMRB cannot be combined with TTRR or ARD')
        if args.mcpr_rank != 32 or args.mcpr_kernel_size != 5:
            parser.error('Step22.28 DMRB rank/kernel are frozen at 32/5')
        if not args.mcpr_delta_checkpoint:
            parser.error('Step22.28 DMRB requires the auditable delta checkpoint format')
        if args.post_train_evaluation != 'validation':
            parser.error('Step22.28 DMRB is frozen to validation-only evaluation')
    else:
        if args.dmrb_l5_enabled or args.dmrb_l20_enabled:
            parser.error('DMRB branch switches require --use_dmrb')
        if args.mcpr_objective == 'dual_scale_residual_basis_distill':
            parser.error('dual_scale_residual_basis_distill requires --use_dmrb')
        if args.mcpr_delta_checkpoint:
            parser.error('MCPR delta checkpoints are currently restricted to DMRB')
    pba_readonly_diagnostic = bool(
        args.stage22_31_pba_diagnostic and args.is_training == 0
    )
    if args.use_pba_cmvd:
        if args.is_training != 1 and not pba_readonly_diagnostic:
            parser.error('PBA-CMVD is a training-only adaptation path')
        if not args.use_mipp:
            parser.error('PBA-CMVD requires the PG95/MIPP source identity')
        if any((
            args.use_mcpr, args.use_dmrb, args.use_mrra, args.use_radr,
            args.use_cfr, args.use_fusr, args.use_bf_cmvd,
        )):
            parser.error('PBA-CMVD cannot be combined with another innovation path')
        if args.e_layers != 8 or args.pba_tail_layers != 2:
            parser.error('Step22.30 PBA-CMVD freezes e_layers/tail_layers at 8/2')
        if not args.pba_delta_checkpoint:
            parser.error('Step22.30 PBA-CMVD requires pba_delta_checkpoint')
        if not args.pba_backbone_checkpoint or not os.path.isfile(
            args.pba_backbone_checkpoint
        ):
            parser.error('PBA-CMVD requires an existing PG95 source checkpoint')
        if not math.isclose(
            args.pba_target_rate, 0.15, rel_tol=0.0, abs_tol=1e-12
        ):
            parser.error('PBA-CMVD target rate is frozen at 0.15')
        if args.pba_block_lengths != [5, 20]:
            parser.error('PBA-CMVD block lengths are frozen at 5 20')
        if args.pba_nested_seed != args.seed:
            parser.error('PBA-CMVD nested seed must equal the experiment seed')
        if (
            args.pba_original_weight != 0.5
            or args.pba_l5_weight != 0.25
            or args.pba_l20_weight != 0.25
        ):
            parser.error('PBA-CMVD teacher weights are frozen at 0.5/0.25/0.25')
        if args.post_train_evaluation != 'validation':
            parser.error('PBA-CMVD is frozen to validation-only evaluation')
        if (
            args.optimizer != 'adam'
            or not math.isclose(
                args.learning_rate, 0.001, rel_tol=0.0, abs_tol=1e-12
            )
            or args.batch_size != 16
            or args.patience != 3
            or args.lradj != 'type1'
        ):
            parser.error('PBA-CMVD optimizer/budget is frozen to the Traffic protocol')
        if pba_readonly_diagnostic:
            if not args.stage22_31_checkpoint or not os.path.isfile(
                args.stage22_31_checkpoint
            ):
                parser.error('Step22.31 requires an existing PBA delta checkpoint')
            if not args.stage22_31_backbone_checkpoint or not os.path.isfile(
                args.stage22_31_backbone_checkpoint
            ):
                parser.error('Step22.31 requires an existing PG95 source checkpoint')
            if os.path.abspath(args.stage22_31_backbone_checkpoint) != os.path.abspath(
                args.pba_backbone_checkpoint
            ):
                parser.error('Step22.31 PG95 source must match the PBA training source')
            if args.max_train_batches != -1 or args.max_eval_batches not in {3, -1}:
                parser.error('Step22.31 allows only three-batch smoke or full validation')
            if (
                not math.isclose(
                    args.stage22_31_target_rate, 0.15,
                    rel_tol=0.0, abs_tol=1e-12,
                )
                or args.stage22_31_block_lengths != [5, 20]
                or args.stage22_31_nested_seed != args.seed
                or args.stage22_31_original_weight != 0.5
                or args.stage22_31_l5_weight != 0.25
                or args.stage22_31_l20_weight != 0.25
            ):
                parser.error('Step22.31 frozen teacher protocol changed')
        else:
            smoke_budget = (
                args.train_epochs == 2
                and args.max_train_batches == 2
                and args.max_eval_batches == 3
            )
            formal_budget = (
                args.train_epochs == 15
                and args.max_train_batches == -1
                and args.max_eval_batches == -1
            )
            if not (smoke_budget or formal_budget):
                parser.error('PBA-CMVD requires the frozen smoke or formal budget')
    elif args.pba_delta_checkpoint or args.pba_backbone_checkpoint:
        parser.error('PBA-CMVD checkpoint fields require --use_pba_cmvd')
    elif args.stage22_31_pba_diagnostic:
        parser.error('Step22.31 diagnostic requires --use_pba_cmvd')
    if args.ard_hidden_dim != 16:
        parser.error('Step22.19 ARD hidden dimension is frozen at 16')
    if args.ard_distill_warmup_epochs != 1:
        parser.error('Step22.19 ARD warmup is frozen at one epoch')
    if not math.isclose(args.ard_energy_epsilon, 1e-8, rel_tol=0.0, abs_tol=1e-15):
        parser.error('Step22.19 ARD energy epsilon is frozen at 1e-8')
    if not math.isclose(args.ard_action_epsilon, 1e-6, rel_tol=0.0, abs_tol=1e-15):
        parser.error('Step22.19 ARD action epsilon is frozen at 1e-6')
    if not math.isclose(args.ard_action_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
        parser.error('Step22.19 ARD action weight is frozen at 1.0')
    if not math.isclose(args.ard_scale_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
        parser.error('Step22.19 ARD scale weight is frozen at 1.0')
    if args.use_mcpr:
        if not args.mcpr_backbone_checkpoint:
            parser.error('MCPR requires --mcpr_backbone_checkpoint')
        if not os.path.isfile(args.mcpr_backbone_checkpoint):
            parser.error('MCPR backbone checkpoint does not exist: {}'.format(
                args.mcpr_backbone_checkpoint
            ))
        if args.use_mrra or args.use_radr or args.use_cfr or args.use_fusr:
            parser.error('MCPR cannot be combined with MRRA, RADR, CFR, or FUSR')
    if args.fusr_objective_version == 'v2_frozen_balanced_rank':
        if not args.use_fusr or not args.use_utility_read_gate:
            parser.error('FUSR V2 requires --use_fusr and --use_utility_read_gate')
        if not args.fusr_router_only:
            parser.error('FUSR V2 requires --fusr_router_only')
        if not args.fusr_backbone_checkpoint:
            parser.error('FUSR V2 requires --fusr_backbone_checkpoint')
        if not os.path.isfile(args.fusr_backbone_checkpoint):
            parser.error('FUSR V2 backbone checkpoint does not exist: {}'.format(
                args.fusr_backbone_checkpoint
            ))
        if args.fusr_feature_set != 'intervention14':
            parser.error('FUSR V2 requires --fusr_feature_set intervention14')
        if args.fusr_gate_score != 'sign_logit':
            parser.error('FUSR V2 requires --fusr_gate_score sign_logit')
        if args.fusr_gate_prior_mode != 'none':
            parser.error('FUSR V2 requires --fusr_gate_prior_mode none')
        if args.fusr_cache_split != 'val':
            parser.error('FUSR V2 cache must use the validation split')
        fractions = (args.fusr_fit_fraction, args.fusr_dev_fraction)
        if not all(math.isfinite(value) and value > 0.0 for value in fractions):
            parser.error('FUSR V2 fit/dev fractions must be finite and positive')
        if args.fusr_fit_fraction + args.fusr_dev_fraction >= 1.0:
            parser.error('FUSR V2 fit_fraction + dev_fraction must be less than 1')
        if args.fusr_embargo_windows < 0:
            parser.error('FUSR V2 embargo windows must be non-negative')
        if args.fusr_router_epochs <= 0 or args.fusr_router_batch_size <= 0:
            parser.error('FUSR V2 router epochs and batch size must be positive')
        if not math.isfinite(args.fusr_router_lr) or args.fusr_router_lr <= 0.0:
            parser.error('FUSR V2 router learning rate must be finite and positive')
        if args.fusr_router_patience < 0:
            parser.error('FUSR V2 router patience must be non-negative')
        if not math.isfinite(args.fusr_router_weight_decay) or args.fusr_router_weight_decay < 0.0:
            parser.error('FUSR V2 router weight decay must be finite and non-negative')
        if not math.isfinite(args.fusr_rank_weight) or args.fusr_rank_weight < 0.0:
            parser.error('FUSR V2 rank weight must be finite and non-negative')
        if args.fusr_cache_max_batches == 0 or args.fusr_cache_max_batches < -1:
            parser.error('FUSR V2 cache_max_batches must be -1 or a positive integer')
    
    
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu and not args.cpu else False
    args.device = torch.device('cuda:{}'.format(args.gpu) if args.use_gpu else 'cpu')
    if "electricity" in args.data_path:
        args.d_var = 321
    elif "traffic" in args.data_path:
        args.d_var= 862
    elif "exchange_rate" in args.data_path:
        args.d_var = 8
    elif "weather" in args.data_path:
        args.d_var = 21
    elif "climate" in args.data_path:
        args.d_var = 9
    elif "ETT2" in args.data_path:
        args.d_var = 7
    elif "ETTh1" in args.data_path:
        args.d_var = 7        
    elif "ETTm1" in args.data_path:
        args.d_var = 7        
    elif "Solar" in args.data_path:
        args.d_var = 137    
    elif "FaceDetection" in args.data_path:
        args.d_var = 144
    elif "LSST" in args.data_path:
        args.d_var = 6
    elif 'FingerMovements' in args.data_path:
        args.d_var = 28
    elif 'MotorImagery' in args.data_path:
        args.d_var = 64
    elif 'Heartbeat' in args.data_path:
        args.d_var = 61    
    elif 'ArticularyWordRecognition' in args.data_path:
        args.d_var = 9  
    elif "simulation" in args.data_path:
        args.d_var = 10
    else:
        # The public code infers d_var from case-sensitive filename tokens.
        # Generated or user-provided assets may use a different filename while
        # retaining the exact configured feature width.  Fall back to enc_in so
        # data identity is not coupled to filename capitalization.
        args.d_var = args.enc_in
        print('D_VAR_FALLBACK data_path={} d_var={} source=enc_in'.format(
            args.data_path, args.d_var
        ))

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')    
    print(args)

    if args.model in ["GRUD","S4_Fill0",'S4_GRUD']:
        Exp = Exp_Long_Term_Forecast2
    elif args.model in ['Transformer','Transformer_M','Autoformer','CARD','iTransformer']:
        Exp = exp_transformer
        args.label_len = int(args.seq_len/2)
    elif args.model in ['grafiti']:
        Exp = Exp_grafiti
        args.data = 'grafiti'
    elif args.model in ['BiaTCGNet']:
        Exp = exp_BiaTCGNet
        args.data = 'custom2'
    elif args.model in ['S4M']:
        Exp = Exp_pretrain1
        args.data = 'custom4'
    elif args.model in ['Brits']:
        Exp = Exp_brits
        args.data = 'Brits'
    elif args.model in ['CRU']:
        Exp = Exp_cru
        args.data = 'cru'
    
    
    if args.model=="S4_Fill0" and args.data=="transformer_impute":
        Exp = exp_transformer
        args.label_len = int(args.seq_len/2)
        
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy('file_system')

    os.environ.setdefault('WANDB_MODE', args.wandb_mode)
    wandb.init(config=vars(args), project='S4', name=args.model, mode=args.wandb_mode)
    # W&B initialization may consume global random state. Re-seed immediately
    # before model construction and DataLoader creation for repeatable runs.
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fix_seed)
    
    if args.is_training==1:
            
        for ii in range(args.itr):
            mask =1 if args.mask else 0
            # setting record of experiments
            setting = '{}_{}_ft{}_ll{}_pl{}_dm{}_el{}_eb{}_dt{}_{}_data_{}tag1_fill_{}_drop_{}_dG_{}_pattn_{}_n_{}_slen_{}_W_{}_ts1_{}_ts2_{}_K_{}_n_{}_M_{}_pre_{}_en_{}_s_{}_top{}_P_{}_m_{}_t_{}'.format(
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.e_layers,
                args.embed,
                args.des,
                args.class_strategy, ii,args.data_path[:-4],args.fillna,str(args.drop),str(args.num_dG),str(args.num_pattn),str(args.n),str(args.short_len),str(args.W),
                str(args.thres1),
                str(args.thres2),
                str(args.K),
                str(args.n),
                str(args.M),
                str(args.pretrain),
                str(args.en_conv_hidden_size),
                str(args.moving_avg),
                str(args.topK),
                str(args.per_mem_size),
                str(args.memnet),
                str(args.test))

            args.check_path = setting
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if args.post_train_evaluation == 'validation':
                print('>>>>>>>post-train validation : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.post_train_validation(setting)
            else:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                if args.test=='test':
                    exp.test(setting)
                else:
                    exp.test1(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()

    elif args.is_training == 0 and args.stage22_25_cuda_stream_comparison:
        setting = '{}_{}_stage22_25_cuda_stream_{}_{}'.format(
            args.model, args.data, args.stage21_split, args.model_id
        )
        exp = Exp(args)
        print('>>>>>>>loading stage22.25 checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(args.stage21_checkpoint))
        exp.load_model(args.stage21_checkpoint)
        exp.stage22_25_cuda_stream_comparison(
            setting, split_name=args.stage21_split
        )
        torch.cuda.empty_cache()

    elif args.is_training == 0 and args.stage22_24_batch_parallel_comparison:
        setting = '{}_{}_stage22_24_batch_parallel_{}_{}'.format(
            args.model, args.data, args.stage21_split, args.model_id
        )
        exp = Exp(args)
        print('>>>>>>>loading stage22.24 checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(args.stage21_checkpoint))
        exp.load_model(args.stage21_checkpoint)
        exp.stage22_24_batch_parallel_comparison(
            setting, split_name=args.stage21_split
        )
        torch.cuda.empty_cache()

    elif args.is_training == 0 and args.stage21_equal_rate_multiscale_evaluation:
        setting = '{}_{}_stage21_equal_rate_multiscale_{}_{}'.format(
            args.model, args.data, args.stage21_split, args.model_id
        )
        exp = Exp(args)
        print('>>>>>>>loading stage21 checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(args.stage21_checkpoint))
        exp.load_model(args.stage21_checkpoint)
        exp.stage21_equal_rate_multiscale_evaluation(
            setting, split_name=args.stage21_split
        )
        torch.cuda.empty_cache()

    elif args.is_training == 0 and args.stage19_equal_rate_multiscale_test:
        setting = '{}_{}_stage19_equal_rate_multiscale_{}'.format(
            args.model, args.data, args.model_id
        )
        exp = Exp(args)
        print('>>>>>>>loading stage19 checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(args.stage19_checkpoint))
        exp.load_model(args.stage19_checkpoint)
        exp.stage19_equal_rate_multiscale_test(setting)
        torch.cuda.empty_cache()

    elif args.is_training == 0 and args.stage18_equal_rate_multiscale_validation:
        setting = '{}_{}_stage18_equal_rate_multiscale_{}'.format(
            args.model, args.data, args.model_id
        )
        exp = Exp(args)
        print('>>>>>>>loading stage18 checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(args.stage18_checkpoint))
        exp.load_model(args.stage18_checkpoint)
        exp.stage18_equal_rate_multiscale_validation(setting)
        torch.cuda.empty_cache()

    elif args.is_training == 0 and args.stage15_nested_ensemble_test:
        setting = '{}_{}_stage15_nested_ensemble_{}'.format(
            args.model, args.data, args.model_id
        )
        exp = Exp(args)
        print('>>>>>>>loading stage15 checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(args.stage15_checkpoint))
        exp.load_model(args.stage15_checkpoint)
        exp.stage15_nested_ensemble_test(setting)
        torch.cuda.empty_cache()

    elif args.is_training == 0 and args.stage14_uncertainty_diagnostic:
        setting = '{}_{}_stage14_uncertainty_{}'.format(
            args.model, args.data, args.model_id
        )
        exp = Exp(args)
        print('>>>>>>>loading stage14 checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(args.stage14_checkpoint))
        exp.load_model(args.stage14_checkpoint)
        exp.stage14_uncertainty_diagnostic(setting)
        torch.cuda.empty_cache()

    elif args.is_training == 0 and args.stage12_validation_diagnostic:
        setting = '{}_{}_stage12_val_{}'.format(args.model, args.data, args.model_id)
        exp = Exp(args)
        print('>>>>>>>loading stage12 checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(args.stage12_checkpoint))
        exp.load_model(args.stage12_checkpoint)
        exp.stage12_validation_diagnostic(setting)
        torch.cuda.empty_cache()

    elif args.is_training == 0 and args.stage22_inference_timing:
        setting = '{}_{}_stage22_timing_{}'.format(args.model, args.data, args.model_id)
        exp = Exp(args)
        print('>>>>>>>loading stage22 timing checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(args.stage22_timing_checkpoint))
        exp.load_model(args.stage22_timing_checkpoint)
        exp.stage22_inference_timing_validation(setting)
        torch.cuda.empty_cache()

    elif args.is_training == 0 and args.stage22_ard_information_diagnostic:
        setting = '{}_{}_stage22_ard_diag_{}'.format(
            args.model, args.data, args.model_id
        )
        exp = Exp(args)
        print(
            '>>>>>>>loading stage22 ARD diagnostic checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(
                args.stage22_ard_diagnostic_checkpoint
            )
        )
        exp.load_model(args.stage22_ard_diagnostic_checkpoint)
        exp.stage22_ard_information_diagnostic(setting)
        torch.cuda.empty_cache()

    elif args.is_training == 0 and args.stage22_27_bf_cmvd_diagnostic:
        setting = '{}_{}_stage22_27_bf_cmvd_diag_{}'.format(
            args.model, args.data, args.model_id
        )
        exp = Exp(args)
        print(
            '>>>>>>>loading stage22.27 BF-CMVD checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(
                args.stage22_27_checkpoint
            )
        )
        exp.load_model(args.stage22_27_checkpoint)
        exp.stage22_27_bf_cmvd_diagnostic(setting)
        torch.cuda.empty_cache()

    elif args.is_training == 0 and args.stage22_31_pba_diagnostic:
        setting = '{}_{}_stage22_31_pba_diag_{}'.format(
            args.model, args.data, args.model_id
        )
        exp = Exp(args)
        print(
            '>>>>>>>loading stage22.31 PBA-CMVD checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(
                args.stage22_31_checkpoint
            )
        )
        exp.load_model(args.stage22_31_checkpoint)
        exp.stage22_31_pba_cmvd_diagnostic(setting)
        torch.cuda.empty_cache()

    elif args.is_training == 0 and args.stage22_29_dmrb_diagnostic:
        setting = '{}_{}_stage22_29_dmrb_diag_{}'.format(
            args.model, args.data, args.model_id
        )
        exp = Exp(args)
        print(
            '>>>>>>>loading stage22.29 DMRB checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(
                args.stage22_29_checkpoint
            )
        )
        exp.load_model(args.stage22_29_checkpoint)
        exp.stage22_29_dmrb_diagnostic(setting)
        torch.cuda.empty_cache()

    elif args.is_training == 0 and args.stage22_validation_diagnostic:
        setting = '{}_{}_stage22_val_{}'.format(args.model, args.data, args.model_id)
        exp = Exp(args)
        print('>>>>>>>loading stage22 checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(args.stage22_checkpoint))
        exp.load_model(args.stage22_checkpoint)
        exp.stage22_validation_diagnostic(setting)
        torch.cuda.empty_cache()

    elif args.is_training == 0 and args.hcup_diagnostic:
        setting = '{}_{}_hcup_{}_{}'.format(
            args.model, args.data, args.hcup_split, args.model_id
        )
        exp = Exp(args)
        print('>>>>>>>loading HCUP checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(args.hcup_checkpoint))
        exp.load_model(args.hcup_checkpoint)
        exp.hcup_diagnostic(setting)
        torch.cuda.empty_cache()

    elif args.is_training == 0 and args.gate_h_diagnostic:
        if not args.gate_h_checkpoint:
            raise ValueError('--gate_h_checkpoint is required for Gate H diagnostic mode')
        if not args.gate_h_metrics_path or not args.gate_h_samples_path:
            raise ValueError('Gate H metrics and samples paths are required')
        setting = '{}_{}_gate_h_{}'.format(args.model, args.data, args.model_id)
        exp = Exp(args)
        print('>>>>>>>loading Gate H checkpoint : {}<<<<<<<<<<<<<<<<<<<<'.format(args.gate_h_checkpoint))
        exp.load_model(args.gate_h_checkpoint)
        exp.gate_h_diagnostic(setting)
        torch.cuda.empty_cache()

    elif args.is_training == 2:
        for ii in range(args.itr):
            mask =1 if args.mask else 0
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_mask_{}_data_{}tag1_encoder_{}_fill_{}_drop_{}_pos_emb_{}_dG_{}_pattn_{}_n_{}_short_len_{}_W_{}_M_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii,str(mask),args.data_path[:-4],args.encoder_type,args.fillna,str(args.drop),str(args.pos_emb),str(args.num_dG),str(args.num_pattn),str(args.n),str(args.short_len),str(args.W),str(args.M))

            exp = Exp(args)
            exp.get_input(setting)
        
        torch.cuda.empty_cache()
