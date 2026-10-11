import torch
import torch.nn as nn
import torch.nn.functional as F
from S4.s4.models.s4.s4 import S4Block as S4
from S4.s4.models.s4.s4_mask1 import S4Block as S4_mask
from model.Bank import MTNet
from model.MRRA import MaskResponsiveResidualAdapter
from model.MCPR import (
    DualScaleMaskConditionedResidualBasis,
    MaskConditionedParetoResidual,
)
import os
import math
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean



class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = torch.nn.parameter.Parameter(torch.Tensor(output_size, input_size))
        self.b = torch.nn.parameter.Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * torch.autograd.Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma
    


class Model(nn.Module):
    """
    Mamba
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.activation = F.relu 
        self.dropout = nn.Dropout(configs.dropout)
        self.mask = configs.mask
        self.decay1 = nn.Linear(configs.d_var,configs.d_var)
        self.decay2 = nn.Linear(configs.d_var,configs.d_var)
        self.classification = configs.classification
        self.plot = configs.plot
        self.d_model = configs.d_model

        self.use_mrra = bool(getattr(configs, 'use_mrra', False))
        self.mrra = None
        if self.use_mrra:
            self.mrra = MaskResponsiveResidualAdapter(
                num_variables=configs.d_var,
                context_size=configs.en_conv_hidden_size,
                rank=int(getattr(configs, 'mrra_rank', 32)),
                kernel_size=int(getattr(configs, 'mrra_kernel_size', 5)),
                version=str(getattr(configs, 'mrra_version', 'v1')),
            )
        self.last_mrra_outputs = None

        self.use_mcpr = bool(getattr(configs, 'use_mcpr', False))
        self.use_dmrb = bool(getattr(configs, 'use_dmrb', False))
        self.use_ttrr = bool(getattr(configs, 'use_ttrr', False))
        self.use_ard = bool(getattr(configs, 'use_ard', False))
        self.mcpr = None
        if self.use_mcpr:
            if self.use_dmrb:
                self.mcpr = DualScaleMaskConditionedResidualBasis(
                    hidden_size=configs.d_model,
                    num_variables=configs.d_var,
                    rank=int(getattr(configs, 'mcpr_rank', 32)),
                    kernel_size=int(getattr(configs, 'mcpr_kernel_size', 5)),
                    l5_enabled=bool(getattr(configs, 'dmrb_l5_enabled', False)),
                    l20_enabled=bool(getattr(configs, 'dmrb_l20_enabled', False)),
                )
            else:
                self.mcpr = MaskConditionedParetoResidual(
                    hidden_size=configs.d_model,
                    num_variables=configs.d_var,
                    rank=int(getattr(configs, 'mcpr_rank', 32)),
                    kernel_size=int(getattr(configs, 'mcpr_kernel_size', 5)),
                    use_ttrr=self.use_ttrr,
                    use_ard=self.use_ard,
                    ard_hidden_dim=int(getattr(configs, 'ard_hidden_dim', 16)),
                )
        self.last_mcpr_outputs = None

        # Training-only HFD-PBA feature capture. These plain Python attributes do
        # not add parameters/buffers and are disabled for every normal forward.
        self.hfd_feature_capture_layers = ()
        self.last_hfd_features = None


        self.mem_net = MTNet(configs)

        
        self.linear1 = nn.Linear(configs.en_conv_hidden_size,self.d_model)



        self.norm = nn.LayerNorm(configs.d_model)
    
        self.s4_layers = nn.ModuleList(
        [
            S4_Layer(
                S4_mask(configs.en_conv_hidden_size, transposed=False),
                S4_mask(configs.en_conv_hidden_size, transposed=False),
                configs.en_conv_hidden_size,
                configs
            )
        ]
        +
        [
            S4_Layer(
                S4(configs.d_model, transposed=False),
                S4(configs.d_model, transposed=False),
                configs.d_model,
                configs
            )
            for i in range(configs.e_layers-1)
        ]
        )
        self.mask_trans = nn.Linear(configs.d_var,configs.en_conv_hidden_size)
        if not self.classification:
            self.out_proj=nn.Linear(configs.d_model, configs.d_var, bias=True)
        else:
            self.out_proj=nn.Linear(configs.d_model,configs.num_class)
        self.hist_pool = []
        self.q_pool = []
        
    def embedding(self,x):
        return self.mem_net.encoder_m(x)
        
    def warmup(self,seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value,mipp_pair=None):
        if self.mem_net.use_mipp and mipp_pair is not None:
            primary = self._decay_impute(
                seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value
            )
            paired = self._decay_impute(
                mipp_pair['seq_x'],
                mipp_pair['seq_x_mask'],
                mipp_pair['max_idx'],
                mipp_pair['min_idx'],
                mipp_pair['max_value'],
                mipp_pair['min_value'],
            )
            self.mem_net.warm(primary, paired, seq_mask=seq_x_mask)
        else:
            self.mem_net.warm(seq_x)
        return None

    def set_prototype_enabled(self, enabled):
        self.mem_net.set_prototype_enabled(enabled)

    def set_collect_retrieval_diagnostics(self, enabled):
        self.mem_net.set_collect_retrieval_diagnostics(enabled)

    def get_last_retrieval_diagnostics(self):
        return self.mem_net.last_retrieval_diagnostics

    def get_last_mipp_outputs(self):
        return self.mem_net.get_last_mipp_outputs()

    def get_last_mipp_warm_diagnostics(self):
        return self.mem_net.get_last_mipp_warm_diagnostics()

    def get_last_cfr_outputs(self):
        return self.mem_net.get_last_cfr_outputs()

    def set_fusr_mode(self, mode):
        """Select routed, always-on, or always-off prototype reads at evaluation."""
        self.mem_net.set_fusr_mode(mode)

    def get_last_fusr_outputs(self):
        return self.mem_net.get_last_fusr_outputs()

    def get_last_mrra_outputs(self):
        return self.last_mrra_outputs

    def get_last_mcpr_outputs(self):
        return self.last_mcpr_outputs

    def set_hfd_feature_capture(self, enabled, layers=None):
        """Enable ephemeral S4 hidden-state capture for HFD-PBA training."""
        if not enabled:
            self.hfd_feature_capture_layers = ()
            self.last_hfd_features = None
            return
        resolved_layers = tuple(sorted(set(int(layer) for layer in (layers or ()))))
        if not resolved_layers:
            raise ValueError('HFD feature capture requires at least one layer')
        if resolved_layers[0] < 0 or resolved_layers[-1] >= len(self.s4_layers):
            raise ValueError(
                f'HFD capture layers out of range: {resolved_layers}'
            )
        self.hfd_feature_capture_layers = resolved_layers
        self.last_hfd_features = None

    def get_last_hfd_features(self):
        if self.last_hfd_features is None:
            raise RuntimeError('HFD feature capture was not enabled for the last forward')
        missing = set(self.hfd_feature_capture_layers) - set(self.last_hfd_features)
        if missing:
            raise RuntimeError(f'HFD feature capture missed layers: {sorted(missing)}')
        return self.last_hfd_features

    def _decay_impute(self,seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value):
        max_idx = (max_idx-torch.mean(max_idx,dim=1,keepdims=True))/torch.std(max_idx,dim=1,keepdims=True)
        min_idx = (max_idx-torch.mean(min_idx,dim=1,keepdims=True))/torch.std(min_idx,dim=1,keepdims=True)
        gamma2 = torch.exp(-1*(self.decay1(max_idx)))
        gamma1 = torch.exp(-1*(self.decay2(min_idx)))
        gamma = gamma1+gamma2
        gamma1 = gamma1/gamma
        gamma2 = gamma2/gamma
        return (gamma1*max_value+gamma2*min_value)*(1-seq_x_mask)+seq_x*seq_x_mask

    def _forecast_from_context(self, context, mask, observation_mask=None):
        """Run the shared S4 forecasting backbone from one memory context."""
        capture_layers = self.hfd_feature_capture_layers
        self.last_hfd_features = {} if capture_layers else None
        x = self.s4_layers[0](context, mask)
        if 0 in capture_layers:
            self.last_hfd_features[0] = x
        x = self.linear1(x)
        for i in range(1, self.configs.e_layers):
            x = self.s4_layers[i](x)
            if i in capture_layers:
                self.last_hfd_features[i] = x
        x = self.norm(x)
        if self.classification:
            out = self.out_proj(x.mean(dim=1))
            return F.softmax(out, dim=1)
        out = self.out_proj(x)
        self.last_mcpr_outputs = None
        if self.use_mcpr:
            if observation_mask is None:
                raise RuntimeError('MCPR requires the historical observation mask')
            self.last_mcpr_outputs = self.mcpr(
                x, observation_mask, backbone_forecast=out
            )
            out = out + self.last_mcpr_outputs['residual']
        return out

    @staticmethod
    def _capture_rng_state(include_cuda):
        state = {'cpu': torch.get_rng_state()}
        if include_cuda:
            state['cuda'] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state):
        torch.set_rng_state(state['cpu'])
        if 'cuda' in state:
            torch.cuda.set_rng_state_all(state['cuda'])

    def forward(self,seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value,mipp_pair=None):

        max_idx = (max_idx-torch.mean(max_idx,dim=1,keepdims=True))/torch.std(max_idx,dim=1,keepdims=True)
        min_idx = (max_idx-torch.mean(min_idx,dim=1,keepdims=True))/torch.std(min_idx,dim=1,keepdims=True)
        seq_mask = seq_x_mask[:,:,0:1]
        gamma2 = torch.exp(-1*(self.decay1(max_idx))) 
        gamma1 = torch.exp(-1*(self.decay2(min_idx))) 
        gamma = gamma1+gamma2
        gamma1 = gamma1/gamma
        gamma2 = gamma2/gamma
        z_s = (gamma1*max_value+gamma2*min_value)*(1-seq_x_mask)+seq_x*seq_x_mask# decay imputation
        self.last_mrra_outputs = None
        self.last_mcpr_outputs = None
        mrra_context_residual = None
        if self.use_mrra:
            self.last_mrra_outputs = self.mrra(z_s, seq_x_mask)
            mrra_context_residual = self.last_mrra_outputs['context_residual']
        pair_z_s = None
        pair_mask = None
        if self.mem_net.use_mipp and mipp_pair is not None:
            pair_z_s = self._decay_impute(
                mipp_pair['seq_x'],
                mipp_pair['seq_x_mask'],
                mipp_pair['max_idx'],
                mipp_pair['min_idx'],
                mipp_pair['max_value'],
                mipp_pair['min_value'],
            )
            pair_mask = mipp_pair['seq_x_mask']
        
        
        # memory bank
        c_s = self.mem_net(
            z_s,
            seq_mask=seq_x_mask,
            pair_Q=pair_z_s,
            pair_mask=pair_mask,
        )
        if mrra_context_residual is not None:
            c_s = c_s + mrra_context_residual
        mask = self.mask_trans(seq_x_mask)
        fusr_outputs = self.mem_net.get_last_fusr_outputs()
        if self.mem_net.use_fusr and self.training:
            if fusr_outputs is None:
                raise RuntimeError('FUSR training forward produced no router outputs')
            context_on = fusr_outputs.get('context_on')
            context_off = fusr_outputs.get('context_off')
            if context_on is None or context_off is None:
                raise RuntimeError('FUSR training requires paired prototype contexts')
            if mrra_context_residual is not None:
                context_on = context_on + mrra_context_residual
                context_off = context_off + mrra_context_residual

            # Common random numbers: both interventions see identical dropout
            # masks, so their loss difference isolates the prototype read.
            rng_state = self._capture_rng_state(context_on.is_cuda)
            prediction_on = self._forecast_from_context(context_on, mask, seq_x_mask)
            self._restore_rng_state(rng_state)
            prediction_off = self._forecast_from_context(context_off, mask, seq_x_mask)
            gate = fusr_outputs['effective_gate'].view(-1, 1, 1)
            out = gate * prediction_on + (1.0 - gate) * prediction_off
            fusr_outputs.update({
                'prediction_on': prediction_on,
                'prediction_off': prediction_off,
                'prediction_routed': out,
            })
        elif self.mem_net.use_cfr and self.training:
            cfr_outputs = self.mem_net.get_last_cfr_outputs()
            if cfr_outputs is None:
                raise RuntimeError('CFR training forward produced no paired contexts')
            context_on = cfr_outputs.get('context_on')
            context_off = cfr_outputs.get('context_off')
            if context_on is None or context_off is None:
                raise RuntimeError('CFR training requires prototype-on/off contexts')
            if mrra_context_residual is not None:
                context_on = context_on + mrra_context_residual
                context_off = context_off + mrra_context_residual

            # Common random numbers ensure that the paired prediction gap is
            # caused by prototype retrieval instead of different dropout masks.
            rng_state = self._capture_rng_state(context_on.is_cuda)
            prediction_on = self._forecast_from_context(context_on, mask, seq_x_mask)
            self._restore_rng_state(rng_state)
            prediction_off = self._forecast_from_context(context_off, mask, seq_x_mask)
            out = prediction_on
            cfr_outputs.update({
                'prediction_on': prediction_on,
                'prediction_off': prediction_off,
            })
        else:
            # Evaluation uses one deployable downstream path.  MTNet has
            # already applied the hard routed/on/off context decision.
            out = self._forecast_from_context(c_s, mask, seq_x_mask)

        return out

class S4_Layer(nn.Module):
    def __init__(self, s4_1,s4_2,d_model,configs):
        super(S4_Layer, self).__init__()
        self.s4_1 = s4_1
        self.s4_2 = s4_2
        d_ff = configs.d_ff
        # self.bidirectional = configs.bidirectional
        self.activation = F.relu 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    def forward(self, x,m=None):
        if m is None:
            new_x = self.s4_1(x)[0]         
        else:
            new_x = self.s4_1(x,m)[0]        
        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)
