import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from S4.s4.models.s4.s4 import S4Block as S4
from model.Memory import FeatureMemory
from model.MIPP import MIPPPurifier
from model.FUSR import ForecastUtilityRouter
import numpy as np


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                # print(z.shape)
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
    


class Encoder2(nn.Module):
    def __init__(self, config,short_len,n=None):
        super(Encoder2, self).__init__()
        if n is None:
            self.n = config.n
        else:
            self.n = n
        self.config = config
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=config.en_conv_hidden_size,
                               kernel_size=(config.W, config.d_var),
                               stride=1,
                               padding=0)  # Assuming padding='VALID' equivalent to padding=0
        self.dropout = nn.Dropout(p=1-config.input_keep_prob)
        self.short_len =short_len
        self.s4 = S4_Layer(d_model = config.en_conv_hidden_size,dropout =1-config.output_keep_prob)  # B,L,D
        self.attention_layer = AttentionLayer(
                        FullAttention(False, config.factor, attention_dropout=config.dropout,
                                      output_attention=config.output_attention), config.en_conv_hidden_size, config.n_heads)
        
        Tc = self.short_len - self.config.W + 1
        self.Tc = Tc
        # Dropout for RNN layers

        self.linear = nn.Linear(Tc,1)

    def forward(self, x):   #
        b,n,l,d = x.shape
        x = x.reshape(-1,1,self.short_len,self.config.d_var)                # Reshape input_x : <batch_size * n,1, T, D>
        batch_size = int(x.shape[0]/n)
        # print("Tag1",x.shape)
        
        last_rnn_hidden_size = self.config.en_rnn_hidden_sizes[-1]


        # CNN layer
        h_conv1 = F.relu(self.conv1(x))  # (batch_size*n,en_conv_hidden_size,Tc,1)
        # print("tag2",h_conv1.shape)
        # print("h_conv1.shape",h_conv1.shape)

        if self.config.input_keep_prob < 1:
            h_conv1 = self.dropout(h_conv1)


        input = h_conv1.reshape(-1,self.Tc,self.config.en_conv_hidden_size) # (batch_size*n,Tc,en_conv_hidden_size)
        # print("input.shape",input.shape)
        att_output,_ = self.attention_layer(input,input,input,attn_mask=None)
        # print("att_output.shape",att_output.shape)
        att_output = self.s4(att_output)
        
        y = self.linear(att_output.transpose(2,1))
        y = y.transpose(2,1)
        # print("y.shape",y.shape)
        y = y.reshape(batch_size,n,-1)
        return y  #  # <batch_size, n, last_rnn_hidden_size>


class S4_Layer(nn.Module):
    def __init__(self, d_model,dropout):
        super(S4_Layer, self).__init__()
        self.s4_1 = S4(d_model, transposed=False)
        d_ff = d_model
        # self.bidirectional = configs.bidirectional
        self.activation = F.relu 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    def forward(self, x):
        # print("***************x.shape",x.shape)
        new_x = self.s4_1(x)[0]               
        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


    
class MTNet(nn.Module):
    def __init__(self, config):
        super(MTNet, self).__init__()
        self.config = config


        self.encoder_m = Encoder2(config,short_len=config.short_len)
        self.encoder_u = Encoder2(config,short_len= config.short_len) 
        
        for param in self.encoder_m.parameters():
            param.requires_grad = False
                      
        self.dropout_u = nn.Dropout(p=1-config.input_keep_prob)
    
        last_rnn_hid_size = config.en_conv_hidden_size
        self.pred_w = nn.Parameter(torch.randn(last_rnn_hid_size * 2 + config.d_var,last_rnn_hid_size))
        self.pred_b = nn.Parameter(torch.randn(last_rnn_hid_size))

        self.dropout = nn.Dropout(1 - config.input_keep_prob)
        self.unfold = torch.nn.Unfold( kernel_size = (config.short_len,config.d_var),padding=(config.short_len-1,0),stride =(1,config.d_var))

        self.memory = FeatureMemory(config)

        self.K = config.K
        self.momentum = config.momentum
        self.count = 0
        self.is_training = config.is_training
        # Diagnostic-only intervention switch. The default preserves the
        # original S4M retrieval path exactly.
        self.prototype_enabled = True
        self.collect_retrieval_diagnostics = False
        self.last_retrieval_diagnostics = None
        self.use_mipp = bool(getattr(config, 'use_mipp', False))
        self.use_content_mask_factorization = bool(
            getattr(config, 'use_content_mask_factorization', False)
        )
        self.use_stability_write_gate = bool(
            getattr(config, 'use_stability_write_gate', False)
        )
        self.mipp_write_gate_mode = str(
            getattr(config, 'mipp_write_gate_mode', 'absolute')
        )
        self.mipp_write_budget = int(
            getattr(config, 'mipp_write_budget', config.M)
        )
        self.mipp = None
        if self.use_mipp:
            self.mipp = MIPPPurifier(
                hidden_size=last_rnn_hid_size,
                num_variables=config.d_var,
                variance_target=getattr(config, 'mipp_variance_target', 1.0),
            )
        self.last_mipp_outputs = None
        self.last_mipp_warm_diagnostics = None
        self.use_cfr = bool(getattr(config, 'use_cfr', False))
        self.last_cfr_outputs = None
        self.use_fusr = bool(getattr(config, 'use_fusr', False))
        self.use_utility_read_gate = bool(
            getattr(config, 'use_utility_read_gate', False)
        )
        self.use_fusr_spectral_uncertainty = bool(
            getattr(config, 'use_fusr_spectral_uncertainty', False)
        )
        self.fusr_objective_version = str(
            getattr(config, 'fusr_objective_version', 'v1_joint')
        )
        self.fusr_feature_set = str(
            getattr(config, 'fusr_feature_set', 'base8')
        )
        self.fusr_gate_score = str(
            getattr(config, 'fusr_gate_score', 'utility_lcb')
        )
        self.fusr_gate_prior_mode = str(
            getattr(config, 'fusr_gate_prior_mode', 'learned')
        )
        self.use_fusr_intervention_features = bool(
            self.use_fusr
            and self.fusr_objective_version == 'v2_frozen_balanced_rank'
            and self.fusr_feature_set == 'intervention14'
        )
        self.fusr = None
        if self.use_fusr:
            self.fusr = ForecastUtilityRouter(
                hidden_dim=int(getattr(config, 'fusr_hidden_dim', 32)),
                soft_gate_temperature=float(
                    getattr(config, 'fusr_gate_temperature', 0.25)
                ),
                gate_threshold=float(getattr(config, 'fusr_delta', 0.0)),
                uncertainty_penalty=float(
                    getattr(config, 'fusr_uncertainty_penalty', 1.0)
                ),
                objective_version=self.fusr_objective_version,
                feature_set=self.fusr_feature_set,
                gate_score=self.fusr_gate_score,
                gate_prior_mode=self.fusr_gate_prior_mode,
            )
        self.fusr_mode = 'route'
        self.last_fusr_outputs = None

    def set_prototype_enabled(self, enabled):
        self.prototype_enabled = bool(enabled)

    def set_collect_retrieval_diagnostics(self, enabled):
        self.collect_retrieval_diagnostics = bool(enabled)
        if not self.collect_retrieval_diagnostics:
            self.last_retrieval_diagnostics = None

    def get_last_mipp_outputs(self):
        return self.last_mipp_outputs

    def get_last_mipp_warm_diagnostics(self):
        return self.last_mipp_warm_diagnostics

    def get_last_cfr_outputs(self):
        return self.last_cfr_outputs

    def set_fusr_mode(self, mode):
        if mode not in ('route', 'on', 'off'):
            raise ValueError("FUSR mode must be one of: route, on, off")
        self.fusr_mode = mode

    def get_last_fusr_outputs(self):
        return self.last_fusr_outputs

    def _fuse_context(self, retrieved, query, raw_history, batch, length, hidden):
        retrieved = retrieved.reshape(batch, length, hidden)
        query = query.reshape(batch, length, hidden)
        context = torch.cat([retrieved, query, raw_history], dim=-1)
        context = torch.matmul(context, self.pred_w) + self.pred_b
        return context + query

    @torch.no_grad()
    def update_k_encoder_weights(self):
        """ manually update key encoder weights with momentum and no_grad"""
        # update k_encoder.parameters 0.99
        for p_u, p_m in zip(self.encoder_u.parameters(), self.encoder_m.parameters()):
            p_m.data = p_m.data*self.momentum + (1.0 - self.momentum)*p_u.data
            p_m.requires_grad = False
            
    def warm(self,X,pair_X=None,seq_mask=None):
        with torch.no_grad():
            B,L,D = X.shape
            Q = int(L/self.config.short_len)
            X = X[:,-Q*self.config.short_len:,:].view(B,-1,self.config.short_len,D)
            print("X.shape",X.shape)
            # X = X.reshape(B,-1,self.config.short_len,D)
            m = self.encoder_m(X) 
            self.last_mipp_warm_diagnostics = None
            if self.use_mipp:
                m = F.normalize(m, dim=-1)
                _, m = self.mipp.encode_content(m)
                if pair_X is not None:
                    pair_X = pair_X[:,-Q*self.config.short_len:,:].view(
                        B,-1,self.config.short_len,D
                    )
                    pair_m = F.normalize(self.encoder_m(pair_X), dim=-1)
                    _, pair_m = self.mipp.encode_content(pair_m)
                    stability = F.cosine_similarity(m, pair_m, dim=-1)
                    flat_m = m.reshape(-1, m.size(-1))
                    mask_features = None
                    if self.mipp_write_gate_mode == 'mask_regime_balanced_rank':
                        if seq_mask is None:
                            raise ValueError(
                                'mask-regime admission requires the primary history mask'
                            )
                        mask_features = seq_mask[:, -Q*self.config.short_len:, :].reshape(
                            B, Q, self.config.short_len, seq_mask.size(-1)
                        )
                    # Bio.Cluster evaluates up to 20 clusters, so keep at least
                    # 21 candidates during initial clustering while still
                    # choosing the most mask-stable candidates.
                    minimum_candidates = min(21, flat_m.size(0))
                    eligible, selection = self.mipp.select_write_candidates(
                        stability,
                        mode=self.mipp_write_gate_mode,
                        threshold=float(self.config.mipp_stability_threshold),
                        budget=self.mipp_write_budget,
                        minimum_candidates=minimum_candidates,
                        features=(
                            m
                            if self.mipp_write_gate_mode == 'stability_diverse_rank'
                            else None
                        ),
                        mask_features=mask_features,
                    )
                    flat_eligible = eligible.reshape(-1)
                    m = flat_m[flat_eligible].unsqueeze(0)
                    self.last_mipp_warm_diagnostics = {
                        'warm_pair_cosine': float(stability.mean().item()),
                        'warm_write_acceptance_rate': float(
                            selection['write_effective_acceptance_rate'].item()
                        ),
                        'warm_raw_absolute_acceptance_rate': float(
                            selection['write_raw_absolute_acceptance_rate'].item()
                        ),
                        'warm_write_stability_std': float(
                            selection['write_stability_std'].item()
                        ),
                        'warm_write_effective_cutoff': float(
                            selection['write_effective_cutoff'].item()
                        ),
                        'warm_candidates_used': int(flat_eligible.sum().item()),
                        'warm_candidates_total': int(flat_eligible.numel()),
                        'warm_write_budget': int(self.mipp_write_budget),
                        'warm_write_gate_mode': self.mipp_write_gate_mode,
                    }
                    for key in (
                        'write_candidate_pairwise_cosine',
                        'write_selected_pairwise_cosine',
                        'write_selected_stability',
                        'write_diversity_gain',
                        'write_mask_candidate_pairwise_distance',
                        'write_mask_selected_pairwise_distance',
                        'write_mask_coverage_distance',
                        'write_mask_topk_coverage_distance',
                        'write_mask_coverage_gain_vs_topk',
                        'write_mask_candidate_missing_rate',
                        'write_mask_selected_missing_rate',
                    ):
                        if key in selection:
                            self.last_mipp_warm_diagnostics[f'warm_{key}'] = float(
                                selection[key].item()
                            )
            self.memory.cluster(m)

    def forward(self,Q,seq_mask=None,pair_Q=None,pair_mask=None):# Q->z_s
        # Q (B,L,D)    (B*L,1,D)
        B,L,D = Q.shape
        M_Q = Q.detach()
        Q = Q.unsqueeze(0)
        raw_history = Q.squeeze(0)
        
        unfold_q = self.unfold(Q)[:,:,:L]  # B*L*D -> B*L*L1*D
        
        sliding_q = unfold_q.view(B,self.config.short_len,D,L).permute(0,3,1,2) # B L l D
        u = self.encoder_u(sliding_q)  # B,L,H

        H = u.shape[-1]
        u = u.reshape(-1,H)  #n_u, H
        u = nn.functional.normalize(u,dim=1)  # normalize query
        content_a_raw = None
        content_a = None
        content_b_raw = None
        content_b = None
        self.last_mipp_outputs = None
        if self.use_mipp:
            content_a_raw, content_a = self.mipp.encode_content(u.reshape(B,L,H))
            u = content_a.reshape(-1,H)
            if pair_Q is not None:
                pair_B,pair_L,pair_D = pair_Q.shape
                if (pair_B,pair_L,pair_D) != (B,L,D):
                    raise ValueError('MIPP paired view shape must match the primary history view')
                pair_unfold = self.unfold(pair_Q.unsqueeze(0))[:,:,:L]
                pair_sliding = pair_unfold.view(
                    B,self.config.short_len,D,L
                ).permute(0,3,1,2)
                pair_u = F.normalize(self.encoder_u(pair_sliding), dim=-1)
                content_b_raw, content_b = self.mipp.encode_content(pair_u)
                if seq_mask is None or pair_mask is None:
                    raise ValueError('MIPP paired training requires both primary and paired masks')
                paired = self.mipp.paired_objective(
                    content_a_raw,
                    content_a,
                    seq_mask,
                    content_b_raw,
                    content_b,
                    pair_mask,
                    self.use_content_mask_factorization,
                )
                paired['total_loss'] = (
                    float(self.config.mipp_inv_weight) * paired['invariance_loss']
                    + float(self.config.mipp_var_weight) * paired['variance_loss']
                    + float(self.config.mipp_cov_weight) * paired['covariance_loss']
                    + float(self.config.mipp_dec_weight) * paired['decorrelation_loss']
                    + float(self.config.mipp_mask_weight) * paired['mask_reconstruction_loss']
                )
                self.last_mipp_outputs = paired
        self.last_retrieval_diagnostics = None
        self.last_fusr_outputs = None
        self.last_cfr_outputs = None
        retrieval_statistics = None
        if self.prototype_enabled:
            m = self.memory.repre.clone().detach().to(u.device) # fetch memory on the active device

            if m.shape[0]>self.config.topM:   # random sample topM memory
                ids = torch.randperm(m.size(0))[:self.config.topM]
                m = m[ids,:]

            # print("m.shape,u.shape",m.shape,u.shape)
            #*****************************normalize
            i = torch.matmul(m,u.transpose(1,0)) # n_memory n_u*

            if self.config.topK<m.shape[0]:   # get the topK most correlated memory
                values,idxs = torch.topk(i,k=self.config.topK,dim=0)
                selec_m = m[idxs,:]
                p = F.softmax(values, dim=0)
                retrieved = torch.sum(p.unsqueeze(-1)*selec_m,dim=0)
                selected_values = values
            else:
                p = F.softmax(i, dim=0)    # n_memory n_us
                retrieved = torch.sum(p.unsqueeze(-1)*m.unsqueeze(1),dim=0)
                selected_values = i

            if self.collect_retrieval_diagnostics or self.use_fusr:
                top1_by_time = torch.max(i, dim=0).values.reshape(B,L)
                top1_similarity = top1_by_time.mean(dim=1)
                selected_similarity = selected_values.mean(dim=0).reshape(B,L).mean(dim=1)
                entropy_by_time = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=0)
                entropy_by_time = entropy_by_time.reshape(B,L)
                entropy = entropy_by_time.mean(dim=1)
                retrieval_statistics = {
                    'top1_similarity': top1_similarity,
                    'selected_similarity': selected_similarity,
                    'retrieval_entropy': entropy,
                    'softmax_support_count': int(p.shape[0]),
                    'candidate_count': int(m.shape[0]),
                }
                if self.use_fusr_intervention_features or self.collect_retrieval_diagnostics:
                    support = max(int(p.shape[0]), 2)
                    normalized_entropy_by_time = (
                        entropy_by_time / float(np.log(support))
                    ).clamp(0.0, 1.0)
                    retrieved_by_time = retrieved.reshape(B,L,H)
                    query_by_time = u.reshape(B,L,H)
                    query_norm = query_by_time.norm(dim=-1).clamp_min(1e-6)
                    retrieval_statistics.update({
                        'top1_temporal_std': top1_by_time.std(
                            dim=1, unbiased=False
                        ),
                        'normalized_entropy_temporal_std': (
                            normalized_entropy_by_time.std(dim=1, unbiased=False)
                        ),
                        'retrieved_query_norm_ratio': (
                            retrieved_by_time.norm(dim=-1) / query_norm
                        ).mean(dim=1),
                        'retrieved_query_cosine_mean': F.cosine_similarity(
                            retrieved_by_time,
                            query_by_time,
                            dim=-1,
                            eps=1e-6,
                        ).mean(dim=1),
                    })
            if self.collect_retrieval_diagnostics:
                self.last_retrieval_diagnostics = {
                    'top1_similarity': top1_similarity.detach(),
                    'selected_similarity': selected_similarity.detach(),
                    'retrieval_entropy': entropy.detach(),
                    'candidate_count': int(m.shape[0]),
                }
                if self.use_fusr_intervention_features or self.collect_retrieval_diagnostics:
                    self.last_retrieval_diagnostics.update({
                        key: retrieval_statistics[key].detach()
                        for key in (
                            'top1_temporal_std',
                            'normalized_entropy_temporal_std',
                            'retrieved_query_norm_ratio',
                            'retrieved_query_cosine_mean',
                        )
                    })
        else:
            # Operational intervention: remove only the retrieved prototype
            # vector while preserving the query encoder, raw-input channel,
            # learned projection, residual, and downstream S4 backbone.
            retrieved = torch.zeros_like(u)

        if self.use_fusr:
            if seq_mask is None:
                raise ValueError('FUSR requires the historical observation mask')
            if retrieval_statistics is None:
                zeros = raw_history.new_zeros(B)
                retrieval_statistics = {
                    'top1_similarity': zeros,
                    'selected_similarity': zeros,
                    'retrieval_entropy': zeros,
                    'softmax_support_count': 1,
                    'candidate_count': 0,
                }
                if self.use_fusr_intervention_features:
                    retrieval_statistics.update({
                        'top1_temporal_std': zeros,
                        'normalized_entropy_temporal_std': zeros,
                        'retrieved_query_norm_ratio': zeros,
                        'retrieved_query_cosine_mean': zeros,
                    })

            # V2 intervention features require the two inference-time hidden
            # contexts, so they are deliberately constructed before routing.
            # The V1 branch retains its original feature/router/context order.
            context_on = None
            context_off = None
            if self.use_fusr_intervention_features:
                context_on = self._fuse_context(
                    retrieved, u, raw_history, B, L, H
                )
                context_off = self._fuse_context(
                    torch.zeros_like(retrieved), u, raw_history, B, L, H
                )

            base_features = self.fusr.build_features(
                raw_history,
                seq_mask,
                retrieval_statistics['top1_similarity'],
                retrieval_statistics['selected_similarity'],
                retrieval_statistics['retrieval_entropy'],
                retrieval_statistics['softmax_support_count'],
                use_spectral=self.use_fusr_spectral_uncertainty,
            )
            if self.use_fusr_intervention_features:
                context_delta = context_on - context_off
                context_off_flat = context_off.reshape(B, -1)
                context_on_flat = context_on.reshape(B, -1)
                context_relative_perturbation_norm = (
                    context_delta.reshape(B, -1).norm(dim=1)
                    / context_off_flat.norm(dim=1).clamp_min(1e-6)
                )
                context_on_off_cosine = F.cosine_similarity(
                    context_on_flat,
                    context_off_flat,
                    dim=1,
                    eps=1e-6,
                )
                features = self.fusr.build_intervention_features(
                    base_features,
                    retrieval_statistics['top1_temporal_std'],
                    retrieval_statistics['normalized_entropy_temporal_std'],
                    retrieval_statistics['retrieved_query_norm_ratio'],
                    retrieval_statistics['retrieved_query_cosine_mean'],
                    context_relative_perturbation_norm,
                    context_on_off_cosine,
                )
            else:
                features = base_features
            fusr_outputs = self.fusr(features)
            # A stable alias keeps the training/evaluation code explicit about
            # the uncertainty-penalized lower confidence bound.
            fusr_outputs['utility_lcb'] = fusr_outputs['conservative_utility']
            if not self.use_utility_read_gate or self.fusr_mode == 'on':
                effective_gate = fusr_outputs['gate'].new_ones(B)
            elif self.fusr_mode == 'off' or not self.prototype_enabled:
                effective_gate = fusr_outputs['gate'].new_zeros(B)
            else:
                effective_gate = fusr_outputs['gate']

            if context_on is None:
                context_on = self._fuse_context(
                    retrieved, u, raw_history, B, L, H
                )
                context_off = self._fuse_context(
                    torch.zeros_like(retrieved), u, raw_history, B, L, H
                )
            routed_retrieved = (
                retrieved.reshape(B, L, H)
                * effective_gate.reshape(B, 1, 1)
            ).reshape(-1, H)
            o = self._fuse_context(
                routed_retrieved, u, raw_history, B, L, H
            )
            fusr_outputs.update({
                'features': features,
                'effective_gate': effective_gate,
                'context_on': context_on,
                'context_off': context_off,
                'retrieval_candidate_count': retrieval_statistics['candidate_count'],
                'retrieval_softmax_support_count': retrieval_statistics['softmax_support_count'],
            })
            self.last_fusr_outputs = fusr_outputs
        elif self.use_cfr and self.training:
            # CFR reuses one retrieval and one memory update.  The paired
            # no-prototype context exists only during training and does not
            # introduce a learned router or any inference-time branch.
            context_on = self._fuse_context(
                retrieved, u, raw_history, B, L, H
            )
            context_off = self._fuse_context(
                torch.zeros_like(retrieved), u, raw_history, B, L, H
            )
            o = context_on
            self.last_cfr_outputs = {
                'context_on': context_on,
                'context_off': context_off,
            }
        else:
            o = self._fuse_context(retrieved, u, raw_history, B, L, H)

        # update memory bank
        with torch.no_grad():
            Q = int(M_Q.shape[1]/self.config.short_len)
            lim = Q*self.config.short_len
            q = M_Q[:,-lim:,:].reshape(B,-1,self.config.short_len,D)
            q = self.encoder_m(q)
            q = torch.nn.functional.normalize(q.reshape(-1,H),dim=1)
            q = q.reshape(B,-1,H)
            eligible_mask = None
            write_diagnostics = {}
            if self.use_mipp:
                _, q = self.mipp.encode_content(q)
                if pair_Q is not None:
                    pair_q = pair_Q[:,-lim:,:].reshape(
                        B,-1,self.config.short_len,D
                    )
                    pair_q = F.normalize(self.encoder_m(pair_q), dim=-1)
                    _, pair_q = self.mipp.encode_content(pair_q)
                    write_stability = F.cosine_similarity(q, pair_q, dim=-1)
                    if self.use_stability_write_gate:
                        mask_features = None
                        if self.mipp_write_gate_mode == 'mask_regime_balanced_rank':
                            if seq_mask is None:
                                raise ValueError(
                                    'mask-regime admission requires the primary history mask'
                                )
                            mask_features = seq_mask[:, -lim:, :].reshape(
                                B, Q, self.config.short_len, seq_mask.size(-1)
                            )
                        eligible_mask, selection = self.mipp.select_write_candidates(
                            write_stability,
                            mode=self.mipp_write_gate_mode,
                            threshold=float(self.config.mipp_stability_threshold),
                            budget=self.mipp_write_budget,
                            features=(
                                q
                                if self.mipp_write_gate_mode == 'stability_diverse_rank'
                                else None
                            ),
                            mask_features=mask_features,
                        )
                    write_diagnostics = {
                        'teacher_pair_cosine': write_stability.mean(),
                        'stable_write_acceptance_rate': (
                            eligible_mask.float().mean()
                            if eligible_mask is not None
                            else write_stability.new_ones(())
                        ),
                        'write_stability': write_stability.mean(),
                        'write_stability_std': write_stability.std(unbiased=False),
                    }
                    if eligible_mask is not None:
                        write_diagnostics.update({
                            'write_effective_cutoff': selection['write_effective_cutoff'],
                            'write_raw_absolute_acceptance_rate': (
                                selection['write_raw_absolute_acceptance_rate']
                            ),
                        })
                        for key in (
                            'write_candidate_pairwise_cosine',
                            'write_selected_pairwise_cosine',
                            'write_selected_stability',
                            'write_diversity_gain',
                            'write_mask_candidate_pairwise_distance',
                            'write_mask_selected_pairwise_distance',
                            'write_mask_coverage_distance',
                            'write_mask_topk_coverage_distance',
                            'write_mask_coverage_gain_vs_topk',
                            'write_mask_candidate_missing_rate',
                            'write_mask_selected_missing_rate',
                        ):
                            if key in selection:
                                write_diagnostics[key] = selection[key]
            if self.is_training:
                self.update_k_encoder_weights()
                update = self.memory.add_features_from_sample_learned(
                    q, eligible_mask=eligible_mask
                )
                if self.use_mipp and self.last_mipp_outputs is not None:
                    self.last_mipp_outputs.update(write_diagnostics)
                    self.last_mipp_outputs.update({
                        'write_eligible_count': int(update['eligible_count']),
                        'write_sampled_count': int(update['sampled_count']),
                        'write_updated_count': int(update['updated_count']),
                        'write_budget': int(self.mipp_write_budget),
                    })
        
        return o  # (B,L,D)
