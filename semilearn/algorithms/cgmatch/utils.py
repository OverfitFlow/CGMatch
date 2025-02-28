# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
from copy import deepcopy
from collections import Counter
from semilearn.algorithms.utils import concat_all_gather
from semilearn.algorithms.hooks import MaskingHook


class FixedThresholdingHook(MaskingHook):
    """
    Dynamic Threshold.
    """
    def __init__(self, num_classes, n_sigma=2, momentum=0.999, per_class=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_classes = num_classes
        self.n_sigma = n_sigma
        self.per_class = per_class
        self.m = momentum

        # initialize Gaussian mean and variance
        if not self.per_class:
            self.prob_max_mu_t = torch.tensor(1.0 / self.num_classes)
            self.prob_max_var_t = torch.tensor(1.0)
        else:
            self.prob_max_mu_t = torch.ones((self.num_classes)) / self.args.num_classes
            self.prob_max_var_t =  torch.ones((self.num_classes))

        self.count_gap_mu_t = None
    
    @torch.no_grad()
    def update(self, algorithm, probs_x_ulb, count_gaps):
        if algorithm.distributed and algorithm.world_size > 1:
            probs_x_ulb = self.concat_all_gather(probs_x_ulb)
        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        if not self.per_class:
            prob_max_mu_t = torch.mean(max_probs) # torch.quantile(max_probs, 0.5)
            prob_max_var_t = torch.var(max_probs, unbiased=True)
            self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t.item()
            self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t.item()
        else:
            prob_max_mu_t = torch.zeros_like(self.prob_max_mu_t)
            prob_max_var_t = torch.ones_like(self.prob_max_var_t)
            for i in range(self.num_classes):
                prob = max_probs[max_idx == i]
                if len(prob) > 1:
                    prob_max_mu_t[i] = torch.mean(prob)
                    prob_max_var_t[i] = torch.var(prob, unbiased=True)
            self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t
            self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t
        
        if self.count_gap_mu_t is None:
            self.count_gap_mu_t = torch.tensor(sum(algorithm.count_gap.values()) / len(algorithm.count_gap)).cuda(algorithm.gpu)
        else:
            count_gap_mu_t = torch.sum(count_gaps) / len(count_gaps)
            self.count_gap_mu_t = self.m * self.count_gap_mu_t + (1 - self.m) * count_gap_mu_t.item()
        
        return max_probs, max_idx
    
    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, idx_ulb, softmax_x_ulb=True, *args, **kwargs):
        if not self.prob_max_mu_t.is_cuda:
            self.prob_max_mu_t = self.prob_max_mu_t.to(logits_x_ulb.device)
        if not self.prob_max_var_t.is_cuda:
            self.prob_max_var_t = self.prob_max_var_t.to(logits_x_ulb.device)

        if softmax_x_ulb:
            # probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
            probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        count_gaps = torch.tensor([algorithm.count_gap[guid] for guid in idx_ulb.cpu().tolist()]).to(logits_x_ulb.device)
        self.update(algorithm, probs_x_ulb, count_gaps)
        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        
        # compute weight
        if not self.per_class:
            mu = self.prob_max_mu_t
        else:
            mu = self.prob_max_mu_t[max_idx]
        
        count_gap_mu = self.count_gap_mu_t

        if algorithm.args.dataset == 'svhn':
            mu = torch.clamp(mu, min=0.9, max=0.95)
        # easy_to_learn: max_probs >= mu
        easy_mask = max_probs.ge(mu).to(max_probs.dtype)
        
        # ambiguous: count_gap >= count_gap_mu & max_probs < mu
        ambiguous_mask = (count_gaps.ge(count_gap_mu)) & (max_probs < mu)
        
        return easy_mask, ambiguous_mask.to(max_probs.dtype)
