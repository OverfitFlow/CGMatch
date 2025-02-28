
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch 
import torch.nn as nn 
from torch.nn import functional as F

from .cross_entropy import ce_loss



def consistency_loss(logits, targets, name='ce', mask=None):
    """
    consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagation, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()



class ConsistencyLoss(nn.Module):
    """
    Wrapper for consistency loss
    """
    def forward(self, logits, targets, name='ce', mask=None):
        return consistency_loss(logits, targets, name, mask)
    
class GCELoss(nn.Module):
    def __init__(self, num_classes, q=0.7):
        super(GCELoss, self).__init__()
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels, mask):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        
        if mask is not None:
        # mask must not be boolean type
            loss = loss * mask
        
        return loss.mean()
