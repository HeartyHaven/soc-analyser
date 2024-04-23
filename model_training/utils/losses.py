import functools
import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F


__all__ = ['L1Loss', 'MSELoss','L1_CCLoss']


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()

    return loss.sum()


def mask_reduce_loss(loss, weight=None, reduction='mean', sample_wise=False):
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        if weight.size(1) == 1:
            weight = weight.expand_as(loss)
        eps = 1e-12

        if sample_wise:
            weight = weight.sum(dim=[1, 2, 3], keepdim=True)
            loss = (loss / (weight + eps)).sum() / weight.size(0)
        else:
            loss = loss.sum() / (weight.sum() + eps)

    return loss

def masked_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                sample_wise=False,
                **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = mask_reduce_loss(loss, weight, reduction, sample_wise)
        return loss

    return wrapper

@masked_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@masked_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

class L1Loss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction='mean', sample_wise=False):
        super().__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


class PearsonCorrelationLoss(nn.Module):
    def __init__(self, loss_weight=100.0):
        super().__init__()
        self.loss_weight = loss_weight
        
    def forward(self, x, y):
        # print('isnan(x).shape:',torch.isnan(x).shape)
        mask = torch.isnan(x) & torch.isnan(y) 
        # print('mask shape:',mask.shape)
        x = x[~mask]
        y = y[~mask]

        corr = torch.corrcoef(torch.stack([x, y],dim=0))[0,1]
        
        loss = (1 - corr)**2 * self.loss_weight
        # print('ccloss: ',loss)
        return loss.squeeze()
    

class MSELoss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction='mean', sample_wise=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)

class L1_CCLoss(nn.Module):
    def __init__(self, loss_weight=80.0):
        super().__init__()
        self.l1_loss = L1Loss(loss_weight)
        self.pearson_loss = PearsonCorrelationLoss(100-loss_weight)

    def forward(self, pred, target):
        l1_loss = self.l1_loss(pred, target)
        pearson_loss = self.pearson_loss(pred, target)
        return l1_loss + pearson_loss