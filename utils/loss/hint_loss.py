#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from .balanced_l1_loss import BalancedL1Loss
from .gfocal_loss import GFocalLoss

class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 1.0,
    ) -> None:
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if mask is not None:
            pred, target = pred[mask], target[mask]
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) *
                            (1 - target)) * pt.pow(self.gamma)
        loss = focal_weight * F.binary_cross_entropy_with_logits(pred, target,
                reduction='none')
        loss = loss.mean() #sum() / (target>=.05).float().sum()
        return loss # different from .focal_loss import FocalLoss

class HintLoss(nn.Module):

    def __init__(
        self,
        mode: str = 'pdf',
        loss_weight: float = 5.0,
        use_focal=True,
    ) -> None:
        super(HintLoss, self).__init__()

        self.mode = mode
        self.loss_weight = loss_weight
        print('Using {} mode...'.format(self.mode))

        self.l1_loss = BalancedL1Loss()

        if False:
            self.conf_loss = nn.CrossEntropyLoss()
            self.conf_loss = nn.KLDivLoss()
        if use_focal:
            self.conf_loss = FocalLoss()
        else:
            self.conf_loss = GFocalLoss()
        self.T = 1

    def dist_loss(self, t, s):
        # https://discuss.pytorch.org/t/is-this-loss-function-for-knowledge-distillation-correct/69863
        prob_t = F.softmax(t/self.T, dim=1)
        log_prob_s = F.log_softmax(s/self.T, dim=1)
        dist_loss = -(prob_t*log_prob_s).sum(dim=1).mean()
        # doesnt have 2nd term: prob_t * log_prob_t
        return dist_loss

    def forward(
        self,
        pred_t: torch.Tensor,
        pred_s: torch.Tensor,
        masks=None,
        task='det',
    ) -> torch.Tensor:

        loss = {}

        if all(f not in self.mode for f in ['mse', 'pdf', 'def', 'soft']):
            return loss

        # segmentation task

        if task == 'seg':
            seg_t, fea_t = pred_t['seg'], pred_t['feature']
            seg_t, fea_t = seg_t.detach(), fea_t.detach()
            seg_s, fea_s = pred_s['seg'], pred_s['feature_s']

            loss['loss_kd'] = ((fea_s-fea_t)**2).mean() * self.loss_weight

            if "soft" in self.mode:
                weight = re.search("soft(\d*(?:[eE][+\-]?\d+))", self.mode)
                weight = eval(weight.group(1)) if weight is not None else 5e1
                # loss['loss_s'] = weight * self.conf_loss(seg_s, seg_t.sigmoid())
                loss['loss_s'] = weight * nn.KLDivLoss()(seg_s, seg_t)

            return loss

        # detection task, original implementation

        loc_t, conf_t, fea_t = pred_t['loc'], pred_t['conf'], pred_t['feature']
        loc_t, conf_t, fea_t = loc_t.detach(), conf_t.detach(), fea_t.detach()
        loc_s, conf_s, fea_s = pred_s['loc'], pred_s['conf'], pred_s['feature_d']

        if "soft" in self.mode:
            loss['loss_l'] = 1e0 * self.l1_loss(loc_s, loc_t)
            # breakpoint()
            # KLDivergence
            # loss['loss_c'] = 1e0 * self.conf_loss(F.log_softmax(conf_s/self.T, dim=2),
            #                 F.softmax(conf_t/self.T,dim=2)) * self.T * self.T
            # * alpha + det * (1 - alpha)
            # choosing 5e3 instead of 5e4 because it should not dominate other
            # losses, and shouldn't be too good so can be improved.
            weight = re.search("soft(\d*(?:[eE][+\-]?\d+))", self.mode)
            weight = eval(weight.group(1)) if weight is not None else 5e3
            loss['loss_c'] = weight * self.conf_loss(conf_s, conf_t.sigmoid())

        if 'mse' in self.mode:
            loss['loss_kd'] = ((fea_s-fea_t)**2).mean() * self.loss_weight

        if 'pdf' in self.mode:
            with torch.no_grad():
                x1 = conf_t.sigmoid()
                x2 = conf_s.sigmoid()
                disagree = (x1 - x2) ** 2
                weight = disagree.sum(-1).unsqueeze(1).sqrt()
                # 6 anchor per location
                weight = F.avg_pool1d(weight, kernel_size=6, stride=6, padding=0)
                weight = weight.permute(0,2,1).expand_as(fea_t)
                weight = weight / weight.sum()
            loss_pdf = (weight*((fea_s-fea_t)**2)).sum() * self.loss_weight

            loss_cls = disagree * F.binary_cross_entropy_with_logits(conf_s, x1,
                    reduction='none')
            loss_cls = loss_cls.sum() / (x1>0.5).float().sum()

            loss_reg = F.l1_loss(loc_s, loc_t)

            loss['loss_kd'] = loss_pdf + loss_cls + loss_reg

        if 'def' in self.mode and masks is not None: # defeat and def256
            from utils.loss.defeat import defeat_loss
            losses = defeat_loss(fea_s, masks, fea_t)
            weight = 1. if 'def256' not in self.mode else 256.
            loss['loss_kd'] = (losses['losskd_neck'] + losses['losskd_neck_back'])/weight

        return loss

            # https://github.com/ggjy/DeFeat.pytorch/blob/c46a793df414a42f64f9fad9c7106ee60b44c9b3/mmdet/models/detectors/single_stage_kd.py
            # tools/train_kd.py -> apis/train.py: train_detector_kd ->
            # apis/train.py: batch_processor_kd
            # main/mmdet/apis/train.py#L299
            # I don't use the head-cls. I found that the boost from head-cls is limited, sometimes you can do the grid search to find the best weight for head-cls, but it can only bring about 0.1-0.2% improvement. So I don't use the head-cls in our paper.

        raise NotImplementedError
