#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from ..box import match, mutual_match, encode
from .focal_loss import FocalLoss
from .gfocal_loss import GFocalLoss
from .balanced_l1_loss import BalancedL1Loss

class MultiBoxLoss(nn.Module):
    """ Object Detection Loss """

    def __init__(
        self,
        mutual_guide: bool = True,
        nneg=0,
        npos=0.75,
        lneg=0.4,
        use_focal=False,
        topk=0,
        tau=1.0,
        diff_img=False,
        diff_sim='',
        same_det=False,
        topk_base='conf',
        conf_sim='',
        use_seggt='',
        iou_base='gt',
        sim_query='',
        reduce=False,
        queue=0,
    ) -> None:
        super(MultiBoxLoss, self).__init__()
        self.mutual_guide = mutual_guide
        self.focal_loss = FocalLoss()
        self.gfocal_loss = GFocalLoss()
        self.l1_loss = BalancedL1Loss()
        self.out_log = True
        self.tau = tau
        self.nneg = nneg
        self.lneg = lneg
        self.npos = npos
        self.use_focal = use_focal
        self.topk = topk
        self.diff_img = diff_img
        self.diff_sim = diff_sim
        self.same_det = same_det
        self.conf_sim = conf_sim
        self.topk_base = topk_base
        self.use_seggt = use_seggt
        self.iou_base = iou_base
        self.sim_query = sim_query
        self.reduce = reduce

        self.bins_topk = torch.arange(0, 1, 0.005).cuda()
        self.bins_iou = torch.arange(0, 1, 0.05).cuda()

    def forward(
        self,
        predictions: dict,
        priors: torch.Tensor,
        targets: list,
        seggt: torch.Tensor = None,
        seg_overlap=None,
        seg_conf=None,
        get_heat=False,
        note="",
    ) -> tuple:
        (loc_data, conf_data) = predictions['loc'], predictions['conf']
        (num, num_priors, num_classes) = conf_data.size()
        result = {}

        if self.mutual_guide:

            # match priors (default boxes) and ground truth boxes
            loc_t = torch.zeros(num, num_priors, 4).cuda()
            conf_t = torch.zeros(num, num_priors).cuda().long()
            overlap_t = torch.zeros(num, num_priors).cuda()
            pred_t = torch.zeros(num, num_priors).cuda()
            for idx in range(num):
                truths = targets[idx][:, :-1]
                labels = targets[idx][:, -1].long()
                regress = loc_data[idx, :, :]
                classif = conf_data[idx, :, :]
                mutual_match(truths, priors, regress, classif, labels, loc_t, conf_t, overlap_t, pred_t, idx)
            loc_t.requires_grad = False
            conf_t.requires_grad = False
            overlap_t.requires_grad = False
            pred_t.requires_grad = False

            # Localization Loss (Smooth L1)
            pos = pred_t >= 3.0
            priors_ = priors.unsqueeze(0).expand_as(loc_data)
            mask = pos.unsqueeze(-1).expand_as(loc_data)

            weights = (pred_t-3.0).relu().unsqueeze(-1).expand_as(loc_data)
            weights = weights[mask].view(-1, 4)
            weights = weights / weights.sum()

            loc_p = loc_data[mask].view(-1, 4)
            loc_t = loc_t[mask].view(-1, 4)
            priors_ = priors_[mask].view(-1, 4)
            result['loss_l']= self.l1_loss(loc_p, encode(loc_t, priors_), weights=weights)

            # Classification Loss
            neg = overlap_t <= 1.0
            conf_t[neg] = 0

            batch_label = torch.zeros(num * num_priors, num_classes + 1).cuda().scatter_(1, conf_t.view(-1, 1), 1)
            batch_label = batch_label[:, 1:].view(num, num_priors, num_classes)  # shape: (batch_size, num_priors, num_classes)
            score = (overlap_t-3.0).relu().unsqueeze(-1).expand_as(batch_label)

            if self.use_focal:
                mask = batch_label >= 0
                result['loss_c']= self.focal_loss(conf_data, batch_label, mask)
            else:
                batch_label = batch_label * score
                result['loss_c']= self.gfocal_loss(conf_data, batch_label)

        else:

            # match priors (default boxes) and ground truth boxes
            overlap_t = torch.zeros(num, num_priors).cuda()
            loc_t = torch.zeros(num, num_priors, 4).cuda()
            conf_t = torch.zeros(num, num_priors).cuda().long()
            for idx in range(num):
                truths = targets[idx][:, :-1]
                labels = targets[idx][:, -1].long()
                match(truths, priors, labels, loc_t, conf_t, overlap_t, idx)
            overlap_t.requires_grad = False
            loc_t.requires_grad = False
            conf_t.requires_grad = False

            pos = overlap_t >= (self.npos if self.same_det else 0.5)
            neg = overlap_t < (0.001 if self.same_det else 0.4) #self.lneg)
            ign = (~pos) & (~neg)

            mov = None
            if self.same_det:
                if self.topk:
                    idx, pos = self.choose_topk(overlap_t, conf_t, conf_data, num)
                else:
                    # idx, pos = self.choose_iou_nneg(overlap_t, conf_t, conf_data, num)
                    idx, pos, mov = self.choose_fixed_topk(overlap_t, conf_t, conf_data, num)
                # don't need neg because conf_t is changed in-line within these
                ign = (~pos) & (conf_t>0) # (~pos) & (~neg) # neg = (conf_t==0)
            else:
                pos = overlap_t >= 0.5
                ign = (overlap_t < 0.5) * (overlap_t >= 0.4) #self.lneg)
                neg = overlap_t < 0.4 #self.lneg
                conf_t[neg] = 0

            # Localization Loss (Smooth L1)
            priors_ = priors.unsqueeze(0).expand_as(loc_data)
            mask = pos.unsqueeze(-1).expand_as(loc_data)
            loc_p = loc_data[mask].view(-1, 4)
            loc_t = loc_t[mask].view(-1, 4)
            priors_ = priors_[mask].view(-1, 4)

            result['loss_l']= self.l1_loss(loc_p, encode(loc_t, priors_))

            # Classification Loss
            batch_label = torch.zeros(num * num_priors, num_classes + 1).cuda().scatter_(1, conf_t.view(-1, 1), 1)
            batch_label = batch_label[:, 1:].view(num, num_priors, num_classes)  # shape: (batch_size, num_priors, num_classes)
            ign = ign.unsqueeze(-1).expand_as(batch_label)  # shape: (batch_size, num_priors, num_classes)
            batch_label[ign] *= -1
            mask = batch_label >= 0
            if self.use_focal:
                result['loss_c']= self.focal_loss(conf_data, batch_label, mask)
            else:
                result['loss_c']= self.gfocal_loss(conf_data, batch_label, mask)

            result['det_pos'] = pos
            result['det_neg'] = neg

        return result
