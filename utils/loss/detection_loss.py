import re
import torch
import torch.nn as nn
from ..box import match, mutual_match, encode, center_size
from .focal_loss import FocalLoss
from .gfocal_loss import GFocalLoss
from .balanced_l1_loss import BalancedL1Loss

class DetectionLoss(nn.Module):
    """ Object Detection Loss """

    def __init__(
        self,
        mutual_guide: bool = True,
        use_focal=False,
    ) -> None:
        super(DetectionLoss, self).__init__()
        self.mutual_guide = mutual_guide
        self.focal_loss = FocalLoss()
        self.gfocal_loss = GFocalLoss()
        self.l1_loss = BalancedL1Loss()
        self.use_focal = use_focal

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
                mutual_match(truths, priors, regress, classif, labels, loc_t,
                        conf_t, overlap_t, pred_t, idx)
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

            batch_label = torch.zeros(num * num_priors, num_classes +
                    1).cuda().scatter_(1, conf_t.view(-1, 1), 1)
            # shape: (batch_size, num_priors, num_classes)
            batch_label = batch_label[:, 1:].view(num, num_priors, num_classes)
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
            self.overlap_t = overlap_t.clone()
            self.conf_t = conf_t.clone()

            pos = overlap_t >= 0.5
            neg = overlap_t < 0.4
            ign = (~pos) & (~neg)
            conf_t[neg] = 0

            # Localization Loss (Smooth L1)
            priors_ = priors.unsqueeze(0).expand_as(loc_data)
            mask = pos.unsqueeze(-1).expand_as(loc_data)
            loc_p = loc_data[mask].view(-1, 4)
            loc_t = loc_t[mask].view(-1, 4)
            priors_ = priors_[mask].view(-1, 4)
            if "oracloc" in note:
                if "oracloc_strict" in note:
                    loc_p = encode(loc_t, priors_)
                thr = re.search('_([\d.]+)', note)
                if thr is not None:
                    neg = overlap_t < float(thr.group(1))
                    pos = overlap_t >= float(thr.group(1))
                self.nneg = 0 # set 0 negative for contrastive loss
                ign = ign | neg # ignoring the background anchors
            result['loss_l']= self.l1_loss(loc_p, encode(loc_t, priors_))

            # Classification Loss
            batch_label = torch.zeros(num * num_priors, num_classes +
                    1).cuda().scatter_(1, conf_t.view(-1, 1), 1)

            # shape: (batch_size, num_priors, num_classes)
            batch_label = batch_label[:, 1:].view(num, num_priors, num_classes)
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
