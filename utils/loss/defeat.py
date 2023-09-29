# https://github.com/ggjy/DeFeat.pytorch/blob/c46a793df414a42f64f9fad9c7106ee60b44c9b3/mmdet/apis/train.py#L40

import torch
import torch.nn.functional as F
from collections import OrderedDict
import torch.distributed as dist

def KLDivergenceLoss(y, teacher_scores, mask=None, T=1):
    if mask is not None:
        if mask.sum() > 0:
            p = F.log_softmax(y/T, dim=1)[mask]
            q = F.softmax(teacher_scores/T, dim=1)[mask]
            l_kl = F.kl_div(p, q, reduce=False)
            loss = torch.sum(l_kl)
            loss = loss / mask.sum()
        else:
            loss = torch.Tensor([0]).cuda()
    else:
        p = F.log_softmax(y/T, dim=1)
        q = F.softmax(teacher_scores/T, dim=1)
        l_kl = F.kl_div(p, q, reduce=False)
        loss = l_kl.sum() / l_kl.size(0)
    return loss * T**2


def BCELoss(y, teacher_scores, mask):
    p = F.softmax(y, dim=1)[mask]
    q = F.softmax(teacher_scores, dim=1)[mask]
    loss = F.binary_cross_entropy(p, q.detach()) * 10.0
    return loss


def l1loss(pred_s, pred_t, target):
    assert pred_s.size() == pred_t.size() == target.size() and target.numel() > 0
    loss_s_t = torch.abs(pred_s - pred_t).sum(1) / 4.0
    loss_s_gt = torch.abs(pred_s - target).sum(1) / 4.0
    loss = loss_s_t[loss_s_t<=loss_s_gt].sum() + loss_s_gt[loss_s_gt<loss_s_t].sum()
    return loss / target.size(0)


def l1rpnloss(pred_s, pred_t, target, weights):
    assert pred_s.size() == pred_t.size() == target.size()
    loss_s_t = torch.abs(pred_s * weights - pred_t * weights).sum(1) / 4.0
    loss_s_gt = torch.abs(pred_s * weights - target * weights).sum(1) / 4.0
    loss = loss_s_t[loss_s_t<=loss_s_gt].sum() + loss_s_gt[loss_s_gt<loss_s_t].sum()
    return loss, weights.sum()/4


def mseloss(pred_s, pred_t, target, weights):
    if weights is not None:
        pred_t = pred_t[weights.type(torch.bool)]
        pred_s = pred_s[weights.type(torch.bool)]
        if weights.sum() > 0:
            pred_s = pred_s.sigmoid()
            pred_t = pred_t.sigmoid()
            loss = F.mse_loss(pred_s, pred_t, reduction='none')
            return loss.sum(), weights.sum()
        else:
            return 0., 0.
    else:
        pred_s = pred_s.sigmoid()
        pred_t = pred_t.sigmoid()
        loss = F.mse_loss(pred_s, pred_t, reduction='none')
        return loss.sum(), loss.size(0)

def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars

def defeat_loss(neck_feat, neck_mask_batch, neck_feat_t,
        bb_feat=None, bb_mask_batch=None, bb_feat_t=None,
        neck_adapt=None, bb_adapt=None,
        kd_warm=dict(), kd_decay=1., epoch=0,
        kd_cfg='neck-adapt,neck-decouple,mask-neck-gt,pixel-wise',
        hint_neck_w=1,#3,
        hint_neck_back_w=1,#12,
        bb_indices=(1,2,3),hint_bb_back_w=0,hint_bb_w=0
    ):

    losses = {}
    if 'neck' in kd_cfg:
        losskd_neck = torch.Tensor([0]).cuda()
        if 'neck-decouple' in kd_cfg:
            losskd_neck_back = torch.Tensor([0]).cuda()
        _neck_feat = neck_feat
        mask_hint = neck_mask_batch.unsqueeze(-1)
        norms = max(1.0, mask_hint.sum() * 2)
        if 'neck-adapt' in kd_cfg and neck_adapt is not None:
            # neck_feat_adapt = neck_adapt[i](_neck_feat)
            pass
        else:
            neck_feat_adapt = _neck_feat

        if 'pixel-wise' in kd_cfg:
            if 'L1' in kd_cfg:
                diff = torch.abs(neck_feat_adapt - neck_feat_t)#[i])
                loss = torch.where(diff < 1.0, diff, diff**2)
                losskd_neck += (loss * mask_hint).sum() / norms
            elif 'Div' in kd_cfg:
                losskd_neck += (torch.pow(1 -
                    neck_feat_adapt / (neck_feat_t + 1e-8), 2) # neck_feat_t[i]
                    * mask_hint).sum() / norms
            elif 'neck-decouple' in kd_cfg:
                norms_back = max(1.0, (1 - mask_hint).sum() * 2)
                losskd_neck_back += (torch.pow(
                    neck_feat_adapt - neck_feat_t, 2) * (1 - mask_hint)).sum() / norms_back
                losskd_neck += (torch.pow(
                    neck_feat_adapt - neck_feat_t, 2) * mask_hint).sum() / norms
            else:
                losskd_neck += (torch.pow(
                    neck_feat_adapt - neck_feat_t, 2) * mask_hint).sum() / norms

        if 'pixel-wise' in kd_cfg:
            losskd_neck = losskd_neck / len(neck_feat)
            losskd_neck = losskd_neck * hint_neck_w
            if 'decay' in kd_cfg:
                losskd_neck *= kd_decay
            if kd_warm.get('hint', False):
                losskd_neck *= 0.
            losses['losskd_neck'] = losskd_neck

        if 'neck-decouple' in kd_cfg:
            losskd_neck_back = losskd_neck_back / len(neck_feat)
            losskd_neck_back = losskd_neck_back * hint_neck_back_w
            if 'decay' in kd_cfg:
                losskd_neck_back *= kd_decay
            if kd_warm.get('hint', False):
                losskd_neck_back *= 0.
            losses['losskd_neck_back'] = losskd_neck_back

    # kd: backbone imitation w/ or w/o adaption layer
    if 'bb' in kd_cfg:
        losskd_bb = torch.Tensor([0]).cuda()
        if 'bb-decouple' in kd_cfg:
            losskd_bb_back = torch.Tensor([0]).cuda()
        mask_hint = bb_mask_batch.unsqueeze(1)
        for i, indice in enumerate(bb_indices):
            if 'bb-adapt' in kd_cfg and bb_adapt is not None:
                bb_feat_adapt = bb_adapt[i](bb_feat[indice])
            else:
                bb_feat_adapt = bb_feat[indice]
            c, h, w = bb_feat_adapt.shape[1:]
            mask_bb = F.interpolate(mask_hint, size=[h, w], mode="nearest").repeat(1, c, 1, 1)
            norms = max(1.0, mask_bb.sum() * 2)
            if 'bb-decouple' in kd_cfg:
                losskd_bb += (torch.pow(bb_feat_adapt - bb_feat_t[indice], 2) * mask_bb).sum() / norms
                norms_back = max(1, (1 - mask_bb).sum() * 2)
                losskd_bb_back += (torch.pow(bb_feat_adapt - bb_feat_t[indice], 2) * (1 - mask_bb)).sum() / norms_back
            else:
                losskd_bb += (torch.pow(bb_feat_adapt - bb_feat_t[indice], 2) * mask_bb).sum() / norms

        losskd_bb /= len(bb_indices)
        losskd_bb *= hint_bb_w
        if 'bb-decouple' in kd_cfg:
            losskd_bb_back /= len(bb_indices)
            losskd_bb_back *= hint_bb_back_w
        if 'decay' in kd_cfg:
            losskd_bb *= kd_decay
            if 'bb-decouple' in kd_cfg:
                losskd_bb_back *= kd_decay
        if kd_warm.get('hint', False):
            losskd_bb *= 0
            if 'bb-decouple' in kd_cfg:
                losskd_bb_back *= 0
        losses['losskd_bb'] = losskd_bb
        if 'bb-decouple' in kd_cfg:
            losses['losskd_bb_back'] = losskd_bb_back

    return losses
    loss, log_vars = parse_losses(losses)
    outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=10)

    return outputs
