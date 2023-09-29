#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import numpy as np
from .box_utils import decode
from ..box import match


def Detect(
    predictions: torch.Tensor,
    prior: torch.Tensor,
    scale: torch.Tensor,
    eval_thresh: float = 0.01,
    nms_thresh: float = 0.5,
) -> tuple:
    """ Detect layer at test time """

    (loc, conf) = predictions['loc'], predictions['conf']
    assert loc.size(0) == 1,  'ERROR: Batch size = {} during evaluation'.format(loc.size(0))

    (loc, conf) = loc.squeeze(0), conf.squeeze(0)
    decoded_boxes = decode(loc, prior).clamp(min=0, max=1)
    decoded_boxes *= scale  # scale each detection back up to the image
    conf_scores = conf.sigmoid()

    keep = conf_scores.max(1)[0] > eval_thresh
    if not keep.any():
        num_classes = conf.size(1)
        return (np.empty([0, 4]), np.empty([0, num_classes]))
    decoded_boxes=decoded_boxes[keep]
    conf_scores=conf_scores[keep]

    keep = torchvision.ops.nms(decoded_boxes, conf_scores.max(1)[0], iou_threshold=nms_thresh)
    decoded_boxes=decoded_boxes[keep]
    conf_scores=conf_scores[keep]

    (decoded_boxes, conf_scores) = decoded_boxes.cpu().numpy(), conf_scores.cpu().numpy()
    return (decoded_boxes, conf_scores)

def transfuse(conf_data, loc_data, targets, priors, seggt,
    base_anchor: float,
    size: int,
    base_size: int,
    npos=0.75,
):

    with torch.no_grad():
        (num, num_priors, num_classes) = conf_data.size()

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

        pos = overlap_t >= npos
        neg = overlap_t < npos # HERE is the difference
        conf_t[neg] = 0

        labels = torch.zeros(num*num_priors, num_classes+1).cuda().scatter_( 1,
            conf_t.view(-1, 1), 1)
    #return labels[:,1:].unsqueeze(0) # torch.cat(scores, 0).unsqueeze(0)

    import math
    from PIL import Image

    # conf_data = (12750, 20)
    # 12750 = (40 ** 2 + 20 ** 2 + 10 ** 2 + 5 ** 2) * 6 anchors
    # the real order is (6 * 1600) + (6 * 400) + (6 * 100) + (6 * 25)
    # the first 6 entries of 12750 belong to the pixel at spatial coordiates (0, 0)

    # cf = conf_data[:9600, :].reshape(1, 40, 40, 6, 20)
    # cf[0, 0, 1, 0] = conf_data[0, 1, 0] # 2nd anchor of 1st pixel
    # cf[0, 1, 0, 0] = conf_data[0, 6, 0] # 1st pixel of pixel (0, 1)
    if base_size == 320:
        repeat = 4
    elif base_size == 512:
        repeat = 5
    else:
        raise ValueError('Error: Sorry size {} is not supported!'.format(base_size))

    sizes = [math.ceil(size / 2 ** (3 + i)) for i in range(repeat)]
    conf_data = conf_data.sigmoid().squeeze()

    scores = []
    prev = 0
    sem_ = []
    for s in sizes:

        next = prev + (s*s*6)
        conf_ = conf_data[prev:next, :]
        #if s < 40:
        #    scores.append(conf_)
        #    continue
        cf = conf_.reshape(s, s, 6, 20)
        sem = np.array(Image.fromarray(seggt).resize((s, s), Image.NEAREST))
        sem[sem == 255] = 0
        sem = torch.Tensor(sem).unsqueeze(-1).expand_as(cf[..., 0])
        sem_.append(sem.reshape(1, -1))
        #score = torch.zeros((s, s, 6, num_classes+1)).scatter_(3, # +1 = BG
        #        sem.type(torch.int64).unsqueeze(-1), 1)
        #scores.append(score[...,1:].reshape(-1, 20).cuda()) # [1:] to remove BG entry
        #scores[-1][conf_ < .4] = 0
        prev = next

    sem_ = torch.cat(sem_, 1).cuda()
    conf_t[conf_t == 0] = sem_[conf_t == 0].type(torch.int64)
    labels = torch.zeros(num*num_priors, num_classes+1).cuda().scatter_( 1,
            conf_t.view(-1, 1), 1)
    return labels[:,1:].unsqueeze(0) # torch.cat(scores, 0).unsqueeze(0)

def contrast_loss(feat_data, conf_data, loc_data, targets, priors, scale,
        npos=.75):

    breakpoint()
    nneg = -100
    temp = 1
    out_log = True

    with torch.no_grad():
        (num, num_priors, num_classes) = feat_data.size()

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

        pos = overlap_t >= npos
        neg = overlap_t < 0.001
        ign = (~pos) * (~neg)
        conf_t[neg] = 0

        idx_t = pos
        if nneg > 0:

            topk_idx = torch.topk(conf_data.sigmoid().max(2)[0]*neg, nneg, dim=1)[1]
            idx_p = torch.zeros_like(idx_t).cuda().scatter_(1, topk_idx, 1)

        else:
            sigmoid = conf_data.sigmoid().max(2)[0].view(-1)
            topk_idx = torch.topk(sigmoid*neg.view(-1),
                    num*abs(nneg), dim=0)[1]
            idx_p = torch.zeros_like(idx_t.view(-1)).cuda().scatter_(0, topk_idx, 1)
            idx_p = idx_p.view(num, -1)


    (loc_data, conf_data) = loc_data.squeeze(0), conf_data.squeeze(0)
    decoded_boxes = decode(loc_data, priors).clamp(min=0, max=1)
    decoded_boxes *= scale

    keep = idx_t.squeeze()
    conf_data = conf_data[keep].cpu().numpy()


    return  decoded_boxes[keep].cpu().numpy(), \
            conf_t.squeeze()[keep].cpu().numpy(), \
            overlap_t.squeeze()[keep].cpu().numpy()


