#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import numpy as np


def xyxy_to_xywh(xyxy):
    "https://github.com/facebookresearch/Detectron/blob/60f66a1780cc4e8c8d49520050d6522b88c6f82c/detectron/utils/boxes.py#L92"
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')

def point_form(
    boxes: torch.Tensor,
) -> torch.Tensor:
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax) """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)


def center_size(
    boxes: torch.Tensor,
) -> torch.Tensor:
    """ Convert prior_boxes to (cx, cy, w, h) """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]), 1)


def jaccard(
    box_a: torch.Tensor,
    box_b: torch.Tensor,
) -> torch.Tensor:
    """ Compute the jaccard overlap of two sets of boxes """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


@torch.no_grad()
def gen_fg_mask(
    truths: torch.Tensor,
    priors: torch.Tensor,
    mask: torch.Tensor,
    idx: int,
) -> None:
    """
        Match each prior box with the ground truth box
            truths: list of ground truth boxes, in (xmin, ymin, xmax, ymax)
            priors: list of anchor boxes, in (cx, cy, w, h) form
    """
    priors = priors[::6, :2]
    mask_  = truths[:, 0].unsqueeze(-1) <= priors[:, 0].unsqueeze(0)
    mask_ &= truths[:, 2].unsqueeze(-1) >= priors[:, 0].unsqueeze(0)
    mask_ &= truths[:, 1].unsqueeze(-1) <= priors[:, 1].unsqueeze(0)
    mask_ &= truths[:, 3].unsqueeze(-1) >= priors[:, 1].unsqueeze(0)

    mask[idx] = (mask_.sum(0) > 0).double()

@torch.no_grad()
def match(
    truths: torch.Tensor,
    priors: torch.Tensor,
    labels: torch.Tensor,
    loc_t: torch.Tensor,
    conf_t: torch.Tensor,
    overlap_t: torch.Tensor,
    idx: int,
) -> None:
    """ Match each prior box with the ground truth box """
    overlaps = jaccard(truths, point_form(priors))
    (best_truth_overlap, best_truth_idx) = overlaps.max(0)
    (best_prior_overlap, best_prior_idx) = overlaps.max(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 1)  # ensure best prior
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    overlap_t[idx] = best_truth_overlap  # [num_priors] jaccord for each prior
    conf_t[idx] = labels[best_truth_idx]  # [num_priors] top class label for each prior
    loc_t[idx] = truths[best_truth_idx]  # Shape: [num_priors,4]


@torch.no_grad()
def mutual_match(
    truths: torch.Tensor,
    priors: torch.Tensor,
    regress: torch.Tensor,
    classif: torch.Tensor,
    labels: torch.Tensor,
    loc_t: torch.Tensor,
    conf_t: torch.Tensor,
    overlap_t: torch.Tensor,
    pred_t: torch.Tensor,
    idx: int,
    topk: int = 15,
    sigma: float =2.0,
) -> None:
    """Classify to regress and regress to classify, Mutual Match for label assignement """
    reg_overlaps = jaccard(truths, decode(regress, priors))
    pred_classifs = jaccard(truths, point_form(priors))
    classif = classif.sigmoid().t()[labels - 1, :]
    # pred_classifs = pred_classifs ** ((sigma - classif + 1e-6) / sigma)
    pred_classifs = (pred_classifs * torch.exp(classif / sigma)).clamp_(max=1, min=0)
    reg_overlaps[reg_overlaps != reg_overlaps.max(dim=0, keepdim=True)[0]] = 0.0
    pred_classifs[pred_classifs != pred_classifs.max(dim=0, keepdim=True)[0]] = 0.0

    for (reg_overlap, pred_classif) in zip(reg_overlaps, pred_classifs):
        num_pos = max(1, torch.topk(reg_overlap, topk, largest=True)[0].sum().int())
        num_pos = min(num_pos, (reg_overlap > 0).sum())
        pos_mask = torch.topk(reg_overlap, num_pos, largest=True)[1]
        reg_overlap[pos_mask] += 3.0
        num_pos = max(1, torch.topk(pred_classif, topk, largest=True)[0].sum().int())
        num_pos = min(num_pos, (pred_classif > 0).sum())
        pos_mask = torch.topk(pred_classif, num_pos, largest=True)[1]
        pred_classif[pos_mask] += 3.0

    ## for classification ###
    (best_truth_overlap, best_truth_idx) = reg_overlaps.max(dim=0)
    overlap_t[idx] = best_truth_overlap  # [num_priors] jaccord for each prior
    conf_t[idx] = labels[best_truth_idx]  # [num_priors] top class label for each prior
    ## for regression ###
    (best_truth_overlap, best_truth_idx) = pred_classifs.max(dim=0)
    pred_t[idx] = best_truth_overlap  # [num_priors] jaccord for each prior
    loc_t[idx] = truths[best_truth_idx]  # Shape: [num_priors,4]


def encode(
    matched: torch.Tensor,
    priors: torch.Tensor,
    variances: list = [0.1, 0.2],
) -> torch.Tensor:
    """ Encode from the priorbox layers to ground truth boxes """
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= variances[0] * priors[:, 2:]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    targets = torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]
    return targets


def decode(
    loc: torch.Tensor,
    priors: torch.Tensor,
    variances: list = [0.1, 0.2],
) -> torch.Tensor:
    """ Decode locations from predictions using priors
        priors: in (cx, cy, w, h) format
    """
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                       priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)

    # converting from (cx, cy, w, h) to (x1, y1, x2, y2) format
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
