import re
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..box import match, mutual_match, encode, center_size
from .focal_loss import FocalLoss
from .gfocal_loss import GFocalLoss
from .balanced_l1_loss import BalancedL1Loss

from typing import List
from fvcore.nn import sigmoid_focal_loss_jit, giou_loss, smooth_l1_loss

# from ..box_regression import Box2BoxTransformLinear as box2box_transform

"""ref
/share/home/leh/anaconda3/envs/bmvc22/lib/python3.9/site-packages/detectron2/modeling/meta_arch/fcos.py
"""

class FCOSLoss(nn.Module):
    """ Object Detection Loss """

    def __init__(
        self,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
    ) -> None:
        super(FCOSLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.l1_loss = BalancedL1Loss()

        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.box2box_transform = Box2BoxTransformLinear(normalize_by_size=False)

    def forward(
        self,
        predictions: dict,
        anchors: torch.Tensor,
        gt_instances: list
    ):
        """
        This method is almost identical to :meth:`RetinaNet.losses`, with an extra
        "loss_centerness" in the returned dict.
        """
        self.num_anchors_per_level = [x.size(1) for x in predictions['loc']]
        pred_anchor_deltas = torch.cat(predictions['loc'], dim=1).float()
        pred_logits = torch.cat(predictions['conf'], dim=1)
        pred_centerness = predictions['cnter']
        (_, _, self.num_classes) = pred_logits.size()

        gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
        gt_labels = torch.stack(gt_labels).long()  # (M, R)
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        normalizer = num_pos_anchors
        # TODO: necessary?
        # normalizer = self._ema_update("loss_normalizer", max(num_pos_anchors, 1), 300)

        # classification and regression loss
        gt_labels_target = F.one_hot(gt_labels, num_classes=self.num_classes + 1)[
            :, :, :-1
        ]  # no loss for the last (background) class

        loss_cls = sigmoid_focal_loss_jit(
            pred_logits,
            gt_labels_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )
        loss_box_reg = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            self.num_anchors_per_level,
            box_reg_loss_type="giou",
        )

        ctrness_targets = self.compute_ctrness_targets(anchors, gt_boxes)  # (M, R)
        pred_centerness = torch.cat(pred_centerness, dim=1).squeeze(dim=2)  # (M, R)
        ctrness_loss = F.binary_cross_entropy_with_logits(
            pred_centerness[pos_mask], ctrness_targets[pos_mask], reduction="sum"
        )
        return {
            "loss_c": loss_cls / normalizer,
            "loss_l": loss_box_reg / normalizer,
            "loss_n": ctrness_loss / normalizer,
        }

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Same interface as :meth:`RetinaNet.label_anchors`, but implemented with FCOS
        anchor matching rule.

        Unlike RetinaNet, there are no ignored anchors.
        """

        gt_labels, matched_gt_boxes = [], []

        for inst in gt_instances:
            if len(inst) > 0:
                # TODO: incompatible with current setup
                gt_boxes = inst[:, :-1] * 320 # TODO: box dim not normalzied
                gt_classes = inst[:, -1] - 1 # TODO: class indx from 0 to (nclas-1)
                match_quality_matrix = self._match_anchors(gt_boxes, anchors)

                # Find matched ground-truth box per anchor. Un-matched anchors are
                # assigned -1. This is equivalent to using an anchor matcher as used
                # in R-CNN/RetinaNet: `Matcher(thresholds=[1e-5], labels=[0, 1])`
                match_quality, matched_idxs = match_quality_matrix.max(dim=0)
                matched_idxs[match_quality < 1e-5] = -1

                matched_gt_boxes_i = gt_boxes[matched_idxs.clip(min=0)]
                gt_labels_i = gt_classes[matched_idxs.clip(min=0)]

                # Anchors with matched_idxs = -1 are labeled background.
                gt_labels_i[matched_idxs < 0] = self.num_classes
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors)
                gt_labels_i = torch.full(
                    (len(matched_gt_boxes_i),),
                    fill_value=self.num_classes,
                    dtype=torch.long,
                    device=matched_gt_boxes_i.device,
                )

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    @torch.no_grad()
    def _match_anchors(self, gt_boxes, anchors,
                       center_sampling_radius: float = 1.5,
                       ):
        """
        Match ground-truth boxes to a set of multi-level anchors.

        Args:
            gt_boxes: Ground-truth boxes from instances of an image.
            anchors: List of anchors for each feature map (of different scales).

        Returns:
            torch.Tensor
                A tensor of shape `(M, R)`, given `M` ground-truth boxes and total
                `R` anchor points from all feature levels, indicating the quality
                of match between m-th box and r-th anchor. Higher value indicates
                better match.
        """
        # Naming convention: (M = ground-truth boxes, R = anchor points)
        # Anchor points are represented as square boxes of size = stride.
        def get_centers(boxes):
            return torch.stack(((boxes[:, 2]+boxes[:, 0])/2, # slicing reduces dim
                                (boxes[:, 3]+boxes[:, 1])/2)).T
        def area(boxes):
            return (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])

        # anchors = Boxes.cat(anchors)  # (R, 4)
        # anchor_centers = get_centers(anchors)  # (R, 2)
        anchor_centers = anchors[:, :2].clone() # (R, 2)
        # anchor_sizes = anchors[:, 2] - anchors[:, 0]  # (R, )
        anchor_sizes = anchors[:, 2].clone() # (R, )

        k = 0
        scale = [8, 16, 32, 64] if len(self.num_anchors_per_level)==4 else [8, 16, 32, 64, 128]
        for num, s in zip(self.num_anchors_per_level, scale):
            anchor_sizes[k: k+num] = s
            k = k+num

        lower_bound = anchor_sizes * 4 # / 320
        lower_bound[: self.num_anchors_per_level[0]] = 0
        upper_bound = anchor_sizes * 8 # / 320
        upper_bound[-self.num_anchors_per_level[-1] :] = float("inf")

        gt_centers = get_centers(gt_boxes)

        # FCOS with center sampling: anchor point must be close enough to
        # ground-truth box center.
        center_dists = (anchor_centers[None, :, :] - gt_centers[:, None, :]).abs_()
        # sampling_regions = center_sampling_radius * 1 / anchor_sizes[None, :]
        sampling_regions = center_sampling_radius * anchor_sizes[None, :]

        match_quality_matrix = center_dists.max(dim=2).values < sampling_regions

        pairwise_dist = pairwise_point_box_distance(anchor_centers, gt_boxes)
        pairwise_dist = pairwise_dist.permute(1, 0, 2)  # (M, R, 4)

        # The original FCOS anchor matching rule: anchor point must be inside GT.
        match_quality_matrix &= pairwise_dist.min(dim=2).values > 0

        # Multilevel anchor matching in FCOS: each anchor is only responsible
        # for certain scale range.
        pairwise_dist = pairwise_dist.max(dim=2).values
        match_quality_matrix &= (pairwise_dist > lower_bound[None, :]) & (
            pairwise_dist < upper_bound[None, :]
        )
        # Match the GT box with minimum area, if there are multiple GT matches.
        gt_areas = area(gt_boxes)

        match_quality_matrix = match_quality_matrix.to(torch.float32)
        match_quality_matrix *= 1e8 - gt_areas[:, None]
        return match_quality_matrix  # (M, R)

    def compute_ctrness_targets(self, anchors, gt_boxes):
        reg_targets = [self.box2box_transform.get_deltas(anchors, m) for m in gt_boxes]
        reg_targets = torch.stack(reg_targets, dim=0)  # NxRx4
        if len(reg_targets) == 0:
            return reg_targets.new_zeros(len(reg_targets))
        left_right = reg_targets[:, :, [0, 2]]
        top_bottom = reg_targets[:, :, [1, 3]]
        ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
            top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]
        )
        return torch.sqrt(ctrness)

class Box2BoxTransformLinear:
    """ref
    /share/home/leh/anaconda3/envs/bmvc22/lib/python3.9/site-packages/detectron2/modeling/box_regression.py
    The linear box-to-box transform defined in FCOS. The transformation is parameterized
    by the distance from the center of (square) src box to 4 edges of the target box.
    """

    def __init__(self, normalize_by_size=True):
        """
        Args:
            normalize_by_size: normalize deltas by the size of src (anchor) boxes.
        """
        self.normalize_by_size = normalize_by_size

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx1, dy1, dx2, dy2) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true.
        The center of src must be inside target boxes.

        Args:
            src_boxes (Tensor): square source boxes, e.g., anchors
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        # src_ctr_x = 0.5 * (src_boxes[:, 0] + src_boxes[:, 2])
        # src_ctr_y = 0.5 * (src_boxes[:, 1] + src_boxes[:, 3])
        src_ctr_x = src_boxes[:, 0]
        src_ctr_y = src_boxes[:, 1]

        target_l = src_ctr_x - target_boxes[:, 0]
        target_t = src_ctr_y - target_boxes[:, 1]
        target_r = target_boxes[:, 2] - src_ctr_x
        target_b = target_boxes[:, 3] - src_ctr_y

        deltas = torch.stack((target_l, target_t, target_r, target_b), dim=1)
        if self.normalize_by_size:
            stride_w = src_boxes[:, 2] - src_boxes[:, 0]
            stride_h = src_boxes[:, 3] - src_boxes[:, 1]
            strides = torch.stack([stride_w, stride_h, stride_w, stride_h], axis=1)
            deltas = deltas / strides

        return deltas

    def apply_deltas(self, deltas, boxes, num_anchors_per_level):
        """
        Apply transformation `deltas` (dx1, dy1, dx2, dy2) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        # Ensure the output is a valid box. See Sec 2.1 of https://arxiv.org/abs/2006.09214
        deltas = F.relu(deltas)
        boxes = boxes.to(deltas.dtype)

        # ctr_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
        # ctr_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
        ctr_x = boxes[:, 0]
        ctr_y = boxes[:, 1]

        anchor_sizes = boxes[:, 2].clone() # (R, )

        k = 0
        scale = [8, 16, 32, 64] if len(num_anchors_per_level)==4 else [8, 16, 32, 64, 128]
        for num, s in zip(num_anchors_per_level, scale):
            anchor_sizes[k: k+num] = s
            k = k+num

        deltas = deltas * anchor_sizes.unsqueeze(-1)

        if False: # self.normalize_by_size:
            stride_w = boxes[:, 2] - boxes[:, 0]
            stride_h = boxes[:, 3] - boxes[:, 1]
            strides = torch.stack([stride_w, stride_h, stride_w, stride_h], axis=1)
            deltas = deltas * strides

        l = deltas[:, 0::4]
        t = deltas[:, 1::4]
        r = deltas[:, 2::4]
        b = deltas[:, 3::4]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = ctr_x[:, None] - l  # x1
        pred_boxes[:, 1::4] = ctr_y[:, None] - t  # y1
        pred_boxes[:, 2::4] = ctr_x[:, None] + r  # x2
        pred_boxes[:, 3::4] = ctr_y[:, None] + b  # y2
        return pred_boxes


def _dense_box_regression_loss(
    anchors,
    box2box_transform,
    pred_anchor_deltas,
    gt_boxes,
    fg_mask,
    num_anchors_per_level,
    box_reg_loss_type="smooth_l1",
    smooth_l1_beta=0.0,
):
    """
    ref: /share/home/leh/anaconda3/envs/bmvc22/lib/python3.9/site-packages/detectron2/modeling/box_regression.py
    Compute loss for dense multi-level box regression.
    Loss is accumulated over ``fg_mask``.

    Args:
        anchors: #lvl anchor boxes, each is (HixWixA, 4)
        pred_anchor_deltas: #lvl predictions, each is (N, HixWixA, 4)
        gt_boxes: N ground truth boxes, each has shape (R, 4) (R = sum(Hi * Wi * A))
        fg_mask: the foreground boolean mask of shape (N, R) to compute loss on
        box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou",
            "diou", "ciou".
        smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
            use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
    """
    if box_reg_loss_type == "smooth_l1":
        gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[fg_mask],
            gt_anchor_deltas[fg_mask],
            beta=smooth_l1_beta,
            reduction="sum",
        )
    elif box_reg_loss_type == "giou":
        # breakpoint()
        pred_boxes = [
            box2box_transform.apply_deltas(k, anchors, num_anchors_per_level) for k in pred_anchor_deltas
        ]
        loss_box_reg = giou_loss(
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction="sum"
        )
    else:
        raise ValueError(f"Invalid dense box regression loss type '{box_reg_loss_type}'")
    return loss_box_reg

def pairwise_point_box_distance(points: torch.Tensor, boxes):
    """
    ref: /share/home/leh/anaconda3/envs/bmvc22/lib/python3.9/site-packages/detectron2/structures/boxes.py
    Pairwise distance between N points and M boxes. The distance between a
    point and a box is represented by the distance from the point to 4 edges
    of the box. Distances are all positive when the point is inside the box.

    Args:
        points: Nx2 coordinates. Each row is (x, y)
        boxes: M boxes

    Returns:
        Tensor: distances of size (N, M, 4). The 4 values are distances from
            the point to the left, top, right, bottom of the box.
    """
    x, y = points.unsqueeze(dim=2).unbind(dim=1)  # (N, 1)
    x0, y0, x1, y1 = boxes.unsqueeze(dim=0).unbind(dim=2)  # (1, M)
    return torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)


