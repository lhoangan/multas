#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

def onehot(labels: torch.Tensor, label_num):
    return torch.zeros((labels.shape[0], label_num+1, labels.shape[2], labels.shape[3]),
            device=labels.device).scatter_(1, labels.long(), 1)[:,:label_num,...]

class SegLoss(nn.Module):

    def __init__(
        self,
        ignore_index: int=255
    ) -> None:
        super(SegLoss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        predictions: dict,
        seggt: torch.Tensor = None,
    ) -> torch.Tensor:

        seg_data = predictions['seg'] if 'seg' in predictions else None
        result = {}

        if seg_data is not None and seggt is not None:
            result['loss_seg'] = self.ce_loss(
                seg_data,
                seggt.long().squeeze(1))
        else:
            return None

        return result
