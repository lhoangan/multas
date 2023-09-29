#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from .teacher_detector import Detector_base


def multibox(
    fpn_level: int,
    num_anchors: int,
    num_classes: int,
    fea_channel: int,
    dis_channel: int,
    conv_block: nn.Module,
) -> tuple:
    loc_layers, conf_layers, dist_layers = list(), list(), list()
    loc_channel = num_anchors * 4
    cls_channel = num_anchors * num_classes
    for _ in range(fpn_level):
        loc_layers.append(
            nn.Sequential(
                conv_block(fea_channel, fea_channel, 3, padding=1),
                conv_block(fea_channel, fea_channel, 3, padding=1),
                nn.Conv2d(fea_channel, loc_channel, 1),
            )
        )
        conf_layers.append(
            nn.Sequential(
                conv_block(fea_channel, fea_channel, 3, padding=1),
                conv_block(fea_channel, fea_channel, 3, padding=1),
                nn.Conv2d(fea_channel, cls_channel, 1),
            )
        )
        dist_layers.append(
            nn.Conv2d(fea_channel, dis_channel, 1)
        )
    return (
        nn.ModuleList(loc_layers),
        nn.ModuleList(conf_layers),
        nn.ModuleList(dist_layers),
    )


class Student_Detector(Detector_base):
    """ Student Detector Model """

    def __init__(
        self,
        base_size: int,
        num_classes: int,
        backbone: str,
        neck: str,
        task: str="det",
        noBN=False,
    ) -> None:

        # Backbone network
        if backbone == 'resnet18':
            self.dis_channel = 256
        else:
            raise ValueError('Error: Sorry backbone {} is not supported!'.format(backbone))

        super(Student_Detector, self).__init__(base_size, num_classes, backbone,
                neck, task, noBN=noBN)

    def init_det(self):

        (self.loc, self.conf, self.dist) = multibox(
            self.fpn_level, self.num_anchors, self.num_classes['det'],
            self.fea_channel, self.dis_channel, self.conv_block,
        )
        bias_value = 0
        for modules in self.loc:
            torch.nn.init.normal_(modules[-1].weight, std=0.01)
            torch.nn.init.constant_(modules[-1].bias, bias_value)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for modules in self.conf:
            torch.nn.init.normal_(modules[-1].weight, std=0.01)
            torch.nn.init.constant_(modules[-1].bias, bias_value)

    def deploy(
        self,
    ) -> None:
        for module in self.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        self.eval()

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict:
        base_size = x.size(2)
        x = self.backbone(x)
        fp = self.neck(x)
        results = dict()
        if 'seg' in self.task:
            results['seg'] = self.seghead(fp, base_size)#, extra=[x2, x1])
            fea = [x.permute(0, 2, 3, 1).contiguous() for x in fp]
            fea = torch.cat([o.view(o.size(0), -1) for o in fea], 1)
            results['feature_s'] = fea.view(fea.size(0), -1, self.fea_channel)
        fea = list()
        loc = list()
        conf = list()
        if 'det' in self.task:
            for (x, l, c, d) in zip(fp, self.loc, self.conf, self.dist):
                fea.append(d(x).permute(0, 2, 3, 1).contiguous())
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            fea = torch.cat([o.view(o.size(0), -1) for o in fea], 1)
            # fea2 = torch.cat([o.view(o.size(0), -1) for o in fea2], 1)
            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
            results['loc'] = loc.view(loc.size(0), -1, 4)
            results['conf'] = conf.view(conf.size(0), -1, self.num_classes['det'])
            results['feature_d'] = fea.view(conf.size(0), -1, self.dis_channel)
        return results


    def forward_test(
        self,
        x: torch.Tensor,
        task="",
    ) -> dict:
        base_size = x.size(2)
        x = self.backbone(x)
        fp = self.neck(x)
        loc = list()
        conf = list()
        results = dict()
        if 'seg' in self.task:
            results['seg'] = self.seghead(fp, base_size)#, extra=[x2, x1])
            # results['seg'], _ = self.seghead(fp, base_size, extra=[x2, x1])
        if 'det' in self.task:
            for (x, l, c) in zip(fp, self.loc, self.conf):
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
            results['loc'] = loc.view(loc.size(0), -1, 4)
            results['conf'] = conf.view(conf.size(0), -1, self.num_classes['det'])
        return results
