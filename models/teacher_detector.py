#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from models.base_blocks import BasicConv
import numpy as np

from .hyp_mlr import HorosphericalLayer, PoincareProjector


class SegHead(nn.Module):
    """
        Implementation of Panoptic FPN's semantic segmentation head. Following:
        https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/modeling/meta_arch/semantic_seg.py#L184
    """
    def __init__(self,
        fpn_level: int,
        num_classes: int,
        fea_channel: int,
        backbone_channels: list,
        conv_block: nn.Module=BasicConv,
        out_chans: int=128,
        classif_block_type: str = "conv1x1", # or horospherical
    ) -> tuple:
        super(SegHead, self).__init__()

        classif_dim = 3

        self.conv = list()
        channels =  [fea_channel]*fpn_level

        fpn_level = len(channels)
        # f = factor, 1st fpn_level, f=0 -> factor = (2^1)x upsampling
        for f in range(fpn_level):
            ops = []
            for k in range(f+1):
                c = 1 if k == 0 else 2
                ops.append(conv_block(channels[f] // c, out_chans, kernel_size=3,
                    stride=1, padding=1).cuda())
                ops.append(nn.Upsample(scale_factor=2))
            self.conv.append(nn.Sequential(*ops))
        # then element-wise sum of all fpn levels
        self.conv = nn.ModuleList(self.conv)

        classif_block_type = "conv1x1" # "horospherical"
        print(f">> classif block type = {classif_block_type}")

        proto_types = "uniform"
        protos_path = f"prototypes/prototypes{proto_types}-{classif_dim}d-{num_classes+1}c.npy"
        classif_block = (
            nn.Conv2d(classif_dim, num_classes+1, kernel_size=1, stride=1, padding=0, bias=False)
            if classif_block_type == "conv1x1"
            else
            nn.Sequential(
                PoincareProjector(),
                HorosphericalLayer(
                    lambda_=0.1 * out_chans,
                    protos=torch.from_numpy(np.load(protos_path))),
            )
        )

        # following vedaseg implementation of VOC_FPN segmentation head, at
        # https://github.com/Media-Smart/vedaseg/blob/fa4ff42234176b05ef0dff8759c7e62a17498ab9/configs/voc_fpn.py#L128
        self.seg = nn.Sequential(
            *([nn.Upsample(scale_factor=4)] +
            [conv_block(out_chans, classif_dim if i == 2 else out_chans,
                        kernel_size=3, padding=1).cuda() for i in range(3)] +
            [classif_block]
        ))
        if isinstance(self.seg[-1], nn.Conv2d):
            torch.nn.init.normal_(self.seg[-1].weight, std=0.01)

    def forward(self, fp, base_size):#, extra=None):

        feats = []
        fp =  fp
        for (x, conv_up) in zip(fp, self.conv):
            feats.append(conv_up(x))
        feats = torch.stack(feats, dim=0).sum(dim=0)

        feats = self.seg[:-1](feats)
        return self.seg[-1](feats)#, feats
        return self.seg(feats)


def multibox(
    fpn_level: int,
    num_anchors: int,
    num_classes: int,
    fea_channel: int,
    conv_block: nn.Module,
) -> tuple:
    loc_layers, conf_layers = list(), list()
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
    return (
        nn.ModuleList(loc_layers),
        nn.ModuleList(conf_layers),
    )

class Detector_base(nn.Module):
    """ Teacher Detector Base Model """

    def __init__(
        self,
        base_size: int,
        num_classes: int,
        backbone: str,
        neck: str,
        task: str="det",
        noBN = False,
    ) -> None:
        super(Detector_base, self).__init__()

        assert "det" in task or "seg" in task, \
                f"Only `det` and `seg` task supported, but found: {task}"

        # Params
        # self.num_classes = num_classes - 1 # remove background
        if isinstance(num_classes, int):
            num_classes = {k: num_classes for k in task.split("+")}
        self.num_classes = {k: num_classes[k] - 1 for k in num_classes} # remove background
        self.num_anchors = 6
        self.fpn_level = 4 if base_size < 512 else 5

        # Backbone network
        if backbone == 'resnet18':
            from models.backbone.resnet_backbone import ResNetBackbone
            self.backbone = ResNetBackbone(depth=18, noBN=noBN)
            self.backbone_channels = (256, 512)
            self.fea_channel = 256
            self.conv_block = BasicConv
        elif backbone == 'resnet34':
            from models.backbone.resnet_backbone import ResNetBackbone
            self.backbone = ResNetBackbone(depth=34, noBN=noBN)
            self.backbone_channels = (256, 512)
            self.fea_channel = 256
            self.conv_block = BasicConv
        elif backbone == 'resnet50':
            from models.backbone.resnet_backbone import ResNetBackbone
            self.backbone = ResNetBackbone(depth=50, noBN=noBN)
            self.backbone_channels = (1024, 2048) # resnet50's thing
            self.fea_channel = 256
            self.conv_block = BasicConv
        elif backbone == 'resnet101':
            from models.backbone.resnet_backbone import ResNetBackbone
            self.backbone = ResNetBackbone(depth=101, noBN=noBN)
            self.backbone_channels = (1024, 2048) # resnet101's thing
            self.fea_channel = 256
            self.conv_block = BasicConv
        else:
            raise ValueError('Error: Sorry backbone {} is not supported!'.format(backbone))

        # Neck network
        if neck == 'ssd':
            from models.neck.ssd_neck import SSDNeck
            self.neck = SSDNeck(self.fpn_level, self.backbone_channels,
                                self.fea_channel, self.conv_block, noBN=noBN)
        elif neck == 'fpn':
            from models.neck.fpn_neck import FPNNeck
            self.neck = FPNNeck(self.fpn_level, self.backbone_channels,
                                self.fea_channel, self.conv_block, noBN=noBN)
        elif neck == 'pafpn':
            from models.neck.pafpn_neck import PAFPNNeck
            self.neck = PAFPNNeck(self.fpn_level, self.backbone_channels,
                                  self.fea_channel, self.conv_block, noBN=noBN)
        else:
            raise ValueError('Error: Sorry neck {} is not supported!'.format(neck))

        self.task = task
        # Detection Head
        if 'det' in task:
            self.init_det()
        if 'seg' in task:
            self.seghead = SegHead(self.fpn_level, self.num_classes['seg'],
                self.fea_channel, self.backbone_channels, self.conv_block)

    def init_det(self):
        (self.loc, self.conf) = multibox(self.fpn_level, self.num_anchors,
                                         self.num_classes['det'],
                                         self.fea_channel, self.conv_block,
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
        x = self.backbone(x)
        fp = self.neck(x) if self.neck is not None else x
        fea = list()
        loc = list()
        conf = list()
        if 'det' in self.task:
            for (x, l, c) in zip(fp, self.loc, self.conf):
                fea.append(x.permute(0, 2, 3, 1).contiguous())
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            fea = torch.cat([o.view(o.size(0), -1) for o in fea], 1)
            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
            return {
                'loc': loc.view(loc.size(0), -1, 4),
                'conf': conf.view(conf.size(0), -1, self.num_classes['det']),
                'feature': fea.view(conf.size(0), -1, self.fea_channel),
            }

    def forward_test(
        self,
        x: torch.Tensor,
    ) -> dict:
        x = self.backbone(x)
        fp = self.neck(x)
        loc = list()
        conf = list()
        for (x, l, c) in zip(fp, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        return {
            'loc': loc.view(loc.size(0), -1, 4),
            'conf': conf.view(conf.size(0), -1, self.num_classes['det']),
        }

def projection(
    fpn_level: int,
    ichannels: int,
    ochannels: int,
    conv_block: nn.Module,
) -> tuple:
    """
        projection head is said to help with learning better representation
        see SimCLR paper
    """

    proj_layers = list()

    for _ in range(fpn_level):
        proj_layers.append(
            nn.Sequential(
                conv_block(ichannels, ochannels, 1),
                nn.ReLU(True),
                conv_block(ochannels, ochannels, 1),
            )
        )
    for proj_layer in proj_layers:
        torch.nn.init.normal_(proj_layer[0].weight, std=0.01)
        torch.nn.init.normal_(proj_layer[2].weight, std=0.01)
        if proj_layer[0].bias is not None:
            torch.nn.init.constant_(proj_layer[0].bias, 0)
        if proj_layer[2].bias is not None:
            torch.nn.init.constant_(proj_layer[2].bias, 0)

    return nn.ModuleList(proj_layers)

class Detector(Detector_base):

    def __init__(
        self,
        base_size: int,
        num_classes: int,
        backbone: str,
        neck: str,
        contr_comb: bool = False,
        contr_loc: bool = False,
        contr_conf: bool = True,
        proj_head: bool = False,
        contr_neck: bool = False,
        task: str = "det",
        noBN = False,
    ) -> None:
        """
            feat_type =
                0: loc=False, conf=False
                1: loc=False, conf=True
                2: loc=True,  conf=False
                3: loc=True,  conf=True, 2 individual
                4: loc=True,  conf=True, combining 1 losses

            contr_comb=True assumes proj_head=True, contr_loc=True, contr_conf=True
            contr_comb=False:
                - contr_loc  = T/F x proj_head = T/F
                - contr_conf = T/F x proj_head = T/F
        """
        super(Detector, self).__init__(base_size, num_classes, backbone, neck,
                                       task, noBN)

        self.contr_loc = contr_loc
        self.contr_conf = contr_conf
        self.contr_comb = contr_comb
        self.proj_head = proj_head
        self.contr_neck = contr_neck

        if self.contr_neck:
            self.contr_conf = False
            self.loc_conf = False

        loc_chans = self.fea_channel
        conf_chans = self.fea_channel
        comb_chans = self.fea_channel

        self.loc_projs = projection(self.fpn_level, loc_chans, loc_chans,  nn.Conv2d)
        self.conf_projs = projection(self.fpn_level, conf_chans, conf_chans, nn.Conv2d)
        self.comb_projs = projection(self.fpn_level, loc_chans + conf_chans,
                comb_chans, nn.Conv2d)
        self.neck_projs = projection(self.fpn_level, self.fea_channel,
                self.fea_channel, nn.Conv2d)

    def forward_test(self, x, task=""):

        results = dict()
        base_size = x.size(2)

        # backbone
        x = self.backbone(x)
        # list [(8,256,40,40),(...20,20),(...10,10),(...5,5)]
        fp = self.neck(x)  if self.neck is not None else x
        # outsize = [f.size(-1) for f in fp] # assuming width = height

        reshape = lambda f, nc: f.permute(0, 2, 3, 1).reshape(f.size(0), -1, nc)
        # when activating this, the random seed will be changed
        results['contr_neck'] = []
        if self.contr_neck: # fencing off from currently running code
            neck = [np(i) for i, np in zip(fp,self.neck_projs)] if self.proj_head else fp
            neck = [reshape(i, i.size(1)) for i in neck]
            results['contr_neck'] = torch.cat(neck, 1)

        # detection
        loc = list()
        conf = list()

        feat_loc = list()
        feat_conf = list()
        feat_comb = list()

        if 'seg' in self.task and ('seg' in task or task == ""):
            # results['seg'], results['seg_feats'] = self.seghead(fp, base_size)#, extra=[x2, x1])
            results['seg'] = self.seghead(fp, base_size)#, extra=[x2, x1])

        if 'det' in self.task and ('det' in task or task == ""):
            for (x, l, c, lp, cp, mp) in zip(fp,self.loc,       self.conf,
                                                self.loc_projs, self.conf_projs,
                                                self.comb_projs):
                b, _, w, h = x.size()
                # reshape = lambda f, nc: f.permute(0, 2, 3, 1).reshape(b, -1, nc)

                xloc = l[:-1](x)
                loc.append(reshape(l[-1](xloc), 4))

                xconf = c[:-1](x)
                conf.append(reshape(c[-1](xconf), self.num_classes['det']))
                pconf = cp(xconf)
                pcomb = mp(torch.concat((xloc, xconf), dim=1))

                if self.contr_comb:
                    feat_comb.append(reshape(pcomb, pcomb.size(1)))
                else:
                    if self.contr_conf:
                        if self.proj_head:
                            feat_conf.append(reshape(pconf, pconf.size(1)))
                        else:
                            feat_conf.append(reshape(xconf, xconf.size(1)))

            loc = torch.cat(loc, 1)
            conf = torch.cat(conf, 1)

            results['loc'] = loc
            results['conf'] = conf

            results['contr_loc'] = torch.cat(feat_loc, 1) if len(feat_loc) > 0 else []
            results['contr_conf'] = torch.cat(feat_conf, 1) if len(feat_conf) > 0 else []
            results['contr_comb'] = torch.cat(feat_comb, 1) if len(feat_comb) > 0 else []

        return results
