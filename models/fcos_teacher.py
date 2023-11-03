import math
import torch
import torch.nn as nn
from .teacher_detector import Detector_base
from .teacher_detector import multibox, projection

class FCOS_Detector(Detector_base):

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
        super(FCOS_Detector, self).__init__(base_size, num_classes, backbone,
                                            neck, task, noBN)

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

        self.center_head = nn.Conv2d(self.fea_channel, 1, 1)
        torch.nn.init.normal_(self.center_head.weight, std=0.01)
        torch.nn.init.constant_(self.center_head.bias, 0)

    def init_det(self):
        self.num_anchors = 1
        super(FCOS_Detector, self).init_det()

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
        cnter = list()

        feat_loc = list()
        feat_conf = list()
        feat_comb = list()

        if 'seg' in self.task and ('seg' in task or task == ""):
            # results['seg'], results['seg_feats'] = self.seghead(fp, base_size)#, extra=[x2, x1])
            results['seg'] = self.seghead(fp, base_size)#, extra=[x2, x1])

        if 'det' in self.task and ('det' in task or task == ""):
            for (x, l, c, _, cp, mp) in zip(fp, self.loc,       self.conf,
                                                self.loc_projs, self.conf_projs,
                                                self.comb_projs):
                xloc = l[:-1](x)
                xconf = c[:-1](x)

                # loc.append(se(reshape(l[-1](xloc), 4)))
                loc.append(reshape(l[-1](xloc), 4))
                conf.append(reshape(c[-1](xconf), self.num_classes['det']))

                # if self.fcos == "loc":
                cnter.append(reshape(self.center_head(xloc), 1))
                # elif self.fcos == "conf":
                # cnter.append(reshape(self.center_head(xconf), 1))

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

            # loc = torch.cat(loc, 1)
            # conf = torch.cat(conf, 1)

            results['loc'] = loc
            results['conf'] = conf
            results['cnter'] = cnter

            results['contr_loc'] = feat_loc#torch.cat(feat_loc, 1) if len(feat_loc) > 0 else []
            results['contr_conf'] = feat_conf#torch.cat(feat_conf, 1) if len(feat_conf) > 0 else []
            results['contr_comb'] = feat_comb#torch.cat(feat_comb, 1) if len(feat_comb) > 0 else []

        return results
