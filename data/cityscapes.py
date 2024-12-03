
import os
import torch
import fnmatch
import numpy as np
import random
import cv2
import torch.nn.functional as F
from .voc0712 import VOCDetection
from datetime import datetime as dt
import albumentations as A

def build_from_module(cfg, module, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        module (:obj:`module`): The module to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = getattr(module, obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} module'.format(
                obj_type, module))
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)

class RandomScaleCrop(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """
    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        img_ = F.interpolate(img[None, :, i:i + h, j:j + w],
                size=(height, width),
                mode='bilinear', align_corners=True).squeeze(0)
        label_ = F.interpolate(label[None, None, i:i + h, j:j + w],
                size=(height, width),
                mode='nearest').squeeze(0).squeeze(0)
        depth_ = F.interpolate(depth[None, :, i:i + h, j:j + w],
                size=(height, width), mode='nearest').squeeze(0)
        _sc = sc
        _h, _w, _i, _j = h, w, i, j

        return img_, label_, depth_ / sc, torch.tensor(
                [_sc, _h, _w, _i, _j, height, width])

# official label list
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
# CITYSCAPES = ('flat', 'surfaces', 'humans', 'vehicles', 'constructions',
#             'objects', 'nature', 'sky', 'void'
#       )
CITYSCAPES = ('flat', 'constructions', 'objects', 'nature', # vegetation, terrain
                'sky', 'humans', # person, rider
                'vehicles', # car, truck, buss, caravan, trailer, train, motorcycle, bike
                'void'
        )

class Cityscapes(VOCDetection):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, is_training=True, ignored_label=-1):
        self.is_training = is_training
        self.root = os.path.expanduser('datasets/cityscapes/')
        self.now = dt.now().strftime("%Y%m%d_%H%M%S.%f")
        self.ignore_label = ignored_label
        self.class_names = CITYSCAPES[:-1] # to show per-class evaluation
        self.num_classes = 8

        # R\read the data file
        if self.is_training:
            self.data_path = self.root + '/train'
        else:
            self.data_path = self.root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'),
            '*.npy'))

        self.task = 'seg'
        self.size = (128, 256)
        self.imgs = None # for cached images of VOCDetection

        self.augment = self.make_augment()
        self.preprocess = self.make_preprocess()

    def pull_img(self, index: int):
        return (np.load(self.data_path + '/image/{:d}.npy'.format(index))*255).astype(np.uint8)

    def pull_seg(self, index: int):
        return np.load(self.data_path + '/label_7/{:d}.npy'.format(index))

    def __getitem__(self, index):

        output = {'image': self.pull_img(index),
                  'mask': self.pull_seg(index)
                }

        if self.is_training:
            output = self.augment(**output)
        output = self.preprocess(**output)

        output['image'] = torch.from_numpy(
                output['image'].transpose((2, 0, 1)))
        output['mask'] = torch.from_numpy(
                output['mask'][..., None].transpose((2, 0, 1)))

        return output


    def __len__(self):
        return self.data_len

    def _cache_images(self,) -> None:
        pass

    def make_preprocess(self):
        """
        Preprocess data, for both training and inference. Update in-place.
        """
        img_norm_cfg = dict(
            max_pixel_value=255.0,
            std=(0.229, 0.224, 0.225),
            mean=(0.485, 0.456, 0.406),
        )
        transforms = [
            dict(type='Resize', height=self.size[0], width=self.size[1], p=1.0),
            dict(type='Normalize', **img_norm_cfg),
        ]
        return self.compose_aug(transforms)

    def make_augment(self):
        """
        Augmentation for training
        """

        image_pad_value = (123.675, 116.280, 103.530)

        transforms = [
            # dict(type='RandomScale', scale_limit=(.5, 2),
            #      interpolation=cv2.INTER_LINEAR),
            dict(type='PadIfNeeded', min_height=self.size[0],
                min_width=self.size[1], value=image_pad_value,
                mask_value=self.ignore_label, border_mode=cv2.BORDER_CONSTANT),
            dict(type='Rotate', limit=10, interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT, value=image_pad_value,
                mask_value=self.ignore_label, p=0.5),
            dict(type='GaussianBlur', blur_limit=7, p=0.5),
            dict(type='HorizontalFlip', p=0.5),
            dict(type='RandomBrightnessContrast', p=0.3),
            dict(type='RGBShift', r_shift_limit=30, g_shift_limit=30,
                b_shift_limit=30, p=0.3),
            dict(type='RandomBrightnessContrast', p=.5),
            dict(type='RandomGamma', p=.5),
            dict(type='CLAHE', p=.5),
        ]
        return self.compose_aug(transforms)

    def compose_aug(self, transforms: list):

        modules = []
        for t in transforms:
            if t is None:
                continue
            modules.append(build_from_module(t, A))

        return A.Compose(modules, #, additional_targets={'bbox': 'bboxes'})
                        bbox_params=A.BboxParams(
                            format='albumentations' #, min_area=1024,
                            # min_visibility=0.1, label_fields=['class_labels']
                        ) if "det" in self.task else None
                )

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from PIL import Image

    train_sets = Cityscapes(is_training=True)
    valid_sets = Cityscapes(is_training=False)

    index = 0
    output = train_sets.__getitem__(index)
    seg_= Image.fromarray(output['mask'].astype(np.uint8), mode="P")
    seg_= output['mask']
    plt.imshow(output['image']), plt.figure(), plt.imshow(seg_),
    plt.show()

