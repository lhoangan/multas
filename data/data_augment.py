#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random
import math
import torch

EP=9 # threshold for each dimension

def good_mask(mask, ignored_class):

    mask = mask.squeeze()
    xs, ys = np.nonzero((mask != 0) & (mask != ignored_class))

    # not a thorough check, there could be 2 small blobs (< EP) lying far away
    if len(xs) > 0 and len(ys) > 0:
        return (xs.max() - xs.min() > EP and ys.max() - ys.min() > EP)
    return False

def _crop_expand_(
    data: dict, # of np.ndarray
    min_scale: float = 0.25,
    max_scale: float = 1.75,
    min_ratio: float = 0.5,
    max_ratio: float = 1.0,
    min_shift: float = 0.4,
    max_shift: float = 0.6,
    min_iou: float = 0.75,
    max_iou: float = 0.25,
    max_try: int = 10,
    img_mean: float = 114.0,
    ignored_class:int=0,
    p: float = 0.75,
) -> dict: # of np.ndarray

    def matrix_iou(a, b):
        lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
        area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        return area_i / area_a[:, np.newaxis]

    # output = {k: data[k].copy() for k in data}

    if random.random() > p:
        return # output

    (height, width, depth) = data['image'].shape
    assert height == width

    depths = {'image': depth, 'mask': 1}
    default_vals = {'image': img_mean, 'mask': ignored_class}

    for _ in range(max_try):
        new_h = random.uniform(min_scale, max_scale)
        if random.randrange(2):
            new_w = new_h * random.uniform(min_ratio, max_ratio)
        else:
            new_w = new_h / random.uniform(min_ratio, max_ratio)

        for _ in range(max_try):
            center_y = random.uniform(min_shift, max_shift)
            center_x = random.uniform(min_shift, max_shift)
            corner_y = center_y - new_h/2
            corner_x = center_x - new_w/2

            cropped_y1 = max(0, corner_y)
            cropped_x1 = max(0, corner_x)
            cropped_y2 = min(1.0, corner_y+new_h)
            cropped_x2 = min(1.0, corner_x+new_w)
            expand_y1 = max(0, -corner_y)
            expand_x1 = max(0, -corner_x)

            real_cropped_y1 = int(cropped_y1 * height)
            real_cropped_x1 = int(cropped_x1 * width)
            real_cropped_y2 = int(cropped_y2 * height)
            real_cropped_x2 = int(cropped_x2 * width)
            real_expand_y1 = int(expand_y1 * height)
            real_expand_x1 = int(expand_x1 * width)
            real_expand_y2 = real_expand_y1 + real_cropped_y2 - real_cropped_y1
            real_expand_x2 = real_expand_x1 + real_cropped_x2 - real_cropped_x1

            cropped = dict()
            for k in data:
                if k != 'image' and k != 'mask':
                    continue
                depth = depths[k]
                image = data[k]

                cropped_image = image[
                    real_cropped_y1 : real_cropped_y2, real_cropped_x1 : real_cropped_x2
                ]
                expand_image = np.ones(
                    (math.ceil(height * new_h), math.ceil(width * new_w), depth)
                ) * default_vals[k]
                expand_image[
                    real_expand_y1:real_expand_y2, real_expand_x1:real_expand_x2
                ] = cropped_image

                cropped[k] = expand_image

            # no objects in gt segmentation
            if 'mask' in cropped and not good_mask(cropped['mask'], ignored_class):
                # print ("Retry", mi, mj)
                continue

            if 'boxes' not in data:
                for k in cropped:
                    data[k] = cropped[k]
                return # output

            # if 'boxes' in data: # has to account for bbox size after cropped
            boxes = data['boxes']
            labels = data['labels']

            roi = np.array((cropped_x1, cropped_y1, cropped_x2, cropped_y2))
            iou = matrix_iou(boxes, roi[np.newaxis])
            iou = iou[iou < min_iou]
            iou = iou[iou > max_iou]
            if len(iou) > 0: # this changes if 'boxes' not in data
                continue

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if len(boxes_t) == 0: # this changes if 'boxes' not in data
                continue

            # if this is good cropping

            boxes_t[:, 0] = np.maximum(0, boxes_t[:, 0]-corner_x) / new_w
            boxes_t[:, 1] = np.maximum(0, boxes_t[:, 1]-corner_y) / new_h
            boxes_t[:, 2] = np.minimum(new_w, boxes_t[:, 2]-corner_x) / new_w
            boxes_t[:, 3] = np.minimum(new_h, boxes_t[:, 3]-corner_y) / new_h

            cropped['boxes'] = boxes_t
            cropped['labels'] = labels_t

            for k in cropped:
                data[k] = cropped[k]
            return # output

    return # output

def _distort(
    image: np.ndarray,
) -> np.ndarray:

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _mirror_(
    data: dict, # of np.array
    vertical: bool=False,
) -> None: # editing inline

    if random.randrange(2):
        data['image'] = data['image'][:, ::-1]
        if 'boxes' in data:
            data['boxes'][:, 0::2] = 1.0 - data['boxes'][:, 2::-2]
        if 'mask' in data:
            data['mask'] = data['mask'][:, ::-1]
    if vertical and random.randrange(2): # TODO: to be TESTED
        data['image'] = data['image'][::-1, :]
        if 'boxes' in data:
            data['boxes'][:, 1::2] = 1.0 - data['boxes'][:, 3::-2]
        if 'mask' in data:
            data['mask'] = data['mask'][::-1, :]

    return

def preproc_for_test(
    image: np.ndarray,
    insize: int,
    mean: list = (0.485, 0.456, 0.406),
    std: list = (0.229, 0.224, 0.225),
    swap: list = (2, 0, 1),
) -> tuple:

    image = cv2.resize(image, (insize, insize), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = image[:, :, ::-1]
    image /= 255.0
    image -= mean
    image /= std
    image = image.transpose(swap)
    image = np.ascontiguousarray(image, dtype=np.float32)
    return image

def preproc_for_test_(
    data: dict, # of np.array
    insize: int,
    mean: list = (0.485, 0.456, 0.406),
    std: list = (0.229, 0.224, 0.225),
    swap: list = (2, 0, 1),
) -> tuple:

    resample = {'image': cv2.INTER_LINEAR, 'mask': cv2.INTER_NEAREST}
    types    = {'image': np.float32, 'mask': np.int64}
    for k in resample:
        if k not in data or insize == 0:
            continue
        data[k] = cv2.resize(data[k], (insize, insize), interpolation=resample[k])

    data['image'] = data['image'].astype(np.float32)
    data['image'] = data['image'][:, :, ::-1] # BGR to RGB
    data['image'] /= 255.0
    data['image'] -= mean
    data['image'] /= std
    if 'mask' in data:
        data['mask'] = data['mask'][..., None] \
                if len(data['mask'].shape) == 2 else data['mask']

    for k in types:
        if k not in data:
            continue
        data[k] = data[k].transpose(swap)
        data[k] = np.ascontiguousarray(data[k], dtype=types[k])
        data[k] = torch.from_numpy(data[k][None, ...]) # adding batch_size dim

    if 'boxes' in data and 'labels' in data:
        data['labels'] = np.expand_dims(data['labels'], 1)
        data['bboxes'] = np.hstack((data['boxes'], data['labels']))
    if 'bboxes' in data:
        data['bboxes'] = [torch.from_numpy(data['bboxes']).float()]

    return data

def preproc_for_train_(
    data: dict, # of np.array
    insize: int,
    task: str = 'det',
    no_augment: bool=False, # when we don't want to do augmentation
    ignored_class: int = 0,
) -> tuple:
    # data['image'], data['mask'], and data['bboxes']

    assert 'image' in data and ('bboxes' in data or 'mask' in data), "Lacking data"

    # copy data over to output, so leaving input untouched
    output = {k: data[k].copy() for k in data}

    if 'bboxes' in output:
        output['boxes'] = output['bboxes'][:, :-1]# .copy()
        output['labels'] = output['bboxes'][:, -1]# .copy()
    if 'mask' in output:
        output['mask'] = output['mask'][..., None] \
                if len(output['mask'].shape) == 2 else output['mask']

    output['image'] = _distort(output['image'])
    _crop_expand_(output, ignored_class=ignored_class)   # inline changing
    _mirror_(output)            # inline changing
    return output

def detection_collate_org(
    batch: tuple,
) -> tuple:

    """ Custom collate fn for images and boxes """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                annos.requires_grad = False
                targets.append(annos)
    return (torch.stack(imgs, 0), targets)

def detection_collate_dict(
    batch: dict,
) -> tuple:

    """ Custom collate fn for images and boxes """

    targets = []
    imgs = []
    segs = []

    for _, sample in enumerate(batch):
        # processing image
        # img = torch.from_numpy(np.ascontiguousarray(
        #     sample['image'].transpose((2, 0, 1)), np.float32))
        img = sample['image']
        while len(img.shape) < 4:
            img = img.unsqueeze(0)
        imgs.append(img)

        # processing bounding boxes
        if 'bboxes' in sample:
            targets.extend(sample['bboxes'])
            targets[-1].requires_grad = False

        # processing semantic segmentation
        if 'mask' in sample:
            # seg = torch.from_numpy(np.ascontiguousarray(sample['mask']).astype(np.float))
            seg = sample['mask']
            while len(seg.shape) < 4:
                seg = seg.unsqueeze(0)
            seg.requires_grad = False
            segs.append(seg)

    if len(segs) > 0:
        return (torch.concat(imgs, 0), targets, torch.concat(segs, 0))

    return (torch.concat(imgs, 0), targets)
