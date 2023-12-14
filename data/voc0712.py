#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, os.path as osp
import pickle
import os.path
import torch
import torch.utils.data as data
from PIL import Image
import cv2
import numpy as np
from .voc_eval import voc_eval
# from .data_augment import preproc_for_train, preproc_for_test
from .data_augment import preproc_for_train_, preproc_for_test_
import xml.etree.ElementTree as ET
from datetime import datetime as dt
import albumentations as A

import json
from utils import xyxy_to_xywh


VOC_CLASSES = ('__background__', # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White
def visualize_bbox(img, bbox, class_name, w, h, color=None, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, x_max = (bbox[::2] * w).astype(int)
    y_min, y_max = (bbox[1::2] * h).astype(int)
    palette = VOCDetection.label_palette()
    if color is None:
        color = palette[VOC_CLASSES.index(class_name)].astype(np.uint8).tolist()

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
            (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img

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

class AnnotationTransform(object):
    """ Transforms a VOC annotation into a Tensor of bbox  """

    def __init__(self, keep_difficult=True, classes=VOC_CLASSES, cl2ind=None):
        self.cl2ind = dict(zip(classes, range(len(classes)))) if cl2ind is None else cl2ind
        self.keep_difficult = keep_difficult

    def __call__(self, target):

        width = float(target.find('size').find('width').text)
        height = float(target.find('size').find('height').text)

        res = np.empty((0,5))
        for obj in target.iter('object'):

            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue

            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = float(bbox.find(pt).text)
                bndbox.append(cur_pt)
            for i, pt in enumerate([width, height, width, height]):
                bndbox[i] /= pt

            name = obj.find('name').text.lower().strip()
            label_idx = self.cl2ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res,bndbox))  # [xmin, ymin, xmax, ymax, label_ind]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class VOCDetection(data.Dataset):
    """ VOC Detection Dataset Object """

    def __init__(
        self,
        image_sets: list,
        size: int = 320,
        dataset_name: str = 'VOC0712',
        cache: bool = True,
        imgset: str = "Main",
        double_aug: bool = False,
        ignore_label: int = 255,
        is_training: bool = False,
        task: str="det",
        both_task: bool=False,
    ) -> None:
        self.root = os.path.join('datasets/', 'VOCdevkit/')
        self.results_file_prefix = dt.now().strftime("%Y%m%d_%H%M%S.%f")
        self.now = dt.now().strftime("%Y%m%d_%H%M%S.%f")
        self.image_set = image_sets
        self.imgset = imgset
        self.size = size
        self.both_task = both_task
        self.task = task if not both_task else "det+seg"
        self.ignore_label = ignore_label
        self.is_training = is_training
        self.double_aug = double_aug
        self.target_transform = AnnotationTransform()
        self.name = dataset_name+"_"+imgset+"_".join(
                [''.join(k[-1::-1]) for k in image_sets])
        self.num_classes = len(self.pull_classes())
        self._seggtpath = os.path.join('%s', 'SegmentationClass', '%s.png')
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.class_names = VOC_CLASSES[1:]

        self.load_data(image_sets)

        self.imgs = None

    def load_data(self, image_sets):
        for (year, name) in image_sets:
            self._year = year
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', self.imgset, name + '.txt')):
                self.ids.append((rootpath, line.strip()))
        self.anno_rootpath = os.path.join(self.root, 'VOC' + self._year)

    def pull_classes(
        self,
    ) -> tuple:

        return VOC_CLASSES

    def __getitem__(
        self,
        index,
    ) -> list:
        data = dict()
        data['image'] = self.pull_image(index, resize=True)
        if 'det' in self.task and 'seg' not in self.task: # == 'det':
            data['bboxes'] = self.pull_anno(index)
        if 'seg' in self.task: # == 'seg': # TODO: elif or if?
            data['bboxes'] = self.pull_anno(index)
            mask = self.pull_segment(index, resize=True)
            if mask is not None:
                data['mask'] = mask
            else:
                # if it's a single-task dataset, raise Exception
                assert self.task!="seg", "Loading seg failed! File not exists!"

        if self.is_training:
            data1 = preproc_for_train_(data, self.size, ignored_class=self.ignore_label)
            data2 = preproc_for_train_(data, self.size, ignored_class=self.ignore_label)
            data = data1
        data = preproc_for_test_(data, self.size)
        if self.double_aug and self.is_training:
            data2 = preproc_for_test_(data2, self.size)
            if 'bboxes' in data:
                data['bboxes'].extend(data2['bboxes'])
            for k in data:
                if not torch.is_tensor(data[k]):
                    continue
                data[k] = torch.cat((data[k], data2[k]), dim=0)

        return data # torch tensor with (B, C, W, H) shapes for image and seggt

    def __len__(
        self,
    ) -> int:
        return len(self.ids)

    def pull_segment(
        self,
        index: int,
        resize: bool = False,
    ) -> np.ndarray:
        img_id = self.ids[index]
        if not osp.exists(self._seggtpath % img_id):
            image = Image.fromarray(np.zeros((self.size, self.size)) + self.ignore_label)
        else:
            image = Image.open(self._seggtpath % img_id)
        if resize:
            image = image.resize((self.size, self.size), resample=Image.NEAREST)
        image = np.array(image)
        return image

    def pull_anno(
        self,
        index: int,
    ) -> np.ndarray:
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        return self.target_transform(target)

    def pull_image(
        self,
        index: int,
        resize: bool = False,
    ) -> np.ndarray:
        img_id = self.ids[index]
        image = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        if resize:
            image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return image

    @staticmethod
    def label_palette(n=256):
        """
            https://github.com/weiliu89/VOCdevkit/blob/master/VOCcode/VOClabelcolormap.m
        """

        # bitget will return 1 whenever the specified bit is 1, regardless of
        # the state of the other bits. So bitget(n,10) will return 1 not only
        # for 2**10 = 1024, but also for any other value with that bit set; in
        # this case, any value in the range 1024..2047
        # https://stackoverflow.com/questions/44557516/translating-matlabs-bitget-extreme-value-to-python
        bitget = lambda x, b: 1 if x & int(2 ** b) else 0
        palette = np.zeros((n,3), dtype=np.uint8)
        for i in range(n):
            id = i
            r, g, b = 0, 0, 0
            for j in range(7):
                r = r | (bitget(id, 0) << (7-j))#bitor(r, bitshift(bitget(id,1),7 - j));
                g = g | (bitget(id, 1) << (7-j))#bitor(g, bitshift(bitget(id,2),7 - j));
                b = b | (bitget(id, 2) << (7-j))#bitor(b, bitshift(bitget(id,3),7 - j));
                id = id >> 3                    #bitshift(id,-3);
            palette[i] = np.array([r, g, b])
        return palette

    @staticmethod
    def vis_seg(gts, preds):
        # Visualization and output
        palette = VOCDetection.label_palette().flatten().tolist()
        from PIL import Image
        os.makedirs("out_tbd", exist_ok=True)
        for i, (gt, pred) in enumerate(zip(gts, preds)):
            mask = (gt > 0)
            pred = pred * mask
            # .resize(gt.size, resample=Image.NEAREST)
            gt = Image.fromarray(gt, mode='P')
            pr = Image.fromarray(pred.astype(np.uint8), mode='P')
            gt.putpalette(palette)
            pr.putpalette(palette)
            mosaic = Image.new('RGB', (gt.width + pr.width + 10, gt.height))
            mosaic.paste(gt, (0, 0))
            mosaic.paste(pr, (gt.width + 10, 0))
            mosaic.save(f"out_tbd/{i:03d}.png")

    def _cache_images(
        self,
    ) -> None:
        cache_file = self.root + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            print(
                "Caching images for the first time..."
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), self.size, self.size, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.pull_image(x, resize=True),
                range(len(self.ids)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.ids))
            for k, out in pbar:
                self.imgs[k] = out.copy()
            self.imgs.flush()
            pbar.close()

        print("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), self.size, self.size, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def evaluate_detections(
        self,
        all_boxes: list,
        all_thresh=True,
        per_class=False
    ) -> float:
        output_dir = os.path.join(self.root, 'eval')
        os.makedirs(output_dir, exist_ok=True)
        self._write_voc_results_file(all_boxes)
        # self._write_coco_bbox_results_file(all_boxes)
        results = []
        results_ = {}
        if all_thresh:
            range = np.arange(0.5,1,0.05)
        else:
            range = [.5]

        for thresh in range:
            result_ = self._do_python_eval(output_dir, thresh)
            result = np.mean(list(result_.values()))
            results_ = result_ if len(results)==0 else {
                                k:results_[k]+result_[k] for k in result_}
            results.append(result)
            print('----AP@iou{:.2f} = {:.3f}'.format(thresh, result*100))

        if per_class:
            [print(f"{cl: <13}{results_[cl]/len(range):.3f}") for cl in results_]

        print(f'mAP results: \t AP@50={results[0]*100:.3f}')
        if len(results) > 5:
            print(f'mAP results: \t AP@75={results[5]*100:.3f}, '
                    f'AP={sum(results)/len(range)*100:.3f}')
        return sum(results)/len(range)

    def _get_voc_results_file_template(
        self,
    ) -> str:
        filename = self.results_file_prefix + '_{:s}.txt'
        filedir = os.path.join(
            self.root, 'results', 'VOC' + self._year, self.imgset)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_coco_bbox_results_file(self, all_boxes):
        # ref: https://github.com/facebookresearch/Detectron/blob/60f66a1780cc4e8c8d49520050d6522b88c6f82c/detectron/datasets/json_dataset_evaluator.py#L166
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        res_file = self._get_voc_results_file_template().format("all") + ".json"
        for cls_ind, cls in enumerate(self.pull_classes()):
            if cls == '__background__':
                continue
            if cls_ind >= len(all_boxes):
                break
            # cat_id = self.category_to_id_map[cls] # TODO: TO BE TESTED
            cat_id = cls_ind
            results.extend(self._coco_bbox_results_one_category(
                all_boxes[cls_ind], cat_id))
        print('Writing bbox results json to: {}'.format(osp.abspath(res_file)))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

        return res_file

    def _coco_bbox_results_one_category(self, boxes, cat_id):
        results = []
        # image_ids = self.COCO.getImgIds() # TODO: ??
        # image_ids.sort()
        # assert len(boxes) == len(image_ids)
        # for i, image_id in enumerate(image_ids):
        for i, image_id in enumerate(self.ids):
            # like index in _write_voc_results_file, but comparing to
            # https://s3.amazonaws.com/images.cocodataset.org/external/external_PASCAL_VOC.zip
            # the image_id is an int: 000012.jpg -> 12, 2008_00008.jpg -> 200800008
            image_id = int(image_id[1].replace('.jpg', '').replace('_', ''))
            dets = boxes[i]
            if isinstance(dets, list) and len(dets) == 0:
                continue
            dets = dets.astype(float)
            scores = dets[:, -1]
            xywh_dets = xyxy_to_xywh(dets[:, 0:4])
            xs = xywh_dets[:, 0]
            ys = xywh_dets[:, 1]
            ws = xywh_dets[:, 2]
            hs = xywh_dets[:, 3]
            results.extend(
                [{'image_id': image_id,
                  'category_id': cat_id,
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_voc_results_file(
        self,
        all_boxes: list,
    ) -> None:
        for cls_ind, cls in enumerate(self.pull_classes()):
            cls_ind = cls_ind
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets.size == 0:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                dets[k, 0] + 1, dets[k, 1] + 1,
                                dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(
        self,
        output_dir: str = 'output',
        thresh: float = 0.5,
    ) -> float:
        name = self.image_set[0][1]
        annopath = os.path.join(
                                self.anno_rootpath,
                                'Annotations',
                                '{:s}.xml')
        imagesetfile = os.path.join(
                                self.anno_rootpath,
                                'ImageSets',
                                self.imgset,
                                name+'.txt')
        cachedir = os.path.join(self.root, 'annotations_cache', self.now)
        aps = {}
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self.pull_classes()):

            if cls == '__background__':
                continue

            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls,
                    cachedir, ovthresh=thresh, use_07_metric=use_07_metric)
            aps[cls] = ap
            # UNCOMMENT to print per-class evaluation
            # print('AP for {} = {:.4f}'.format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        return aps #np.mean(aps)

    def evaluate_perimg(
        self,
        im,
        all_boxes: list,
        all_thresh=True
    ) -> float:
        results = {}
        if all_thresh:
            for thresh in np.arange(0.5,1,0.05):
                result = self.perimg_eval(im, thresh, all_boxes)
                results[f"{thresh:.2f}"] = result
            return results
        else:
            thresh = 0.5
            result = self.perimg_eval(im, thresh, all_boxes)
            return result

    def perimg_eval(
        self,
        im_ind,
        thresh: float = 0.5,
        all_boxes: list = None,
    ) -> float:
        annopath = os.path.join(
                                self.anno_rootpath,
                                'Annotations',
                                '{:s}.xml')
        cachedir = os.path.join(self.root, 'annotations_cache', self.now)
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

        # processing each image in the list
        im = self.ids[im_ind][1]
        fn = f"temp_eval-{im}-" + dt.now().strftime("%Y%m%d_%H%M%S.%f")
        open(fn, 'w').write(im + '\n')

        aps = []
        for i, cls in enumerate(self.pull_classes()):

            if cls == '__background__':
                continue

            filename = self._get_voc_results_file_template().format(cls)

            # write detection to file customized from
            # _write_voc_results_file
            # to only write the image we are processing
            output = False
            with open(filename, 'wt') as f:
                dets = all_boxes[i][im_ind]
                if len(dets) == 0:
                    continue
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(im, dets[k, -1],
                            dets[k, 0] + 1, dets[k, 1] + 1,
                            dets[k, 2] + 1, dets[k, 3] + 1))
                    output = True

            if not output:
                continue

            rec, prec, ap = voc_eval(filename, annopath, fn, cls,
                    cachedir, ovthresh=thresh, use_07_metric=use_07_metric)
            aps += [ap]

        os.remove(fn)

        return np.mean(aps) * 100

    def evaluate_detections_w_stats(
        self,
        all_boxes: list,
        all_thresh=True
    ) -> float:
        output_dir = os.path.join(self.root, 'eval')
        os.makedirs(output_dir, exist_ok=True)
        self._write_voc_results_file(all_boxes)
        results = []
        if all_thresh:
            statss = []
            for thresh in np.arange(0.5,1,0.05):
                result, stats = self._do_python_eval_w_stats(output_dir, thresh)
                results.append(result)
                print('----AP@iou{:.2f} = {:.3f}'.format(thresh, result*100))
                statss.append(stats)

            print('mAP results: AP@50={:.3f}, AP@75={:.3f}, AP={:.3f}'.format(
                results[0] * 100, results[5]*100, sum(results)/10*100))
            return sum(results)/10, statss
        else:
            thresh = 0.5
            result, stats = self._do_python_eval_w_stats(output_dir, thresh)
            print('----thresh={:.2f}, AP={:.3f}'.format(thresh, result*100))
            return result, [stats]

    def _do_python_eval_w_stats(
        self,
        output_dir: str = 'output',
        thresh: float = 0.5,
    ) -> float:
        name = self.image_set[0][1]
        annopath = os.path.join(
                                self.anno_rootpath,
                                'Annotations',
                                '{:s}.xml')
        imagesetfile = os.path.join(
                                self.anno_rootpath,
                                'ImageSets',
                                self.imgset,
                                name+'.txt')
        cachedir = os.path.join(self.root, 'annotations_cache', self.now)
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        outputs = {'rec':[],'ap':[],'prec':[],'tp':[],'fp1':[],'fp2':[],'ovd':[]}
        for i, cls in enumerate(self.pull_classes()):

            if cls == '__background__':
                continue

            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap, tp, fp1, fp2, ovd = voc_eval(filename, annopath,
                    imagesetfile, cls, cachedir, ovthresh=thresh,
                    use_07_metric=use_07_metric, output_tp_fp=True)
            aps += [ap]
            outputs['rec'].append(rec)
            outputs['prec'].append(prec)
            outputs['ap'].append(ap)
            outputs['tp'].append(tp)
            outputs['fp1'].append(fp1)
            outputs['fp2'].append(fp2)
            outputs['ovd'].append(ovd)
        return np.mean(aps), outputs

    def evaluate_segmentation(self, preds, gts, per_class=False):
        """
        Args:
            preds: list of 2D numpy arrays or 3D logits (class dimension = 0)
            gts: list of 2D numpy arrays
        """

        # reshape back to GT size
        all_segs = []
        all_gts = []
        for i, pred in enumerate(preds):
            gt = self.pull_segment(i, resize=False)
            all_gts.append(gt)
            # breakpoint()
            all_segs.append(np.array(Image.fromarray(pred.astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), resample=Image.NEAREST)))

        self._conf_matrix = np.zeros((self.num_classes+1, self.num_classes+1),
                dtype=np.int64)
        self._b_conf_matrix = np.zeros((self.num_classes+1, self.num_classes+1),
                dtype=np.int64)
        self._predictions = []
        self._compute_boundary_iou = True

        for pred, gt in zip(preds, gts):
            if len(pred.shape) == 3:
                pred = pred.argmax(dim=0)
            gt[gt == self.ignore_label] = self.num_classes

            self._conf_matrix += np.bincount(
                (self.num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))

                self._b_conf_matrix += np.bincount(
                    (self.num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

        return self.evaluate_conf_matrix(per_class)


    def evaluate_conf_matrix(self, per_class=False):
        """
        https://github.com/facebookresearch/detectron2/blob/96c752ce821a3340e27edd51c28a00665dd32a30/detectron2/evaluation/sem_seg_evaluation.py

        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """

        # # TODO: logistics, skipped for now
        # if self._output_dir:
        #     PathManager.mkdirs(self._output_dir)
        #     file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
        #     with PathManager.open(file_path, "w") as f:
        #         f.write(json.dumps(self._predictions))

        acc = np.full(self.num_classes, np.nan, dtype=np.float64)
        iou = np.full(self.num_classes, np.nan, dtype=np.float64)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float64)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float64)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float64)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        if self._compute_boundary_iou:
            b_iou = np.full(self.num_classes, np.nan, dtype=np.float64)
            b_tp = self._b_conf_matrix.diagonal()[:-1].astype(np.float64)
            b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(np.float64)
            b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(np.float64)
            b_union = b_pos_gt + b_pos_pred - b_tp
            b_iou_valid = b_union > 0
            b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self.class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            if self._compute_boundary_iou:
                res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]
                res[f"min(IoU, B-Iou)-{name}"] = 100 * min(iou[i], b_iou[i])
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self.class_names):
            res[f"ACC-{name}"] = 100 * acc[i]
            # UNCOMMENT to print per-class evaluation
            if per_class:
                print (f"Class {name}: {100 * acc[i]}%")

        # # TODO: logistics, skipped for now
        # if self._output_dir:
        #     file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
        #     with PathManager.open(file_path, "wb") as f:
        #         torch.save(res, f)
        # results = OrderedDict({"sem_seg": res})
        return res

    def _mask_to_boundary(self, mask: np.ndarray, dilation_ratio=0.02):
        assert mask.ndim == 2, "mask_to_boundary expects a 2-dimensional image"
        h, w = mask.shape
        diag_len = np.sqrt(h**2 + w**2)
        dilation = max(1, int(round(dilation_ratio * diag_len)))
        kernel = np.ones((3, 3), dtype=np.uint8)

        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        eroded_mask_with_padding = cv2.erode(padded_mask, kernel, iterations=dilation)
        eroded_mask = eroded_mask_with_padding[1:-1, 1:-1]
        boundary = mask - eroded_mask
        return boundary

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # unit-test for old-style dataloader and augmentation
    # running from data/.. with python -m data.voc0712
    from data.data_augment import detection_collate

    train_sets = [('2012', 'train')]
    dataset = VOCDetection(train_sets, 320, imgset='Segmentation',
            cache=True, double_aug=False, is_training=False,
            ignore_label=255, task='seg')

    loader = iter(data.DataLoader(dataset, batch_size=2, shuffle=False,
            num_workers=1, collate_fn=detection_collate))
    imgs, targets, seggt = next(loader)
    imgs = imgs.cpu().numpy().transpose((0, 2, 3, 1))
    seggt = seggt.cpu().numpy().squeeze()
    seggt[seggt==255]=22
    plt.imshow(imgs[0]), plt.figure(), plt.imshow(seggt[0]), plt.figure()
    plt.imshow(imgs[0]), plt.imshow(seggt[0], alpha=.5)
    plt.show()
    plt.imshow(imgs[1]), plt.figure(), plt.imshow(seggt[1]), plt.figure()
    plt.imshow(imgs[1]), plt.imshow(seggt[1], alpha=.5)
    plt.show()

    # -------------
    # loading with albumentation
    train_sets = [('2012', 'train')]
    dataset = VOCDetection(train_sets, 320, imgset='Segmentation',
            cache=True, double_aug=False, is_training=True,
            ignore_label=255, task='det+seg')
    output = dataset.__getitem__(0)

    # visualization
    for bbox_cat in output['bboxes']:
        visualize_bbox(output['image'], np.array(bbox_cat[:4]), str(bbox_cat[-1]),
                (255, 0, 0), output['image'].shape[1], output['image'].shape[0])

    seg_= Image.fromarray(output['mask'].astype(np.uint8), mode="P")
    seg_.putpalette(VOCDetection.label_palette())
    plt.imshow(output['image']), plt.figure(), plt.imshow(seg_),
    # plt.figure(), plt.imshow(output['mask'])
    plt.show()


    index = 0
    img = dataset.pull_image(index, resize=True)
    seg = dataset.pull_segment(index, resize=True)

    img_norm_cfg = dict(
        max_pixel_value=255.0,
        std=(0.229, 0.224, 0.225),
        mean=(0.485, 0.456, 0.406),
    )
    ignore_label = 255
    crop_size_h, crop_size_w = 513, 513
    test_size_h, test_size_w = 513, 513
    image_pad_value = (123.675, 116.280, 103.530)

    transforms = [
        dict(type='RandomScale', scale_limit=(0.5, 2),
             interpolation=cv2.INTER_LINEAR),
        dict(type='PadIfNeeded', min_height=crop_size_h,
            min_width=crop_size_w, value=image_pad_value,
            mask_value=ignore_label, border_mode=0),
        dict(type='RandomCrop', height=crop_size_h, width=crop_size_w),
        dict(type='Rotate', limit=10, interpolation=cv2.INTER_LINEAR,
            border_mode=0, value=image_pad_value,
            mask_value=ignore_label, p=0.5),
        dict(type='GaussianBlur', blur_limit=7, p=0.5),
        dict(type='HorizontalFlip', p=0.5),
        dict(type='RandomBrightnessContrast', p=0.3),
        dict(type='RGBShift', r_shift_limit=30, g_shift_limit=30,
            b_shift_limit=30, p=0.3),
        dict(type='RandomBrightnessContrast', p=.5),
        dict(type='RandomGamma', p=.5),
        dict(type='CLAHE', p=.5),
        # dict(type='Normalize', **img_norm_cfg),
        # dict(type='ToTensorV2'),
    ]

    img_id = dataset.ids[index]
    img = cv2.imread(dataset._imgpath % img_id, cv2.IMREAD_COLOR)
    box = dataset.target_transform(ET.parse(dataset._annopath % img_id).getroot())
    seg = np.array(Image.open(dataset._seggtpath % img_id))
    seg_= Image.fromarray(seg, mode="P")
    seg_.putpalette(VOCDetection.label_palette())

    # visualize original data
    img_ = img.copy()
    for bbox_cat in box:
        visualize_bbox(img_, bbox_cat[:4], str(bbox_cat[-1]), (255, 0, 0),
                img_.shape[1], img_.shape[0])
    plt.imshow(img_), plt.figure(), plt.imshow(seg_),  plt.show()

    modules = []
    for t in transforms:
        modules.append(build_from_module(t, A))
    aug = A.Compose(modules,
            # additional_targets={'bbox': 'bboxes'},
            bbox_params=A.BboxParams(format='albumentations'
                #, min_area=1024, min_visibility=0.1, label_fields=['class_labels']
            )
        )

    for i in range(1):
        # augmentation
        output = aug(image=img, mask=seg, bboxes=box)
        # visualization
        for bbox_cat in output['bboxes']:
            visualize_bbox(output['image'], np.array(bbox_cat[:4]), str(bbox_cat[-1]),
                    (255, 0, 0), output['image'].shape[1], output['image'].shape[0])

        seg_= Image.fromarray(output['mask'], mode="P")
        seg_.putpalette(VOCDetection.label_palette())
        plt.imshow(output['image']), plt.figure(), plt.imshow(seg_),
        # plt.figure(), plt.imshow(output['mask'])
        plt.show()
