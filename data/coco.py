#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, os.path as osp
import pickle
import torch.utils.data as data
import cv2
import numpy as np
import json
import io
from builtins import str as unicode
from .data_augment import preproc_for_train_, preproc_for_test_
from datetime import datetime as dt

from PIL import Image

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.cocostuffeval import COCOStuffeval
from pycocotools.cocostuffhelper import segmentationToCocoResult
from pycocotools.cocostuffhelper import cocoSegmentationToSegmentationMap

class COCODetection(data.Dataset):
    """ COCO Detection Dataset Object """

    def __init__(self, image_sets, size, dataset_name='COCO2017', cache=True,
                subset=None, is_training: bool = False,
                task="det", ignore_label=0,
            ):
        self.root = osp.join('datasets/', 'coco2017/')
        self.now = dt.now().strftime("%Y%m%d_%H%M%S.%f")
        self.cache_path = osp.join(self.root, 'cache')
        self.image_set = image_sets
        self.size = size
        self.name = dataset_name+str(self.image_set)+str(self.size)
        self.ids = list()
        self.task = task
        self.ignore_label = ignore_label
        # self.is_training = not is_training
        self.is_training = is_training
        self.annotations = list()
        subset = osp.join(self.root, 'subsets', subset) if subset is not None else None
        for (year, image_set) in image_sets:
            coco_name = image_set+year
            annofile = self._get_ann_file(coco_name) # "toy2017")#
            self._COCO = COCO(annofile) #"overfit2017"
            self.coco_name = coco_name
            cats = self._COCO.loadCats(self._COCO.getCatIds())
            self._classes = tuple(['__background__'] + [c['name'] for c in cats])
            self.num_classes = len(self._classes)
            self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
            self._class_to_coco_cat_id = dict(zip([c['name'] for c in cats], self._COCO.getCatIds()))
            indexes = self._COCO.getImgIds() if subset is None else [int(k)
                        for k in open(osp.join(subset, image_set)).read().splitlines()]
            self.image_indexes = indexes
            self.ids.extend([self.image_path_from_index(coco_name, index) for index in indexes])
            if image_set.find('test') != -1:
                print('test set will not load annotations!')
            else:
                self.annotations.extend(self._load_coco_annotations(indexes, self._COCO))
                # if image_set.find('val') != -1 and not is_training:
                if image_set.find('val') != -1:
                    print('val set will not remove non-valid images!')
                else:
                    ids, annotations = [], []
                    for i, a in zip(self.ids, self.annotations):
                        # excluding images with no annotations
                        if a.shape[0] > 0:
                            ids.append(i)
                            annotations.append(a)
                    self.ids = ids
                    self.annotations = annotations
        if cache:
            self._cache_images()
        else:
            self.imgs = None

    def pull_classes(self):
        return self._classes

    def image_path_from_index(self, name, index):
        """ Construct an image path """
        file_name = (str(index).zfill(12) + '.jpg')
        image_path = osp.join(self.root, name, file_name)
        # assert osp.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def _get_ann_file(self, name):
        if self.task == "det":
            prefix = 'instances' if name.find('test') == -1 else 'image_info'
        elif self.task == "seg":
            prefix = 'things' if name.find('test') == -1 else 'image_info'
        else:
            raise NotImplementedError('Unknown task {}!'.format(self.task))
        return osp.join(self.root, 'annotations', prefix + '_' + name + '.json')

    def _load_coco_annotations(self, indexes, _COCO):
        annotation_from_index = self._annotation_from_index if self.task == "det" \
                                    else self._segmentation_from_index
        gt_roidb = [annotation_from_index(index, _COCO) for index in indexes]
        return gt_roidb

    def _segmentation_from_index(self, index, _COCO):

        # annIds = _COCO.getAnnIds(imgIds=self.ids[index], iscrowd=None)
        annIds = _COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = _COCO.loadAnns(annIds)
        return np.array([k['segmentation']['size'] for k in objs]) # dummy anno

    def _annotation_from_index(self, index, _COCO):
        """ Loads COCO bounding-box instance annotations """
        im_ann = _COCO.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = _COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = _COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and (x2-x1) > 6 and (y2-y1) > 6:
                obj['clean_bbox'] = [x1/width, y1/height, x2/width, y2/height]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        # Lookup table to map from COCO category ids to our internal class indices
        coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                          self._class_to_ind[cls])
                                         for cls in self._classes[1:]])

        for ix, obj in enumerate(objs):
            cls = coco_cat_id_to_class_ind[obj['category_id']]
            res[ix, 0:4] = obj['clean_bbox']
            res[ix, 4] = cls

        return res

    def _cache_images(self):
        cache_file = self.root + "/img_resized_cache_" + self.name + ".array"
        if not osp.exists(cache_file):
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

    def pull_image(self, index, resize=False):
        ''' Returns the original image object at index '''
        img_id = self.ids[index]
        image = cv2.imread(img_id, cv2.IMREAD_COLOR)
        if resize:
            image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return image

    def pull_anno(self, index):
        return self.annotations[index]

    def pull_segment(self, index, resize=False):
        labelMap = cocoSegmentationToSegmentationMap(self._COCO,
                                                     self.image_indexes[index])
        if resize:
            labelMap = Image.fromarray(labelMap).resize(
                    (self.size, self.size), resample=Image.NEAREST)
        return np.array(labelMap)

    def __getitem__(self, index):
        # target = self.annotations[index]
        # if self.imgs is not None:
        #     img = self.imgs[index].copy()
        # else:
        #     img = self.pull_image(index, resize=True)
        # img, target = preproc_for_train(img, target, self.size)
        # return img, target
        data = dict()
        if self.imgs is not None:
            data['image'] = self.imgs[index].copy()
        else:
            data['image'] = self.pull_image(index, resize=True)
        if self.task == 'det':
            data['bboxes'] = self.pull_anno(index)
        elif self.task == 'seg': # TODO: elif or if?
            # data['bboxes'] = self.pull_anno(index)
            data['mask'] = self.pull_segment(index, resize=True)

        if self.is_training:
            data = preproc_for_train_(data, self.size)
        data = preproc_for_test_(data, self.size)

        return data # torch tensor with (B, C, W, H) shapes for image and seggt


    def __len__(self):
        return len(self.ids)

    def _print_detection_eval_metrics(self, coco_eval, perclass=False):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
               '~~~~'.format(IoU_lo_thresh, IoU_hi_thresh))
        print('all: {:.1f}'.format(100 * ap_default))
        if perclass:
            for cls_ind, cls in enumerate(self._classes):
                if cls == '__background__':
                    continue
                # minus 1 because of __background__
                precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
                ap = np.mean(precision[precision > -1])
                print('{}: {:.1f}'.format(cls, 100 * ap))

        print('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()
        return ap_default

    def _do_detection_eval(self, res_file, output_dir, perclass=False):
        ann_type = 'bbox'
        coco_dt = self._COCO.loadRes(res_file)
        coco_eval = COCOeval(self._COCO, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        ap_default = self._print_detection_eval_metrics(coco_eval, perclass)
        eval_file = osp.join(output_dir, f'detection_results_{self.now}.pkl')
        with open(eval_file, 'wb') as fid:
            pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote COCO eval results to: {}'.format(eval_file))
        return ap_default

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_indexes):
            dets = boxes[im_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
              [{'image_id' : index,
                'category_id' : cat_id,
                'bbox' : [xs[k], ys[k], ws[k], hs[k]],
                'score' : scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_coco_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            #print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind, self.num_classes ))
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind], coco_cat_id))
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def evaluate_detections(self, all_boxes, all_thresh=True, per_class=False):
        output_dir = osp.join(self.root, 'eval')
        os.makedirs(output_dir, exist_ok=True)
        res_file = osp.join(output_dir, ("detections_" + self.coco_name + '_results'))
        res_file += '_' + self.now + '.json'
        self._write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self.coco_name.find('test') == -1:
            return self._do_detection_eval(res_file, output_dir)
        # Optionally cleanup results json file

    def evaluate_segmentation(self, all_segs, all_thresh=True, per_class=False):

        # reshape back to GT size
        preds = []
        for i, pred in enumerate(all_segs):
            gt = self.pull_segment(i, resize=False)
            # breakpoint()
            preds.append(np.array(Image.fromarray(pred.astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), resample=Image.NEAREST)))

        # output to JSON file
        output_dir = osp.join(self.root, 'eval')
        os.makedirs(output_dir, exist_ok=True)
        res_file = osp.join(output_dir, ("segmentation_" + self.coco_name + '_results'))
        res_file += '_' + self.now + '.json'
        # breakpoint()
        self._labelMapToCocoResult(preds, res_file)
        # Only do evaluation on non-test sets
        if self.coco_name.find('test') == -1:
            return self._do_segmentation_eval(res_file, output_dir)
        # Optionally cleanup results json file

    def _labelMapToCocoResult(self, sem_preds, res_file, resType='examples', indent=None):
        '''
        Converts a folder of .png images with segmentation results back
        to the COCO result format. 
        :param dataDir: location of the COCO root folder
        :param resType: identifier of the result annotation file
        :param indent: number of whitespaces used for JSON indentation
        :return: None
        ref: https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/cocostuff/pngToCocoResultDemo.py
        '''

        class BytesEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, bytes):
                    return obj.decode('utf-8')
                return json.JSONEncoder.default(self, obj)

        assert len(self.image_indexes) == len(sem_preds)
        # Init
        annCount = 0
        imgCount = len(sem_preds)
        with io.open(res_file, 'w', encoding='utf8') as output:
            print('Writing results to: %s' % res_file)

            # Annotation start
            output.write(unicode('[\n'))

            for i, (labelMap, imgId) in enumerate(zip(sem_preds, self.image_indexes)):

                anns = segmentationToCocoResult(labelMap, imgId, stuffStartId=1)

                # Write JSON
                str_ = json.dumps(anns, indent=None, cls=BytesEncoder)
                str_ = str_[1:-1]
                if len(str_) > 0:
                    output.write(unicode(str_))
                    annCount = annCount + 1

                # Add comma separator
                if i < imgCount-1 and len(str_) > 0:
                    output.write(unicode(','))

                # Add line break
                output.write(unicode('\n'))

            # Annotation end
            output.write(unicode(']'))

            # Create an error if there are no annotations
            if annCount == 0:
                raise Exception('The output file has 0 annotations and will not'
                                ' work with the COCO API!')


    def _do_segmentation_eval(self, resFile, output_dir, perclass=False):
        "https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/cocostuff/cocoStuffEvalDemo.py"
        cocoRes = self._COCO.loadRes(resFile)
        cocoEval = COCOStuffeval(self._COCO, cocoRes, stuffStartId=1,
                                 stuffEndId=91, addOther=True)

        cocoEval.evaluate()
        cocoEval.summarize()

        [miou, fwiou, macc, pacc, ious, maccs] = cocoEval._computeMetrics(cocoEval.confusion)

        evalFile = osp.join(output_dir, f'segmentation_results_{self.now}.pkl')
        with open(evalFile, 'wb') as fid:
            pickle.dump(cocoEval, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote COCO eval results to: {}'.format(evalFile))
        return {'mIoU': miou*100, 'fwIOU': fwiou*100,
                'macc': macc*100, 'pacc': pacc*100}
                # 'ious': ious, 'maccs': maccs}

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # unit-test for dataloader and augmentation
    # running from data/.. with python -m data.coco
    from .voc0712 import visualize_bbox
    std=(0.229, 0.224, 0.225)
    mean=(0.485, 0.456, 0.406)

    dataset = COCODetection([('2017', 'train')], size=320, cache=True,
                                is_training=False,
                            )
    # detection
    for i in range(1):
        data = dataset.__getitem__(i)
        img = ((data['image'][0].numpy().transpose((1, 2, 0)) * std + mean)*255).astype(np.uint8)
        img = img[..., -1::-1].copy()
        plt.figure()
        for bbox in data['bboxes'][0].numpy():
            cat = bbox[-1]
            bbox = bbox[:-1] # format xmin, ymin, xmax, ymax
            visualize_bbox(img, bbox, str(cat),
                w=img.shape[0], h=img.shape[1], color = (255, 0, 0)
            )
        plt.imshow(img)
        plt.show()

    # segmentation
    dataset = COCODetection([('2017', 'train')], size=320, cache=True, task='seg',
                                is_training=False,)
    for i in range(10):
        data = dataset.__getitem__(i)
        img = ((data['image'][0].numpy().transpose((1, 2, 0)) * std + mean)*255).astype(np.uint8)
        img = img[..., -1::-1].copy()
        plt.figure()
        plt.imshow(img)
        plt.imshow(data['mask'][0].squeeze(), alpha=.8)
        plt.figure()
        plt.imshow(data['mask'][0].squeeze())
        plt.show()
