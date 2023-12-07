#!/usr/bin/python
# -*- coding: utf-8 -*-

import os.path as osp
import cv2
import numpy as np
#from .data_augment import preproc_for_vedai
# from .data_augment import preproc_for_train
from .voc0712 import VOCDetection
import warnings
warnings.filterwarnings("ignore")


CLASSES = ('__background__', # always index 0
    'car', 'truck', 'pickup', 'tractor', 'camper', 'ship', 'van', 'others'
    )

# class VedaiDetection(data.Dataset):
class VedaiDetection(VOCDetection):

    def __init__(self, data_root, imgset, size, normalize_data=False, is_training=False):
        self.class_names = ['__background__', '0', '1', '2', '3', '4', '5', '6', '7']

        self.name = f'VEDAI25-{imgset}'
        self.size = size
        self.num_classes = len(self.class_names)
        self.norm = normalize_data

        self.task = "det"
        self._imgpath = osp.join(data_root, 'JPEGImages', '%s.png')
        self._annopath = osp.join(data_root, 'Annotations512', '%s.xml')
        self.ids = open(osp.join(data_root, 'splits', f'{imgset}.txt')).read().splitlines()
        self.target_transform = self.parse_xml
        self.is_training = is_training
        self.ignore_label = 255

        if self.is_training:
            # adding augmented files
            self.ids += open(osp.join(data_root, 'splits', 'train.txt')).read().splitlines()
            self.ids += [f"{f}_{i}" for f in self.ids for i in range(1,4,1)]
            self.augment = self.make_augment()
        self.preprocess = self.make_preprocess()

        self.class_to_ind = dict(zip( self.class_names, range(len(self.class_names))))

        self.use_seggt = False
        self.double_aug = False

        # self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # self.thermal_filenames, self.visible_filenames, self.fusion_targets = list(), list(), list()

        # filenames = sorted(os.listdir(annotation_folder))
        # filenames = [filenames[l] for l in ids] if ids is not None else filenames


        # for filename in filenames:

        #     # thermal_filename = os.path.join(image_folder, filename.replace('.xml','_ir.png'))
        #     visible_filename = os.path.join(image_folder, filename.replace('.xml','.png'))

        #     filename = os.path.join(annotation_folder, filename)

        #     if not filename.endswith('.xml'):
        #         warnings.warn('WARN: passing {}'.format(filename))
        #         continue

        #     fusion_res = self.parse_xml(filename)

        #     # verify the image pair contains bounding boxes (only for training)
        #     if 'train' in annotation_folder:
        #         if len(fusion_res)==0:
        #             warnings.warn('WARN: passing {}... No bounding boxes found...'.format(filename))
        #             continue

        #     self.thermal_filenames.append(
        #             os.path.join(thermal_filename)
        #         )
        #     self.visible_filenames.append(
        #             os.path.join(visible_filename)
        #         )
        #     self.fusion_targets.append(fusion_res)

    # def __getitem__(self, index):
    #     img = self.pull_image(index)
    #     targets = self.fusion_targets[index]
    #     img, targets = preproc_for_train(img, targets,
    #             self.size)#, norm=self.norm)
    #     return img, targets

    def __len__(self):
        return len(self.ids)
        return len(self.thermal_filenames)

    def pull_classes(self):
        return self.class_names

    def to_be_filtered(self, bndbox):
        #return False
        return bndbox[0] >= bndbox[2] or bndbox[1] >= bndbox[3]

    def parse_xml(self, target):

        # target = ET.parse(xml_file).getroot()
        width = float(target.find('size').find('width').text)
        height = float(target.find('size').find('height').text)

        res = np.empty((0,5))
        for obj in target.iter('object'):

            name = obj.find('name').text.strip()
            if name not in self.class_to_ind:
                continue
            label_idx = self.class_to_ind[name]

            bndbox = []
            for i, pt in enumerate(['xmin', 'ymin', 'xmax', 'ymax']):
                bndbox.append(
                    float(obj.find('bndbox').find(pt).text)
                )
            bndbox = [
                    min(bndbox[0],bndbox[2]), min(bndbox[1],bndbox[3]),
                    max(bndbox[0],bndbox[2]), max(bndbox[1],bndbox[3]),
            ]
            for i, x in enumerate([width, height, width, height]):
                bndbox[i] /= x
            # clamp to (0, 1)
            bndbox = [min(max(b, 0), 1) for b in bndbox]
            bndbox.append(label_idx)

            # filter non valid boxes!!
            #bndbox[2] = bndbox[2] + 1 if bndbox[2] == bndbox[0] else bndbox[2]
            #bndbox[3] = bndbox[3] + 1 if bndbox[3] == bndbox[1] else bndbox[3]

            if self.to_be_filtered(bndbox):
                continue

            assert bndbox[0]<bndbox[2] and bndbox[1]<bndbox[3], \
            f"should be: {bndbox[0]}<{bndbox[2]} and {bndbox[1]}<{bndbox[3]}"
            res = np.vstack((res, bndbox))

        return res

    def pull_image(self, index, resize=False):

        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        # Reading 8 bits thermal image
        #img1 = cv2.imread(self.thermal_filenames[index], cv2.IMREAD_GRAYSCALE)
        #img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        # img =  cv2.imread(self.visible_filenames[index], cv2.IMREAD_COLOR)

        # Reisze thermal image to visible image size
        # img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        return img


    def evaluate_detections(self, all_boxes, ids=None, all_thresh=True):
        # reference to detectron2 pascal voc evaluation for more info
        # https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/pascal_voc_evaluation.html

        if all_thresh:
            all_thresh = [0.25, .1]
        else:
            all_thresh = [.1]

        for thresh in all_thresh:
            result = self.evaluate_thres(all_boxes, thresh=thresh, ids=ids)
            print('----AP@iou{:.2f} = {:.2f}%'.format(thresh, result))

        return result / 100 # for thresh = .1

    def evaluate_thres(self, all_boxes, thresh, ids=None):
        aps = list()
        if ids is None:
            ids = range(len(self))
        for j in range(1, self.num_classes):

            # prepare gt
            class_recs = dict()
            npos = 0
            #breakpoint()
            for i in ids: #range(len(self)):
                R = dict()
                # anno = self.fusion_targets[i]
                anno = self.pull_anno(i)
                inds = np.where(anno[:, -1] == j)[0]
                if len(inds) == 0:
                    R['bbox'] = np.empty([0, 4], dtype=np.float32)
                else:
                    R['bbox'] = anno[inds, :4]
                R['det'] = [False] * len(inds)
                class_recs[i] = R
                npos += len(inds)

            # parse det
            image_ids = list()
            confidence = list()
            BB = np.empty([0, 4], dtype=np.float32)
            for i in ids: #range(len(self)):
                for det in all_boxes[j][i]:
                    image_ids.append(i)
                    confidence.append(det[-1])
                    BB = np.vstack((BB,det[np.newaxis, :4]))
            image_ids = np.array(image_ids)
            confidence = np.array(confidence)

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            image_ids = image_ids[sorted_ind]
            BB = BB[sorted_ind, :]

            # mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[int(image_ids[d])]
                BBGT = R['bbox'].astype(float)
                bb = BB[d, :].astype(float)
                ovmax = -np.inf

                if BBGT.size > 0:
                    # compute overlaps
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = (bb[2] - bb[0]) * (bb[3] - bb[1]) + \
                          (BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1]) - \
                          inters
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if (ovmax > thresh) and (not R['det'][jmax]):
                    R['det'][jmax] = True
                    tp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes v
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            aps.append(ap)
            print('class {} = {:.2f}% AP'.format(self.class_names[j], ap*100))

        ap = np.nanmean(aps)
        return ap*100


    def evaluate_withPR(self, all_boxes, thresh, ids=None):
        # reference to detectron2 pascal voc evaluation for more info
        # https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/pascal_voc_evaluation.html

        aps = list()
        prs = list() # mean precision of all classes
        res = list() # mean recall of all classes
        prc = dict() # info for pr-curve
        if ids is None:
            ids = range(len(self))
        for j in range(1, self.num_classes):

            # prepare gt
            class_recs = dict()
            npos = 0
            #breakpoint()
            for i in ids: #range(len(self)):
                R = dict()
                # anno = self.fusion_targets[i]
                anno = self.pull_anno(i)
                inds = np.where(anno[:, -1] == j)[0]
                if len(inds) == 0:
                    R['bbox'] = np.empty([0, 4], dtype=np.float32)
                else:
                    R['bbox'] = anno[inds, :4]
                R['det'] = [False] * len(inds)
                class_recs[i] = R
                npos += len(inds)

            # parse det
            image_ids = list()
            confidence = list()
            BB = np.empty([0, 4], dtype=np.float32)
            for i in ids: #range(len(self)):
                for det in all_boxes[j][i]:
                    image_ids.append(i)
                    confidence.append(det[-1])
                    BB = np.vstack((BB,det[np.newaxis, :4]))
            image_ids = np.array(image_ids)
            confidence = np.array(confidence)

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            image_ids = image_ids[sorted_ind]
            BB = BB[sorted_ind, :]

            # mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[int(image_ids[d])]
                BBGT = R['bbox'].astype(float)
                bb = BB[d, :].astype(float)
                ovmax = -np.inf

                if BBGT.size > 0:
                    # compute overlaps
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = (bb[2] - bb[0]) * (bb[3] - bb[1]) + \
                          (BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1]) - \
                          inters
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if (ovmax > thresh) and (not R['det'][jmax]):
                    R['det'][jmax] = True
                    tp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

            prs.append(np.mean(prec))
            res.append(np.mean(rec))
            # 11 point metric
            ap = 0.0
            for ei, t in enumerate(np.arange(0.0, 1.01, 0.01)):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                if ei not in prc:
                    prc[ei] = [t]
                prc[ei].append(p)

            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes v
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            aps.append(ap)
            # print('class {} = {:.2f}% AP'.format(self.class_names[j], ap*100))

        prc_mat = np.vstack([prc[l] for l in range(len(prc))])
        ap = np.nanmean(aps)
        return aps, prc_mat

if __name__ == "__main__":

    from .voc0712 import visualize_bbox
    import matplotlib.pyplot as plt
    std=(0.229, 0.224, 0.225)
    mean=(0.485, 0.456, 0.406)

    image_folder = 'datasets/VEDAI_Heng/JPEGImages/'
            # for testing
    dataset = VedaiDetection('datasets/VEDAI_Heng', 'fold01test',
            512, normalize_data=False, is_training=False)

    for i in range(100):
        # output = dataset.__getitem__(i)

        # # visualization
        # for bbox_cat in output['bboxes']:
        #     visualize_bbox(output['image'], np.array(bbox_cat[:4]), str(bbox_cat[-1]),
        #             output['image'].shape[1], output['image'].shape[0], color=(255, 0, 0))

        # plt.imshow(output['image']), plt.show()

        data = dataset.__getitem__(i)
        print (data[1]) # bbox

        img = ((data[0].numpy().transpose((1, 2, 0)) * std + mean)*255).astype(np.uint8).copy()
        for bbox in data[1]:
            cat = bbox[-1] # the first entry is empty
            bbox = bbox[:-1] # format xmin, ymin, xmax, ymax
            # bbox = torch.cat((bbox[:2] - bbox[2:] / 2, bbox[:2] + bbox[2:] / 2), 0)
            visualize_bbox(img, bbox, str(cat),
                           w=data[0].size(2), h=data[0].size(1), color = (255, 0, 0)
                           )
        plt.imshow(img)
        plt.show()
