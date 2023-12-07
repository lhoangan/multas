import torch
import os, os.path as osp
import numpy as np
from .voc0712 import visualize_bbox
from .vedai_dataset import VedaiDetection
# from .data_augment import preproc_for_train
from .data_augment import preproc_for_train_, preproc_for_test_
from datetime import datetime as dt
from sklearn.metrics import confusion_matrix

from PIL import Image

ISPRS = ('imperv_surfaces', 'buildings', 'low_vegetation',
        'trees', 'cars', 'background'
        )

# https://github.com/nshaud/DeepNetsForEO/blob/master/SegNet_PyTorch_v2.ipynb
ISPRS_LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names

# Parameters
WINDOW_SIZE = (256, 256) # Patch size
STRIDE = 32 # Stride for testing
BATCH_SIZE = 10 # Number of samples in a mini-batch


crop_set_ = "cropped_320_BuildingTreeCar_IRRG_resize69"

PATH_TEMPL = {
    "Potsdam": { # cropped and with object detection added
        "stem":"top_potsdam_",
        "RGB": "images/{}{}.png",
        "Box": "bboxes/{}{}.txt",
        "Seg": "semantic/{}{}.png",
        "SegE":"semantic_erode/{}{}.png"
    },
    "Vaihingen": {
        "stem":"top_mosaic_09cm_area",
        "RGB": "images/{}{}.png",
        "Box": "bboxes/{}{}.txt",
        "Seg": "semantic/{}{}.png",
        "SegE":"semantic_erode/{}{}.png",
    },
}
import copy
PATH_TEMPL = {**PATH_TEMPL, **{"P1": copy.deepcopy(PATH_TEMPL["Potsdam"]),
                               "P2": copy.deepcopy(PATH_TEMPL["Potsdam"])}}

class ISPRSDataset(VedaiDetection):

    def __init__(self, path,
                    # image_sets: list,
                    size: int = 320,
                    dataset_name: str = 'ISPRS',
                    imgset: str = "Potsdam", # potsdam or vaihingen
                    ignore_label: int = 255,
                    is_training: bool = False,
                    task: str="det",
                    both_task: bool=False,
                    seg3: bool=False, # if using 3 classes [b,t,c]for seg
                    *args, **kwargs
                ):

        self.is_training = is_training
        self.root = os.path.expanduser('datasets/isprs/')
        self.now = dt.now().strftime("%Y%m%d_%H%M%S.%f")
        self.task = task if not both_task else "det+seg"
        self.class_names = ISPRS_LABELS if not seg3 else ['buildings', 'trees', 'cars']
        self.ignore_label = len(self.class_names) # for segmentation
        if task == 'det':
            self.class_names = ['__background__', 'buildings', 'trees', 'cars']
        self.num_classes = len(self.class_names)
        self.size = size
        self.seg3 = seg3
        self.imgset = imgset

        self.ids = list()
        crop_set = crop_set_ #+ ("_IRRG_resize69" if imgset == "Potsdam" else "")
        self._imgpath = osp.join(self.root, "{}", crop_set, PATH_TEMPL[imgset]['RGB'])

        if 'seg' in self.task:
            if is_training:
                self._segpath = osp.join(self.root, "{}", crop_set, PATH_TEMPL[imgset]['Seg'])
            else:
                self._segpath = osp.join(self.root, "{}", crop_set, PATH_TEMPL[imgset]['SegE'])

        if 'det' in self.task:
            self._boxpath = osp.join(self.root, "{}", crop_set, PATH_TEMPL[imgset]['Box'])

        self.load_data()

    def __len__(self):
        return len(self._imgpath)
        if self.task == 'seg' or self.imgset == 'Potsdam':
            return len(self.ids)
        else:
            return super(ISPRSDataset, self).__len__()

    def load_data(self):
        crop_set = crop_set_ #+ ("_IRRG_resize69" if self.imgset == "Potsdam" else "")
        if self.is_training:
            iss = [self.imgset]
            path = {s: osp.join(self.root, s, f"{crop_set}/train.txt") for s in iss}
        else:
            # iss = ["Potsdam", "Vaihingen"]
            iss = [self.imgset]
            path = {s: osp.join(self.root, s, f"{crop_set}/test.txt") for s in iss}

        ids = {s: open(path[s]).read().splitlines() for s in iss}

        self._imgpath = [self._imgpath.format(s, PATH_TEMPL[s]['stem'], i) for s in iss for i in ids[s]]
        if 'seg' in self.task:
            self._segpath = [self._segpath.format(s, PATH_TEMPL[s]['stem'], i) for s in iss for i in ids[s]]
        if 'det' in self.task:
            self._boxpath = [self._boxpath.format(s, PATH_TEMPL[s]['stem'], i) for s in iss for i in ids[s]]

    @staticmethod
    def label_palette(n=256):
        """
        ref: https://github.com/nshaud/DeepNetsForEO/blob/master/SegNet_PyTorch_v2.ipynb
        """
        # ISPRS color palette
        palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)        # Undefined (black)
        }
        return palette

    @staticmethod
    def invert_palette():
        return {v: k for k, v in ISPRSDataset.label_palette().items()}

    @staticmethod
    def class_to_color(arr_2d):
        """ Numeric labels to RGB-color encoding
        ref: https://github.com/nshaud/DeepNetsForEO/blob/master/SegNet_PyTorch_v2.ipynb
        """
        palette=ISPRSDataset.label_palette()
        arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

        for c, i in palette.items():
            m = arr_2d == c
            arr_3d[m] = i

        return arr_3d

    @staticmethod
    def color_to_class(arr_3d):
        """ RGB-color encoding to grayscale labels
        ref: https://github.com/nshaud/DeepNetsForEO/blob/master/SegNet_PyTorch_v2.ipynb
        """
        palette = ISPRSDataset.invert_palette()
        arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

        for c, i in palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i

        return arr_2d

    def __getitem__(self, index):
        data = {"image": self.pull_image(index)}
        if 'seg' in self.task: # == 'seg':
            data['mask'] = self.pull_segment(index)[:, :, None]
        if 'det' in self.task: # == "det":
            data['bboxes'] = self.pull_anno(index)

        if self.is_training:
            data = preproc_for_train_(data, self.size,
                                      ignored_class=self.ignore_label)

        data = preproc_for_test_(data, self.size)

        return data

    def pull_image(self, index, withlabel=False):
        # print (self.ids[index])
        return np.array(Image.open(self._imgpath[index]))

    def pull_anno(self, index):
        # this function is used for evaluation
        path = self._boxpath[index]
        boxes = np.array([l.split() for l in open(path).read().splitlines()], np.float32)
        # converting from [cat, Cx, Cy, W, H] to [xmin, ymin, xmax, ymax, cat]
        # print (boxes)
        boxes = np.hstack((boxes[:, 1:3] - boxes[:, 3:]/2,
                           boxes[:, 1:3] + boxes[:, 3:]/2,
                           boxes[:, 0, None]+1)) # class counting from 1
        return boxes

    def pull_segment(self, index, resize=False):
        seg = np.array(Image.open(self._segpath[index]))
        return self.transf_seg(ISPRSDataset.color_to_class(seg))
        return torch.from_numpy(ISPRSDataset.color_to_class(seg)).float()

    def transf_seg(self, seg):
        if not self.seg3:
            return seg
        else:
            maps = {0: 3, 1: 0, 2: 3, 3: 1, 4: 2, 5: 3, 6: 3}
            seg_out = seg.copy()
            for i in range(len(maps)):
                seg_out[seg == i] = maps[i]
            return seg_out

    def evaluate_detections(self, all_boxes, ids=None, all_thresh=True, per_class=False):
        # reference to detectron2 pascal voc evaluation for more info
        # https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/pascal_voc_evaluation.html

        results = []
        for thresh in np.arange(0.5,1,0.05):
            result = self.evaluate_thres(all_boxes, thresh=thresh, ids=ids)
            results.append(result)
            print('----AP@iou{:.2f} = {:.2f}%'.format(thresh, result))

        print('mAP results: AP@50={:.3f}, AP@75={:.3f}, AP={:.3f}'.format(
            results[0], results[5], sum(results)/10))
        return (sum(results)/10) / 100 # expected output in range [0, 1]

    def evaluate_segmentation(self, preds, gts, per_class=False):

        # breakpoint()
        # computing confusion matrix, F1 score following ISPRS benchmarks
        self.metrics(preds.ravel(), gts.ravel(), label_values=self.class_names)

        # compute PascalVOC mIOU
        return super(ISPRSDataset, self).evaluate_segmentation(preds, gts, per_class)

    def metrics(self, predictions, gts, label_values=ISPRS_LABELS):
        """
        ref: https://github.com/nshaud/DeepNetsForEO/blob/master/SegNet_PyTorch_v2.ipynb
        """
        cm = confusion_matrix(
                gts,
                predictions,
                labels=range(len(label_values)))

        print("Confusion matrix :")
        print(cm)

        print("---")

        # Compute global accuracy
        total = sum(sum(cm))
        accuracy = sum([cm[x][x] for x in range(len(cm))])
        accuracy *= 100 / float(total)
        print("{} pixels processed".format(total))
        print("Total accuracy : {}%".format(accuracy))

        print("---")

        # Compute F1 score
        F1Score = np.zeros(len(label_values))
        for i in range(len(label_values)):
            try:
                F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
            except:
                # Ignore exception if there is no element in class i for test set
                pass
        print("F1Score :")
        for l_id, score in enumerate(F1Score):
            print("{}: {}".format(label_values[l_id], score))

        print("---")

        # Compute kappa coefficient
        total = np.sum(cm)
        pa = np.trace(cm) / float(total)
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
        kappa = (pa - pe) / (1 - pe);
        print("Kappa: " + str(kappa))
        return accuracy

class ISPRS3Dataset(ISPRSDataset):

    def __init__(self, task, *args, **kwargs):
        super(ISPRS3Dataset, self).__init__(*args, **kwargs)
        self.ignore_label = 3 #ignore_label
        if task == 'seg':
            self.classes = ["buildings", "trees", "cars"]
            self.num_classes = len(self.classes)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    std=(0.229, 0.224, 0.225)
    mean=(0.485, 0.456, 0.406)

    # detection NEW
    dataset = ISPRSDataset('datasets/isprs/potsdam_od/train.txt', is_training=True,
                imgset="Potsdam", batch_size=1, size=320, task="det",
                           both_task=True, seg3=True)
    # detection
    for i in range(10,50,1):
        # data = dataset.__getitem__(i)
        # img = ((data['image'][0].numpy().transpose((1, 2, 0)) * std + mean)*255).astype(np.uint8)
        # img = img[..., -1::-1].copy()
        data = {"image": dataset.pull_image(i),
                "bboxes": torch.Tensor([dataset.pull_anno(i)]),
                "mask": torch.Tensor([dataset.pull_segment(i)]).unsqueeze(0)}
        img = data['image'].astype(np.uint8)
        plt.imshow(img)
        plt.figure()
        maps = {1: "building", 2: "tree", 3: "car"}
        for bbox in data['bboxes'][0]:
            cat = bbox[-1].cpu().numpy()
            bbox = bbox[:-1].cpu().numpy() # format xmin, ymin, xmax, ymax
            visualize_bbox(img, bbox, maps[int(cat)],
                w=img.shape[0], h=img.shape[1], color = (255, 0, 0)
            )
        plt.imshow(img)
        if 'mask' in data:
            mask = data['mask'][0]
            mask[mask==255] = mask[mask!=255].max()+1
            plt.figure()
            plt.imshow(mask.numpy().transpose((1, 2, 0)))
        plt.show()

    # segmentation
    dataset = ISPRSDataset('datasets/isprs/potsdam_od/train.txt', imgset="Potsdam",
                    is_training=True, batch_size=1, size=320, task="seg")
    for i in range(10):
        data = dataset.__getitem__(i)
        img = ((data['image'][0].numpy().transpose((1, 2, 0)) * std + mean)*255).astype(np.uint8)
        plt.imshow(img[...,-1::-1])
        plt.figure()
        mask = data['mask'][0]
        mask[mask==255] = mask[mask!=255].max()+1
        plt.imshow(mask.numpy().transpose((1, 2, 0)))
        plt.show()

    # detection OLD
    dataset = ISPRSDataset(path="datasets/isprs/potsdam_od/train.txt")
    # create a pseudo predicted boxes, last column is score
    # pred_bbox = [np.hstack((b[:,1:], b[:, 0, None] + 1)) for b in dataset.labels]
    pred_bbox = [np.hstack((b[:,1:3]-b[:,3:]/2, b[:,1:3]+b[:,3:]/2, 1+b[:,0,None]))
                 for b in dataset.labels]
    # pred_bbox[cl_indx][img_indx][box_indx]
    pred_bbox = [[], pred_bbox] # 1st entry = BG, 2nd= class1
    dataset.evaluate_detections(pred_bbox)

    for i in range(50, 100):
        data = dataset.__getitem__(i)
        print (data[1]) # bbox

        img = data[0].numpy().transpose((1, 2, 0)).copy().astype(np.uint8)
        for bbox in data[1]:
            cat = bbox[-1] # the first entry is empty
            bbox = bbox[:-1] # format xmin, ymin, xmax, ymax
            # bbox = torch.cat((bbox[:2] - bbox[2:] / 2, bbox[:2] + bbox[2:] / 2), 0)
            visualize_bbox(img, bbox, str(cat),
                           w=data[0].size(2), h=data[0].size(1), color = (255, 0, 0)
                           )
        plt.imshow(img)
        plt.show()
