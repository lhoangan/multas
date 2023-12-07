import torch
import numpy as np
from .voc0712 import VOCDetection, AnnotationTransform, VOC_CLASSES
from PIL import Image

# Multi-Exclusive Category dataset
MXT_CLASSES = ('__background__', # always index 0
    'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
    'animal', 'furniture', 'person'
    )

label_map = {'__background__': "__background__",
        'aeroplane':'aeroplane',
        'bicycle'  :'bicycle',
        'boat'     :'boat',
        'bus'      :'bus',
        'car'      :'car',
        'motorbike':'motorbike',
        'train'    : 'train',

        'cat': "animal", 'bird': "animal", 'cow': "animal", 'dog': "animal",
        'horse': "animal", 'sheep': "animal",

        'bottle': "furniture", 'chair': "furniture", 'diningtable': "furniture",
        'sofa': "furniture", 'pottedplant': "furniture", 'tvmonitor': "furniture",

        'person': "person",
    }


class MXTDetection(VOCDetection):

    def __init__(self, *args, **kwargs):
        super(MXTDetection, self).__init__(*args, **kwargs)
        self.class_names = MXT_CLASSES[1:]
        # convert VOC to MXT to index
        voc2ind = {k: MXT_CLASSES.index(label_map[k]) for k in label_map}
        self.index_map = {VOC_CLASSES.index(k): voc2ind[k] for k in VOC_CLASSES}
        self.target_transform = AnnotationTransform(classes=MXT_CLASSES, cl2ind=voc2ind)

    def pull_classes(
        self,
    ) -> tuple:

        return MXT_CLASSES

    def pull_segment(
        self,
        index: int,
        resize: bool = False,
    ) -> np.ndarray:
        img_id = self.ids[index]
        image = Image.open(self._seggtpath % img_id)
        if resize:
            image = image.resize((self.size, self.size), resample=Image.NEAREST)
        image = np.array(image)
        for i in self.index_map:
            image[image==i] = self.index_map[i]
        return image

    def __getitem__(
        self,
        index,
    ) -> list:

        data = super(MXTDetection, self).__getitem__(index)

        self.remove_empty(data)
        return data

    def remove_empty(self, data):

        # defining common classes in this dataset space -> VOC
        common_class = {1: 1, 2: 2, 3: 4, 4: 6, 5: 7, 6: 14, 7: 19, 10: 15}
        bboxes = []
        # [print(k) for k in data['bboxes'][0]]
        bboxes = [k for k in data['bboxes'][0] if int(k[-1]) in common_class]
        # print ("BEFORE: ", bboxes)
        for k in bboxes:
            k[-1] = common_class[int(k[-1])]
        # print ("AFTER: ", bboxes)
        data['bboxes'][0] = torch.stack(bboxes) if len(bboxes) > 0 else torch.Tensor([0.0])
