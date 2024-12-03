import torch
import numpy as np
from .voc0712 import VOCDetection, AnnotationTransform, VOC_CLASSES
from PIL import Image

# Multi-Exclusive Category dataset
MXS_CLASSES = ('__background__', # always index 0
    'transportation', 'animal', 'furniture', 'person'
    )

# for reference only
index_map = {0: 0,
         1: 1, 2: 1,  4: 1,  6: 1,  7: 1, 14: 1, 19: 1, # transportation
         3: 2, 8: 2, 10: 2, 12: 2, 13: 2, 17: 2, # animal
         5: 3, 9: 3, 11: 3, 16: 3, 18: 3, 20: 3, # furniture
        15: 4
    } # person

label_map = {'__background__': "__background__",
        'aeroplane': "transportation", 'bicycle': "transportation",
        'boat': "transportation", 'bus': "transportation",
        'car': "transportation", 'motorbike': "transportation",
        'train': "transportation",

        'cat': "animal", 'bird': "animal", 'cow': "animal", 'dog': "animal",
        'horse': "animal", 'sheep': "animal",

        'bottle': "furniture", 'chair': "furniture", 'diningtable': "furniture",
        'sofa': "furniture", 'pottedplant': "furniture", 'tvmonitor': "furniture",

        'person': "person",
    }


class MXSDetection(VOCDetection):

    def __init__(self, *args, **kwargs):
        super(MXSDetection, self).__init__(*args, **kwargs)
        self.class_names = MXS_CLASSES[1:]
        # convert VOC to MXS to index
        voc2ind = {k: MXS_CLASSES.index(label_map[k]) for k in label_map}
        self.index_map = {VOC_CLASSES.index(k): voc2ind[k] for k in VOC_CLASSES}
        self.target_transform = AnnotationTransform(classes=MXS_CLASSES, cl2ind=voc2ind)

    def pull_classes(
        self,
    ) -> tuple:

        return MXS_CLASSES

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

        data = super(MXSDetection, self).__getitem__(index)

        self.remove_empty(data)
        return data

    def remove_empty(self, data):

        # return
        common_class = [4]
        bboxes = []
        bboxes = [k for k in data['bboxes'][0] if k[-1] in common_class]
        for k in bboxes:
            k[-1] = 15
        data['bboxes'][0] = torch.stack(bboxes) if len(bboxes) > 0 else torch.Tensor([0.0])
