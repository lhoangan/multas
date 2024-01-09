# MulTaS: MULti-TAsk Self-training of object detection and semantic segmentation

The source code has been used for our papers in the
[ICCV 2023 workshop](https://openaccess.thecvf.com/content/ICCV2023W/LIMIT/html/Le_Self-Training_and_Multi-Task_Learning_for_Limited_Data_Evaluation_Study_on_ICCVW_2023_paper.html)
and [BMVC 2023](https://proceedings.bmvc2023.org/870/).
If you are involving the source code in your research, please consider citing our papers:

```
@InProceedings{Le_2023_ICCV,
    author    = {L\^e, Ho\`ang-\^An and Pham, Minh-Tan},
    title     = {Self-Training and Multi-Task Learning for Limited Data: Evaluation Study on Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2023},
    pages     = {1003-1009}
}

@inproceedings{Le_2023_BMVC,
    author    = {L\^e, Ho\`ang-\^An and Pham, Minh-Tan},
    title     = {Data exploitation: multi-task learning of object detection and semantic segmentation on partially annotated data},
    booktitle = {34th British Machine Vision Conference 2023, {BMVC} 2023, Aberdeen, UK, November 20-24, 2023},
    publisher = {BMVA},
    year      = {2023}
}
```

## Installation

### Dependencies

We use the anaconda for managing environment, all the packages and installation can be found in the `environments.yml` and installed by running the following command.

```conda env create --name envname --file=environments.yml```

### Dataset

#### Pascal VOC

Download the Pascal [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
and [VOC212](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) datasets and place
them in the `datasets` directories following the follow structure.

```
datasets/VOCdevkit
|-- VOC2007
|   |-- Annotations
|   |-- ImageSets
|   |-- JPEGImages
|   |-- SegmentationClass
|   `-- SegmentationObject
|-- VOC2012
|   |-- Annotations
|   |-- ImageSets
|   |-- JPEGImages
|   |-- SegmentationClass
|   `-- SegmentationObject
```

The `ImageSets` directories contain the splits used in the paper and are
provided at `datasets/imgsetVOC`, you can create a symlink to it by

```
cd multas/datasets
ln -s imgsetVOC VOCdevkit
```

Scripts are provided in `data/scripts` to automate the process and can be run by
the following command

```
./data/scripts/VOC2007.sh datasets/
```

#### Augmented VOC (SBD) dataset

Download the [SBD dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)
and read the mat files using `scipy.io.loadmat` on python. The segmentation can
be accessed via `mat["GTcls"][0]["Segmentation"][0]`.

## Training

### Self-Training on Object Detection

**Training teacher network**

```
python train.py --seed 0 --size 320 --batch_size 5 --lr 0.01 --eval_epoch 1\
                --double_aug --match mg --conf_loss gfc \
                --backbone resnet50 --neck pafpn \
                --dataset VOC --imgset Half
```

where `imgset` is in `Half`, `Quarter`, or `Eighth`

**Training student network**

```
python distil.py    --seed 0 --size 320 --batch_size 10 --lr 0.01 --eval_epoch 1\
                    --double_aug --match iou --conf_loss gfc \
                    --backbone resnet18 --neck fpn \
                    --dataset VOC  --imgset Half \
                    --teacher_backbone resnet50 --teacher_neck pafpn \
                    --kd hard+pdf --tdet_weights [path/to/teacher/weights.pth]
```

where

- `imgset` is `Main`, `Half`, `Quarter`, `Eighth` for overlapping training sets or
`Half2`, `3Quarter`, `7Eighth` for disjoint training sets
- `kd` is `hard` (supervised training), `soft`, `soft+mse`, `soft+pdf`, `soft+defeat`
(self-training training). As such the `Main` split should only be used with
`soft`-based distillation.

### Multi-task learning

Coming soon

### Reference

The repo is based on this [repo](https://github.com/zhanghengdev/MutualGuide) by @zhanghengdev.

