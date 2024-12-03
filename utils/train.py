import os, os.path as osp
import sys

import argparse
import numpy as np
import torch
import torch.utils.data as data
from data import detection_collate_dict, DataPrefetcher, preproc_for_test
from utils import Detect


import wandb
from loguru import logger
fmt = "<green>{time:MM-DD HH:mm:ss}</green> | "\
        "<cyan>{file}</cyan>:<cyan>{line}</cyan> | "\
        "<level>{message}</level>"
logger.remove()
logger.add(sys.stderr, format=fmt)

def init_wandb(args):
    wandb.init(job_type=args.job_type, group=(None if args.group=="" else args.group))
    if wandb.run.dir is not None and osp.exists(wandb.run.dir):
        args.save_folder = wandb.run.dir
    wandb.config.update(args)

def parse_arguments():
    from datetime import datetime as dt
    expid = dt.now().strftime("%m%d.%H%M")

    parser = argparse.ArgumentParser(description='Pytorch Training')
    # wandb grouping
    parser.add_argument('--group', default=expid, type=str)
    parser.add_argument('--job_type', default="teacher", type=str)

    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--neck', default='fpn')
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--dataset', default='VOC',
                        choices=["VOC",  "VOCaug", "MXE", "ISPRS", "COCO", "MXS",
                                "Cityscapes", "MXE+Cityscapes", "MXE+MXS",
                                 "MXE+MXT", "CNES"
                                 ])
    parser.add_argument('--imgset', default='Main',
            choices=["Main", "Segmentation",            # VOC imgset
                    "Half", "Half2",                    # VOC imgset
                    "Quarter", "3Quarter",              # VOC imgset
                    "Eighth","7Eighth", "Ei2ghth",      # VOC imgset
                    "Eighth+Ei2ghth",                   # VOC imgset
                    "SegHf", "SegHs",                   # VOC imgset
                    "det", "det2", "seg", "seg2", "all",# MXE imgset
                    "det+seg", "det+seg2", "det2+seg",  # MXE imgset
                    "Potsdam", "Vaihingen",
                    "Potsdam+Vaihingen", "Vaihingen+Potsdam",
                    "P1+P2", "P2+P1", "P1+Vaihingen", "P2+Vaihingen",
                    "Eighth+Eighth",
                ])
    parser.add_argument('--save_folder', default='weights/')
    parser.add_argument('--match', default='iou', choices=['iou', 'mg'])
    parser.add_argument('--conf_loss', default='fc', choices=['gfc', 'fc'])
    parser.add_argument('--base_anchor_size', default=24.0, type=float)
    parser.add_argument('--size', default=320, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--warm_iter', default=500, type=int)
    parser.add_argument('--eval_epoch', default=0, type=int)
    parser.add_argument('--double_aug', action='store_true')
    parser.add_argument('--noBN', action='store_true')
    parser.add_argument('--note', default='', type=str)
    parser.add_argument('--load_weights', default='', type=str)
    parser.add_argument('--nepoch', default=70, type=int)
    parser.add_argument('--task', default="det", type=str,
                        choices=['det', 'seg', 'det+seg', 'det+det', 'seg+seg'])
    parser.add_argument('--task_weights', default="1.0", type=str)

    return parser

def preprocess_args(args):
    imgset = "+".join([l[:5] for l in args.imgset.split('+')])
    args.group = f"{args.group}-{args.task}:{args.dataset}".strip("-")
    args.job_type = f"{args.job_type[:3]}_{imgset}"

    init_wandb(args)

    # to keep wandb record concise, only update imgset after having init wandb
    if "+" in args.task and "+" not in args.task_weights:
        args.task_weights = "+".join([args.task_weights] * 2)
    if "+" in args.task and "+" not in args.dataset:
        args.dataset = "+".join([args.dataset] * 2)
    if "+" in args.task and "+" not in args.imgset:
        args.imgset = "+".join([args.imgset] * 2)
    if "MXE" in args.dataset: # det+seg to MXEdet+MXEseg
        args.imgset = "+".join([d+i if d=="MXE" else i for d,i in
                                zip(args.dataset.split("+"), args.imgset.split("+"))])

    assert ("+" in args.task) == ("+" in args.task_weights), \
        'The number of task weights is different from the number of tasks.'
    assert ("+" in args.task) == ("+" in args.imgset), \
        'The number of imgsets is different from the number of tasks.'
    assert ("+" in args.task) == ("+" in args.dataset),\
        'The number of datasets is different from the number of tasks'

def get_path(args, suffixe=""):
    mg = '_MG' if args.match=='mg' else ''
    BN = '_NOBatchNorm' if args.noBN else ''
    save_path = os.path.join(args.save_folder,
        f'{args.dataset}_retina_{args.neck}_{args.backbone}{BN}_size{args.size}_'\
        f"anchor{args.base_anchor_size}{mg}_{suffixe}.pth"
        )
    return save_path

def save_weights(model, args, suffixe=""):
    save_path = get_path(args, suffixe)
    print('Saving to {}'.format(save_path))
    torch.save(model.state_dict(), save_path)
    if 'WANDB_EXP' in os.environ:
        open(os.environ['WANDB_EXP'], 'w').write(save_path)
    return save_path

def test_model(args, model, priors, valid_sets, all_thresh=True,
               per_class=False, output_coco=False
               ):
    logger.info('Evaluating...')
    model.eval()
    num_images = len(valid_sets)
    all_boxes = [
        [ None for _ in range(num_images) ] for _ in range(valid_sets.num_classes)
    ]
    for i in range(num_images):

        # prepare image to detect
        img = valid_sets.pull_image(i)
        scale = torch.Tensor(
                [ img.shape[1], img.shape[0], img.shape[1], img.shape[0] ]
            ).cuda() if "VEDAI" not in args.dataset and "ISPRS" not in args.dataset else 1
        x = torch.from_numpy(
            preproc_for_test(img, args.size)
        ).unsqueeze(0).cuda()
        # model inference
        with torch.no_grad():
            out = model.forward_test(x)

        priors_ = priors

        (boxes, scores) = Detect(
                out, priors_, scale, eval_thresh=0.05, nms_thresh=0.5,
        )
        for j in range(1, valid_sets.num_classes):
            inds = np.where(scores[:, j-1] > 0.05)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
            else:
                all_boxes[j][i] = np.hstack(
                        (boxes[inds], scores[inds, j-1:j])
                    ).astype(np.float32)

    return valid_sets.evaluate_detections(all_boxes, all_thresh=all_thresh,
                                          per_class=per_class,
                                          output_coco=output_coco) * 100

def test_segmentation(model, testset, wgt=True, per_class=False, vis=False):
    logger.info('Evaluating segmentation...')
    model.eval()
    num_images = len(testset)
    preds = []
    gts = []
    batch_size = 5 # if wgt else 1
    loader = data.DataLoader(testset, batch_size=batch_size, shuffle=False,
            num_workers=5, collate_fn=detection_collate_dict)
    prefetcher = DataPrefetcher(loader)
    for _ in range(0, num_images, batch_size):
        with torch.no_grad():
            x, _, seg = prefetcher.next()
            x = x.cuda()
            out = model.forward_test(x, task="seg")
            pred = out['seg'].cpu().numpy().argmax(axis=1)
            preds.append(pred)
            if wgt:
                gts.append(seg.cpu().numpy().astype(np.int64))
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0).squeeze() if wgt else []
    logger.info (f"Total: {num_images} | Len of preds: {preds.shape[0]}")

    if vis:
        from data import VOCDetection
        #NOTE:must go before eval which changes 255 to 21
        VOCDetection.vis_seg(gts, preds, vis)
    res = testset.evaluate_segmentation(preds, gts, per_class=per_class)

    for k in sorted(res.keys()):
        if len(k) < 7:
            print (k, f"{res[k]:.4f}")
    return res['mIoU']

def load_dataset(args):
    args.task_weights = dict(zip(args.task.split("+"),
        [float(tw) for tw in args.task_weights.split("+")]))

    train_sets = {}
    valid_sets = {}
    for i, (task, dataset, imgset) in enumerate(zip(args.task.split('+'),
                                                args.dataset.split('+'),
                                                args.imgset.split('+'))):
        task_ = f"{i}{task}"
        if dataset == 'VOC':
            from data import VOCDetection
            train_names = [('2007', 'trainval'), ('2012', 'trainval')]
            train_sets[task_] = VOCDetection(train_names, args.size,
                                             imgset=args.imgset,
                    cache=True,
                    double_aug=args.double_aug, is_training=True, task=task)
            valid_sets[task_] = VOCDetection([('2007', 'test')], args.size,
                                             cache=False)
        elif dataset == 'VOCaug':
            from data import VOCDetection
            train_names = [('2012_segaug', 'train')]
            train_sets[task_] = VOCDetection(train_names, args.size,
                                             imgset=imgset, cache=True,
                                             double_aug=args.double_aug,
                                             is_training=True, task=task)
            valid_sets[task_] = VOCDetection([('2012', 'val')], args.size,
                                             cache=False, imgset=imgset,
                                             task=task)
        elif dataset == 'MXE':
            from data import VOCDetection
            train_names = [('2007', 'train'), ('2012', 'train')]
            train_sets[task_]= VOCDetection(
                train_names, args.size, imgset=imgset, cache=True,
                double_aug=args.double_aug, is_training=True, task=task,
                both_task=("both_task" in args.note),
                pseudo_seg=("pse-msk" in args.note)
            )
            valid_sets[task_] = VOCDetection([('2012', 'val')], args.size,
                    imgset=imgset, cache=False, task=task)
        elif dataset == "Cityscapes":
            from data import Cityscapes
            train_sets[task_] = Cityscapes(is_training=True)
            valid_sets[task_] = Cityscapes(is_training=False)
        elif dataset == 'COCO':
            from data import COCODetection
            subset=imgset if imgset!="Main" else None
            train_sets[task_] = COCODetection([('2017', 'train')], args.size,
                                             cache=False, is_training=True,
                                             subset=subset, task=task,
                                            pseudo_seg=("pse-msk" in args.note),
                                            )
            valid_sets[task_] = COCODetection([('2017', 'val')], args.size,
                                              is_training=False,
                                              cache=False, task=task
                                            )
        elif dataset == "MXS":
            imgset = "MXE" + imgset
            from data import MXSDetection
            train_names = [('2007', 'train'), ('2012', 'train')]
            train_sets[task_]= MXSDetection(train_names, args.size,
                                            imgset=imgset,
                                            double_aug=args.double_aug,
                                            is_training=True, task=task)
            valid_sets[task_] = MXSDetection([('2012', 'val')], args.size,
                                             imgset=imgset, cache=False,
                                             task=task)
        elif dataset == "MXT":
            imgset = "MXE" + imgset
            from data import MXTDetection
            train_names = [('2007', 'train'), ('2012', 'train')]
            train_sets[task_]= MXTDetection(train_names, args.size,
                                            imgset=imgset,
                                            double_aug=args.double_aug,
                                            is_training=True, task=task)
            valid_sets[task_] = MXTDetection([('2012', 'val')], args.size,
                                             imgset=imgset, cache=False,
                                             task=task)
        else:
            raise NotImplementedError('Unkown dataset {}!'.format(dataset))

    [logger.info(f"Training set for task {t[1:]}: {d} {len(train_sets[t])} imgs")
            for d, t in zip(args.dataset.split("+"), sorted(train_sets.keys()))]
    [logger.info(f"Validation set for task {t[1:]}: {d} {len(valid_sets[t])} imgs")
            for d, t in zip(args.dataset.split("+"), sorted(valid_sets.keys()))]

    epoch_size = max([len(train_sets[s]) for s in train_sets]) // args.batch_size
    max_iter = args.nepoch * epoch_size
    num_classes = {t[1:]: train_sets[t].num_classes for t in train_sets}

    return train_sets, valid_sets, epoch_size, max_iter, num_classes
