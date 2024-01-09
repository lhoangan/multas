import os, os.path as osp
import sys

sys.path.insert(0, "/share/home/berg/cocostuffapi/PythonAPI")
sys.path.insert(0, "/share/home/berg/cocostuffapi/PythonAPI/pycocotools")

assert os.path.exists("/share/home/berg/cocostuffapi/PythonAPI")

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import geoopt
from data import detection_collate_dict, DataPrefetcher, preproc_for_test
from utils import Detect
from utils import PriorBox
from utils import Timer, ModelEMA
from utils import adjust_learning_rate, tencent_trick
from utils import DetectionLoss
from utils import SegLoss


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
                                 "MXE+MXT",
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

def test_model(args, model, priors, valid_sets, all_thresh=True, per_class=False):
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
        # from data.voc0712 import visualize_bbox
        # from PIL import Image
        # # std = (.229, .224, .225)
        # # mean = (.485, .456, .406)
        # # myimg = ((x.cpu().numpy().squeeze().transpose((1,2,0))*std+mean)*255)
        # target = valid_sets.pull_anno(i)
        # myimg = img
        # myimg = myimg.astype(np.uint8).copy()
        for j in range(1, valid_sets.num_classes):
            inds = np.where(scores[:, j-1] > 0.05)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
            else:
                all_boxes[j][i] = np.hstack(
                        (boxes[inds], scores[inds, j-1:j])
                    ).astype(np.float32)
            # argmax = np.argmax(scores[inds, j-1:j])
            # for b,s in zip([boxes[inds][argmax]], [scores[inds, j-1:j][argmax]]):
            # for b in target:
            #     visualize_bbox(myimg, b[:4], str(b[4]), myimg.shape[1], myimg.shape[0],
            #                    color=(0, 0, 255))
            # for b,s in zip(boxes[inds][:], scores[inds, j-1:j][:]):
            # for b,s in zip(boxes[inds][:10], scores[inds, j-1:j][:10]):
            #     b1 = b.copy()
            #     # b[0], b[2] = b1[0] / myimg.shape[1], b1[2] / myimg.shape[1]
            #     # b[1], b[3] = b1[1] / myimg.shape[0], b1[3] / myimg.shape[0]
            #     visualize_bbox(myimg, b, str(j-1), myimg.shape[1], myimg.shape[0],
            #                    color=(255, 0, 0))
        # import matplotlib.pyplot as plt
        # plt.imshow(myimg), plt.show()
        # Image.fromarray(myimg).save(f'vis_debug_{i:03d}.png')

    return valid_sets.evaluate_detections(all_boxes, all_thresh=all_thresh,
                                          per_class=per_class) * 100

def test_segmentation(model, testset, wgt=True, per_class=False):
    logger.info('Evaluating segmentation...')
    model.eval()
    # model.backbone.apply(deactivate_batchnorm)
    num_images = len(testset)
    preds = []
    gts = []
    batch_size = 5 # if wgt else 1
    loader = data.DataLoader(testset, batch_size=batch_size, shuffle=False,
            num_workers=5, collate_fn=detection_collate_dict)
    prefetcher = DataPrefetcher(loader)
    for i in range(0, num_images, batch_size):
        with torch.no_grad():
            x, _, seg = prefetcher.next()
            # if wgt: # not COCO
            #     x, _, seg = prefetcher.next()
            # else: # is COCO
            #     x = torch.from_numpy(testset.pull_image(i)).unsqueeze(0)
            x = x.cuda()
            out = model.forward_test(x, task="seg")
            pred = out['seg'].cpu().numpy().argmax(axis=1)#.squeeze()
            preds.append(pred)
            if wgt:
                gts.append(seg.cpu().numpy().astype(np.int64))#.squeeze())
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0).squeeze() if wgt else []
    logger.info (f"Total: {num_images} | Len of preds: {preds.shape[0]}")

    # breakpoint()
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
                                                args.imgset.split('+')
                                    )):
        task_ = f"{i}{task}"
        if dataset == 'VOC':
            from data import VOCDetection
            train_names = [('2007', 'trainval'), ('2012', 'trainval')]
            train_sets[task_] = VOCDetection(train_names, args.size, imgset=args.imgset,
                    cache=True,
                    double_aug=args.double_aug, is_training=True, task=task)
            valid_sets[task_] = VOCDetection([('2007', 'test')], args.size, cache=False)
            # valid_sets[task] = VOCDetection([('2012', 'val')], args.size,
            #                                 imgset="MXEdet", cache=False)
        elif dataset == 'VOCaug':
            from data import VOCDetection
            train_names = [('2012_segaug', 'train')]
            train_sets[task_] = VOCDetection(train_names, args.size, imgset=imgset,
                    cache=True,
                    double_aug=args.double_aug, is_training=True, task=task)
            valid_sets[task_] = VOCDetection([('2012', 'val')], args.size, cache=False,
                    imgset=imgset, task=task)
        elif dataset == 'MXE':
            from data import VOCDetection
            train_names = [('2007', 'train'), ('2012', 'train')]
            train_sets[task_]= VOCDetection(train_names, args.size, imgset=imgset,
                    cache=True,
                    double_aug=args.double_aug, is_training=True, task=task,
                                            both_task=("both_task" in args.note)
                    )
            valid_sets[task_] = VOCDetection([('2012', 'val')], args.size,
                    imgset=imgset, cache=False, task=task)
        elif dataset == "Cityscapes":
            from data import Cityscapes
            train_sets[task_] = Cityscapes(is_training=True)
            valid_sets[task_] = Cityscapes(is_training=False)
        elif dataset == 'ISPRS':
            from data import ISPRSDataset
            train_sets[task_] = ISPRSDataset('datasets/isprs/potsdam_od/train.txt',
                        imgset=imgset, batch_size=1, img_size=args.size, task=task,
                        is_training=True, both_task=("both_task" in args.note),
                        seg3="seg3" in args.note)
            valid_sets[task_] = ISPRSDataset('datasets/isprs/potsdam_od/test.txt',
                    imgset=imgset, batch_size=1, img_size=args.size, task=task,
                    seg3="seg3" in args.note)
        elif dataset == 'COCO':
            from data import COCODetection
            subset=imgset if imgset!="Main" else None
            train_sets[task_] = COCODetection([('2017', 'train')], args.size,
                                             cache=False, is_training=True,
                                             subset=subset, task=task
                                            )
            valid_sets[task_] = COCODetection([('2017', 'val')], args.size,
                                              is_training=False,
                                              cache=False, task=task
                                            )
            args.nepoch = 140 # TODO currently hardcoded for COCO
        elif dataset == "MXS":
            imgset = "MXE" + imgset
            from data import MXSDetection
            train_names = [('2007', 'train'), ('2012', 'train')]
            train_sets[task_]= MXSDetection(train_names, args.size, imgset=imgset,
                    double_aug=args.double_aug, is_training=True, task=task
                    )
            valid_sets[task_] = MXSDetection([('2012', 'val')], args.size,
                    imgset=imgset, cache=False, task=task)
        elif dataset == "MXT":
            imgset = "MXE" + imgset
            from data import MXTDetection
            train_names = [('2007', 'train'), ('2012', 'train')]
            train_sets[task_]= MXTDetection(train_names, args.size, imgset=imgset,
                    double_aug=args.double_aug, is_training=True, task=task
                    )
            valid_sets[task_] = MXTDetection([('2012', 'val')], args.size,
                    imgset=imgset, cache=False, task=task)
        else:
            raise NotImplementedError('Unkown dataset {}!'.format(dataset))

    [logger.info(f"Training set for task {t[1:]}:   {d} {len(train_sets[t])} images")
            for d, t in zip(args.dataset.split("+"), sorted(train_sets.keys()))]
    [logger.info(f"Validation set for task {t[1:]}: {d} {len(valid_sets[t])} images")
            for d, t in zip(args.dataset.split("+"), sorted(valid_sets.keys()))]

    epoch_size = max([len(train_sets[s]) for s in train_sets]) // args.batch_size
    max_iter = args.nepoch * epoch_size
    num_classes = {t[1:]: train_sets[t].num_classes for t in train_sets}

    return train_sets, valid_sets, epoch_size, max_iter, num_classes

if __name__ == '__main__':

    parser = parse_arguments()
    args = parser.parse_args()
    preprocess_args(args)
    logger.info("Running with arguments: \n"+
        "\n".join([f"{k: >8}:{v}" for k, v in sorted(vars(args).items())]))

    ### For Reproducibility ###
    if args.seed is not None:
        logger.info('Fixing seed to {}'.format(args.seed))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        cudnn.benchmark = False
        cudnn.deterministic = True
        cudnn.enabled = True

    logger.info('Loading Dataset...')
    train_sets, valid_sets, epoch_size, max_iter, num_classes = load_dataset(args)

    logger.info('Loading Optimizer & Network...')
    from models.teacher_detector import Detector
    model = Detector(args.size, num_classes, args.backbone, args.neck,
                     task=args.task, noBN=args.noBN
        ).cuda()

    eval_det = (lambda *a, **kwa: { k: test_model(*a, **kwa,
                                                  valid_sets=valid_sets[k])
        for k in valid_sets if 'det' in k and args.task_weights['det'] > 0 })
    eval_seg = (lambda *a, **kwa: { k: test_segmentation(*a, **kwa,
                                                         testset=valid_sets[k])
        for k in valid_sets if 'seg' in k and args.task_weights['seg'] > 0 })

    if args.load_weights:
        state_dict = torch.load(args.load_weights)
        state_dict = state_dict['model'] if 'model' in state_dict else state_dict
        # # removing keys is only applicable for single task finetuning
        # rmkeys = []
        # if args.task == 'seg': # remove all segmentation head
        #     rmkeys = [k for k in state_dict if 'seghead.' in k]
        # elif args.task == 'det': # remove all detection head
        #     rmkeys = [k for k in state_dict if 'conf.' in k or 'loc.' in k]
        # [state_dict.pop(k) for k in rmkeys]

        model.load_state_dict(state_dict, strict=True)

        priors = PriorBox(args.base_anchor_size, args.size,
                          base_size=args.size).cuda()
        # precision = {"det": 0, "seg": 0}
        # precision["det"] = test_model(args, model, priors.clone().detach(),
        #                     valid_sets['det']) if 'det' in valid_sets else 0
        # logger.info("det mAP={}".format(precision['det']))
        # precision["seg"] = test_segmentation(model, valid_sets['seg']) \
        #         if 'seg' in valid_sets else 0
        # logger.info('seg mAP={}'.format(precision['seg']))
        # print({k: test_model(args, model, priors.clone().detach(), valid_sets[k])
        #         for k in valid_sets if 'det' in k and args.task_weights['det'] > 0})
        # print({k: test_segmentation(model, valid_sets[k], wgt="COCO" not in args.dataset)
        #         for k in valid_sets if 'seg' in k and args.task_weights['seg'] > 0})
        print(eval_det(args, model, priors.clone().detach()))
        print(eval_seg(model, wgt=("COCO" not in args.dataset)))

    logger.info('Loading teacher Network...')

    params_to_train = tencent_trick(model)


    classif_block_type = os.getenv("CLASSIF_BLOCK_TYPE", "conv1x1") # "horospherical"
    if classif_block_type in ("horospherical", "hyperbolic"):
        optimizer = geoopt.optim.RiemannianSGD(
                params_to_train, lr=args.lr,
                momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = optim.SGD(params_to_train, lr=args.lr,
                  momentum=0.9, weight_decay=0.0005)
    scaler = torch.cuda.amp.GradScaler()

    ema_model = ModelEMA(model)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Total trainable param is : {:e}'.format(num_param))

    logger.info('Preparing Criterion & AnchorBoxes...')
    criterion = DetectionLoss(
            mutual_guide=(args.match=='mg'),
            use_focal=(args.conf_loss=='fc'),
        )

    if 'seg' in '_'.join(train_sets):
        k = [key for key in train_sets if 'seg' in key][0]
        criterion_seg = SegLoss(ignore_index=train_sets[k].ignore_label)

    priors = PriorBox(args.base_anchor_size, args.size, base_size=args.size).cuda()

    os.makedirs(args.save_folder, exist_ok=True)
    epoch = 0
    mAPs = {}
    best_epoch = 0
    longest_ep = 5
    precision = {}
    best_maps = {}
    timer = Timer()
    args.eval_epoch = int(max_iter/epoch_size/1.75) if args.eval_epoch == 1 else args.eval_epoch
    logger.info(f"EPOCH SIZE: {epoch_size}")
    done = False

    val = lambda x: np.array([x[k] * args.task_weights[k[1:]] for k in x]).sum()

    for iteration in range(max_iter):
        results = {}
        if iteration % epoch_size == 0:

            log_info = {"Loss/loc": [], "Loss/conf": [], "Loss/contrast": [],
                    "Loss/seg": [], "Loss": [], "LR": [],
                }

            epoch += 1
            # create batch iterator
            # rand_loader = data.DataLoader(
            #     train_sets, args.batch_size, shuffle=True, num_workers=4, collate_fn=detection_collate
            # )
            # prefetcher = DataPrefetcher(rand_loader)
            prefetcher = {d: DataPrefetcher(data.DataLoader(
                train_sets[d], args.batch_size, shuffle=True, num_workers=4,
                collate_fn=detection_collate_dict
            )) for d in train_sets}
            # prefetcher = {d: DataPrefetcher(rand_loader[d]) for d in rand_loader}
            model.train()

            # if iteration > 0:
            #     ema_model.update(model)
            if args.eval_epoch and epoch > args.eval_epoch:
                priors = PriorBox(args.base_anchor_size, args.size, base_size=args.size).cuda()
                # precision["det"] = test_model(args, ema_model.ema, priors.clone().detach(),
                #                               valid_sets['det']) if 'det' in valid_sets else 0
                # logger.info("det mAP={}".format(precision['det']))

                # precision["seg"] = test_segmentation(ema_model.ema, valid_sets['seg']) \
                #         if 'seg' in valid_sets else 0
                # logger.info('seg mAP={}'.format(precision['seg']))
                # precision.update(
                #     {k: test_model()
                #             for k in valid_sets if 'det' in k and args.task_weights['det'] > 0})
                # precision.update({k: test_segmentation(ema_model.ema, valid_sets[k])
                #             for k in valid_sets if 'seg' in k and args.task_weights['seg'] > 0})
                path = save_weights(ema_model.ema, args, '')
                precision.update(eval_det(args, ema_model.ema, priors.clone().detach()))
                precision.update(eval_seg(ema_model.ema, wgt=("COCO" not in args.dataset)))
                [logger.info(f"{k} mAP={precision[k]:.2f}") for k in precision]

                if val(precision) > val(best_maps) + 7e-2:
                    # path = save_weights(ema_model.ema, args,
                    #         'ep{}_mAPd{:.4f}_mAPs{:.4f}'.format(epoch-1,
                    #             precision['det'], precision['seg']))
                    path = save_weights(ema_model.ema, args, f'ep{epoch-1}_'+
                        '_'.join([f'{k}{precision[k]:.2f}' for k in sorted(precision.keys())])
                        )
                            # 'ep{}_mAPd{:.4f}_mAPs{:.4f}'.format(epoch-1,
                    # Update best precision
                    mAPs[path] = val(precision)
                    best_maps = precision.copy()
                    if best_epoch != 0 and longest_ep < epoch - best_epoch:
                        longest_ep = epoch - best_epoch
                        logger.info(f"Update max streak to: {longest_ep}")
                    best_epoch = epoch

                if epoch >= 50 and epoch-best_epoch > 1.5 * longest_ep:
                    logger.warning(
                        f"MAP has not been improved for {int(longest_ep*1.5)} ep, "
                        f"since Epoch {best_epoch}. Now stopping..."
                    )
                    break

                ema_model.ema.train()

        # traning iteration
        results = {'loss_l': torch.tensor(0.0).cuda(),
                   'loss_c': torch.tensor(0.0).cuda(),
                   'loss_s':torch.tensor(0.0).cuda()}
        loss = torch.tensor(0.0).cuda()
        timer.tic()
        out = dict()
        for task in prefetcher:
            adjust_learning_rate(optimizer, args.lr, iteration, args.warm_iter, max_iter)
            seggt = None
            output = prefetcher[task].next()

            if output[0] is None or len(output[0]) < args.batch_size:
                prefetcher[task] = DataPrefetcher(data.DataLoader(
                    train_sets[task], args.batch_size, shuffle=True, num_workers=5,
                    collate_fn=detection_collate_dict
                ))
                output = prefetcher[task].next()
            if "seg" in task:
                (images, targets, seggt) = output
                new_size = args.size
            else:
                (images, targets) = output

                if iteration >= 0.8*max_iter:
                    new_size = args.size
                elif args.size == 320:
                    new_size = 64 * (5 + random.choice([-1,0,1]))
                elif args.size == 512:
                    new_size = 128 * (4 + random.choice([-1,0,1]))
                images = nn.functional.interpolate(
                        images, size=(new_size, new_size), mode="bilinear", align_corners=False
                    )
            priors = PriorBox(args.base_anchor_size, new_size, base_size=args.size).cuda()

            segs = None
            overlap = None
            conf = None
            with torch.cuda.amp.autocast():
                if "skip_empty" in args.note:
                    if args.task_weights[task[1:]] == 0:
                        continue

                out[task] = model.forward_test(images, task)

                if "det" in task:
                    result = criterion(out[task], priors, targets, seggt=seggt,
                        seg_overlap=overlap, seg_conf=conf
                    )
                    results['loss_l'] += result['loss_l'] * args.task_weights['det']
                    results['loss_c'] += result['loss_c'] * args.task_weights['det']
                elif "seg" in task:
                    result = criterion_seg(out[task], seggt)
                    results['loss_s'] += result['loss_seg'] * args.task_weights['seg']
                else:
                    raise NotImplementedError('Unkown task {}!'.format(task))

        loss = results['loss_l'] + results['loss_c'] + results['loss_s']
        log_info["Loss/loc"].append(results['loss_l'].item() if 'loss_l' in results else 0)
        log_info["Loss/conf"].append(results['loss_c'].item() if 'loss_c'in results else 0)
        log_info["Loss/seg"].append(results['loss_s'].item() if 'loss_s' in results else 0)
        log_info["Loss"].append(loss.item())
        log_info["LR"].append(optimizer.param_groups[0]['lr'])
        wandb.log({k: np.average(log_info[k]) for k in log_info}, step=epoch)
        if any(best_maps.values()):
            ks = []
            for t in sorted(best_maps.keys()):
                k = t[1:] if t[1:] not in ks else t
                wandb.log({f"{k}/MAP": best_maps[t]}, step=epoch)
                ks.append(k)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ema_model.update(model)
        load_time = timer.toc()

        # logging
        if iteration % 100 == 0:
            logger.info(
                    'ep {} it {}/{} lr {:.4f}, l:loc {:.2f} l:cls {:.2f} '\
                    'l:seg {:.2f} (LOSS {:.3f}) '\
                    #'d:best {:.2f} s:best {:.2f}, '\
                    'best:{} '\
                    'time {:.2f}s eta {:.2f}h'.format(
                epoch, iteration, max_iter, optimizer.param_groups[0]['lr'],
                results['loss_l'].item() if 'loss_l' in results else 0,
                results['loss_c'].item() if 'loss_c' in results else 0,
                results['loss_s'].item() if 'loss_s' in results else 0,
                loss.item(),
                #best_maps['det'], best_maps['seg'],
                ' '.join([f"{k} {best_maps[k]:.2f}"for k in best_maps]),
                load_time, load_time * (max_iter - iteration) / 3600,
                ))
            timer.clear()

    priors = PriorBox(args.base_anchor_size, args.size, base_size=args.size).cuda()
    # precision['det'] = test_model(args, ema_model.ema, priors.clone().detach(),
    #         valid_sets['det']) if 'det' in valid_sets else 0
    # logger.info('det mAP={}'.format(precision['det']))
    # precision['seg'] = test_segmentation(ema_model.ema, valid_sets['seg']) \
    #         if 'seg' in valid_sets else 0
    # logger.info('seg mAP={}'.format(precision['seg']))
    # precision.update(
    #     {k: test_model(args, ema_model.ema, priors.clone().detach(), valid_sets[k])
    #             for k in valid_sets if 'det' in k and args.task_weights['det'] > 0})
    # precision.update({k: test_segmentation(ema_model.ema, valid_sets[k])
    #             for k in valid_sets if 'seg' in k and args.task_weights['seg'] > 0})
    precision.update(eval_det(args, ema_model.ema, priors.clone().detach()))
    precision.update(eval_seg(ema_model.ema, wgt=("COCO" not in args.dataset)))
    [logger.info(f"{k} mAP={precision[k]:.2f}") for k in precision]

    if val(precision) > val(best_maps) + 7e-2 or len(mAPs) == 0:
        # path = save_weights(ema_model.ema, args,
        #     'ep{}_mAPd{:.4f}_mAPs{:.4f}'.format(epoch-1,
        #         precision['det'], precision['seg']))
        path = save_weights(ema_model.ema, args, f'ep{epoch-1}_'+
            '_'.join([f'{k}{precision[k]:.2f}' for k in sorted(precision.keys())]))

        mAPs[path] = val(precision)
        best_maps = precision.copy()
        ks = []
        for t in sorted(best_maps.keys()):
            k = t[1:] if t[1:] not in ks else t
            wandb.log({f"{k}/MAP": best_maps[t]}, step=epoch)
            ks.append(k)

        # exit program correctly
        if len(mAPs) == 0:
            exit()

    k = max(zip(mAPs.values(), mAPs.keys()))[1]
    for v in mAPs:
        if v != k:
            logger.info(f"Removing: {v}")
            if 'WANDB_EXP' in os.environ:
                assert v != open(os.environ['WANDB_EXP']).read()
            os.remove(v)

    logger.info(f"Checksum: loading and rerunning saved model: {k}")
    state_dict = torch.load(k)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    ema_model.ema.load_state_dict(state_dict, strict=True)
    # precision = test_model(args, ema_model.ema, priors.clone().detach(),
    #         valid_sets['det'], all_thresh=True) if 'det' in valid_sets else 0
    # precision = test_segmentation(ema_model.ema, valid_sets['seg']) \
    #         if 'seg' in valid_sets else 0
    # print({k: test_model(args, ema_model.ema, priors.clone().detach(), valid_sets[k],
    #         per_class=True) for k in valid_sets if 'det' in k and args.task_weights['det'] > 0})
    # print({k: test_segmentation(ema_model.ema, valid_sets[k], per_class=True)
    #         for k in valid_sets if 'seg' in k and args.task_weights['seg']> 0})
    print(eval_det(args, ema_model.ema, priors.clone().detach(), per_class=True))
    print(eval_seg(ema_model.ema, per_class=True, wgt=("COCO" not in args.dataset)))
