import os

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from data import detection_collate_dict, DataPrefetcher, preproc_for_test
from utils import Detect
from utils import Timer, ModelEMA
from utils import adjust_learning_rate, tencent_trick
from utils import FCOSLoss
from utils import SegLoss

from train import parse_arguments, load_dataset, preprocess_args
from train import save_weights, test_segmentation
from train import logger, wandb

def PriorBox(
    base_anchor: float,
    size: int,
    base_size: int,
) -> torch.Tensor:
    """Overwritting utils.box.prior_box for fcos
    """

    import math
    from itertools import product as product
    repeat = 4 if base_size == 320 else 5
    feature_map = [math.ceil(size / 2 ** (3 + i)) for i in range(repeat)]

    output = []
    for (k, (f_h, f_w)) in enumerate(zip(feature_map, feature_map)):
        for (i, j) in product(range(f_h), range(f_w)):

            cy = (i + 0.5) / f_h * size
            cx = (j + 0.5) / f_w * size

            anchor = base_anchor * 2 ** k # / size
            output += [cx, cy, anchor, anchor]

    output = torch.Tensor(output).view(-1, 4)
    # output.clamp_(max=1, min=0)
    # output.clamp_(max=size, min=0)
    return output

def Detect(
    predictions: torch.Tensor,
    prior: torch.Tensor,
    org_size: torch.Tensor,
    eval_thresh: float = 0.01,
    nms_thresh: float = 0.5,
) -> tuple:
    """ Detect layer at test time """

    from torchvision.ops import boxes as box_ops

    num_anchors_per_level = [x.size(1) for x in predictions['loc']]

    loc = torch.cat(predictions['loc'], dim=1)
    conf = torch.cat(predictions['conf'], dim=1)
    cnter = torch.cat(predictions['cnter'], dim=1)
    assert loc.size(0) == 1,  'ERROR: Batch size = {} during evaluation'.format(loc.size(0))
    (loc, conf, cnter) = loc.squeeze(0), conf.squeeze(0), cnter.squeeze(0)

    test_score_thresh = eval_thresh # .2
    test_topk_candidates = 1000
    test_nms_thresh = nms_thresh # .6
    max_detection_per_image = 100

    # Eq4 in https://arxiv.org/pdf/2006.09214.pdf
    pred_scores = torch.sqrt(conf.sigmoid_() * cnter.sigmoid_())

    # Apply two filtering to make NMS faster
    # 1. Keep boxes with confidence score higher than threshold
    keep_idxs = pred_scores > test_score_thresh
    pred_scores = pred_scores[keep_idxs]
    topk_idxs = torch.nonzero(keep_idxs) # K x 2

    # 2. Keep top k top scoring boxes only
    topk_idxs_size = topk_idxs.shape[0]
    if isinstance(topk_idxs_size, torch.Tensor):
        # It's a tensor in tracing
        num_topk = torch.clamp(top_idxs_size, max=test_topk_candidates)
    else:
        num_topk = min(topk_idxs_size, test_topk_candidates)

    pred_scores, idxs = pred_scores.topk(num_topk)
    topk_idxs = topk_idxs[idxs]

    anchor_idxs, classes_idxs = topk_idxs.unbind(dim=1)

    from utils.loss.fcos_loss import Box2BoxTransformLinear
    transform = Box2BoxTransformLinear(normalize_by_size=False)
    decoded_boxes = transform.apply_deltas(loc, prior, num_anchors_per_level)[anchor_idxs]

    # from torch.nn import functional as F
    # deltas = F.relu(loc)
    # boxes = prior.to(deltas.dtype)

    # ctr_x = boxes[:, 0]
    # ctr_y = boxes[:, 1]
    # anchor_sizes = boxes[:, 2].clone().unsqueeze(-1) # (R, )

    # k = 0
    # scale = [8, 16, 32, 64] if len(num_anchors_per_level)==4 else [8, 16, 32, 64, 128]
    # for num, s in zip(num_anchors_per_level, scale):
    #     anchor_sizes[k: k+num] = s
    #     k = k+num

    # # deltas = deltas * anchor_sizes.unsqueeze(-1)

    # l = deltas[:, 0::4] * anchor_sizes / 320 * org_size[0]
    # t = deltas[:, 1::4] * anchor_sizes / 320 * org_size[1]
    # r = deltas[:, 2::4] * anchor_sizes / 320 * org_size[2]
    # b = deltas[:, 3::4] * anchor_sizes / 320 * org_size[3]

    # pred_boxes = torch.zeros_like(deltas)

    # pred_boxes[:, 0::4] = ctr_x[:, None] - l  # x1
    # pred_boxes[:, 1::4] = ctr_y[:, None] - t  # y1
    # pred_boxes[:, 2::4] = ctr_x[:, None] + r  # x2
    # pred_boxes[:, 3::4] = ctr_y[:, None] + b  # y2

    # decoded_boxes = pred_boxes[anchor_idxs]

    # Note: Torchvision already has a strategy
    # (see https://github.com/pytorch/vision/issues/1311)
    # to decide whether to use coordinate trick or for loop to implement batched_nms.
    # So we just call it directly.
    # float16 does not have enough range for batched NMS, so adding float()
    # return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)
    keep = box_ops.batched_nms(decoded_boxes.float(),
                               pred_scores,
                               classes_idxs,
                               iou_threshold=test_nms_thresh)

    decoded_boxes=(decoded_boxes[keep[: max_detection_per_image]]).cpu().numpy()
    conf_scores = pred_scores[keep[: max_detection_per_image]].cpu()
    classes_idxs = classes_idxs[keep[: max_detection_per_image]].cpu()

    # recover the shape of scores
    scores = torch.zeros((classes_idxs.shape[0], conf.shape[1]))
    scores = scores.scatter(1, classes_idxs.unsqueeze(-1), conf_scores.unsqueeze(-1)).numpy()

    return (decoded_boxes, scores)
# ~/anaconda3/envs/bmvc22/lib/python3.9/site-packages/detectron2/modeling/meta_arch/dense_detector.py

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

        priors_ = priors # /args.size * scale

        (boxes, scores) = Detect(
                out, priors_, scale, eval_thresh=0.05, nms_thresh=0.5,
        )
        boxes = boxes / args.size * scale.cpu().numpy()
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
            # # argmax = np.argmax(scores[inds, j-1:j])
            # # for b,s in zip([boxes[inds][argmax]], [scores[inds, j-1:j][argmax]]):
            # for b in target:
            #     visualize_bbox(myimg, b[:4], str(b[4]), myimg.shape[1], myimg.shape[0],
            #                    color=(0, 0, 255))
            # for b,s in zip(boxes[inds][:], scores[inds, j-1:j][:]):
            # # for b,s in zip(boxes[inds][:10], scores[inds, j-1:j][:10]):
            #     b1 = b.copy()
            #     b[0], b[2] = b1[0] / myimg.shape[1], b1[2] / myimg.shape[1]
            #     b[1], b[3] = b1[1] / myimg.shape[0], b1[3] / myimg.shape[0]
            #     visualize_bbox(myimg, b, str(j-1), myimg.shape[1], myimg.shape[0],
            #                    color=(255, 0, 0))
        # Image.fromarray(myimg).save(f'vis_debug_{i:03d}.png')

    return valid_sets.evaluate_detections(all_boxes, all_thresh=all_thresh,
                                          per_class=per_class) * 100

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
    from models.fcos_teacher import FCOS_Detector as Detector
    model = Detector(args.size, num_classes, args.backbone, args.neck,
                     task=args.task, noBN=args.noBN
                    ).cuda()

    if args.load_weights:
        state_dict = torch.load(args.load_weights)
        state_dict = state_dict['model'] if 'model' in state_dict else state_dict
        model.load_state_dict(state_dict, strict=True)

        priors = PriorBox(args.base_anchor_size, args.size,
                          base_size=args.size).cuda()
        print({k: test_model(args, model, priors.clone().detach(), valid_sets[k])
                for k in valid_sets if 'det' in k and args.task_weights['det'] > 0})
        print({k: test_segmentation(model, valid_sets[k])
                for k in valid_sets if 'seg' in k and args.task_weights['seg'] > 0})

    logger.info('Loading teacher Network...')

    params_to_train = tencent_trick(model)

    optimizer = optim.SGD(params_to_train, lr=args.lr,
            momentum=0.9, weight_decay=0.0005)
    scaler = torch.cuda.amp.GradScaler()

    ema_model = ModelEMA(model)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Total trainable param is : {:e}'.format(num_param))

    logger.info('Preparing Criterion & AnchorBoxes...')
    criterion = FCOSLoss()

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
                    "Loss/seg": [], "Loss/cnt": [], "Loss": [], "LR": [],
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

            if args.eval_epoch and epoch > args.eval_epoch:
                priors = PriorBox(args.base_anchor_size, args.size, base_size=args.size).cuda()
                # precision["det"] = test_model(args, ema_model.ema, priors.clone().detach(),
                #                               valid_sets['det']) if 'det' in valid_sets else 0
                # logger.info("det mAP={}".format(precision['det']))

                # precision["seg"] = test_segmentation(ema_model.ema, valid_sets['seg']) \
                #         if 'seg' in valid_sets else 0
                # logger.info('seg mAP={}'.format(precision['seg']))
                path = save_weights(ema_model.ema, args, '')
                precision.update(
                    {k: test_model(args, ema_model.ema, priors.clone().detach(), valid_sets[k])
                            for k in valid_sets if 'det' in k and args.task_weights['det'] > 0})
                precision.update({k: test_segmentation(ema_model.ema, valid_sets[k])
                            for k in valid_sets if 'seg' in k and args.task_weights['seg'] > 0})
                [logger.info(f"{k} mAP={precision[k]:.2f}") for k in precision]

                if val(precision) > val(best_maps) + 7e-2:
                    # # path = save_weights(ema_model.ema, args,
                    # #         'ep{}_mAPd{:.4f}_mAPs{:.4f}'.format(epoch-1,
                    # #             precision['det'], precision['seg']))
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
                   'loss_n': torch.tensor(0.0).cuda(),
                   'loss_s':torch.tensor(0.0).cuda()
                   }
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
                    result = criterion(out[task], priors, targets)
                    results['loss_l'] += result['loss_l'] * args.task_weights['det']
                    results['loss_c'] += result['loss_c'] * args.task_weights['det']
                    results['loss_n'] += result['loss_n'] * args.task_weights['det']
                elif "seg" in task:
                    result = criterion_seg(out[task], seggt)
                    results['loss_s'] += result['loss_seg'] * args.task_weights['seg']
                else:
                    raise NotImplementedError('Unkown task {}!'.format(task))

        loss = results['loss_l'] + results['loss_c'] + results['loss_s'] + results['loss_n']
        log_info["Loss/loc"].append(results['loss_l'].item() if 'loss_l' in results else 0)
        log_info["Loss/conf"].append(results['loss_c'].item() if 'loss_c'in results else 0)
        log_info["Loss/seg"].append(results['loss_s'].item() if 'loss_s' in results else 0)
        log_info["Loss/cnt"].append(results['loss_n'].item() if 'loss_n' in results else 0)
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
                    'l:cnt {:.2f} l:seg {:.2f} (LOSS {:.3f}) '\
                    #'d:best {:.2f} s:best {:.2f}, '\
                    'best:{} '\
                    'time {:.2f}s eta {:.2f}h'.format(
                epoch, iteration, max_iter, optimizer.param_groups[0]['lr'],
                results['loss_l'].item() if 'loss_l' in results else 0,
                results['loss_c'].item() if 'loss_c' in results else 0,
                results['loss_n'].item() if 'loss_c' in results else 0,
                results['loss_s'].item() if 'loss_s' in results else 0,
                loss.item(),
                #best_maps['det'], best_maps['seg'],
                ' '.join([f"{k} {best_maps[k]:.2f}"for k in best_maps]),
                load_time, load_time * (max_iter - iteration) / 3600,
                ))
            timer.clear()

    # TODO: save weights for debugging training process

    path = save_weights(ema_model.ema, args, f'ep{epoch-1}_'+
            '_'.join([f'{k}{precision[k]:.2f}' for k in sorted(precision.keys())]))

    priors = PriorBox(args.base_anchor_size, args.size, base_size=args.size).cuda()
    # precision['det'] = test_model(args, ema_model.ema, priors.clone().detach(),
    #         valid_sets['det']) if 'det' in valid_sets else 0
    # logger.info('det mAP={}'.format(precision['det']))
    # precision['seg'] = test_segmentation(ema_model.ema, valid_sets['seg']) \
    #         if 'seg' in valid_sets else 0
    # logger.info('seg mAP={}'.format(precision['seg']))
    precision.update(
        {k: test_model(args, ema_model.ema, priors.clone().detach(), valid_sets[k])
                for k in valid_sets if 'det' in k and args.task_weights['det'] > 0})
    precision.update({k: test_segmentation(ema_model.ema, valid_sets[k])
                for k in valid_sets if 'seg' in k and args.task_weights['seg'] > 0})
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
    print({k: test_model(args, ema_model.ema, priors.clone().detach(), valid_sets[k],
            per_class=True) for k in valid_sets if 'det' in k and args.task_weights['det'] > 0})
    print({k: test_segmentation(ema_model.ema, valid_sets[k], per_class=True)
            for k in valid_sets if 'seg' in k and args.task_weights['seg']> 0})
