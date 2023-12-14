import os
import random
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from data import detection_collate_dict, DataPrefetcher, preproc_for_test
from utils import Detect
from utils import PriorBox
from utils import Timer, ModelEMA
from utils import adjust_learning_rate, tencent_trick
from utils import DetectionLoss
from utils import SegLoss

from train import parse_arguments, load_dataset, preprocess_args
from train import save_weights, test_model, test_segmentation
from train import logger, wandb


from utils.bestie import refine_box_generation, center_map_gen, gaussian

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def gen_pseudo_boxes(args, model, x, priors, seggt, vis=False):
    from utils import Detect
    from data import preproc_for_test
    model.eval()

    # all_boxes = [
    #     [ None for _ in range(num_images) ] for _ in range(valid_sets.num_classes)
    # ]

    # prepare image to detect
    # scale = torch.Tensor(
    #         [ img.shape[1], img.shape[0], img.shape[1], img.shape[0] ]
    #     ).cuda() if "VEDAI" not in args.dataset and "ISPRS" not in args.dataset else 1
    # x = torch.from_numpy(
    #     preproc_for_test(img, args.size)
    # ).unsqueeze(0).cuda()

    # model inference
    with torch.no_grad():
        out = model.forward_test(x)

    center_maps = []
    offset_maps = []
    label = []
    all_boxes = []
    torch.unique(seggt)
    for b in range(x.shape[0]):
        (boxes, scores) = Detect(
            {k: out[k][b, None] for k in out if len(out[k]) > 0},
            priors, args.size, # torch.Tensor([args.size, args.size, args.size, args.size]),
            eval_thresh=0.05, nms_thresh=0.5#, out_cpu=False
        )

        all_boxes.append({'boxes': boxes, 'scores': scores})
        center_map, offset_map = generate_offset_from_boxes(boxes, scores,
                                                            args.size, args.size)
        center_maps.append(center_map)
        offset_maps.append(offset_map.transpose(2, 0, 1))

        # gt = seggt[b]
        # # BESTIE assumes label is 1 lower than segmap
        # label.append(torch.zeros(20).cuda().scatter(0, gt[(gt!=0) & (gt!=255)]-1, 1))

    # label = torch.stack(label)
    params = {'refine_thresh': 0.1, 'sigma': 6, 'beta': 3.0, 'kernel': 41 }

    refined = refine_box_generation(
        out['seg'].double(),
        torch.from_numpy(np.stack(center_maps)),
        torch.from_numpy(np.stack(offset_maps)).cuda(),
        # label,
        seggt,
        Struct(**params)
        )

    if not vis:
        return refined['targets']

    # visualization
    std = (.229, .224, .225)
    mean = (.485, .456, .406)
    import matplotlib.pyplot as plt
    import flow_vis

    for b in range(x.shape[0]):
        from data.voc0712 import visualize_bbox
        img = ((x[b].detach().cpu().numpy().squeeze().transpose((1,2,0))*std+mean)*255)
        img = img.astype(np.uint8).copy()

        fig, axs = plt.subplots(2, 2)
        fig.tight_layout()

        offset = flow_vis.flow_to_color(refined['offset'][0].cpu().numpy().transpose(1, 2, 0))
        offset
        # axs[1, 0].imshow(offset)#, plt.figure()

        # [0, 0]: predicted boxes
        pred = img.copy()
        for j in range(all_boxes[b]['scores'].shape[1]):
            i = np.where(all_boxes[b]['scores'][:, j-1] > 0.5)[0]
            for box, score in zip(all_boxes[b]['boxes'][i], all_boxes[b]['scores'][i, j-1]):
                visualize_bbox(pred, box/args.size, f"{j}/{score:.1f}",
                           pred.shape[1], pred.shape[0], color=(255, 0, 0))
        axs[0, 0].imshow(pred)
        axs[0, 0].imshow(center_maps[b].sum(0), alpha=.5)#, plt.figure()

        # [1, 0]: pseudo/generated boxes
        pseudo = img.copy()
        for box in refined['targets'][b]:
            visualize_bbox(pseudo, box[:4].cpu().numpy(),
                           f"{box[-2].item():.0f}/{box[-1].item():.1f}",
                           pseudo.shape[1], pseudo.shape[0], color=(255, 0, 0))
        axs[1, 0].imshow(pseudo)
        axs[1, 0].imshow(refined['center'][b].sum(0), alpha=.5)#, plt.figure()

        axs[0, 1].imshow(out['seg'][b].max(0)[1].detach().cpu().numpy())

        gt_seg_map = seggt[b].cpu()
        gt_seg_map[gt_seg_map == 255] = 0
        axs[1, 1].imshow(gt_seg_map.cpu().numpy().squeeze()), plt.show()

    return refined['targets']

def generate_offset_from_boxes(boxes, scores, img_w, img_h):
    """
    Generating offset and centers from a list of boxes
    Arguments:
        boxes: list [N, 4]
        scores: list [C, N] with C = number of classes (20 for VOC)
    Returns:
        center_map: A Tensor of shape [B, C, H, W]. output center map.
        offset_map: A Tensor of shape [B, 2, H, W]. output offset map. 2 = [y x]
        label: A Tensor of shape [B, C]. one-hot image-level label.
    """
    center_map = np.zeros((scores.shape[1], img_h, img_w))
    offset_map = np.zeros((img_h, img_w, 2))
    bsizes_map = np.zeros((img_h, img_w)) + img_h * img_w
    xv, yv = np.meshgrid(range(img_w), range(img_h), indexing="xy")

    for j in range(scores.shape[1]):
        inds = np.where(scores[:, j-1] > 0.5)[0]
        for box, score in zip(boxes[inds], scores[inds, j-1]):
            center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 # X, Y
            area = (box[2] - box[0]) * (box[3] - box[1]) # box_w * box_h

            # slicing the region inside the box
            mask = (box[0] < xv) * (xv < box[2]) * (box[1] < yv) * (yv < box[3])
            # filtering out those pixels known to belong to smaller boxes
            mask = mask * (bsizes_map > area)

            offset_map[mask, 0] = center[1] # offset map c-0 is y
            offset_map[mask, 1] = center[0] # offset map c-1 is x
            bsizes_map[mask] = area

            # generate center map
            center_map = center_map_gen(center_map, center[0], center[1], j-1, # BESTIE removes BG
                                        8, gaussian(8)) # * score)
    empty = offset_map == 0
    offset_map =  offset_map - np.dstack((yv, xv)) # offset map is yx-order
    offset_map = offset_map / np.sqrt(np.power(offset_map, 2).sum(axis=2))[..., None]
    offset_map[empty] = 0

    return center_map, offset_map

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

    if  args.load_weights: # TODO:: reactivate
        state_dict = torch.load(args.load_weights)
        state_dict = state_dict['model'] if 'model' in state_dict else state_dict
        model.load_state_dict(state_dict, strict=True)

        priors = PriorBox(args.base_anchor_size, args.size,
                          base_size=args.size).cuda()
        # print(eval_det(args, model, priors.clone().detach()))
        # print(eval_seg(model, wgt=("COCO" not in args.dataset)))

    logger.info('Loading teacher Network...')

    params_to_train = tencent_trick(model)

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
            # if iteration == 7687: # 377:
            #     breakpoint()
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
                out[task] = model.forward_test(images)

                if "det" in task:
                    result = criterion(out[task], priors, targets, seggt=seggt,
                        seg_overlap=overlap, seg_conf=conf
                    )
                    results['loss_l'] += result['loss_l'] * args.task_weights['det']
                    results['loss_c'] += result['loss_c'] * args.task_weights['det']

                elif "seg" in task:
                    result = criterion_seg(out[task], seggt)
                    results['loss_s'] += result['loss_seg'] * args.task_weights['seg']

                    # TODO: adding weak detection loss
                    ## generating targets
                    if False: #iteration % epoch_size == 0:
                        targets_ = gen_pseudo_boxes(args, ema_model.ema, images,
                                                    priors, seggt, vis=True)
                    else:
                        targets_ = gen_pseudo_boxes(args, ema_model.ema, images,
                                                    priors, seggt)
                    ## computing loss
                    try:
                        mk = torch.Tensor([len(t) > 0 for t in targets_])
                        result = criterion(
                            {k: out[task][k] for k in out[task]},
                            priors, targets_, seggt=seggt,
                        seg_overlap=overlap, seg_conf=conf
                        )
                        if  torch.isnan(result['loss_l']) or \
                            torch.isinf(result['loss_l']) or \
                            torch.isnan(result['loss_c']) or \
                            torch.isinf(result['loss_c']):
                            raise
                    except:
                        print (iteration)
                        breakpoint()
                        targets_ = gen_pseudo_boxes(args, ema_model.ema, images, priors, seggt)

                    if "0L" not in args.note:
                        results['loss_l'] += result['loss_l'] * args.task_weights['det']
                    if "0C" not in args.note:
                        results['loss_c'] += result['loss_c'] * args.task_weights['det']

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
                    'best:{} '\
                    'time {:.2f}s eta {:.2f}h'.format(
                epoch, iteration, max_iter, optimizer.param_groups[0]['lr'],
                results['loss_l'].item() if 'loss_l' in results else 0,
                results['loss_c'].item() if 'loss_c' in results else 0,
                results['loss_s'].item() if 'loss_s' in results else 0,
                loss.item(),
                ' '.join([f"{k} {best_maps[k]:.2f}"for k in best_maps]),
                load_time, load_time * (max_iter - iteration) / 3600,
                ))
            timer.clear()

    priors = PriorBox(args.base_anchor_size, args.size, base_size=args.size).cuda()
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
    print(eval_det(args, ema_model.ema, priors.clone().detach(), per_class=True))
    print(eval_seg(ema_model.ema, per_class=True, wgt=("COCO" not in args.dataset)))
