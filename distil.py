#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, os.path as osp
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from data import detection_collate_dict, DataPrefetcher
from utils import PriorBox
from utils import DetectionLoss, HintLoss, SegLoss
from utils import Timer, ModelEMA
from utils import adjust_learning_rate, tencent_trick

from train import parse_arguments, load_dataset, preprocess_args
from train import save_weights, test_model, test_segmentation
from train import logger, wandb

def parse_arguments_student(parser):

    parser.add_argument('--teacher_neck', default='pafpn')
    parser.add_argument('--teacher_backbone', default='')
    parser.add_argument('--kd', default='pdf', help='Hint loss')
    parser.add_argument('--tdet_weights', default='', type=str)
    parser.add_argument('--tseg_weights', default='', type=str)
    args = parser.parse_args()

    args.job_type = "student" if args.job_type == "teacher" else args.job_type

    return args

if __name__ == '__main__':

    parser = parse_arguments()
    args = parse_arguments_student(parser)
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

    logger.info('Loading student Network...')
    from models.student_detector import Student_Detector

    model = Student_Detector(args.size, num_classes, args.backbone,
                             args.neck, task=args.task, noBN=args.noBN).cuda()
    stu_num_param = sum(p.numel() for p in model.parameters())
    logger.info(f"Student backbone: {args.backbone} | Neck: {args.neck}")
    logger.info(f'Total trainable param of student model is: {stu_num_param:e}')
    priors = PriorBox(args.base_anchor_size, args.size, base_size=args.size).cuda()
    if args.load_weights:
        logger.info (f"Loading from pretrained weights: {args.load_weights}")

        state_dict = torch.load(args.load_weights)
        state_dict = state_dict['model'] if 'model' in state_dict else state_dict
        # # removing keys is only applicable for single task finetuning
        # rmkeys = []
        # if args.task == 'seg': # remove all segmentation head
        #     rmkeys = [k for k in state_dict if 'seghead.' in k]
        # elif args.task == 'det': # remove all detection head
        #     rmkeys = [k for k in state_dict if 'conf.' in k or 'loc.' in k]
        # [state_dict.pop(k) for k in rmkeys]
        model.load_state_dict(state_dict, strict=False)

        print({k: test_model(args, model, priors.clone().detach(), valid_sets[k])
                for k in valid_sets if 'det' in k})
        print({k: test_segmentation(model, valid_sets[k])
                for k in valid_sets if 'seg' in k})

    logger.info('Loading teacher Network...')
    if args.teacher_backbone != '':
        backbone = args.teacher_backbone
    else:
        raise NotImplementedError
    neck = args.teacher_neck
    from models.teacher_detector import Detector_base as Detector

    teacher = {t: Detector(args.size, num_classes, backbone, neck, task=t,
                           noBN=args.noBN).cuda()
            for t in args.task.split("+")}
    logger.info(f"Teacher backbone: {backbone} | Neck: {args.teacher_neck}")

    # Load teacher weight for each task
    for t in args.task.split("+"):

        if t == 'seg' and ('0harddet' in args.kd or '0tseg' in args.kd):
            continue

        num_param = sum(p.numel() for p in teacher[t].parameters())
        logger.info(f'Total param of teacher model is : {num_param:e}')
        logger.info(f'Ratio student/teacher: {stu_num_param/num_param:.3f}')

        trained_model = eval(f"args.t{t}_weights")
        if args.kd == "hard":
            teacher[t].deploy()
            continue
        assert osp.exists(trained_model), f"args.t{t}_weights = {trained_model} " \
                "do not exist"
        logger.info(f'Loading teacher weights from {trained_model}')
        state_dict = torch.load(trained_model)
        if "model" in state_dict:
            state_dict = state_dict['model']
        teacher[t].load_state_dict(state_dict, strict=False)
        teacher[t].deploy()


    logger.info('Preparing Optimizer & AnchorBoxes...')
    optimizer = optim.SGD(tencent_trick(model), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scaler = torch.cuda.amp.GradScaler()
    ema_model = ModelEMA(model)
    criterion_det = DetectionLoss(
            mutual_guide=(args.match=='mg'),
            use_focal=(args.conf_loss=='fc')
        )
    criterion_kd = HintLoss(args.kd)
    if 'seg' in '_'.join(train_sets):
        k = [key for key in train_sets if 'seg' in key][0]
        criterion_seg = SegLoss(ignore_index=train_sets[k].ignore_label)

    os.makedirs(args.save_folder, exist_ok=True)
    timer = Timer()
    epoch = 0
    best_maps = 0
    best_epoch = 0
    precision = {}
    best_maps = {}
    longest_ep = 5
    mAPs = {}
    args.eval_epoch = int(max_iter/epoch_size/1.75) if args.eval_epoch == 1 else args.eval_epoch
    for iteration in range(max_iter):
        if iteration % epoch_size == 0:

            epoch += 1

            log_info = {"Loss/loc": [], "Loss/conf": [], "Loss/contrast": [],
                    "Loss/hint": [], "Loss/seg": [], "Loss": [], "LR": []
                }

            # create batch iterator
            prefetcher = {d: DataPrefetcher(data.DataLoader(
                train_sets[d], args.batch_size, shuffle=True, num_workers=4,
                collate_fn=detection_collate_dict
            )) for d in train_sets}
            model.train()

            if args.eval_epoch and epoch > args.eval_epoch:
                priors = PriorBox(args.base_anchor_size, args.size, base_size=args.size).cuda()

                precision.update(
                    {k: test_model(args, ema_model.ema, priors.clone().detach(), valid_sets[k])
                            for k in valid_sets if 'det' in k})
                precision.update({k: test_segmentation(ema_model.ema, valid_sets[k])
                            for k in valid_sets if 'seg' in k})
                [logger.info(f"{k} mAP={precision[k]:.2f}") for k in precision]

                # precision["det"] = test_model(args, ema_model.ema, priors.clone().detach(),
                #         valid_sets['det']) if 'det'in valid_sets else 0
                # logger.info("det mAP={}".format(precision['det']))

                # precision["seg"] = test_segmentation(ema_model.ema, valid_sets['seg']) \
                #         if 'seg' in valid_sets else 0
                # logger.info('seg mAP={}'.format(precision['seg']))

                if sum(precision.values()) > sum(best_maps.values()) + 4e-2:
                    path = save_weights(ema_model.ema, args, f'ep{epoch-1}_'+
                        '_'.join([f'{k}{precision[k]:.4f}' for k in sorted(precision.keys())])
                        )
                            # 'ep{}_mAPd{:.4f}_mAPs{:.4f}'.format(epoch-1,
                            #     precision['det'], precision['seg']))

                    # Update best precision
                    mAPs[path] = sum(precision.values())
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
                model.train()

        # traning iteratoin
        timer.tic()
        loss = torch.tensor(0.0).cuda()
        loss_l = torch.Tensor([0]).cuda()
        loss_c = torch.Tensor([0]).cuda()
        loss_s = torch.Tensor([0]).cuda()
        loss_k = torch.Tensor([0]).cuda()
        for task in prefetcher:
            adjust_learning_rate(optimizer, args.lr, iteration, args.warm_iter, max_iter)
            output = prefetcher[task].next()
            if output[0] is None or len(output[0]) < args.batch_size:
                prefetcher[task] = DataPrefetcher(data.DataLoader(
                    train_sets[task], args.batch_size, shuffle=True, num_workers=4,
                    collate_fn=detection_collate_dict
                ))
                output = prefetcher[task].next()
            if "seg" in task:
                (images, _, seggt) = output
            else:
                (images, targets) = output

            if 'det' in task:
                # random resize
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

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    out_t = {t[1:]: teacher[t[1:]](images) for t in prefetcher}

                out = model(images)

                if "hard" in args.kd:
                    if 'det' in task and not "0harddet" in args.kd:
                        result_det = criterion_det(out, priors, targets)
                        loss_l += result_det['loss_l'] * args.task_weights['det']
                        loss_c += result_det['loss_c'] * args.task_weights['det']
                    elif "seg" in task:
                        result = criterion_seg(out, seggt)
                        loss_s += result['loss_seg'] * args.task_weights['seg']
                # if "soft" not in args.kd or "hard" in args.kd:
                #     result_det = criterion_det(out, priors, targets)
                #     loss_l = result_det['loss_l']
                #     loss_c = result_det['loss_c'] * (0 if 'wc0' in args.note else 1)
                masks = None
                if "def" in args.kd:
                    from utils.box.box_utils import gen_fg_mask
                    (num, num_priors, _) = out['conf'].size()
                    masks = torch.zeros(num, int(num_priors/6)).cuda()
                    for idx in range(num):
                        truths = targets[idx][:, :-1]
                        gen_fg_mask(truths, priors, masks, idx)
                # if "0harddet" in args.kd and task == "seg":
                #     continue
                if ('det' in task and ('1soft' in args.kd or '0harddet' in args.kd)) or \
                   ("seg" in task and ('0soft' in args.kd)) or '2soft' in args.kd:
                    loss_kd = criterion_kd(out_t['det'], out, masks)# task='det')
                    loss_l += loss_kd['loss_l']*args.task_weights['det'] if 'loss_l' in loss_kd else 0
                    loss_c += loss_kd['loss_c']*args.task_weights['det'] if 'loss_c' in loss_kd else 0
                    loss_k += loss_kd['loss_kd'] if 'loss_kd' in loss_kd else torch.Tensor([0]).cuda()
                    continue

                loss_kd = criterion_kd(out_t[task[1:]], out, masks)
                loss_l += loss_kd['loss_l'] if 'loss_l' in loss_kd else 0
                loss_c += loss_kd['loss_c'] if 'loss_c' in loss_kd else 0
                loss_k += loss_kd['loss_kd'] if 'loss_kd' in loss_kd else torch.Tensor([0]).cuda()
                loss = loss_l + loss_c + loss_k

        loss = loss_l + loss_c + loss_s + loss_k

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ema_model.update(model)
        load_time = timer.toc()

        log_info["Loss/loc"].append(loss_l.item())
        log_info["Loss/conf"].append(loss_c.item())
        log_info["Loss/hint"].append(loss_k.item())
        log_info["Loss/seg"].append(loss_s.item())
        log_info["Loss"].append(loss.item())
        log_info["LR"].append(optimizer.param_groups[0]['lr'])
        wandb.log({k: np.average(log_info[k]) for k in log_info}, step=epoch)
        if any(best_maps.values()):
            ks = []
            for t in sorted(best_maps.keys()):
                k = t[1:] if t[1:] not in ks else t
                wandb.log({f"{k}/MAP": best_maps[t]}, step=epoch)
                ks.append(k)

        # logging
        if iteration % 100 == 0:
            logger.info(
                    'ep {} it {}/{} lr {:.4f}, l:loc {:.2f} l:cls {:.2f} '\
                    'l:seg {:.2f} l:kd {:.2f} (LOSS {:.3f}) '\
                    'best:{}'\
                    'time {:.2f}s eta {:.2f}h'.format(
                epoch, iteration, max_iter, optimizer.param_groups[0]['lr'],
                loss_l.item(), loss_c.item(), loss_s.item(), loss_k.item(),
                loss.item(),
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
    precision.update(
        {k: test_model(args, ema_model.ema, priors.clone().detach(), valid_sets[k])
                for k in valid_sets if 'det' in k})
    precision.update({k: test_segmentation(ema_model.ema, valid_sets[k])
                for k in valid_sets if 'seg' in k})
    [logger.info(f"{k} mAP={precision[k]:.2f}") for k in precision]

    # Save the final weights
    if sum(precision.values()) > sum(best_maps.values()) or len(mAPs) == 0:

        path = save_weights(ema_model.ema, args, f'ep{epoch-1}_'+
            '_'.join([f'{k}{precision[k]:.4f}' for k in sorted(precision.keys())]))
        # path = save_weights(ema_model.ema, args,
        #     'ep{}_mAPd{:.4f}_mAPs{:.4f}'.format(epoch-1,
        #         precision['det'], precision['seg']))
        ks = []
        for t in sorted(best_maps.keys()):
            k = t[1:] if t[1:] not in ks else t
            wandb.log({f"{k}/MAP": best_maps[t]}, step=epoch)
            ks.append(k)
        # Update best precision
        mAPs[path] = sum(precision.values())
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

    print({k: test_model(args, ema_model.ema, priors.clone().detach(), valid_sets[k])
                for k in valid_sets if 'det' in k})
    print({k: test_segmentation(ema_model.ema, valid_sets[k])
                for k in valid_sets if 'seg' in k})
