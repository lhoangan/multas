import os

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from data import detection_collate_dict, DataPrefetcher
from utils import PriorBox
from utils import Timer, ModelEMA
from utils import adjust_learning_rate, tencent_trick
from utils import DetectionLoss
from utils import SegLoss

from train import parse_arguments, load_dataset, preprocess_args
from train import save_weights, test_model, test_segmentation
from train import logger, wandb

def parse_arguments_student(parser):
    pass

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
        print({k: test_model(args, model, priors.clone().detach(), valid_sets[k])
                for k in valid_sets if 'det' in k})
        print({k: test_segmentation(model, valid_sets[k])
                for k in valid_sets if 'seg' in k})

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
                precision.update(
                    {k: test_model(args, ema_model.ema, priors.clone().detach(), valid_sets[k])
                            for k in valid_sets if 'det' in k})
                precision.update({k: test_segmentation(ema_model.ema, valid_sets[k])
                            for k in valid_sets if 'seg' in k})
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

            if len(output) == 2:
                (images, targets) = output
            else:
                (images, targets, seggt) = output

            new_size = args.size
            if "det" in task:
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
                out = model.forward_test(images)

                if 'seg' in task:
                    # detection in segmentation-phase (MXS dataset)
                    out_ = {'loc': [], 'conf': []}
                    targets_ = []
                    for bi in range(len(targets)): # go through batch dim
                        if len(targets[bi].shape) != 1: # targets[bi] != tensor([0.])
                            out_['loc'].append(out['loc'][bi])
                            out_['conf'].append(out['conf'][bi])
                            targets_.append(targets[bi])
                    targets = targets_
                    if len(targets) > 0:
                        out['loc'] = torch.stack(out_['loc'])
                        out['conf'] = torch.stack(out_['conf'])

                if len(targets) > 0: # 'det' in task: #
                    # breakpoint()
                    result = criterion(out, priors, targets, seggt=seggt,
                        seg_overlap=overlap, seg_conf=conf
                    )
                    result['loss_l'] = result['loss_l'] * args.task_weights['det']
                    result['loss_c'] = result['loss_c'] * args.task_weights['det']
                    loss += result['loss_l'] + result['loss_c']
                    # print ("Loss DET: ", result['loss_l'] + result['loss_c'])

                if 'seg' in task:
                    result = criterion_seg(out, seggt)
                    result['loss_seg'] = result['loss_seg'] * args.task_weights['seg']
                    loss += result['loss_seg']
                    # print ("Loss SEG: ", result['loss_seg'])

            results = {**result, **results}

        log_info["Loss/loc"].append(results['loss_l'].item() if 'loss_l' in results else 0)
        log_info["Loss/conf"].append(results['loss_c'].item() if 'loss_c'in results else 0)
        log_info["Loss/seg"].append(results['loss_seg'].item() if 'loss_seg' in results else 0)
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
                results['loss_seg'].item() if 'loss_seg' in results else 0,
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
    precision.update(
        {k: test_model(args, ema_model.ema, priors.clone().detach(), valid_sets[k])
                for k in valid_sets if 'det' in k})
    precision.update({k: test_segmentation(ema_model.ema, valid_sets[k])
                for k in valid_sets if 'seg' in k})
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
    print({k: test_model(args, ema_model.ema, priors.clone().detach(),
            valid_sets[k], per_class=True) for k in valid_sets if 'det' in k})
    print({k: test_segmentation(ema_model.ema, valid_sets[k], per_class=True)
                for k in valid_sets if 'seg' in k})
