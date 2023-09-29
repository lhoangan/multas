import math
import torch
import torch.nn as nn


def adjust_learning_rate(
    optimizer: torch.optim,
    base_lr: float,
    iteration: int,
    warm_iter: int,
    max_iter: int,
) -> float:
    """ warmup + cosine lr decay """
    start_lr = base_lr / 10
    if iteration <= warm_iter:
        lr = start_lr + (base_lr - start_lr) * iteration / warm_iter
    else:
        lr = start_lr + (base_lr - start_lr) * 0.5 * (1 + math.cos((iteration - warm_iter) * math.pi / (max_iter - warm_iter)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def tencent_trick(
    model: nn.Module,
    excl_names=[]
) -> list:
    """ no weight decay for bn and conv bias """
    (decay, no_decay) = ([], [])
    for (name, param) in model.named_parameters():
        if not param.requires_grad or name in excl_names:
            continue  # frozen weights
        elif len(param.shape) == 1 or name.endswith('.bias'):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay}]
