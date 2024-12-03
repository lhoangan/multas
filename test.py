import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utils import PriorBox

from utils.train import parse_arguments, load_dataset, preprocess_args
from utils.train import test_model, test_segmentation
from utils.train import logger


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
        print({k: test_model(args, model, priors.clone().detach(), valid_sets[k],
                             per_class=True) for k in valid_sets if 'det' in k})
        print({k: test_segmentation(model, valid_sets[k], per_class=True)
                for k in valid_sets if 'seg' in k})
