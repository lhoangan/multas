import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utils import PriorBox

from train import parse_arguments, load_dataset, preprocess_args
from train import test_model, test_segmentation
from train import logger

import flow_vis
from utils.bestie import refine_label_generation, center_map_gen, gaussian
from utils.bestie import pseudo_label_generation
import cv2

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def pred_score_of_gt_seg(softmax_seg_map, gt_seg_map):

    _gt_seg_map = torch.zeros_like(softmax_seg_map)
    for b in range(softmax_seg_map.shape[0]):
        for _cls in range(softmax_seg_map.shape[1]):
            _gt_seg_map[b, _cls] = ((gt_seg_map[b] == _cls) * (_cls != 0)) * 3
            _gt_seg_map[b, _cls] += softmax_seg_map[b, _cls]
    return _gt_seg_map # .max(dim=1)[1]

def test_model(args, model, priors, valid_sets, all_thresh=True, per_class=False):
    from utils import Detect
    from data import preproc_for_test

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
        from data.voc0712 import visualize_bbox
        from PIL import Image
        target = valid_sets.pull_anno(i)
        std = (.229, .224, .225)
        mean = (.485, .456, .406)
        myimg = ((x.cpu().numpy().squeeze().transpose((1,2,0))*std+mean)*255)
        myimg = (myimg[..., -1::-1]).astype(np.uint8).copy()
        # myimg = img

        box = boxes.copy()
        box[:, 0], box[:, 2] = box[:, 0] / img.shape[1], box[:, 2] / img.shape[1]
        box[:, 1], box[:, 3] = box[:, 1] / img.shape[0], box[:, 3] / img.shape[0]
        center_map, offset_map = generate_offset_from_boxes(box*320, scores,
                                                            myimg.shape[1],
                                                            myimg.shape[0])
        seg_map = out['seg'] # Tensor [B, C+1, 320, 320 ]
        gt_seg_map = cv2.resize(valid_sets.pull_segment(i), (320, 320),
                                interpolation=cv2.INTER_NEAREST)
        label = np.zeros(20)
        # BESTIE assumes label is 1 lower than segmap
        label[[j-1 for j in np.unique(gt_seg_map) if j!=255 and j!=0]] = 1
        params = {'refine_thresh': 0.1,
                  'sigma': 6,
                  'beta': 3.0,
                  'kernel': 41
                  }

        # suppressing multi-center per segmentation
        # pseudo = pseudo_label_generation("cls", gt_seg_map,)

        refined = refine_label_generation(
            seg_map.cpu(),
            torch.from_numpy(center_map[None, ...]),
            torch.from_numpy(offset_map.transpose(2, 0, 1)[None, ...]),
            torch.from_numpy(label[None, ...]),
            torch.from_numpy(gt_seg_map[None, ...]),
            Struct(**params)
            )

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3)
        fig.tight_layout()
        offset = flow_vis.flow_to_color(refined['offset'][0].numpy().transpose(1, 2, 0))
        axs[1, 0].imshow(offset)#, plt.figure()

        # def merge_gt_pred_segmap(softmax_seg_map, gt_seg_map):

        #     _gt_seg_map = torch.zeros_like(softmax_seg_map)
        #     for b in range(softmax_seg_map.shape[0]):
        #         for _cls in range(softmax_seg_map.shape[1]):
        #             _gt_seg_map[b, _cls] = ((gt_seg_map[b] == _cls) * (_cls != 0)) * 3 + softmax_seg_map[b, _cls]
        #     breakpoint()
        #     return _gt_seg_map.max(dim=1)[1]
        # gt_seg_map = merge_gt_pred_segmap(
        #     out['seg'].cpu().softmax(dim=1),
        #     torch.from_numpy(gt_seg_map[None, ...]))[0].numpy()
        gt_seg_map[gt_seg_map == 255] = 0
        axs[1, 2].imshow(gt_seg_map)#, plt.figure()
        axs[1, 1].imshow(refined['center'][0].sum(0), clim=[.4, 1])#, plt.figure()

        for j in range(1, valid_sets.num_classes):
            inds = np.where(scores[:, j-1] > 0.5)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
            else:
                all_boxes[j][i] = np.hstack(
                        (boxes[inds], scores[inds, j-1:j])
                    ).astype(np.float32)
            # visualize target
            # argmax = np.argmax(scores[inds, j-1:j])
            # for b,s in zip([boxes[inds][argmax]], [scores[inds, j-1:j][argmax]]):
            # for b in target:
            #     visualize_bbox(myimg, b[:4], str(b[4]), myimg.shape[1], myimg.shape[0],
            #                    color=(0, 0, 255))
            # for b,s in zip(boxes[inds][:], scores[inds, j-1:j][:]):
            for b,_ in zip(boxes[inds][:10], scores[inds, j-1:j][:10]):
                b1 = b.copy()
                b[0], b[2] = b1[0] / img.shape[1], b1[2] / img.shape[1]
                b[1], b[3] = b1[1] / img.shape[0], b1[3] / img.shape[0]
                visualize_bbox(myimg, b, str(j-1), myimg.shape[1], myimg.shape[0],
                               color=(255, 0, 0))

        axs[0, 0].imshow(myimg)#, plt.figure(), #plt.show()
        # axs[0, 0].imshow(flow_vis.flow_to_color(offset_map))#, plt.figure()
        axs[0, 0].imshow(center_map.sum(0), clim=[.4, 1], alpha=.5)#, plt.figure()
        pred_seg = out['seg'][0].softmax(0).max(0)
        pred_cls = pred_seg[1].detach().cpu().numpy()
        pred_scr = pred_seg[0].detach().cpu().numpy()
        pred_scr = torch.gather(out['seg'][0].softmax(0).cpu(), 0,
                                torch.Tensor(gt_seg_map).long()[None]).squeeze()
        axs[0, 1].imshow(pred_scr),
        axs[0, 2].imshow(pred_cls), plt.show()
        # Image.fromarray(myimg).save(f'vis_debug_{i:03d}.png')

    return valid_sets.evaluate_detections(all_boxes, all_thresh=all_thresh,
                                          per_class=per_class) * 100

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
                                        8, gaussian(8) * score)

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

        # test_model(args, model, priors.clone().detach(), valid_sets['det'],
        #                      per_class=True)
        print ("Train keys: ", train_sets.keys())
        print ("Valid keys: ", valid_sets.keys())
        # test_model(args, model, priors.clone().detach(), valid_sets['0det'],
        #                      per_class=True)
        # test_model(args, model, priors.clone().detach(), train_sets['0det'],
        #                      per_class=True)
        test_model(args, model, priors.clone().detach(), train_sets['1seg'],
                             per_class=True)

        print({k: test_segmentation(model, valid_sets[k], per_class=True)
                for k in valid_sets if 'seg' in k})
