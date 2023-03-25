import argparse
import os
import datetime
import logging
import time
import math
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import re

import torch
import torch.nn.functional as F
import torch.backends.cudnn

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU, get_color_pallete
from core.utils.logger import setup_logger

import setproctitle
import warnings
warnings.filterwarnings('ignore')

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def inference(feature_extractor, classifier, image, label, flip=False):
    size = label.shape[-2:]
    if flip:
        image = torch.cat([image, torch.flip(image, [3])], 0)
    with torch.no_grad():
        output = classifier(feature_extractor(image))
    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    return output.unsqueeze(dim=0)


def transform_color(pred):
    synthia_to_city = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 15,
        14: 17,
        15: 18,
    }
    label_copy = 255 * np.ones(pred.shape, dtype=np.float32)
    for k, v in synthia_to_city.items():
        label_copy[pred == k] = v
    return label_copy.copy()


def test(cfg):
    logger = logging.getLogger("AL-RIPU.tester")
    logger.info("Start testing")
    device = torch.device(cfg.MODEL.DEVICE)

    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)

    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(feature_extractor_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    feature_extractor.eval()
    classifier.eval()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()
    dataset_name = cfg.DATASETS.TEST
    output_folder = '.'
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)

    test_data = build_dataset(cfg, mode='test', is_source=False)

    assert cfg.TEST.BATCH_SIZE == 1, "Test batch size should be 1!"
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4,
                                              pin_memory=True, sampler=None)

    for batch in tqdm(test_loader):
        x, y, name = batch['img'], batch['label'], batch['name']
        name = name[0]
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True).long()

        pred = inference(feature_extractor, classifier, x, y, True)

        output = pred.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        # save the result
        pred = pred.cpu().numpy().squeeze().argmax(0)
        if cfg.MODEL.NUM_CLASSES == 16:
            pred = transform_color(pred)
        mask = get_color_pallete(pred, "city")
        mask_filename = name if len(name.split("/")) < 2 else name.split("/")[1]
        if mask.mode == 'P':
            mask = mask.convert('RGB')
        mask.save(os.path.join(output_folder, mask_filename))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(cfg.MODEL.NUM_CLASSES):
        logger.info(
            '{} {} iou/accuracy: {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i], iou_class[i], accuracy_class[i]))


def main():
    parser = argparse.ArgumentParser(description="Active Domain Adaptive Semantic Segmentation Testing")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    parser.add_argument("--proctitle",
                        type=str,
                        default="AL-RIPU",
                        help="allow a process to change its title", )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    setproctitle.setproctitle(f'{args.proctitle}')
    save_dir = ""
    logger = setup_logger("AL-RIPU", save_dir, 0)
    logger.info(cfg)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg)


if __name__ == "__main__":
    main()
