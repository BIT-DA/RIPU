import argparse
import os
import datetime
import logging
import time

import torch
import torch.nn as nn
import torch.utils
import torch.distributed
from torch.utils.data import DataLoader

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.active.build import PixelSelection, RegionSelection
from core.datasets.dataset_path_catalog import DatasetCatalog
from core.loss.negative_learning_loss import NegativeLearningLoss
from core.loss.local_consistent_loss import LocalConsistentLoss
from core.utils.utils import set_random_seed

import setproctitle
import warnings
warnings.filterwarnings('ignore')


def train(cfg):
    logger = logging.getLogger("AL-RIPU.trainer")

    # create network
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)

    print(classifier)

    # init optimizer
    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()

    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    # load checkpoint
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        classifier.load_state_dict(checkpoint['classifier'])

    # init mask for cityscape
    DatasetCatalog.initMask(cfg)

    # init data loader
    src_train_data = build_dataset(cfg, mode='train', is_source=True)
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False)
    tgt_epoch_data = build_dataset(cfg, mode='active', is_source=False, epochwise=True)

    src_train_loader = DataLoader(src_train_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=4,
                                  pin_memory=True, drop_last=True)
    tgt_train_loader = DataLoader(tgt_train_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=4,
                                  pin_memory=True, drop_last=True)
    tgt_epoch_loader = DataLoader(tgt_epoch_data, batch_size=1, shuffle=False, num_workers=4,
                                  pin_memory=True, drop_last=False)

    # init loss
    sup_criterion = nn.CrossEntropyLoss(ignore_index=255)
    negative_criterion = NegativeLearningLoss(threshold=cfg.SOLVER.NEGATIVE_THRESHOLD)
    local_consistent_loss = LocalConsistentLoss(cfg.MODEL.NUM_CLASSES, cfg.SOLVER.LCR_TYPE).cuda()

    iteration = 0
    start_training_time = time.time()
    end = time.time()
    max_iters = cfg.SOLVER.MAX_ITER
    meters = MetricLogger(delimiter="  ")

    logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
    feature_extractor.train()
    classifier.train()
    active_round = 1
    for batch_index, (src_data, tgt_data) in enumerate(zip(src_train_loader, tgt_train_loader)):
        data_time = time.time() - end

        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters,
                                          power=cfg.SOLVER.LR_POWER)
        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr * 10

        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()

        src_input, src_label = src_data['img'], src_data['label']
        src_input = src_input.cuda(non_blocking=True)
        src_label = src_label.cuda(non_blocking=True)

        # target data
        # tgt_mask is active label, 255 means unlabeled data
        tgt_input, tgt_mask = tgt_data['img'], tgt_data['mask']
        tgt_input = tgt_input.cuda(non_blocking=True)
        tgt_mask = tgt_mask.cuda(non_blocking=True)

        src_size = src_input.shape[-2:]
        src_out = classifier(feature_extractor(src_input), size=src_size)
        tgt_size = tgt_input.shape[-2:]
        tgt_out = classifier(feature_extractor(tgt_input), size=tgt_size)
        predict = torch.softmax(tgt_out, dim=1)

        # source supervision loss
        loss = torch.Tensor([0]).cuda()
        loss_sup = sup_criterion(src_out, src_label)
        meters.update(loss_sup=loss_sup.item())
        loss += loss_sup

        # target active supervision loss
        if torch.sum((tgt_mask != 255)) != 0:  # target has labeled pixels
            loss_sup_tgt = sup_criterion(tgt_out, tgt_mask)
            meters.update(loss_sup_tgt=loss_sup_tgt.item())
            loss += loss_sup_tgt

        # source consistency regularization loss
        if cfg.SOLVER.CONSISTENT_LOSS > 0:
            consistent_loss = local_consistent_loss(src_out, src_label) * cfg.SOLVER.CONSISTENT_LOSS
            meters.update(cr_loss=consistent_loss.item())
            loss += consistent_loss

        # target negative pseudo loss
        if cfg.SOLVER.NEGATIVE_LOSS > 0:
            negative_learning_loss = negative_criterion(predict) * cfg.SOLVER.NEGATIVE_LOSS
            meters.update(nl_loss=negative_learning_loss.item())
            loss += negative_learning_loss

        loss.backward()
        optimizer_fea.step()
        optimizer_cls.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        iteration += 1
        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.02f} GB"
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer_fea.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
                )
            )

        if iteration == cfg.SOLVER.MAX_ITER or iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            filename = os.path.join(cfg.OUTPUT_DIR, "model_iter{:06d}.pth".format(iteration))
            torch.save({'iteration': iteration,
                        'feature_extractor': feature_extractor.state_dict(),
                        'classifier': classifier.state_dict(),
                        'optimizer_fea': optimizer_fea.state_dict(),
                        'optimizer_cls': optimizer_cls.state_dict(),
                        }, filename)

        # active learning
        if iteration in cfg.ACTIVE.SELECT_ITER or cfg.DEBUG:
            if cfg.ACTIVE.SETTING == "RA":
                RegionSelection(cfg=cfg,
                                feature_extractor=feature_extractor,
                                classifier=classifier,
                                tgt_epoch_loader=tgt_epoch_loader)
            elif cfg.ACTIVE.SETTING == 'PA':
                PixelSelection(cfg=cfg,
                               feature_extractor=feature_extractor,
                               classifier=classifier,
                               tgt_epoch_loader=tgt_epoch_loader)
            active_round += 1

        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / cfg.SOLVER.STOP_ITER
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Active Domain Adaptive Semantic Segmentation Training")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str)
    parser.add_argument("--proctitle",
                        type=str,
                        default="AL-RIPU",
                        help="allow a process to change its title",)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    if args.opts is not None:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    setproctitle.setproctitle(f'{args.proctitle}')
    logger = setup_logger("AL-RIPU", output_dir, 0)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    logger.info('Initializing Cityscapes label mask...')

    set_random_seed(cfg.SEED)

    train(cfg)


if __name__ == '__main__':
    main()
