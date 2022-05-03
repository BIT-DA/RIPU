from . import transform
from .dataset_path_catalog import DatasetCatalog


def build_transform(cfg, mode, is_source):
    if mode == "train":
        w, h = cfg.INPUT.SOURCE_INPUT_SIZE_TRAIN if is_source else cfg.INPUT.TARGET_INPUT_SIZE_TRAIN
        trans_list = [
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ]
        if cfg.INPUT.INPUT_SCALES_TRAIN[0] == cfg.INPUT.INPUT_SCALES_TRAIN[1] and cfg.INPUT.INPUT_SCALES_TRAIN[0] == 1:
            trans_list = [transform.Resize((h, w)), ] + trans_list
        else:
            trans_list = [
                             transform.RandomScale(scale=cfg.INPUT.INPUT_SCALES_TRAIN),
                             transform.RandomCrop(size=(h, w), pad_if_needed=True),
                         ] + trans_list
        trans = transform.Compose(trans_list)
    else:
        w, h = cfg.INPUT.INPUT_SIZE_TEST
        trans = transform.Compose([
            transform.Resize((h, w), resize_label=False),
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ])
    return trans


def build_dataset(cfg, mode='train', is_source=True, epochwise=False, empty=False):
    assert mode in ['train', 'val', 'test', 'active']
    transform = build_transform(cfg, mode, is_source)
    print(transform)
    iters = None
    if mode == 'train' or mode == 'active':
        if not epochwise:
            iters = cfg.SOLVER.MAX_ITER * cfg.SOLVER.BATCH_SIZE
        if is_source:
            dataset = DatasetCatalog.get(cfg.DATASETS.SOURCE_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES,
                                         max_iters=iters, transform=transform, cfg=cfg, empty=empty)
        else:
            dataset = DatasetCatalog.get(cfg.DATASETS.TARGET_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES,
                                         max_iters=iters, transform=transform, cfg=cfg, empty=empty)
    elif mode == 'val':
        dataset = DatasetCatalog.get(cfg.DATASETS.TEST, 'val', num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters,
                                     transform=transform, cfg=cfg, empty=empty)
    elif mode == 'test':
        dataset = DatasetCatalog.get(cfg.DATASETS.TEST, cfg.DATASETS.TEST.split('_')[-1],
                                     num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=transform, cfg=cfg,
                                     empty=empty)

    return dataset
