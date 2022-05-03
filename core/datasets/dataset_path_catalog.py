import os
import os.path as op
from .cityscapes import cityscapesDataSet
from .gtav import GTAVDataSet
from .synthia import synthiaDataSet
import numpy as np
import torch
from torch.utils import data
from PIL import Image
from tqdm import tqdm
import errno


class DatasetCatalog(object):
    DATASET_DIR = "datasets"
    DATASETS = {
        "gtav_train": {
            "data_dir": "gtav",
            "data_list": "gtav_train_list.txt"
        },
        "synthia_train": {
            "data_dir": "synthia",
            "data_list": "synthia_train_list.txt"
        },
        "cityscapes_train": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt"
        },
        "cityscapes_val": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_val_list.txt"
        },
    }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None, cfg=None, empty=False):
        if "gtav" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return GTAVDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                               split=mode, transform=transform)
        elif "synthia" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return synthiaDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                  split=mode, transform=transform)

        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return cityscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform, cfg=cfg, empty=empty)

        raise RuntimeError("Dataset not available: {}".format(name))

    @staticmethod
    def initMask(cfg):
        if cfg.DEBUG == 1:
            for i in range(10):
                print("Debug without mask initialization!")
            return
        data_dir = DatasetCatalog.DATASET_DIR
        attrs = DatasetCatalog.DATASETS['cityscapes_train']
        data_list = os.path.join(data_dir, attrs["data_list"])
        root = os.path.join(data_dir, attrs["data_dir"])
        with open(data_list, "r") as handle:
            content = handle.readlines()
        for fname in tqdm(content):
            name = fname.strip()
            path2image = os.path.join(root, "leftImg8bit/%s/%s" % ('train', name))
            path2mask = os.path.join(
                cfg.OUTPUT_DIR,
                "gtMask/%s/%s"
                % (
                    "train",
                    name.split("_leftImg8bit")[0]
                    + "_gtFine_labelIds.png",
                ),
            )
            path2indicator = os.path.join(
                cfg.OUTPUT_DIR,
                "gtIndicator/%s/%s"
                % (
                    "train",
                    name.split("_leftImg8bit")[0]
                    + "_indicator.pth",
                ),
            )
            mask_dir = os.path.join("%s/gtMask/train/%s" % (cfg.OUTPUT_DIR, name.split("/")[0]))
            indicator_dir = os.path.join("%s/gtIndicator/train/%s" % (cfg.OUTPUT_DIR, name.split("/")[0]))

            # mkdir
            mkdir_path(mask_dir)
            mkdir_path(indicator_dir)

            img = Image.open(path2image).convert('RGB')
            h, w = img.size[1], img.size[0]
            mask = np.ones((h, w), dtype=np.uint8) * 255
            mask = Image.fromarray(mask)
            mask.save(path2mask)

            indicator = {
                'active': torch.tensor([0], dtype=torch.bool),
                'selected': torch.tensor([0], dtype=torch.bool),
            }
            torch.save(indicator, path2indicator)


def mkdir_path(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
