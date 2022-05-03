import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image


class cityscapesDataSet(data.Dataset):
    def __init__(
            self,
            data_root,
            data_list,
            max_iters=None,
            num_classes=19,
            split="train",
            transform=None,
            ignore_label=255,
            debug=False,
            cfg=None,
            empty=False,
    ):
        self.active = True if split == 'active' else False
        if split == 'active':
            split = 'train'
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.cfg = cfg
        self.empty = empty
        with open(data_list, "r") as handle:
            content = handle.readlines()

        self.data_list = []
        if empty:
            self.data_list.append(
                {
                    "img": "",
                    "label": "",
                    "label_mask": "",
                    "name": "",
                }
            )
        else:
            for fname in content:
                name = fname.strip()
                self.data_list.append(
                    {
                        "img": os.path.join(
                            self.data_root, "leftImg8bit/%s/%s" % (self.split, name)
                        ),
                        "label": os.path.join(
                            self.data_root,
                            "gtFine/%s/%s"
                            % (
                                self.split,
                                name.split("_leftImg8bit")[0]
                                + "_gtFine_labelIds.png",
                            ),
                        ),
                        "label_mask": os.path.join(
                            self.cfg.OUTPUT_DIR,
                            "gtMask/%s/%s"
                            % (
                                self.split,
                                name.split("_leftImg8bit")[0]
                                + "_gtFine_labelIds.png",
                            ),
                        ),
                        "name": name,
                        'indicator': os.path.join(
                            cfg.OUTPUT_DIR,
                            "gtIndicator/%s/%s"
                            % (
                                "train",
                                name.split("_leftImg8bit")[0]
                                + "_indicator.pth",
                            ),
                        )
                    }
                )

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        # --------------------------------------------------------------------------------
        # A list of all labels
        # --------------------------------------------------------------------------------

        # Please adapt the train IDs as appropriate for your approach.
        # Note that you might want to ignore labels with ID 255 during training.
        # Further note that the current train IDs are only a suggestion. You can use whatever you like.
        # Make sure to provide your results using the original IDs and not the training IDs.
        # Note that many IDs are ignored in evaluation and thus you never need to predict these!

        # labels = [
        #     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        #     Label('unlabeled',                0,      255,    'void',             0,          False,          True,       (0, 0, 0)),
        #     Label('ego vehicle',              1,      255,    'void',             0,          False,          True,       (0, 0, 0)),
        #     Label('rectification border',     2,      255,    'void',             0,          False,          True,       (0, 0, 0)),
        #     Label('out of roi',               3,      255,    'void',             0,          False,          True,       (0, 0, 0)),
        #     Label('static',                   4,      255,    'void',             0,          False,          True,       (0, 0, 0)),
        #     Label('dynamic',                  5,      255,    'void',             0,          False,          True,       (111, 74, 0)),
        #     Label('ground',                   6,      255,    'void',             0,          False,          True,       (81, 0, 81)),
        #     Label('road',                     7,      0,      'flat',             1,          False,          False,      (128, 64, 128)),
        #     Label('sidewalk',                 8,      1,      'flat',             1,          False,          False,      (244, 35, 232)),
        #     Label('parking',                  9,      255,    'flat',             1,          False,          True,       (250, 170, 160)),
        #     Label('rail track',               10,     255,    'flat',             1,          False,          True,       (230, 150, 140)),
        #     Label('building',                 11,     2,      'construction',     2,          False,          False,      (70, 70, 70)),
        #     Label('wall',                     12,     3,      'construction',     2,          False,          False,      (102, 102, 156)),
        #     Label('fence',                    13,     4,      'construction',     2,          False,          False,      (190, 153, 153)),
        #     Label('guard rail',               14,     255,    'construction',     2,          False,          True,       (180, 165, 180)),
        #     Label('bridge',                   15,     255,    'construction',     2,          False,          True,       (150, 100, 100)),
        #     Label('tunnel',                   16,     255,    'construction',     2,          False,          True,       (150, 120, 90)),
        #     Label('pole',                     17,     5,      'object',           3,          False,          False,      (153, 153, 153)),
        #     Label('polegroup',                18,     255,    'object',           3,          False,          True,       (153, 153, 153)),
        #     Label('traffic light',            19,     6,      'object',           3,          False,          False,      (250, 170, 30)),
        #     Label('traffic sign',             20,     7,      'object',           3,          False,          False,      (220, 220, 0)),
        #     Label('vegetation',               21,     8,      'nature',           4,          False,          False,      (107, 142, 35)),
        #     Label('terrain',                  22,     9,      'nature',           4,          False,          False,      (152, 251, 152)),
        #     Label('sky',                      23,     10,     'sky',              5,          False,          False,      (70, 130, 180)),
        #     Label('person',                   24,     11,     'human',            6,          True,           False,      (220, 20, 60)),
        #     Label('rider',                    25,     12,     'human',            6,          True,           False,      (255, 0, 0)),
        #     Label('car',                      26,     13,     'vehicle',          7,          True,           False,      (0, 0, 142)),
        #     Label('truck',                    27,     14,     'vehicle',          7,          True,           False,      (0, 0, 70)),
        #     Label('bus',                      28,     15,     'vehicle',          7,          True,           False,      (0, 60, 100)),
        #     Label('caravan',                  29,     255,    'vehicle',          7,          True,           True,       (0, 0, 90)),
        #     Label('trailer',                  30,     255,    'vehicle',          7,          True,           True,       (0, 0, 110)),
        #     Label('train',                    31,     16,     'vehicle',          7,          True,           False,      (0, 80, 100)),
        #     Label('motorcycle',               32,     17,     'vehicle',          7,          True,           False,      (0, 0, 230)),
        #     Label('bicycle',                  33,     18,     'vehicle',          7,          True,           False,      (119, 11, 32)),
        #     Label('license plate',             -1,    -1,     'vehicle',          7,          False,          True,       (0, 0, 142)),
        # ]

        # GTAV
        self.id_to_trainid = {
            7: 0,
            8: 1,
            11: 2,
            12: 3,
            13: 4,
            17: 5,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            31: 16,
            32: 17,
            33: 18,
        }
        self.trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motocycle",
            18: "bicycle",
        }
        if self.NUM_CLASS == 16:  # SYNTHIA
            self.id_to_trainid = {
                7: 0,
                8: 1,
                11: 2,
                12: 3,
                13: 4,
                17: 5,
                19: 6,
                20: 7,
                21: 8,
                23: 9,
                24: 10,
                25: 11,
                26: 12,
                28: 13,
                32: 14,
                33: 15,
            }
            self.trainid2name = {
                0: "road",
                1: "sidewalk",
                2: "building",
                3: "wall",
                4: "fence",
                5: "pole",
                6: "light",
                7: "sign",
                8: "vegetation",
                9: "sky",
                10: "person",
                11: "rider",
                12: "car",
                13: "bus",
                14: "motocycle",
                15: "bicycle",
            }
        self.transform = transform

        self.ignore_label = ignore_label

        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = np.array(Image.open(datafiles["label"]), dtype=np.uint8)
        label_mask = None
        if self.split == 'train':
            label_mask = np.array(Image.open(datafiles["label_mask"]), dtype=np.uint8)
        else:
            # test or val, mask is useless
            label_mask = np.ones_like(label, dtype=np.uint8) * 255

        # for generate new mask
        origin_mask = torch.from_numpy(label_mask).long()

        active_indicator = torch.tensor([0])
        active_selected = torch.tensor([0])
        if self.active:
            indicator = torch.load(datafiles['indicator'])
            active_indicator = indicator['active']
            active_selected = indicator['selected']
            # if first time load, initialize it
            if active_indicator.size() == (1,):
                active_indicator = torch.zeros_like(origin_mask, dtype=torch.bool)
                active_selected = torch.zeros_like(origin_mask, dtype=torch.bool)

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = np.array(label_copy, dtype=np.uint8)

        origin_label = torch.from_numpy(label).long()

        label.resize(label.shape[0], label.shape[1], 1)
        label_mask.resize(label_mask.shape[0], label_mask.shape[1], 1)

        h, w = label.shape[0], label.shape[1]

        mask_aggregation = np.concatenate((label, label_mask), axis=2)
        mask_aggregation = Image.fromarray(mask_aggregation)

        if self.transform is not None:
            image, mask_aggregation = self.transform(image, mask_aggregation)
            label = mask_aggregation[:, :, 0]
            label_mask = mask_aggregation[:, :, 1]

        ret_data = {
            "img": image,  # data
            'label': label,  # for test
            'mask': label_mask,  # for train
            'name': datafiles['name'],  # for test to store the results
            'path_to_mask': datafiles['label_mask'],  # for active to store new mask
            'path_to_indicator': datafiles['indicator'],  # store new indicator
            'size': torch.tensor([h, w]),  # for active to interpolate the output to original size
            'origin_mask': origin_mask,  # mask without transforms for active
            'origin_label': origin_label,  # label without transforms for active
            'active': active_indicator,  # indicate region or pixels can not be selected
            'selected': active_selected,  # indicate the pixel have been selected, can calculate the class-wise ratio of selected samples
        }

        return ret_data