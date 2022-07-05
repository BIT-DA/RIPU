import os

from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = "deeplabv2_resnet101"
_C.MODEL.NUM_CLASSES = 19
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHTS = ""
_C.MODEL.FREEZE_BN = False

_C.INPUT = CN()
_C.INPUT.SOURCE_INPUT_SIZE_TRAIN = (1280, 720)
_C.INPUT.TARGET_INPUT_SIZE_TRAIN = (1280, 640)
_C.INPUT.INPUT_SIZE_TEST = (1280, 640)
_C.INPUT.INPUT_SCALES_TRAIN = (1.0, 1.0)
_C.INPUT.IGNORE_LABEL = 255
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = False

_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.SOURCE_TRAIN = ""
_C.DATASETS.TARGET_TRAIN = ""
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ""

_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 60000
_C.SOLVER.STOP_ITER = 40000
_C.SOLVER.CHECKPOINT_PERIOD = 10000

_C.SOLVER.LR_METHOD = 'poly'
_C.SOLVER.BASE_LR = 2.5e-4
_C.SOLVER.LR_POWER = 0.9
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005

# 4 images per batch, two for source and two for target
_C.SOLVER.BATCH_SIZE = 2
_C.SOLVER.BATCH_SIZE_VAL = 1

# hyper-parameters
_C.SOLVER.CONSISTENT_LOSS = 0.1
_C.SOLVER.NEGATIVE_LOSS = 1.0
_C.SOLVER.NEGATIVE_THRESHOLD = 0.05

# local consistent loss
_C.SOLVER.LCR_TYPE = "l1"


_C.ACTIVE = CN()

# active strategy
_C.ACTIVE.NAME = 'AL-RIPU'
_C.ACTIVE.SETTING = 'RA'
_C.ACTIVE.SELECT_ITER = [10000, 12000, 14000, 16000, 18000] # for 5 selection rounds
# total selection ratio per image
_C.ACTIVE.RATIO = 0.022
# total selected pixel per image
_C.ACTIVE.PIXELS = 40
# k-neighbor-radius
_C.ACTIVE.RADIUS_K = 1

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1

_C.OUTPUT_DIR = ""
_C.resume = ""
_C.PREPARE_DIR = ""
_C.SEED = -1
_C.DEBUG = 0
