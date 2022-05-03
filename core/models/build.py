from .feature_extractor import resnet_feature_extractor
from .classifier import ASPP_Classifier_V2, DepthwiseSeparableASPP
from .layers import FrozenBatchNorm2d
import torch.nn as nn


def build_feature_extractor(cfg):
    model_name, backbone_name = cfg.MODEL.NAME.split('_')
    if backbone_name.startswith('resnet'):
        backbone = resnet_feature_extractor(backbone_name, pretrained_weights=cfg.MODEL.WEIGHTS, aux=False,
                                            pretrained_backbone=True, freeze_bn=cfg.MODEL.FREEZE_BN)
    else:
        raise NotImplementedError
    return backbone


def build_classifier(cfg):
    deeplab_name, backbone_name = cfg.MODEL.NAME.split('_')
    bn_layer = nn.BatchNorm2d
    if cfg.MODEL.FREEZE_BN:
        bn_layer = FrozenBatchNorm2d

    if deeplab_name == 'deeplabv2':
        classifier = ASPP_Classifier_V2(2048, [6, 12, 18, 24], [6, 12, 18, 24], cfg.MODEL.NUM_CLASSES)
    elif deeplab_name =='deeplabv3plus':
        if backbone_name.startswith('resnet'):
            classifier = DepthwiseSeparableASPP(inplanes=2048, dilation_series=[1, 6, 12, 18],
                                                padding_series=[1, 6, 12, 18], num_classes=cfg.MODEL.NUM_CLASSES,
                                                norm_layer=bn_layer)
    else:
        raise NotImplementedError
    return classifier

