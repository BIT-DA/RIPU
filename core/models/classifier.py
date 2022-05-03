import torch
from torch import nn
import torch.nn.functional as F


class ASPP_Classifier_V2(nn.Module):
    def __init__(self, in_channels, dilation_series, padding_series, num_classes):
        super(ASPP_Classifier_V2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(
                    in_channels,
                    num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                )
            )

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x, size=None):
        x = x['out']
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 norm_layer=None):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.depthwise_bn = norm_layer(in_channels)
        self.depthwise_activate = nn.ReLU(inplace=True)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                        groups=1, bias=bias)
        self.pointwise_bn = norm_layer(out_channels)
        self.pointwise_activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_activate(x)
        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)
        x = self.pointwise_activate(x)
        return x


class DepthwiseSeparableASPP(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes, norm_layer):
        super(DepthwiseSeparableASPP, self).__init__()

        out_channels = 512
        # build aspp net
        self.parallel_branches = nn.ModuleList()
        for idx, dilation in enumerate(dilation_series):
            if dilation == 1:
                branch = nn.Sequential(
                    nn.Conv2d(inplanes, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    norm_layer(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                branch = DepthwiseSeparableConv2d(inplanes, out_channels, kernel_size=3, stride=1, padding=dilation,
                                                  dilation=dilation, bias=False, norm_layer=norm_layer)
            self.parallel_branches.append(branch)

        self.global_branch = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                           nn.Conv2d(inplanes, out_channels, 1, stride=1, padding=0, bias=False),
                                           norm_layer(out_channels),
                                           nn.ReLU(inplace=True))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilation_series) + 1), out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

        # build shortcut
        shortcut_inplanes = 256
        shortcut_out_channels = 48
        self.shortcut = nn.Sequential(
            nn.Conv2d(shortcut_inplanes, shortcut_out_channels, 1, bias=False),
            norm_layer(shortcut_out_channels),
            nn.ReLU(inplace=True)
        )

        decoder_inplanes = 560
        decoder_out_channels = 512
        self.decoder = nn.Sequential(
            DepthwiseSeparableConv2d(decoder_inplanes, decoder_out_channels, kernel_size=3, stride=1, padding=1,
                                     bias=False, norm_layer=norm_layer),
            DepthwiseSeparableConv2d(decoder_out_channels, decoder_out_channels, kernel_size=3, stride=1, padding=1,
                                     bias=False, norm_layer=norm_layer),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_out_channels, num_classes, kernel_size=1, stride=1, padding=0))

        self._init_weight()

    def forward(self, x, size=None):

        # fed to backbone
        low_level_feat = x['low']
        x = x['out']

        # feed to aspp
        aspp_out = []
        for branch in self.parallel_branches:
            aspp_out.append(branch(x))
        global_features = self.global_branch(x)
        global_features = F.interpolate(global_features, size=x.size()[2:], mode='bilinear', align_corners=True)
        aspp_out.append(global_features)
        aspp_out = torch.cat(aspp_out, dim=1)
        aspp_out = self.bottleneck(aspp_out)
        aspp_out = F.interpolate(aspp_out, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)

        # feed to shortcut
        shortcut_out = self.shortcut(low_level_feat)

        # feed to decoder
        feats = torch.cat([aspp_out, shortcut_out], dim=1)
        out = self.decoder(feats)

        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
