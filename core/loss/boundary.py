import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectSPBoundary(nn.Module):
    """
    detect boundary for superpixel, give the superpixel bool mask, return the bool boundary of the superpixel
    """

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, neighbor=8, padding_mode='zeros'):
        """
        padding_mode: 'zeros', 'reflect', 'replicate', 'circular'
        """
        super(DetectSPBoundary, self).__init__()
        # have not been explored
        if kernel_size != 3:
            raise NotImplementedError
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=1, padding=int(kernel_size / 2), bias=False, padding_mode=padding_mode)
        if neighbor == 8:
            a = torch.tensor([[[[-1., -1., -1.],
                                [-1., 8., -1.],
                                [-1., -1., -1.]]]])
        elif neighbor == 4:
            a = torch.tensor([[[[0., -1., 0.],
                                [-1., 4., -1.],
                                [0., -1., 0.]]]])
        else:
            raise NotImplementedError
        # a = a.repeat([1, in_channels, 1, 1])
        a = nn.Parameter(a)
        self.conv.weight = a
        self.conv.requires_grad_(False)

    def forward(self, mask):
        """
        mask:
            (h, w) bool, detect the boundary of the true region
            (b, h, w) long, detect the semantic boundary
        """
        if len(mask.size()) == 2:
            x = mask.float()
            x = x.unsqueeze(dim=0).unsqueeze(dim=0)
            out = self.conv(x)
            out = out.long()
            out = out.squeeze(dim=0).squeeze(dim=0)
            pre_boundary = (out != 0)
            boundary = pre_boundary & mask
            # (h, w)
            return boundary
        elif len(mask.size()) == 3:
            x = mask.float()
            x = x.unsqueeze(dim=1)
            out = self.conv(x)
            out = out.long()
            out = out.squeeze(dim=1)
            pre_boundary = (out != 0)
            # (b, h, w)
            return pre_boundary


class LocalDiscrepancy(nn.Module):

    def __init__(self, in_channels=19, padding_mode='replicate', neighbor=8, l_type="l1"):
        """
        depth-wise conv to calculate the mean of neighbor
        """
        super(LocalDiscrepancy, self).__init__()
        self.type = l_type
        self.mean_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                   stride=1, padding=int(3 / 2), bias=False, padding_mode=padding_mode,
                                   groups=in_channels)
        if neighbor == 8:
            a = torch.tensor([[[[1., 1., 1.],
                                [1., 1., 1.],
                                [1., 1., 1.]]]]) / 9
        elif neighbor == 4:
            a = torch.tensor([[[[0., 1., 0.],
                                [1., 1., 1.],
                                [0., 1., 0.]]]]) / 5
        else:
            raise NotImplementedError
        a = a.repeat([in_channels, 1, 1, 1])
        a = nn.Parameter(a)
        self.mean_conv.weight = a
        self.mean_conv.requires_grad_(False)

    def forward(self, x):
        p = torch.softmax(x, dim=1)
        mean = self.mean_conv(p)
        l = None
        if self.type == "l1":
            l = torch.abs(p - mean).sum(dim=1)
        elif self.type == "kl":
            l = torch.sum(p * torch.log(p / (mean + 1e-6) + 1e-6), dim=1)
        else:
            raise NotImplementedError("not implemented local soft loss: {}".format(self.type))
        return l
