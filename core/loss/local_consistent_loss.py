import torch.nn as nn
from core.loss.boundary import DetectSPBoundary, LocalDiscrepancy


class LocalConsistentLoss(nn.Module):
    def __init__(self, in_channels, l_type='l1'):
        super(LocalConsistentLoss, self).__init__()
        self.semantic_boundary = DetectSPBoundary(padding_mode='zeros')
        self.neighbor_dif = LocalDiscrepancy(in_channels=in_channels, padding_mode='replicate', l_type=l_type)

    def forward(self, x, label):
        discrepancy = self.neighbor_dif(x)
        mask = self.semantic_boundary(label)
        mask = mask & (label != 255)
        loss = discrepancy[mask].mean()
        return loss
