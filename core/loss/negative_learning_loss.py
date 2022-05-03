import torch
import torch.nn as nn
import torch.nn.functional as F


class NegativeLearningLoss(nn.Module):
    def __init__(self, threshold=0.05):
        super(NegativeLearningLoss, self).__init__()
        self.threshold = threshold

    def forward(self, predict):
        mask = (predict < self.threshold).detach()
        negative_loss_item = -1 * mask * torch.log(1 - predict + 1e-6)
        negative_loss = torch.sum(negative_loss_item) / torch.sum(mask)

        return negative_loss


