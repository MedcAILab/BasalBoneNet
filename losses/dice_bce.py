import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape(pred.size(0), -1)
        target = target.reshape(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        pred_sum = pred.sum(dim=1)
        target_sum = target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
