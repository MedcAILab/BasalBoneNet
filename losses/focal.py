# -*- coding:utf-8 -*-

import torch


def focal_loss(predict, target, pos_thresh=0.9, alpha=2.0, beta=4.0, eps=1e-6):
    """CenterNet-style focal loss for sigmoid heatmaps.

    Args:
        predict: predicted heatmap after sigmoid, shape [B, C, D, H, W].
        target: Gaussian target heatmap in [0, 1], same shape as predict.
    """
    predict = torch.clamp(predict, min=eps, max=1.0 - eps)

    pos_inds = (target >= pos_thresh).float()
    neg_inds = (target < pos_thresh).float()
    neg_weights = torch.pow(1.0 - target, beta)

    pos_loss = torch.log(predict) * torch.pow(1.0 - predict, alpha) * pos_inds
    neg_loss = torch.log(1.0 - predict) * torch.pow(predict, alpha) * neg_weights * neg_inds

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    num_pos = pos_inds.sum()

    if num_pos.item() > 0:
        loss = -(pos_loss + neg_loss) / num_pos
    else:
        # Avoid giant loss spikes for batches without confident positive voxels.
        num_neg = neg_inds.sum().clamp_min(1.0)
        loss = -neg_loss / num_neg

    return loss
