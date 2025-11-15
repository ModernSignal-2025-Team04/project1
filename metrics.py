# metrics.py
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


# ------------------------------
# Dice 系数
# ------------------------------
def dice_coeff(pred, target, eps=1e-6):
    """
    pred, target: torch tensors, shape [N,1,H,W], 值为0/1
    """
    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()

def dice_loss(logits, targets, eps=1e-6):
    """
    logits: 未经过 sigmoid 的模型输出
    targets: 0/1 mask
    """
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1 - dice.mean()

# ------------------------------
# surface distance（ASSD、HD95辅助）
# ------------------------------
def _surface_distances(pred, target, spacing=(1.0, 1.0)):
    """
    pred, target: numpy bool arrays, shape [H,W]
    返回 pred 到 target 的表面距离（和相反方向）
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    # 如果两个都是空 mask，则距离为0
    if not pred.any() and not target.any():
        return np.array([0.0])

    # 计算距离变换
    dt_pred = distance_transform_edt(~pred, sampling=spacing)
    dt_target = distance_transform_edt(~target, sampling=spacing)

    # 提取表面距离
    sds_pred_to_target = dt_target[pred]
    sds_target_to_pred = dt_pred[target]

    return np.concatenate([sds_pred_to_target, sds_target_to_pred])


# ------------------------------
# ASSD 平均对称表面距离
# ------------------------------
def assd(pred, target, spacing=(1.0, 1.0)):
    dists = _surface_distances(pred, target, spacing)
    return float(dists.mean())


# ------------------------------
# HD95（95% Hausdorff）
# ------------------------------
def hd95(pred, target, spacing=(1.0, 1.0)):
    dists = _surface_distances(pred, target, spacing)
    return float(np.percentile(dists, 95))
