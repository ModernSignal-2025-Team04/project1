# train_task1.py
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_faz import FazSegDataset
from model_zoo import get_model
from metrics import dice_coeff


# ==========================
# 修改这里来切换增强方式
# 可选: "none", "flip", "flip_rotate"
# ==========================
AUG_MODE = "none"
# ==========================
# 模型名称，可选 "UNet2D", "NestedUNet", "TransUNet", "nnUNet"
# ==========================
MODEL_NAME = "UNet2D"


# Dice Loss（用概率计算）
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Domain1 的 train 数据
    train_set = FazSegDataset(domain_id=1, split="train", aug=AUG_MODE)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)

    # 模型
    model = get_model(MODEL_NAME, in_ch=1, num_classes=1).to(device)

    # 损失函数
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10  
    
    # 保存模型的目录
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{MODEL_NAME}_faz_aug_{AUG_MODE}.pth"

    best_dice = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)

            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_set)
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}")

        # ---------- 训练集上的粗略 Dice ----------
        model.eval()
        all_pred = []
        all_gt = []
        with torch.no_grad():
            for imgs, masks in train_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                preds = (torch.sigmoid(logits) > 0.5).float()

                all_pred.append(preds.cpu())
                all_gt.append(masks.cpu())

        all_pred = torch.cat(all_pred)
        all_gt = torch.cat(all_gt)
        dice = dice_coeff(all_pred, all_gt)
        print(f"Train Dice: {dice:.4f}")

        # 保存最好的模型
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >>> Saved new best model! Dice={best_dice:.4f}")

    print("Training finished. Best Dice:", best_dice)


if __name__ == "__main__":
    train()
