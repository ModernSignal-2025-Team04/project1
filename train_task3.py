# train_task3.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_task3 import Task3AugDataset
from unet import UNet2D
from metrics import dice_loss, dice_coeff

# -------------------------------
# 超参数（保持和 Task1 一致）
# -------------------------------
LR = 1e-3
EPOCHS = 10
BATCH = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_task3():
    print("Using device:", DEVICE)

    # 使用 Task3 的增强数据集
    dataset = Task3AugDataset(use_feddg=True, use_cyclegan=True)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)

    # UNet
    model = UNet2D(in_ch=1, num_classes=1).to(DEVICE)

    bce = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    best_dice = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for imgs, masks in pbar:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            optim.zero_grad()
            logits = model(imgs)

            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # epoch summary
        print(f"Epoch {epoch}: Loss={epoch_loss/len(loader):.4f}")

        # simple dice check
        model.eval()
        with torch.no_grad():
            logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.5).float()
            dice = dice_coeff(preds.cpu(), masks.cpu())

        print(f"Train Dice: {dice:.4f}")

        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), "checkpoints/task3_unet_augmix.pth")
            print(f" >>> Saved new best model! Dice={dice:.4f}")

    print("Training done. Best Dice:", best_dice)


if __name__ == "__main__":
    train_task3()
