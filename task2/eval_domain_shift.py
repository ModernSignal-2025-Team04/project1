# eval_domain_shift.py
# 评估：原始domain1、FedDG结果、CycleGAN结果，相对domain3的特征距离

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
from pathlib import Path
import imageio.v2 as imageio
import torchvision.transforms.functional as F

from dataset_faz import FazSegDataset
from unet import UNet2D


# ---------- 用UNet的encoder做特征提取 ----------

class UNetEncoderFeature(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        # 和之前 UMAP 那个一致：取 down2 层并做全局平均池化
        x1 = self.unet.inc(x)
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        feat = torch.mean(x3, dim=(2, 3))  # [B, C]
        return feat


# ---------- 从 FAZ 数据集中提特征（domain1,3） ----------

def extract_features_faz(domain_id, encoder, device, n_samples=60):
    dataset = FazSegDataset(domain_id, split="train", aug="none")
    feats = []

    for i in range(min(n_samples, len(dataset))):
        img, _ = dataset[i]              # [1,H,W]
        img = img.unsqueeze(0).to(device)  # [1,1,H,W]
        with torch.no_grad():
            f = encoder(img)             # [1,C]
        feats.append(f.cpu().numpy()[0])

    return np.array(feats)


# ---------- 从文件夹中读迁移后的图像并提特征 ----------

def extract_features_from_folder(folder, encoder, device, n_samples=60):
    folder = Path(folder)
    paths = sorted(list(folder.glob("*.png")))
    if len(paths) == 0:
        raise RuntimeError(f"No png images found in {folder}")

    feats = []
    for i, p in enumerate(paths):
        if i >= n_samples:
            break
        img_np = imageio.imread(p)  # H x W (灰度)
        img_np = img_np.astype(np.float32) / 255.0

        img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        # 和训练时保持一致：resize到256x256
        img_t = F.resize(img_t, (256, 256))
        img_t = img_t.to(device)

        with torch.no_grad():
            f = encoder(img_t)
        feats.append(f.cpu().numpy()[0])

    return np.array(feats)


# ---------- 计算两组特征的L2距离 ----------

def mean_l2_distance(A, B):
    """
    A, B: [N,C] 或 [M,C]，先取均值再算L2距离
    """
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    return float(np.linalg.norm(mu_A - mu_B))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # 1. 加载Task1训练好的UNet做特征提取（更有意义）
    ckpt = Path(PROJECT_ROOT) / "checkpoints" / "unet_faz_aug_none.pth"
    unet = UNet2D(in_ch=1, num_classes=1)
    unet.load_state_dict(torch.load(ckpt, map_location=device))
    unet.eval()
    encoder = UNetEncoderFeature(unet).to(device)

    # 2. 提取四类的数据特征
    print("Extracting features for domain1 (original)...")
    feats_d1 = extract_features_faz(1, encoder, device)

    print("Extracting features for domain3 (target)...")
    feats_d3 = extract_features_faz(3, encoder, device)

    print("Extracting features for FedDG(domain1→3)...")
    feats_fed = extract_features_from_folder("fed_dg_results", encoder, device)

    print("Extracting features for CycleGAN(domain1→3)...")
    feats_cyc = extract_features_from_folder("cyclegan_generated", encoder, device)

    # 3. 计算相对domain3的距离
    d_d1_d3  = mean_l2_distance(feats_d1,  feats_d3)
    d_fed_d3 = mean_l2_distance(feats_fed, feats_d3)
    d_cyc_d3 = mean_l2_distance(feats_cyc, feats_d3)

    print("\n===== Feature-space distances (mean L2) to Domain3 =====")
    print(f"Original Domain1  vs Domain3: {d_d1_d3:.4f}")
    print(f"FedDG(1→3)       vs Domain3: {d_fed_d3:.4f}")
    print(f"CycleGAN(1→3)    vs Domain3: {d_cyc_d3:.4f}")

    print("\nInterpretation:")
    print("  数值越小，说明这一类图像在UNet特征空间里越接近Domain3。")
