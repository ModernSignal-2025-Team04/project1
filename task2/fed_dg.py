# fed_dg.py

import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import numpy as np
import os
from dataset_faz import FazSegDataset
from pathlib import Path
from tqdm import tqdm
import imageio.v2 as imageio


# ========== 参数 ==========
# dice 最低的目标域（从 Task1 得出的）
SRC_DOMAIN = 1
TGT_DOMAIN = 3

# 低频比例（0.1~0.2 常用）
LOW_FREQ_RATIO = 0.1

SAVE_DIR = "fed_dg_results"
os.makedirs(SAVE_DIR, exist_ok=True)


def fft_img(img_np):
    F = np.fft.fft2(img_np)
    F_shift = np.fft.fftshift(F)
    return F_shift


def ifft_img(F_shift):
    F = np.fft.ifftshift(F_shift)
    img_back = np.fft.ifft2(F)
    return np.real(img_back)


def fed_dg_transfer(img_src, img_tgt, low_ratio=0.1):
    H, W = img_src.shape

    # FFT
    Fs = fft_img(img_src)
    Ft = fft_img(img_tgt)

    # 计算低频窗口大小
    bH = int(H * low_ratio)  # 低频 height
    bW = int(W * low_ratio)  # 低频 width

    cH, cW = H // 2, W // 2  # 中心位置

    # ========= 替换低频区域 =========
    Fs[cH - bH : cH + bH, cW - bW : cW + bW] = Ft[cH - bH : cH + bH, cW - bW : cW + bW]

    # 逆 FFT
    new_img = ifft_img(Fs)

    # 归一化回 0~255
    new_img = new_img - new_img.min()
    new_img = new_img / (new_img.max() + 1e-6)
    new_img = (new_img * 255).astype(np.uint8)

    return new_img


def generate_fed_dg_images():
    src_set = FazSegDataset(SRC_DOMAIN, split="train", aug="none")
    tgt_set = FazSegDataset(TGT_DOMAIN, split="train", aug="none")

    # 保证 target 和 source 数量一样（如果不一样就循环）
    tgt_imgs = []
    for i in range(len(src_set)):
        tgt_imgs.append(tgt_set[i % len(tgt_set)][0].squeeze(0).numpy())

    print("Generating FedDG images...")

    for idx in tqdm(range(len(src_set))):
        img_src, _ = src_set[idx]
        img_src = img_src.squeeze(0).numpy()

        img_tgt = tgt_imgs[idx]

        new_img = fed_dg_transfer(img_src, img_tgt, LOW_FREQ_RATIO)

        save_path = os.path.join(SAVE_DIR, f"domain1_to_3_{idx}.png")
        imageio.imwrite(save_path, new_img)

    print("Saved FedDG images to:", SAVE_DIR)


if __name__ == "__main__":
    generate_fed_dg_images()
