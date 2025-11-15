import sys
import os

# 把 project1 的根目录加入 Python 搜索路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# fft_visualize.py
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset_faz import FazSegDataset

# 选择要展示的 domain
DOMAIN_ID = 1   # 你可以改成 1~5

def visualize_fft(domain_id):
    # 加载一个 domain 的一个样本（不需要mask）
    dataset = FazSegDataset(domain_id=domain_id, split="train", aug="none")
    img, _ = dataset[0]   # img: [1, H, W]

    img_np = img.squeeze(0).numpy()  # H x W

    # ----- 1. 计算 FFT -----
    F = np.fft.fft2(img_np)
    F_shift = np.fft.fftshift(F)

    amplitude = np.abs(F_shift)
    phase = np.angle(F_shift)

    # ----- 2. 对 amplitude 做对数灰度处理（常用技巧） -----
    amplitude_log = np.log(1 + amplitude)

    # ----- 3. 可视化 -----
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title(f"Domain {domain_id} Image")
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Amplitude Spectrum (log)")
    plt.imshow(amplitude_log, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Phase Spectrum")
    plt.imshow(phase, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_fft(DOMAIN_ID)
