# fft_batch.py

import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from dataset_faz import FazSegDataset

SAVE_DIR = "fft_results"
os.makedirs(SAVE_DIR, exist_ok=True)

N_IMAGES = 5   # 每个 domain 保存前 5 张，你可改成 3/10/20


def save_fft_for_domain(domain_id):
    dataset = FazSegDataset(domain_id=domain_id, split="train", aug="none")

    for i in range(min(N_IMAGES, len(dataset))):
        img, _ = dataset[i]
        img_np = img.squeeze(0).numpy()

        # FFT
        F = np.fft.fft2(img_np)
        F_shift = np.fft.fftshift(F)

        amplitude = np.abs(F_shift)
        amplitude_log = np.log(1 + amplitude)
        phase = np.angle(F_shift)

        # Save figure
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(img_np, cmap="gray")
        plt.title(f"Domain {domain_id} image {i}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(amplitude_log, cmap="gray")
        plt.title("Amplitude (log)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(phase, cmap="gray")
        plt.title("Phase")
        plt.axis("off")

        save_path = os.path.join(SAVE_DIR, f"domain{domain_id}_img{i}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    for dom in range(1, 6):
        save_fft_for_domain(dom)
