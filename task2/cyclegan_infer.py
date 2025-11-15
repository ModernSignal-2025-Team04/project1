# cyclegan_infer.py

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import torch
import imageio.v2 as imageio
from dataset_faz import FazSegDataset
from cyclegan_train import ResnetGenerator
import numpy as np
from tqdm import tqdm

SAVE_DIR = "cyclegan_generated"
os.makedirs(SAVE_DIR, exist_ok=True)

CKPT = "cyclegan_results/G12.pth"   # domain1 → domain3


def generate_images():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # 加载生成器
    G12 = ResnetGenerator().to(device)
    G12.load_state_dict(torch.load(CKPT, map_location=device))
    G12.eval()

    # domain1 训练集
    dataset = FazSegDataset(1, split="train", aug="none")

    print("Generating CycleGAN images...")
    for idx in tqdm(range(len(dataset))):
        img, _ = dataset[idx]
        img = img.to(device)

        with torch.no_grad():
            fake_t = G12(img)                # tensor
            fake_np = fake_t.cpu().numpy()   # numpy array with any shape
            fake_np = np.squeeze(fake_np)    # force to (H, W)




        # CycleGAN 输出是 tanh → [-1,1]，需要恢复到 [0,255]
        fake_np = (fake_np + 1) / 2.0  # normalize
        fake_np = np.clip(fake_np * 255, 0, 255).astype(np.uint8)

        save_path = os.path.join(SAVE_DIR, f"domain1_to_3_cyclegan_{idx}.png")
        imageio.imwrite(save_path, fake_np)

    print("Saved to:", SAVE_DIR)


if __name__ == "__main__":
    generate_images()
