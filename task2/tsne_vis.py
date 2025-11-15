# tsne_vis.py  —— 使用 UMAP 替代 t-SNE（稳定不崩溃）

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
from dataset_faz import FazSegDataset
from unet import UNet2D
import umap
import matplotlib.pyplot as plt


class UNetEncoderFeature(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        x1 = self.unet.inc(x)
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        feat = torch.mean(x3, dim=(2,3))  # → [B,256]
        return feat


def extract_features(domain_id, encoder, n_samples=40):
    dataset = FazSegDataset(domain_id, split="train", aug="none")
    feats = []

    for i in range(min(n_samples, len(dataset))):
        img, _ = dataset[i]
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            f = encoder(img)
        feats.append(f.cpu().numpy()[0])

    return np.array(feats)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    unet = UNet2D()
    encoder = UNetEncoderFeature(unet).to(device)
    unet.eval()

    all_feats = []
    all_labels = []

    print("Extracting features...")

    for dom in range(1, 6):
        feats = extract_features(dom, encoder)
        all_feats.append(feats)
        all_labels += [dom] * len(feats)

    all_feats = np.vstack(all_feats)

    print("Running UMAP (stable)...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )
    X_embedded = reducer.fit_transform(all_feats)

    plt.figure(figsize=(8, 6))
    colors = ["red", "blue", "green", "purple", "orange"]

    for dom in range(1, 6):
        idx = [i for i, l in enumerate(all_labels) if l == dom]
        plt.scatter(
            X_embedded[idx, 0],
            X_embedded[idx, 1],
            s=20,
            c=colors[dom - 1],
            label=f"Domain {dom}",
            alpha=0.7,
        )

    plt.legend()
    plt.title("UMAP across Domain1–5 (UNet encoder features)")
    plt.tight_layout()
    plt.savefig("tsne_domains.png", dpi=200)
    plt.show()

    print("Saved UMAP plot to tsne_domains.png")
