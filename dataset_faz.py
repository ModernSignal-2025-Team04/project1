# dataset_faz.py
import numpy as np
import h5py
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F

from config import get_domain_path


class FazSegDataset(Dataset):
    """
    FAZ h5 数据集 + 固定 resize 到 256x256（最关键）
    """
    def __init__(self, domain_id: int, split: str = "train", aug: str = "none"):
        super().__init__()
        assert split in ["train", "test"]
        self.domain_id = domain_id
        self.split = split
        self.aug = aug

        # 目标尺寸
        self.target_size = (256, 256)

        self.root = get_domain_path(domain_id)

        if split == "train":
            self.data_dir = self.root / "train"
        else:
            self.data_dir = self.root / "test"

        self.h5_paths = sorted(list(self.data_dir.glob("*.h5")))
        if len(self.h5_paths) == 0:
            raise RuntimeError(f"No h5 files found in: {self.data_dir}")

    def __len__(self):
        return len(self.h5_paths)

    def _load_h5(self, idx):
        h5_path = self.h5_paths[idx]
        with h5py.File(h5_path, "r") as f:
            img = f["image"][:]
            mask = f["mask"][:]

        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]

        return img, mask

    def _apply_aug(self, img_t, mask_t):
        if self.aug == "none":
            return img_t, mask_t

        if "flip" in self.aug:
            if torch.rand(1) < 0.5:
                img_t = F.hflip(img_t)
                mask_t = F.hflip(mask_t)
            if torch.rand(1) < 0.5:
                img_t = F.vflip(img_t)
                mask_t = F.vflip(mask_t)

        if "rotate" in self.aug:
            angle = float(torch.empty(1).uniform_(-30, 30))
            img_t = F.rotate(img_t, angle, interpolation=F.InterpolationMode.BILINEAR)
            mask_t = F.rotate(mask_t, angle, interpolation=F.InterpolationMode.NEAREST)

        return img_t, mask_t

    def __getitem__(self, idx):
        img, mask = self._load_h5(idx)

        img = torch.from_numpy(img).float()
        if img.max() > 1:
            img = img / 255.0
        img = img.unsqueeze(0)

        mask = torch.from_numpy(mask).float()
        if mask.max() > 1:
            mask = mask / 255.0
        mask = mask.unsqueeze(0)
        mask = (mask > 0.5).float()

        # ======== 统一 resize 到 256x256 ========
        img = F.resize(img, self.target_size, interpolation=F.InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.target_size, interpolation=F.InterpolationMode.NEAREST)
        # ==================================================

        img, mask = self._apply_aug(img, mask)

        return img, mask
