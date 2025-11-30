# dataset_task3.py
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from PIL import Image

from dataset_faz import FazSegDataset  # 复用 FAZ 数据集


class Task3AugDataset(Dataset):
    """""
    Task3 用的数据集：
    - 只使用 domain1 的标签（mask）
    - 输入图像可以是：
        1）原始 domain1 图像（h5）
        2）FedDG 迁移图像（task2/fed_dg_results）
        3）CycleGAN 迁移图像（task2/cyclegan_generated）
    - 三者按索引一一对应
    """""

    def __init__(self, use_feddg=True, use_cyclegan=True, use_domain1=True):
        super().__init__()

        # 只用 domain1 的 train 作为标签来源
        self.base = FazSegDataset(domain_id=1, split="train", aug="none")

        # 三种来源开关
        self.use_feddg = use_feddg
        self.use_cyclegan = use_cyclegan
        self.use_domain1 = use_domain1

        # project 根目录
        self.project_root = Path(__file__).resolve().parent

        # FedDG 和 CycleGAN 图像目录
        self.feddg_dir = self.project_root / "task2" / "fed_dg_results"
        self.cyc_dir = self.project_root / "task2" / "cyclegan_generated"

    def __len__(self):
        return len(self.base)

    def _load_png_as_tensor(self, path):
        """
        读取单通道 png → [1,256,256] 的 float tensor，范围 0~1
        """
        img = Image.open(path).convert("L")          # 灰度
        img = np.array(img).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).unsqueeze(0)   # [1,H,W]
        # 和 FAZ 数据集保持一致的尺寸
        img_t = F.resize(img_t, (256, 256))
        return img_t

    def __getitem__(self, idx):
        # 先拿到：原图 + mask（mask 来自 domain1）
        img_orig, mask = self.base[idx]   # img_orig: [1,256,256]

        # 准备候选图像来源
        if not self.use_domain1:
            candidates = []
        else:
            candidates = [("orig", None)]

        # 对应的文件名：和我们生成时保持一致
        feddg_path = self.feddg_dir / f"domain1_to_3_{idx}.png"
        cyc_path = self.cyc_dir / f"domain1_to_3_cyclegan_{idx}.png"

        if self.use_feddg and feddg_path.exists():
            candidates.append(("feddg", feddg_path))

        if self.use_cyclegan and cyc_path.exists():
            candidates.append(("cyc", cyc_path))

        # 随机选择一种来源
        kind, path = random.choice(candidates)

        if kind == "orig":
            img = img_orig
        else:
            img = self._load_png_as_tensor(path)

        # 标签始终还是 domain1 的 mask
        return img, mask
