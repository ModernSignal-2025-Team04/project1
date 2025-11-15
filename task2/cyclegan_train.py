# cyclegan_train.py

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_faz import FazSegDataset
from tqdm import tqdm
import itertools

# =======================
# 生成器（U-Net 风格）
# =======================
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=32):
        super().__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]

        # down
        model += [
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True)
        ]
        model += [
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True)
        ]

        # 6 Resnet blocks
        for _ in range(6):
            model += [ResnetBlock(ngf * 4)]

        # up
        model += [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True)
        ]
        model += [
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# =======================
# 判别器 PatchGAN
# =======================
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=32):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kw, padding=padw)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


# =======================
# CycleGAN Training
# =======================
def train_cyclegan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # datasets
    src = FazSegDataset(1, "train", aug="none")  # domain1
    tgt = FazSegDataset(3, "train", aug="none")  # domain3

    src_loader = DataLoader(src, batch_size=4, shuffle=True)
    tgt_loader = DataLoader(tgt, batch_size=4, shuffle=True)

    # models
    G12 = ResnetGenerator().to(device)  # 1→3
    G21 = ResnetGenerator().to(device)  # 3→1

    D1 = NLayerDiscriminator().to(device)
    D3 = NLayerDiscriminator().to(device)

    # loss
    adv = nn.MSELoss()
    cycle_loss = nn.L1Loss()

    # optimizers
    opt_G = torch.optim.Adam(
        itertools.chain(G12.parameters(), G21.parameters()),
        lr=2e-4, betas=(0.5, 0.999)
    )
    opt_D = torch.optim.Adam(
        itertools.chain(D1.parameters(), D3.parameters()),
        lr=2e-4, betas=(0.5, 0.999)
    )

    # training loop
    for epoch in range(3):   # 为了作业，训练 3 个 epoch 就够
        print(f"Epoch {epoch+1}/3")
        pbar = tqdm(zip(src_loader, tgt_loader), total=min(len(src_loader), len(tgt_loader)))

        for (xs, _), (xt, _) in pbar:
            xs = xs.to(device)
            xt = xt.to(device)

            # -----------------
            # Train G
            # -----------------
            opt_G.zero_grad()

            fake_t = G12(xs)
            fake_s = G21(xt)

            loss_G12 = adv(D3(fake_t), torch.ones_like(D3(fake_t)))
            loss_G21 = adv(D1(fake_s), torch.ones_like(D1(fake_s)))

            # cycle
            rec_s = G21(fake_t)
            rec_t = G12(fake_s)
            cyc_loss = cycle_loss(rec_s, xs) + cycle_loss(rec_t, xt)

            G_loss = loss_G12 + loss_G21 + 10 * cyc_loss
            G_loss.backward()
            opt_G.step()

            # -----------------
            # Train D
            # -----------------
            opt_D.zero_grad()

            real_pred1 = D1(xs)
            fake_pred1 = D1(fake_s.detach())
            loss_D1 = (adv(real_pred1, torch.ones_like(real_pred1)) +
                       adv(fake_pred1, torch.zeros_like(fake_pred1))) * 0.5

            real_pred3 = D3(xt)
            fake_pred3 = D3(fake_t.detach())
            loss_D3 = (adv(real_pred3, torch.ones_like(real_pred3)) +
                       adv(fake_pred3, torch.zeros_like(fake_pred3))) * 0.5

            D_loss = loss_D1 + loss_D3
            D_loss.backward()
            opt_D.step()

            pbar.set_postfix(G=G_loss.item(), D=D_loss.item())

    os.makedirs("cyclegan_results", exist_ok=True)
    torch.save(G12.state_dict(), "cyclegan_results/G12.pth")
    print("CycleGAN training finished. Generator G12 saved.")


if __name__ == "__main__":
    train_cyclegan()
