# unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


def pad_to_match(inp, ref):
    diffY = ref.size(2) - inp.size(2)
    diffX = ref.size(3) - inp.size(3)

    inp = F.pad(inp, [
        diffX // 2,
        diffX - diffX // 2,
        diffY // 2,
        diffY - diffY // 2
    ])
    return inp


class UNet2D(nn.Module):
    def __init__(self, in_ch=1, num_classes=1):
        super().__init__()

        self.inc = DoubleConv(in_ch, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = pad_to_match(x, x4)
        x = self.dec1(torch.cat([x4, x], dim=1))

        x = self.up2(x)
        x = pad_to_match(x, x3)
        x = self.dec2(torch.cat([x3, x], dim=1))

        x = self.up3(x)
        x = pad_to_match(x, x2)
        x = self.dec3(torch.cat([x2, x], dim=1))

        x = self.up4(x)
        x = pad_to_match(x, x1)
        x = self.dec4(torch.cat([x1, x], dim=1))

        return self.outc(x)
