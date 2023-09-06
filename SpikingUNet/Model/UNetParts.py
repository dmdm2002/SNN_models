import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate


class DoubleConv(nn.Module):
    def __init__(self, in_dims, out_dims, mid_dims=None):
        super().__init__()
        if not mid_dims:
            mid_dims = out_dims

        spike_grad = surrogate.fast_sigmoid(slope=25)
        # global decay rate for all Leaky neurons in Layer 1
        beta1 = 0.9
        # independent decay rate for each Leaky neuron in Layer 2: [0,1)
        beta2 = torch.rand(out_dims, dtype=torch.float)

        self.conv1 = nn.Conv2d(in_dims, mid_dims, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_dims)
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=spike_grad, init_hidden=True)
        self.conv2 = nn.Conv2d(mid_dims, out_dims, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dims)
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=spike_grad, init_hidden=True, learn_beta=True)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        mem2_rec = []

        for step in range(25):
            cur1 = self.conv1(x)
            cur1 = self.bn1(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.conv2(x)
            cur2 = self.bn2(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x, _ = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1, _ = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)