import torch
import torch.nn as nn
import numpy as np
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
        beta1 = 0.5
        # independent decay rate for each Leaky neuron in Layer 2: [0,1)
        beta2 = torch.rand(out_dims, dtype=torch.float)

        self.conv1 = nn.Conv2d(in_dims, mid_dims, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_dims)
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(mid_dims, out_dims, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dims)
        self.lif2 = snn.Leaky(beta=beta1, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        cur1 = self.conv1(x)
        cur1 = self.bn1(cur1)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = self.conv2(spk1)
        cur2 = self.bn2(cur2)
        spk2, mem2 = self.lif2(cur2, mem2)

        return spk2, mem2


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
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x, _ = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)