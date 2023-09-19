import torch
import torch.nn as nn

from SpikingUNet.Model.UNetParts import Up, Down, DoubleConv, OutConv
# from torchsummary.torchsummary import summary


class SpikingUNet(nn.Module):
    def __init__(self, n_dims, n_classes, bilinear=False):
        super().__init__()
        self.n_dims = n_dims
        self.n_classes = n_classes
        factor = 2 if bilinear else 1

        self.stem = DoubleConv(self.n_dims, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024//factor)

        self.up4 = Up(1024, 512//factor, bilinear)
        self.up3 = Up(512, 256//factor, bilinear)
        self.up2 = Up(256, 128//factor, bilinear)
        self.up1 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1, _ = self.stem(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        out = self.outc(x)

        return out


# model = SpikingUNet(3, 1)
# print(model)
# # summary(model, (3, 224, 224), device='cpu')
