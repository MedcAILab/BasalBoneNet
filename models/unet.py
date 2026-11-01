import torch

from .unet_blocks import DownBlock, UpBlock, DoubleConv3D, OutputHead
from torch import nn


class UNet(nn.Module):
    """3D U-Net with 3 downsampling stages and a bottleneck, restoring the original 128^3 output."""

    def __init__(self, in_channels=1, out_channels=1, base_channels=64, activation='relu', output_activation='sigmoid'):
        super().__init__()

        self.down1 = DownBlock(in_channels, base_channels, activation)
        self.down2 = DownBlock(base_channels, base_channels * 2, activation)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, activation)
        self.bottleneck = DoubleConv3D(base_channels * 4, base_channels * 8, activation)

        self.up3 = UpBlock(base_channels * 8, base_channels * 4, activation)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, activation)
        self.up1 = UpBlock(base_channels * 2, base_channels, activation)

        self.output = OutputHead(base_channels, out_channels, output_activation)

    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)

        x = self.bottleneck(x)

        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        return self.output(x)

