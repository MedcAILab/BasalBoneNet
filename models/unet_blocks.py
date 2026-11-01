import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    """Single 3D convolution block: Conv3d -> normalization -> activation."""

    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = self._build_activation(activation)

    @staticmethod
    def _build_activation(activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        if activation == 'prelu':
            return nn.PReLU()
        if activation == 'elu':
            return nn.ELU(inplace=True)
        raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class DoubleConv3D(nn.Module):
    """Two stacked 3D convolution blocks."""

    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.net = nn.Sequential(
            Conv3DBlock(in_channels, out_channels, activation),
            Conv3DBlock(out_channels, out_channels, activation),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    """Downsampling block: apply convs, then pool to reduce spatial size."""

    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.conv = DoubleConv3D(in_channels, out_channels, activation)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class UpBlock(nn.Module):
    """Upsampling block using interpolation followed by convolution."""

    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv3D(in_channels + out_channels, out_channels, activation)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class OutputHead(nn.Module):
    def __init__(self, in_channels, out_channels=1, out_activation='sigmoid'):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        if out_activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif out_activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif out_activation is None or out_activation == 'identity':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported output activation: {out_activation}")

    def forward(self, x):
        return self.activation(self.conv(x))
