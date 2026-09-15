import torch
from torch import nn
import torchsparse.nn as spnn


__all__ = ["PointNet"]


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc,
                outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()

        self.net = nn.Sequential(
            spnn.Conv3d(
                inc,
                outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(
                outc,
                outc,
                kernel_size=ks,
                dilation=dilation,
                stride=1,
            ),
            spnn.BatchNorm(outc),
        )

        self.downsample = (
            nn.Identity()
            if inc == outc and stride == 1
            else nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, stride=stride),
                spnn.BatchNorm(outc),
            )
        )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        return self.relu(self.net(x) + self.downsample(x))


class PointNet(nn.Module):
    def __init__(self):
        super().__init__()

        cs = [32, 32, 64, 128, 256]

        self.stem = nn.Sequential(
            spnn.Conv3d(3, cs[0], kernel_size=2, stride=2),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=2, stride=2),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2),
            ResidualBlock(cs[0], cs[1]),
            ResidualBlock(cs[1], cs[1]),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2),
            ResidualBlock(cs[1], cs[2]),
            ResidualBlock(cs[2], cs[2]),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2),
            ResidualBlock(cs[2], cs[3]),
            ResidualBlock(cs[3], cs[3]),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2),
            ResidualBlock(cs[3], cs[4]),
            ResidualBlock(cs[4], cs[4]),
        )

        self.weight_initialization()

    def weight_initialization(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)