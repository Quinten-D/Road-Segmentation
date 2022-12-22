import torch
import collections
import torch.nn as nn

kernel_size = 2
stride = 2
dim = 1
pow2 = [1, 2, 4, 8, 16]

block_kernel_size = 3
block_padding = 1


class UNet(nn.Module):
    @staticmethod
    def CNNBlock(num_channels_in, features, namePrefix):
        conv1 = (namePrefix + "conv1",
                 nn.Conv2d(
                     in_channels=num_channels_in,
                     out_channels=features,
                     kernel_size=block_kernel_size,
                     padding=block_padding,
                     bias=False,
                 ))
        norm1 = (namePrefix + "norm1", nn.BatchNorm2d(num_features=features))
        relu1 = (namePrefix + "relu1", nn.ReLU(inplace=True))
        conv2 = (namePrefix + "conv2",
                 nn.Conv2d(
                     in_channels=features,
                     out_channels=features,
                     kernel_size=block_kernel_size,
                     padding=block_padding,
                     bias=False,
                 ))
        norm2 = (namePrefix + "norm2", nn.BatchNorm2d(num_features=features))
        relu2 = (namePrefix + "relu2", nn.ReLU(inplace=True))
        return nn.Sequential(collections.OrderedDict([conv1, norm1, relu1, conv2, norm2, relu2]))

    def __init__(self, num_channels_in=3, num_channels_out=1, features=32):
        super().__init__()

        self.encoder1 = UNet.CNNBlock(num_channels_in, features, namePrefix="enc1")
        self.encoder2 = UNet.CNNBlock(features, features * pow2[1], namePrefix="enc2")
        self.encoder3 = UNet.CNNBlock(features * pow2[1], features * pow2[2], namePrefix="enc3")
        self.encoder4 = UNet.CNNBlock(features * pow2[2], features * pow2[3], namePrefix="enc4")

        self.maxpool1 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.maxpool2 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.maxpool3 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.maxpool4 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        self.bottleneck = UNet.CNNBlock(features * pow2[3], features * pow2[4], namePrefix="b")

        self.upconv4 = nn.ConvTranspose2d(features * pow2[4], features * pow2[3], kernel_size=kernel_size, stride=stride)
        self.upconv3 = nn.ConvTranspose2d(features * pow2[3], features * pow2[2], kernel_size=kernel_size, stride=stride)
        self.upconv2 = nn.ConvTranspose2d(features * pow2[2], features * pow2[1], kernel_size=kernel_size, stride=stride)
        self.upconv1 = nn.ConvTranspose2d(features * pow2[1], features, kernel_size=kernel_size, stride=stride)

        self.decoder4 = UNet.CNNBlock(features * pow2[4], features * pow2[3], namePrefix="dec4")
        self.decoder3 = UNet.CNNBlock(features * pow2[3], features * pow2[2], namePrefix="dec3")
        self.decoder2 = UNet.CNNBlock(features * pow2[2], features * pow2[1], namePrefix="dec2")
        self.decoder1 = UNet.CNNBlock(features * pow2[1], features, namePrefix="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=num_channels_out, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.maxpool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.maxpool2(e2)
        e3 = self.encoder3(p2)
        p3 = self.maxpool3(e3)
        e4 = self.encoder4(p3)
        bottleneck = self.bottleneck(self.maxpool4(e4))
        d4 = self.upconv4(bottleneck)
        d4 = self.decoder4(torch.cat((d4, e4), dim=dim))
        d3 = self.upconv3(d4)
        d3 = self.decoder3(torch.cat((d3, e3), dim=dim))
        d2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat((d2, e2), dim=dim))
        d1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat((d1, e1), dim=dim))

        return torch.sigmoid(self.conv(d1))
