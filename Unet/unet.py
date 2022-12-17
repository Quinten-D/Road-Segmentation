from collections import OrderedDict

import torch
import torch.nn as nn

kernel_size = 2
stride = 2
dim = 1
pow2 = [1, 2, 4, 8, 16]

block_kernel_size = 3
block_padding = 1


class UNet(nn.Module):
    def __init__(self, num_channels_in=3, num_channels_out=1, features=32):
        super().__init__()

        self.encoder1 = UNet.CNNBlock(num_channels_in, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.encoder2 = UNet.CNNBlock(features, features * pow2[1], name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.encoder3 = UNet.CNNBlock(features * pow2[1], features * pow2[2], name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.encoder4 = UNet.CNNBlock(features * pow2[2], features * pow2[3], name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        self.bottleneck = UNet.CNNBlock(features * pow2[3], features * pow2[4], name="b")

        self.upconv4 = nn.ConvTranspose2d(features * pow2[4], features * pow2[3], kernel_size=kernel_size, stride=stride)
        self.decoder4 = UNet.CNNBlock(features * pow2[4], features * pow2[3], name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * pow2[3], features * pow2[2], kernel_size=kernel_size, stride=stride)
        self.decoder3 = UNet.CNNBlock(features * pow2[3], features * pow2[2], name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * pow2[2], features * pow2[1], kernel_size=kernel_size, stride=stride)
        self.decoder2 = UNet.CNNBlock(features * pow2[2], features * pow2[1], name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * pow2[1], features, kernel_size=kernel_size, stride=stride)
        self.decoder1 = UNet.CNNBlock(features * pow2[1], features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=num_channels_out, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)
        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2)
        enc3 = self.encoder3(pool2)
        pool3 = self.pool3(enc3)
        enc4 = self.encoder4(pool3)

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=dim)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=dim)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=dim)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=dim)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def CNNBlock(num_channels_in, features, name):
        conv1 = (name + "conv1",
                 nn.Conv2d(
                     in_channels=num_channels_in,
                     out_channels=features,
                     kernel_size=block_kernel_size,
                     padding=block_padding,
                     bias=False,
                 ))
        norm1 = (name + "norm1", nn.BatchNorm2d(num_features=features))
        relu1 = (name + "relu1", nn.ReLU(inplace=True))
        conv2 = (name + "conv2",
                 nn.Conv2d(
                     in_channels=features,
                     out_channels=features,
                     kernel_size=block_kernel_size,
                     padding=block_padding,
                     bias=False,
                 ))
        norm2 = (name + "norm2", nn.BatchNorm2d(num_features=features))
        relu2 = (name + "relu2", nn.ReLU(inplace=True))
        return nn.Sequential(OrderedDict([conv1, norm1, relu1, conv2, norm2, relu2]))
