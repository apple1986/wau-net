import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ["LinkNet"]

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out

class Encoder(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(in_planes//4),
            nn.ReLU(inplace=True)
        )
        self.tp_conv = nn.Sequential(
            nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(in_planes//4),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_high_level, x_low_level):
        x = self.conv1(x_high_level)
        x = self.tp_conv(x)
        x = center_crop(x, x_low_level.size()[2], x_low_level.size()[3])
        x = self.conv2(x)
        return x

def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    diffy = (h - max_height) // 2
    diffx = (w - max_width) // 2
    return layer[:, :, diffy:(diffy + max_height), diffx:(diffx + max_width)]

class LinkNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)  # Input channels changed to 1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.encoder1 = Encoder(64, 64, 3, 1, 1)
        self.encoder2 = Encoder(64, 128, 3, 2, 1)
        self.encoder3 = Encoder(128, 256, 3, 2, 1)
        self.encoder4 = Encoder(256, 512, 3, 2, 1)

        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)

        self.tp_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.tp_conv2 = nn.ConvTranspose2d(32, classes, 2, 2, 0)

    def forward(self, x):
        x = self.conv1(x)  # 256x256 -> 128x128
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 128x128 -> 64x64

        e1 = self.encoder1(x)  # 64x64
        e2 = self.encoder2(e1)  # 64x64 -> 32x32
        e3 = self.encoder3(e2)  # 32x32 -> 16x16
        e4 = self.encoder4(e3)  # 16x16 -> 8x8

        d4 = e3 + self.decoder4(e4, e3)  # 8x8 -> 16x16
        d3 = e2 + self.decoder3(d4, e2)  # 16x16 -> 32x32
        d2 = e1 + self.decoder2(d3, e1)  # 32x32 -> 64x64
        d1 = x + self.decoder1(d2, x)  # 64x64

        y = self.tp_conv1(d1)  # 64x64 -> 128x128
        y = self.conv2(y)
        y = self.tp_conv2(y)  # 128x128 -> 256x256
        return y

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinkNet(classes=2).to(device)  # 2 classes for binary segmentation
    summary(model, (1, 256, 256))  # Input: 1 channel, 256x256 grayscale image