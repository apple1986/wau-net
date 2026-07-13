import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from torchsummary import summary

# ----------------------------
# Wavelet 模块
# ----------------------------
class WaveletTransform(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave, mode='zero')

    def forward(self, x):
        Yl, Yh = self.dwt(x)
        LH = Yh[0][:, :, 0, :, :]
        HL = Yh[0][:, :, 1, :, :]
        HH = Yh[0][:, :, 2, :, :]
        return Yl, LH, HL, HH

# ----------------------------
# ASPP 模块 with Wavelet
# ----------------------------
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*modules)

class ASPPWithWavelet(nn.Module):
    def __init__(self, in_channels, wt: WaveletTransform, atrous_rates=[1, 3, 5]):
        super().__init__()
        out_channels = in_channels // 4
        self.wt = wt
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.aspp_convs = nn.ModuleList([ASPPConv(in_channels * 3, out_channels, r) for r in atrous_rates])
        # 调整输入通道数为实际总和
        self.project = nn.Sequential(
            nn.Conv2d(out_channels + (len(atrous_rates) * out_channels) + (3 * in_channels), out_channels * 2, 1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        LL, LH, HL, HH = self.wt(x)  # 分解
        res_ll = self.conv1x1(LL)    # LL用1x1
        high_freq_input = torch.cat([LH, HL, HH], dim=1)  # 高频cat，通道数为 in_channels * 3
        res_high = [conv(high_freq_input) for conv in self.aspp_convs]
        # 使用x的实际空间维度进行上采样
        target_size = x.shape[2:]  # (H/16, W/16)
        res_ll = F.interpolate(res_ll, size=target_size, mode='bilinear', align_corners=True)
        res_high = [F.interpolate(r, size=target_size, mode='bilinear', align_corners=True) for r in res_high]
        high_freq = [
            F.interpolate(LH, size=target_size, mode='bilinear', align_corners=True),
            F.interpolate(HL, size=target_size, mode='bilinear', align_corners=True),
            F.interpolate(HH, size=target_size, mode='bilinear', align_corners=True)
        ]
        res = [res_ll] + res_high + high_freq
        return self.project(torch.cat(res, dim=1))

# ----------------------------
# U-Net 基础模块
# ----------------------------
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.seq(x)

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        self.up_trans = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.up_trans(x1)

        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# ----------------------------
# U-Net变体2：瓶颈加ASPP with wave
# ----------------------------
class UNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        wt = WaveletTransform()
        self.aspp = ASPPWithWavelet(512, wt=wt)  # 输出通道为256 (out_channels*2)

        self.inc   = double_conv(1, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.up1 = up(512 + 256, 256)  # x5经ASPP后256，cat x4的512
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.aspp(x5)  # ASPP with wave

        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)

# ----------------------------
# 结构预览
# ----------------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(classes=2).to(device)
    summary(model, (1, 256, 256))