import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ["EDANet"]

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super(DownsamplerBlock, self).__init__()

        self.ninput = ninput
        self.noutput = noutput

        if self.ninput < self.noutput:
            # Wout > Win
            self.conv = nn.Conv2d(ninput, noutput - ninput, kernel_size=3, stride=2, padding=1)
            self.pool = nn.MaxPool2d(2, stride=2)
        else:
            # Wout < Win
            self.conv = nn.Conv2d(ninput, noutput, kernel_size=3, stride=2, padding=1)

        self.bn = nn.BatchNorm2d(noutput)

    def forward(self, x):
        if self.ninput < self.noutput:
            output = torch.cat([self.conv(x), self.pool(x)], 1)
        else:
            output = self.conv(x)

        output = self.bn(output)
        return F.relu(output)

class EDAModule(nn.Module):
    def __init__(self, ninput, dilated, k=40, dropprob=0.02):
        super().__init__()

        self.conv1x1 = nn.Conv2d(ninput, k, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(k)

        self.conv3x1_1 = nn.Conv2d(k, k, kernel_size=(3, 1), padding=(1, 0))
        self.conv1x3_1 = nn.Conv2d(k, k, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(k)

        self.conv3x1_2 = nn.Conv2d(k, k, (3, 1), stride=1, padding=(dilated, 0), dilation=dilated)
        self.conv1x3_2 = nn.Conv2d(k, k, (1, 3), stride=1, padding=(0, dilated), dilation=dilated)
        self.bn2 = nn.BatchNorm2d(k)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x):
        input = x

        output = self.conv1x1(x)
        output = self.bn0(output)
        output = F.relu(output)

        output = self.conv3x1_1(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        output = F.relu(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        output = torch.cat([output, input], 1)
        return output

class EDANetBlock(nn.Module):
    def __init__(self, in_channels, num_dense_layer, dilated, growth_rate):
        super().__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(EDAModule(_in_channels, dilated[i], growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        return out

class EDANet(nn.Module):
    def __init__(self, classes=2):
        super(EDANet, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(1, 15))  # 256x256 -> 128x128
        self.layers.append(DownsamplerBlock(15, 60))  # 128x128 -> 64x64
        self.layers.append(EDANetBlock(60, 5, [1, 1, 1, 2, 2], 40))  # 64x64
        self.layers.append(DownsamplerBlock(260, 130))  # 64x64 -> 32x32
        self.layers.append(EDANetBlock(130, 8, [2, 2, 4, 4, 8, 8, 16, 16], 40))  # 32x32

        self.project_layer = nn.Conv2d(450, classes, kernel_size=1)

        self.weights_init()

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        
        output = self.project_layer(output)
        # 修改上采样倍率为 8，从 32x32 恢复到 256x256
        output = F.interpolate(output, scale_factor=8, mode='bilinear', align_corners=True)
        return output

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EDANet(classes=2).to(device)  # 修改为 2 类
    summary(model, (1, 256, 256))  # 修改输入为单通道 256×256