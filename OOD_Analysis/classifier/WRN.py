import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class WideResNet(nn.Module):
    def __init__(self, num_layers, widen_factor, block, num_classes=10):
        super(WideResNet, self).__init__()
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.get_layers(block, 16, 16 * widen_factor, 1)
        self.layer2 = self.get_layers(block, 16 * widen_factor, 32 * widen_factor, 2)
        self.layer3 = self.get_layers(block, 32 * widen_factor, 64 * widen_factor, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * widen_factor, num_classes)

    def get_layers(self, block, in_channels, out_channels, stride):
        layers = []

        for i in range(self.num_layers):
            if i == 0:
                layers.append(block(in_channels, out_channels, stride))
                continue
            layers.append(block(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        embeding = x

        x = self.avg_pool(x)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x,


def wide_resnet(num_layers=5, widen_factor=2, num_classes=10):
    return WideResNet(num_layers, widen_factor, Block, num_classes)