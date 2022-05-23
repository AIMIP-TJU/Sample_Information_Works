# -*- coding: utf-8 -*-
"""
@Time ： 2022/3/14 21:52
@Auth ： Mashukun
@File ：AlexNet.py
@IDE ：PyCharm
@Project：Active-learning

"""
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            # nn.Linear(512, self.num_classes)
        )

        self.linear = nn.Linear(512, self.num_classes)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        # x = x.view(x.size(0), -1)
        embedding = x.detach()
        x = self.linear(x),
        return x,embedding

def alexnet(input_num,num_classes):
    return AlexNet(num_classes)

if __name__ == '__main__':
    # Example
    net = AlexNet(num_classes=11)
    print(net)
