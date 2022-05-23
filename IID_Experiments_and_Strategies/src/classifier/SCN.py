# -*- coding: utf-8 -*-

"""
Created on 

@file: SCN.py
@author: ZhangZ

"""
from torch import nn
import torch.nn.functional as F


class SCN(nn.Module):
    def __init__(self, input_num, class_num):
        super(SCN, self).__init__()

        self.conv1 = nn.Conv2d(input_num, 6, 5)  # 输入通道数为1，输出通道数为6
        self.conv2 = nn.Conv2d(6, 16, 5)  # 输入通道数为6，输出通道数为16
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 64, 5)
        self.AAPool = nn.AdaptiveAvgPool2d(1)

        self.fc3 = nn.Linear(64, 2)
        self.fc4 = nn.Linear(2, class_num)


    def forward(self, x):
        # 输入x -> conv1 -> relu -> 2x2窗口的最大池化
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # 输入x -> conv2 -> relu -> 2x2窗口的最大池化
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.relu(x)
        # view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        x = self.AAPool(x)
        x = x.squeeze().squeeze()
        x = self.fc3(x)
        x = self.fc4(x)
        return x
