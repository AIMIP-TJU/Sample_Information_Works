"""
包含实现基本测试功能代码的文件
"""

import math
from tqdm.notebook import tqdm
from utils.my_forward import my_forward
import torch.nn.functional as F
import torch
from torch import nn
from scipy import io
import os
import csv

def test(model, device, dataiter, test_set, batch_size):  # 训练函数
    """
    此代码功能为对数据进行一个EPOCH的测试

    输入数据rcs的格式为：
    [batch_size, time_steps, input_channel=1]
    标签label格式为：
    [batch_size, class_num]
    输出output格式为：
    [batch_size, class_num]
    :param model: 深度学习模型
    :param device: 计算处理器设备
    :param dataiter: 数据迭代器
    :param test_set: 测试集
    :param batch_size: 批大小
    :param mission: 任务类型
    :param recoder: 日志记录器
    :param tp: 是否记录结果
    :return: loss, acc
    """
    model.eval()  # 将model设定为训练模式
    with torch.no_grad():
        aloss = 0
        aacc = 0
        iter_times = math.ceil(len(test_set) / batch_size)

        for iteration in tqdm(range(iter_times), leave = False):
            input_data, label, names, label_names = next(dataiter)
            input_data = input_data.clone().detach().float().to(device)
            label = label.clone().detach().to(device)
            output = my_forward( model, input_data, label)

            aloss += output[0].cpu().detach().numpy()
            aacc += output[1].cpu().detach().numpy()

        return aloss / iter_times, aacc / iter_times