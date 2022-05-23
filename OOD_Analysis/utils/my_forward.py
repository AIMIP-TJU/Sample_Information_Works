import math
import time
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np


def my_forward(model, input_data, label):
    """
    进行一次正向传播
    :param mission: 任务类型
    :param model: 深度学习模型
    :param input_data: 输入数据
    :param label: 样本标签
    :param recoder: 日志记录器
    :param tp: 是否记录当前结果
    :return: loss, acc
    """
    loss_function_ce = nn.CrossEntropyLoss()
    loss_function_mse = nn.MSELoss()



    output = model(input_data)
    corrent = torch.eq(torch.argmax(output, dim=1), label)
    # 记录开始记录日志时刻时刻
    time4 = time.time()
    # 记录训练结束时刻
    time3 = time.time()
    loss = loss_function_ce(output, label.long())
    acc = torch.mean(corrent.float())

    return loss, acc

