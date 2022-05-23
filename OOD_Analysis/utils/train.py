import math
from tqdm.notebook import tqdm
from utils.my_forward import my_forward
import torch.nn.functional as F
import torch
from torch import nn

def train(model, device, dataiter, optimizer, train_set, batch_size):  # 训练函数
    """

    此代码功能为对数据进行一个EPOCH的训练

    输入数据rcs的格式为：
    [batch_size, time_steps, input_channel=1]
    标签label格式为：
    [batch_size, class_num]
    输出output格式为：
    [batch_size, class_num]


    :param model: 深度学习模型
    :param device: 计算处理器设备
    :param dataiter: 数据迭代器
    :param optimizer: 优化器
    :param train_set: 训练集
    :param batch_size: 批大小
    :param mission: 任务类型
    :param recoder: 日志记录器
    :param tp: 是否记录结果
    :return: loss, acc
    """
    model.train()  # 将model设定为训练模式
    aloss = 0
    aacc = 0
    iter_times = math.ceil(len(train_set) / batch_size)
    for iteration in tqdm(range(iter_times), leave = False):
        data, label, names, label_names = next(dataiter)
        data = data.clone().float().to(device)
        label = label.clone().detach().to(device)

        loss, acc = my_forward(model, data, label)

        #############
        aacc += acc.cpu().detach().numpy()
        loss.backward()  # 针对损失函数的后向传播
        optimizer.step()  # 反向传播后的梯度下降
        optimizer.zero_grad()  # 清除旧的梯度信息
        aloss += loss.cpu().detach().numpy()

    return aloss / iter_times, aacc / iter_times