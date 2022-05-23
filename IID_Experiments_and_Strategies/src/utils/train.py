import math
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch
from torch import nn
import time



def train(model, classifier_name,device, dataiter, optimizer, train_set, batch_size, recoder, dimension, num_classes,i, EPOCH,tp=False):  # 训练函数
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
    embeddinglist = [[]]*num_classes
    for iteration in tqdm(range(iter_times)):
        data, label, img_name, originlable = next(dataiter)
        labels = label.numpy()
        data = data.clone().float().to(device)
        label = label.clone().detach().to(device)
        loss, acc, embedding, output = my_forward(model,classifier_name, data, label, recoder, tp)
        aacc += acc.cpu().detach().numpy()
        loss.backward()  # 针对损失函数的后向传播
        optimizer.step()  # 反向传播后的梯度下降
        optimizer.zero_grad()  # 清除旧的梯度信息
        aloss += loss.cpu().detach().numpy()
        recoder.log_train_loss(loss.cpu().detach().numpy())
        recoder.log_train_acc(acc.cpu().detach().numpy())
    return aloss / iter_times, aacc / iter_times

def my_forward(model,classifier_name, input_data, label, recoder, tp=False):
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
    # loss_function = vae_loss_function()
    loss_function = nn.CrossEntropyLoss()

    output, embedding = model(input_data)
    embedding = embedding.cpu().numpy()

    # print(embedding)
    corrent = torch.eq(torch.argmax(output, dim=1), label)
    loss = loss_function(output, label.long())
    acc = torch.mean(corrent.float())
    if tp:
        # 记录当前batch的真值和预测结果
        for i in range(len(torch.argmax(output, dim=1).cpu().numpy())):
            recoder.log_test_label(label.cpu().numpy()[i],
                                   torch.argmax(output, dim=1).cpu().numpy()[i])
    return loss, acc, embedding, output
