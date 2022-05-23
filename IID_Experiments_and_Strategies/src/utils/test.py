"""
包含实现基本测试功能代码的文件
"""

import math
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
import time
from torch import nn
# def softmax(output):
#     probab = np.zeros(len(output))
#     sum = 0
#     for i in range(len(output)):
#         sum = math.exp(output[i]) + sum
#     for i in range(len(output)):
#         probab[i] = math.exp(output[i]) / sum
#
#     return probab
# sof = F.softmax()
def test(model,dimension,classifier_name,num_classes, device, dataiter, test_set, batch_size, recoder, log, tp=False):  # 训练函数
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
    model.eval()  # 将model设定为测试模式
    img_names = []
    originlables = []
    embeddinglist = []
    labels = []
    outputlist = []
    losslist = []
    TrainEmbeddinglist = [[]]* num_classes
    with torch.no_grad():
        aloss = 0
        aacc = 0
        iter_times = math.ceil(len(test_set) / batch_size) #1

        # print('测试开始：', iter_times)
        for iteration in tqdm(range(iter_times)):
            input_data, label, img_name,  originlable = next(dataiter)
            input_data = input_data.clone().detach().float().to(device)
            for i in range(len(label)):
                labels.append(label.numpy()[i])
            label = label.clone().detach().to(device)
            loss, acc, embedding, output = my_forward(model, classifier_name,input_data, label, recoder,log, tp)

          # 特征存入列表
            for i in range(len(embedding)):
                TrainEmbeddinglist[label[i]] = TrainEmbeddinglist[label[i]] + [embedding[i]]
                embeddinglist.append(embedding[i])
                img_names.append(img_name[i])
                originlables.append(originlable[i])
                output[i] = output[i].squeeze()
                prob = F.softmax(output[i], dim=0)
                prob = prob.cuda().data.cpu().numpy()
                # 计算softmax
                outputlist.append(prob)

            #######################
            losslist = losslist + list(loss.cpu().detach().numpy())
            aloss += loss.cpu().detach().numpy().mean()
            aacc += acc.cpu().detach().numpy()
            if log is True:
                recoder.log_test_loss(loss.cpu().detach().numpy())
                recoder.log_test_acc(acc.cpu().detach().numpy())

        protolist = np.zeros(shape=(num_classes, dimension))
        embed_maxlist = np.ones(shape=(num_classes, dimension))
        for index_num_classes in range(num_classes):
            if TrainEmbeddinglist[index_num_classes] != []:
                protolist[index_num_classes] = np.array(TrainEmbeddinglist[index_num_classes]).mean(axis=0)
                embed_maxlist[index_num_classes] = np.array(TrainEmbeddinglist[index_num_classes]).max(axis=0)

    return aloss / iter_times, aacc / iter_times, TrainEmbeddinglist,embeddinglist,protolist,embed_maxlist, outputlist, losslist,img_names, originlables, labels#dists


def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def my_forward(model, classifier_name, input_data, label, recoder,log, tp=False):
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
    loss_function = nn.CrossEntropyLoss(reduction='none')
    output, embedding = model(input_data)
    embedding = embedding.cpu().numpy()
    corrent = torch.eq(torch.argmax(output, dim=1), label)
    # 记录开始记录日志时刻时刻
    time4 = time.time()
    # 记录训练结束时刻
    time3 = time.time()
    loss = loss_function(output, label.long())
    acc = torch.mean(corrent.float())
    if tp:
        # 记录当前batch的真值和预测结果
        for i in range(len(torch.argmax(output, dim=1).cpu().numpy())):
            if log is True:
                recoder.log_test_label(label.cpu().numpy()[i],
                                   torch.argmax(output, dim=1).cpu().numpy()[i])
    return loss, acc, embedding, output
