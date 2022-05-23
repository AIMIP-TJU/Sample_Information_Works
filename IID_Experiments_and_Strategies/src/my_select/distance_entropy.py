# -*- coding: utf-8 -*-
"""
@Time ： 2022/5/19 8:38
@Auth ： Mashukun
@File ：distance_difference.py
@IDE ：PyCharm
@Project：IID-ADD
@Motto：ABC(Always Be Coding)
"""
import csv
import math
import shutil
import os
import numpy as np
import torch.nn.functional as F
import torch
import os
from matplotlib import pyplot as plt
from src.utils.my_mkdir import mkdir
from sklearn import manifold,datasets
from ...train_main import args
def distance_entropy_select(method_name, originlables, num_classes, classifier_name,labels,
                                                              dataset_name, embeddinglist,pool_embeddinglist, img_names,
                                                              protolist, select_ratio, select_type, add_ratio):
    # 定义要创建的目录
    global scorelist, select_data
    filepath = './Selection/{}/{}/{}/{}/'.format(dataset_name, classifier_name, method_name,select_type)
    # 调用函数创建目录
    if os.path.exists(filepath):
        pass
    else:
        os.makedirs(filepath)
    # 创建csv文件
    ft = open(str('{}{}_{}.csv'.format(filepath,dataset_name,select_ratio)), 'w', newline='')
    ft_csv = csv.writer(ft)

    # print(pool_embeddinglist)
    # -------计算距离-------
    distancelist = np.zeros((len(pool_embeddinglist), num_classes))
    for i in range(num_classes):
        diff = np.array(pool_embeddinglist) - protolist[i][:]
        distance = np.linalg.norm(diff, axis=1)
        distancelist[:, i] = distance
    distance_softmax = F.softmax(torch.from_numpy(-distancelist), dim=1)
    distance_softmax = distance_softmax.numpy()
    # ------计算距离熵------

    # print(outputlist[i])
    distance_entropy = []
    print(len(distance_softmax))
    for softmaxdis in distance_softmax:
        entropy = 0
        for x in softmaxdis:
            # print(x)
            if x == 0:
                entropy = entropy
            else:
                entropy = entropy + (-x) * math.log(x, 2)
        distance_entropy.append(entropy)
    distance_entropy_list = [[]]*num_classes
    for i in range(len(distance_entropy)):
        distance_entropy_list[labels[i]] = distance_entropy_list[labels[i]] + [distance_entropy[i]]
    distance_entropy_mean = []
    for k in distance_entropy_list:
        distance_entropy_mean.append(np.mean(k))
    # print(distance_entropy_mean)
    with open('./Data/1.txt', "a") as myfile:
        for k in range(len(distance_entropy_mean)):
            myfile.write(str(distance_entropy_mean[k])+' ')
        myfile.write('\n')
    if select_type=='ADD-GOOD':
        # 由大到小
        order = (-np.array(distance_entropy)).argsort()
    elif select_type=='ADD-BAD':
        # 由小到大
        order = np.array(distance_entropy).argsort()
    # 创建列表
    lablelist = []
    dirlist = []

    print("开始筛选")

    for v, i in enumerate(order):
        # ff_csv.writerow([img_names[i]] + [img_names[i].split('_')[0]])
        if v < add_ratio*50000:
            ft_csv.writerow([img_names[i]] + [str(originlables[i])])
            # shutil.copyfile('/root/projects/msk/datasets/CSE/images/' + img_names[i], object_path + str(scorelist[i]) + '_' +img_names[i])
            dirlist.append(img_names[i])
            lablelist.append(str(originlables[i]))
            # if select_type=='ADD-GOOD':
            #     plt.scatter(pool_embeddinglist[i][0], pool_embeddinglist[i][1], c='xkcd:green')
            # elif select_type == 'ADD-BAD':
            #     plt.scatter(pool_embeddinglist[i][0], pool_embeddinglist[i][1], c='xkcd:red')
        else:
            break
    select_data = list(zip(dirlist, lablelist))
    # plt.show()
     # np.save('../Feature/Cifar10/'+ classifier_name +'/test/metric_select/' + str(dataset_name) + '.npy', scorelist)
    return select_data

def distance_entropy_aver(method_name, originlables, num_classes, classifier_name,labels,
                                                              dataset_name, embeddinglist,pool_embeddinglist, img_names,
                                                              protolist, select_ratio, select_type, add_ratio):
    # print(len(pool_embeddinglist))
    # 定义要创建的目录
    global scorelist, select_data
    filepath = './Selection/{}/{}/{}/{}/'.format(dataset_name, classifier_name, method_name,select_type)
    # 调用函数创建目录
    if os.path.exists(filepath):
        pass
    else:
        os.makedirs(filepath)
    # 创建csv文件
    ft = open(str('{}{}_{}.csv'.format(filepath,dataset_name,select_ratio)), 'w', newline='')
    ft_csv = csv.writer(ft)
    originlableslist = [[]]*num_classes
    img_nameslist = [[]]*num_classes
    for i in range(len(img_names)):
        originlableslist[labels[i]] = originlableslist[labels[i]] + [originlables[i]]
        img_nameslist[labels[i]] = img_nameslist[labels[i]] + [img_names[i]]
    # protolist = np.load(str(protofile))
    protolist = protolist
    # print(pool_embeddinglist)
    # -------计算距离-------
    distancelist = np.zeros((len(pool_embeddinglist), num_classes))
    for i in range(num_classes):
        diff = np.array(pool_embeddinglist) - protolist[i][:]
        distance = np.linalg.norm(diff, axis=1)
        distancelist[:, i] = distance
    distance_softmax = F.softmax(torch.from_numpy(-distancelist), dim=1)
    distance_softmax = distance_softmax.numpy()
    # ------计算距离熵------
    # print(outputlist[i])
    distance_entropy = []
    print(len(distance_softmax))
    for softmaxdis in distance_softmax:
        entropy = 0
        for x in softmaxdis:
            # print(x)
            if x == 0:
                entropy = entropy
            else:
                entropy = entropy + (-x) * math.log(x, 2)
        distance_entropy.append(entropy)
    # print(len(distance_entropy))
    distance_entropy_list = [[]]*num_classes
    for i in range(len(distance_entropy)):
        distance_entropy_list[labels[i]] = distance_entropy_list[labels[i]] + [distance_entropy[i]]
    distance_entropy_mean = []
    for k in distance_entropy_list:
        distance_entropy_mean.append(np.mean(k))
    txtpath = './Data/aver_distanceEntropy_distribution.txt'
    with open(txtpath, "a") as myfile:
        for k in range(len(distance_entropy_mean)):
            myfile.write(str(distance_entropy_mean[k]) + ' ')
        myfile.write('\n')
    order = []
    for i in range(len(distance_entropy_list)):
        if select_type == 'ADD-GOOD':
            # 由大到小
            order.append((-np.array(distance_entropy_list[i])).argsort())
        elif select_type == 'ADD-BAD':
            # 由小到大
            order.append((np.array(distance_entropy_list[i])).argsort())   # 创建列表
    # print(len(order))
    lablelist = []
    dirlist = []
    i=0
    print("开始筛选")
    for j in range(len(order)):
        for v, i in enumerate(order[j]):
            # print(len(order[j]))
            if v < add_ratio*500:
                # print(add_ratio*500)
                ft_csv.writerow([img_nameslist[j][i]] + [str(originlableslist[j][i])])
                # shutil.copyfile('/root/projects/msk/datasets/CSE/images/' + img_names[i], object_path + str(scorelist[i]) + '_' +img_names[i])
                dirlist.append(img_nameslist[j][i])
                lablelist.append(str(originlableslist[j][i]))
                # if select_type=='ADD-GOOD':
                #     plt.scatter(pool_embeddinglist[i][0], pool_embeddinglist[i][1], c='xkcd:green')
                # elif select_type == 'ADD-BAD':
                #     plt.scatter(pool_embeddinglist[i][0], pool_embeddinglist[i][1], c='xkcd:red')
            else:
                break
    select_data = list(zip(dirlist, lablelist))

    return select_data

