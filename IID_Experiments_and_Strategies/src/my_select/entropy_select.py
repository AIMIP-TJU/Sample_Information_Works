# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/13 19:53
@Auth ： MSK
@File ：entropy_select.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""

import math
import os

import numpy
import numpy as np
import csv
from .metric_select import *
from src.utils.my_mkdir import mkdir


def entropy_select(select_method, originlables, classifier_name, dataset_name, outputlist, img_names, select_ratio, labels,select_type,add_ratio,num_classes):
    # 定义csv存放的目录
    global select_data
    filepath =  '/home/a611/Projects/msk/NMI/IID-ADD/Selcetion/{}/{}/{}/{}/'.format(dataset_name, classifier_name, select_method,select_type)
    # 调用函数创建目录
    if os.path.exists(filepath):
        pass
    else:
        os.makedirs(filepath)
    ft = open(str('{}{}_{}.csv'.format(filepath,dataset_name,select_ratio)), 'w', newline='')
    ft_csv = csv.writer(ft)

    entropy_list = [[]]*num_classes
    originlableslist = [[]]*num_classes
    img_nameslist = [[]]*num_classes

    # 计算熵
    # Entropy = lambda p: sum(-p * np.log2(p))
    for i in range(len(labels)):
        entropy = 0
        # print(outputlist[i])
        for x in outputlist[i]:
            # print(x)
            if x ==0:
                entropy = entropy
            else:
                entropy = entropy + (-x) * math.log(x, 2)
        # print(entropy)
        entropy_list[labels[i]] = entropy_list[labels[i]] + [entropy]

        originlableslist[labels[i]] = originlableslist[labels[i]] + [originlables[i]]
        img_nameslist[labels[i]] = img_nameslist[labels[i]] + [img_names[i]]
    lablelist = []
    dirlist = []
    order = []
    for i in range(len(entropy_list)):
        if select_type == 'ADD-GOOD':
            order.append((-np.array(entropy_list[i])).argsort())
        elif select_type == 'ADD-BAD':
            order.append(np.array(entropy_list[i]).argsort())

    for j in range(len(order)):
        # print(originlableslist[j][0] + '筛选结果为：')
        # print('-------------------真实-----------------------标签')
        for v, i in enumerate(order[j]):
            if v < add_ratio*500:
                ft_csv.writerow([str(img_nameslist[j][i])] + [str(originlableslist[j][i])])
                # print('{}:'.format(v+1) + str(img_nameslist[j][i]) + ' ' + str(originlableslist[j][i]))
                # if img_nameslist[j][i].split('/')[1] == originlableslist[j][i]:
                #     k[j] = k[j]+1
                dirlist.append(str(img_nameslist[j][i]))
                lablelist.append(str(originlableslist[j][i]))
                select_data = list(zip(dirlist, lablelist))
            else:
                break
    return select_data


