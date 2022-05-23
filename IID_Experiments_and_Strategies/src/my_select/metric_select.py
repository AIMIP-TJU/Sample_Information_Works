# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/11 20:20
@Auth ： MSK
@File ：metric_select.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""

import csv
import os

import numpy as np
from src.utils.my_mkdir import mkdir

def metric_select(method_name, originlables,num_classes, classifier_name, dataset_name, pool_embeddinglist, img_names, protolist, select_amount, labels, select_ratio,select_type,add_ratio):
    # 定义要创建的目录
    filepath = '/home/a611/Projects/msk/NMI/IID-ADD/Selcetion/{}/{}/{}/{}/'.format(dataset_name, classifier_name,
                                                                                   method_name, select_type)
    # 调用函数创建目录
    if os.path.exists(filepath):
        pass
    else:
        os.makedirs(filepath)
    # 创建csv文件
    ft = open(str('{}{}_{}.csv'.format(filepath,dataset_name,select_ratio)), 'w', newline='')
    ft_csv = csv.writer(ft)

    distancelist =  [[]]*num_classes
    originlableslist = [[]]*num_classes
    img_nameslist = [[]]*num_classes
    # 计算距离
    for i in range(len(labels)):
        k = labels[i]
        proto = protolist[labels[i]][:]
        distance = [pool_embeddinglist[i] - proto]
        distance = np.linalg.norm(distance, axis=1)
        distancelist[k] = distancelist[k] + list(distance)
        originlableslist[k] = originlableslist[k] + [originlables[i]]
        img_nameslist[k] = img_nameslist[k] + [img_names[i]]
    order = []
    lablelist = []
    dirlist = []
    if select_type == 'ADD-BAD':
        for i in range(len(distancelist)):
            order.append(np.array(distancelist[i]).argsort())
    elif select_type == 'ADD-GOOD':
        for i in range(len(distancelist)):
            order.append((-np.array(distancelist[i])).argsort())
    # 挑选得分高的样本，保存到CSV文件
    for j in range(len(order)):
        for v, i in enumerate(order[j]):
            if v < add_ratio*500:
                ft_csv.writerow([str(img_nameslist[j][i])] + [str(originlableslist[j][i])])
                dirlist.append(img_nameslist[j][i])
                lablelist.append(str(originlableslist[j][i]))
            else:
                break
    select_data = list(zip(dirlist, lablelist))
    return select_data
    # for j in range(len(order)):
    #     print(originlableslist[j][0] + '正检数量为：' + '{}/{}'.format(select_amount - k[j], select_amount))
    # print('平均正检率为:', 1-(sum(k)/(select_amount*len(k))))
