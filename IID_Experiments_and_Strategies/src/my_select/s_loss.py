import csv
import shutil
import os
import numpy as np
import torch.nn.functional as F
import torch
import os

from matplotlib import pyplot as plt

from src.utils.my_mkdir import mkdir


def s_loss(method_name, originlables, classifier_name, dataset_name,
                                                 outputlist, img_names, select_ratio, labels, select_type,
                                                 add_ratio, num_classes,losslist):
    # 定义要创建的目录
    global scorelist, select_data
    filepath = './Selcetion/{}/{}/{}/{}/'.format(dataset_name, classifier_name, method_name,select_type)
    # 调用函数创建目录
    if os.path.exists(filepath):
        pass
    else:
        os.makedirs(filepath)
    # 创建csv文件
    ft = open(str('{}{}_{}.csv'.format(filepath,dataset_name,select_ratio)), 'w', newline='')
    ft_csv = csv.writer(ft)

    if select_type=='ADD-GOOD':
        # 由大到小
        order = (-np.array(losslist)).argsort()
    elif select_type=='ADD-BAD':
        # 由小到大
        order = np.array(losslist).argsort()
    # 创建列表
    lablelist = []
    dirlist = []
    i=0
    print("开始筛选")
    for v, i in enumerate(order):
        if v < add_ratio*50000:
            ft_csv.writerow([img_names[i]] + [str(originlables[i])])
            dirlist.append(img_names[i])
            lablelist.append(str(originlables[i]))
        else:
            break
    select_data = list(zip(dirlist, lablelist))
    return select_data