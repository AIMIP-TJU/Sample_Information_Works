# -*- coding: utf-8 -*-

"""
Created on 

@file: model_select.py
@author: ZhangZ

"""
def dataset_select(dataset_name):
    global num_input, num_classes, file_Path, train_name, test_name
    elif dataset_name == 'Cifar10':
        file_Path = '/home/a611/Projects/Datasets/cifar10/'
        train_name = '/home/a611/Projects/Datasets/cifar10/trainval.csv'
        test_name = '/home/a611/Projects/Datasets/cifar10/test.csv'
        num_classes = 10
        num_input = 3
    elif dataset_name == 'mini-Imagenet':
        file_Path = '/home/a611/Projects/Datasets/mini-imagenet/images/'
        train_name = '/home/a611/Projects/Datasets/mini-imagenet/msk_label/train_aver_50000.csv'
        test_name = '/home/a611/Projects/Datasets/mini-imagenet/msk_label/test_aver_10000.csv'
        num_classes = 100
        num_input = 3
    return [num_input, num_classes, file_Path, train_name, test_name]