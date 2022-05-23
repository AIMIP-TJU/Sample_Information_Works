# -*- coding: utf-8 -*-

"""
Created on 

@file: model_select.py
@author: ZhangZ

"""
import torchvision
from torch import nn

import model4project


def model_select(model_name, input_num, num_classes, att_type):
    if model_name == 'SCN':
        return model4project.SCN(input_num, num_classes)
    elif model_name == 'ResNet18':
        return model4project.ResNet18(input_num, num_classes, att_type)
    elif model_name == 'ResNet34':
        return model4project.ResNet34(input_num, num_classes)
    elif model_name == 'ResNet50':
        return model4project.ResNet50(input_num, num_classes)
    elif model_name == 'ResNet101':
        return model4project.ResNet101(input_num, num_classes)
    elif model_name == 'ResNet152':
        return model4project.ResNet152(input_num, num_classes)
    elif model_name == 'VGG11':
        return model4project.vgg11(input_num, num_classes)
    elif model_name == 'VGG13':
        return model4project.vgg13(input_num, num_classes)
    elif model_name == 'VGG16':
        return model4project.vgg16(input_num, num_classes)
    elif model_name == 'VGG19':
        return model4project.vgg19(input_num, num_classes)
    elif model_name == 'VGG11BN':
        return model4project.vgg11_bn(input_num, num_classes)
    elif model_name == 'VGG13BN':
        return model4project.vgg13_bn(input_num, num_classes)
    elif model_name == 'VGG16BN':
        return model4project.vgg16_bn(input_num, num_classes)
    elif model_name == 'VGG19BN':
        return model4project.vgg19_bn(input_num, num_classes)

    elif model_name == 'shufflenet_v2_x1_0':
        classifier = model4project.shufflenet_v2_x1_0(num_classes)
        return classifier
    elif model_name == 'WRN-28-2':
        classifier = model4project.wide_resnet(num_layers=4, widen_factor=2, num_classes=num_classes)
        return classifier
    elif model_name == 'WRN-22-8':
        classifier = model4project.wide_resnet(num_layers=3, widen_factor=8, num_classes=num_classes)
        return classifier
    elif model_name == 'VAE':
        classifier = model4project.VAE()
        return classifier