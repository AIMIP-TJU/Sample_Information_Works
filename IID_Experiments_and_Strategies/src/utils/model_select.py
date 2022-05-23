# -*- coding: utf-8 -*-

"""
Created on 

@file: model_select.py
@author: ZhangZ

"""

from src.classifier import Resnet
from src.classifier import vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn
from src.classifier import shufflenet_v2_x1_0
from src.classifier import wide_resnet
from src.classifier import VAE
from src.classifier import alexnet

def model_select(model_name, input_num, num_classes):
    if model_name == 'SCN':
        # return SCN(input_num, num_classes)
        pass
    elif model_name == 'ResNet18':
        return Resnet.ResNet18(input_num, num_classes)
    elif model_name == 'ResNet34':
        return Resnet.ResNet34(input_num, num_classes)
    elif model_name == 'ResNet50':
        return Resnet.ResNet50(input_num, num_classes)
    elif model_name == 'ResNet101':
        return Resnet.ResNet101(input_num, num_classes)
    elif model_name == 'ResNet152':
        return Resnet.ResNet152(input_num, num_classes)
    elif model_name == 'VGG11':
        return vgg11(input_num, num_classes)
    elif model_name == 'VGG13':
        return vgg13(input_num, num_classes)
    elif model_name == 'VGG16':
        return vgg16(input_num, num_classes)
    elif model_name == 'VGG19':
        return vgg19(input_num, num_classes)
    elif model_name == 'VGG11BN':
        return vgg11_bn(input_num, num_classes)
    elif model_name == 'VGG13BN':
        return vgg13_bn(input_num, num_classes)
    elif model_name == 'VGG16BN':
        return vgg16_bn(input_num, num_classes)
    elif model_name == 'VGG19BN':
        return vgg19_bn(input_num, num_classes)

    elif model_name == 'shufflenet_v2_x1_0':
        classifier = shufflenet_v2_x1_0(num_classes)
        return classifier
    elif model_name == 'WRN-22-8':
        classifier = wide_resnet(num_layers=3, widen_factor=8, num_classes=num_classes)
        return classifier
    elif model_name == 'VAE':
        classifier = VAE()
        return classifier
    elif model_name == 'AlexNet':
        classifier = alexnet(input_num,num_classes)
        return classifier