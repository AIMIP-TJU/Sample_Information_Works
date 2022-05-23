# -*- coding: utf-8 -*-

"""
Created on 

@file: model_select.py
@author: ZhangZ

"""
import selectionmethod


def method_select(method_name, proportion):
    if method_name == 'random':
        return selectionmethod.select_random(proportion)
    elif method_name == 'ResNet18':
        return selectionmethod.select_random(proportion)
