# -*- coding: utf-8 -*-

"""
Created on 

@file: model_select.py
@author: Guo X.L.

"""


def dataset_select(dataset_name):
   
      
    file_Path = '/root/projects/datasets/NICO/'
    train_name = '../datasets/NICO/Animal/NICO_animal_test.csv'
    test_name = '../datasets/NICO/Animal/NICO_animal_train.csv'
    num_classes = 10
    num_input = 3 
    
         
    return [num_input, num_classes, file_Path, train_name, test_name]

