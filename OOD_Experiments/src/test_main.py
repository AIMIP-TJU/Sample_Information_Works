import utils
import torch.utils.data as data
from torch import optim
import math
import torch
import numpy as np
from utils.get_log import log_recorder
import os
from config import DEVICE, batch_size, mission, EPOCH, num_workers, lr_list, lrf, transform
from config import dataset_names, classifier_names, method_names, proportions
import itertools
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torch.nn as nn
import torch.multiprocessing
import csv
from collections import Counter
import matplotlib.pyplot as plt
import operator
import scipy.stats
import pdb
import sklearn.metrics as skm
import warnings
from mutual import getmutual
torch.multiprocessing.set_sharing_strategy('file_system')


    
def writecsv(path, method, levels, order):

    filepath = classifier_name + '/' + path
    if not os.path.exists(filepath):
        os.makedirs(filepath)
     
    level = locals() 
    for i in range(1, levels + 1):
        level['f_' + str(i)] = open(('%s/%s-level%d.csv' % (filepath, method, i)),'a', newline = '')
        level[f'f{i}_csv'] = csv.writer(level['f_' + str(i)])
        #print(level['f1_csv'])
    
    for i,k in enumerate(reversed(order)):
        flag = len(order) / levels
        '''
        if i < flag:    
            level['f1_csv'].writerow([img_names[k]] + [img_names[k].split('_')[0]])
        elif i < flag * 2:                                                                  
            level['f2_csv'].writerow([img_names[k]] + [img_names[k].split('_')[0]])    
        elif i < flag * 3:                                                                  
            level['f3_csv'].writerow([img_names[k]] + [img_names[k].split('_')[0]])  
        '''
        if i < len(order)*0.6:
            level['f1_csv'].writerow([img_names[k]] + [img_names[k].split('/')[1]])
        else:
            level['f2_csv'].writerow([img_names[k]] + [img_names[k].split('/')[1]])
        
def write_data_eu(method):
    proto = np.load('%s/%sproto/proto_%s.npy' % (classifier_name, method, dataset_name))     
    dist = np.linalg.norm(proto - data_, axis = 1)
    order = np.argsort(dist)
    writecsv('CSEeu', 'eu', 2, order) 
        
   
if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    for i in itertools.product(dataset_names, classifier_names, method_names, proportions, lr_list):
        [dataset_name, classifier_name, method_name, proportion, lr] = i
        [
            num_input,
            num_classes,
            file_Path,
            train_name,
            test_name
        ] = utils.dataset_select(dataset_name)
        select_method = utils.method_select(method_name, proportion)
        label_map = utils.get_map(test_name)
        test_set = utils.MyDataset(file_Path, test_name, label_map,
                                   None, transform['test'])

        test_loader = data.DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        classifier = utils.model_select(classifier_name, num_input, num_classes).to(DEVICE)
        classifier = nn.DataParallel(classifier.to(DEVICE))  # multi-GPU
        
        #=============Load the feature extractor,============#
        #=========write down your own model path here========#
        checkpoint = torch.load('root/projects/src/Model/')
        classifier.load_state_dict(checkpoint)    
        recoder = log_recorder(dataset_name, classifier_name, method_name, proportion, mission, batch_size, EPOCH, lr)
        
        print('Test')
        test_iter = iter(test_loader)  
        test_loss, test_acc, data_, img_names, labels = utils.test2(classifier, DEVICE, test_iter, test_set, batch_size, mission, recoder)
        print(test_acc)
        
        
        #=============Save feature prototype============#
        proto_path = '%s/CSEproto/' % classifier_name
        if not os.path.exists(proto_path):
            os.makedirs(proto_path)
        np.save(proto_path + , np.mean(data_,axis=0))
        
        #============Split dataset, write down==========#
        #============the dataset's name here============#
        #write_data_eu('CSE')   

        
           
        
