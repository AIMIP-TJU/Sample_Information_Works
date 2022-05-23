import utils
import torch.utils.data as data
from torch import optim
import math
import torch
from utils.get_log import log_recorder
import os
from config import DEVICE, batch_size, mission, EPOCH, num_workers, lr_list, lrf, transform
from config import dataset_names, classifier_names, method_names, proportions
import itertools
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torch.nn as nn
import warnings



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
        train_set = utils.MyDataset(file_Path, train_name, label_map,
                                    select_method, transform['train'])
        train_loader = data.DataLoader(
            dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_set = utils.MyDataset(file_Path, test_name, label_map,
                                   None, transform['test'])
        test_loader = data.DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        classifier = utils.model_select(classifier_name, num_input, num_classes).to(DEVICE)
        classifier = nn.DataParallel(classifier.to(DEVICE))  # multi-GPU
        #==========Load pretrained model==============#
        checkpoint = torch.load('/root/projects/gxl/fewshot/Model/pretrained/mini-imagenet-pretrain_ResNet18_random_1/test_acc_0.766925_epoch_159')
        classifier.load_state_dict(checkpoint)
        classifier.module.linear = nn.Linear(512 , num_classes).to(DEVICE)
        
        optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9,weight_decay=5e-4)
        decay_epoch = [30, 60, 80, 100]
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=decay_epoch, gamma=0.1)

        recoder = log_recorder(dataset_name, classifier_name, method_name, proportion, mission, batch_size, EPOCH, lr)

        top_acc = 0
        model_path = '../Model/NICO/{}_{}_{}_{}_{}/'.format(dataset_name, classifier_name, method_name, proportion,lr)
        if os.path.exists(model_path):
            pass
        else:
            os.mkdir(model_path)
        for i in range(EPOCH):
            print('EPOCH:', i + 1)
            train_iter = iter(train_loader)
            test_iter = iter(test_loader)
            ########################################
            train_loss, train_acc = utils.train(classifier, DEVICE, train_iter, optimizer, train_set, batch_size,
                                                mission,
                                                recoder)
            test_loss, test_acc = utils.test(classifier, DEVICE, test_iter, test_set, batch_size, mission, recoder)
            scheduler.step()
            ########################################
            print('The test loss of this epoch is {}, the test acc is {}'.format(test_loss, test_acc)
            if test_acc > top_acc:
                top_acc = test_acc
                torch.save(classifier.state_dict(), model_path + 'test_acc_{:.6f}_epoch_{}'.format(test_acc, i + 1),
                           _use_new_zipfile_serialization=False)
        recoder.log_close()
