import src.utils as utils
import time
import torch.utils.data as data
from torch import optim
import torch
from torchvision import transforms
from src.my_select import *
import os
from dataset_select import dataset_select
import itertools
import torch.optim.lr_scheduler as lr_scheduler
import argparse
from src.utils.get_log import log_recorder


"""
Training parameters:
lr_list: The list of learning rate.
device: cuda or cpu.
batch_size: batchsize of the training process.
batch_size_tset: batchsize of the testing process.
EPOCH: lraining rounds.
num_workers: Threads.
dataset_names: Dataset name.
classifier_names: Backbone Network.
dimension: Output feature dimension (modified with network parameters).

Selection parameters:
select_ratios：The proportion of training samples in each cycle.
add_ratios：The proportion of adding samples in each cycle.
method_names: The methods of selecting samples.
select_type: The type of selecting samples(goodset or badset).
"""
parser = argparse.ArgumentParser()
parser.add_argument('--lr_list', type = list, default=[0.01])
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--batch_size_tset', type=int, default=256)
parser.add_argument('--EPOCH', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=8, help='how many subprocesses to use for data loading')
parser.add_argument('--dataset_names', type = list, default = ['cifar10'])
parser.add_argument('--classifier_names', type = list, default = ['ResNet18'])
parser.add_argument('--dimension', type=int, default=512)
parser.add_argument('--select_ratios', type=int, default=[0.2,0.4,0.6,0.8,1])#[0.4,0.5,0.6,0.7,0.75,0.80,0.85,0.90,0.95,1]
parser.add_argument('--add_ratios', type=int, default  = [0.2,0.2,0.2,0.2,0])#[0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0]
parser.add_argument('--method_names', type=list, default=['metric_select'])
parser.add_argument('--select_type', type=list, default=['ADD-GOOD','ADD-BAD'],help='ADD-GOOD or ADD-BAD')
# parser.add_argument('--epoch_decay_start', type=int, default=80)
args = parser.parse_args()

batch_size = args.batch_size
batch_size_tset = args.batch_size_tset
EPOCH = args.EPOCH
num_workers = args.num_workers
dimension = args.dimension
DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
# 默认预处理

if __name__ == '__main__':
    select_data = []
    j = 0
    rq = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    txt_path = './Result/{}/'.format(rq[:8])
    txtfile = txt_path + '{}.txt'.format(rq[8:])
    if os.path.exists(txt_path):
        pass
    else:
        os.makedirs(txt_path)

    for i in itertools.product(args.dataset_names, args.classifier_names, args.method_names,  args.lr_list, args.select_type, args.select_ratios):
        [dataset_name, classifier_name, method_name, lr, select_type,select_ratio] = i
        [num_input,num_classes,file_Path,train_name,test_name] = dataset_select(dataset_name)
        if j == len(args.add_ratios):
            j = 0
            select_data = []
        add_ratio = args.add_ratios[j]
        # 图像预处理
        if dataset_name == 'mini-Imagenet-10'or dataset_name == 'CSEv1':
            transform = {
                "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Resize([224, 224]),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                "test": transforms.Compose([transforms.Resize([224,224]),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        elif dataset_name in ['cifar10_train_amount_unbalance','Cifar10']:
            transform = {
                "train": transforms.Compose([transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
                "test": transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
        else:
            transform = {
                "train": transforms.Compose([transforms.RandomResizedCrop(84),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Resize([84, 84]),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                "test": transforms.Compose([transforms.Resize([84,84]),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        # 训练集数据
        if select_ratio == args.select_ratios[0]:
            base_csv_dir = './Selcetion/{}/{}/{}/{}/'.format(
              dataset_name, classifier_name, method_name, select_type,select_ratio)
            if os.path.exists(base_csv_dir):
                pass
            else:
                os.makedirs(base_csv_dir)
            base_csv_path = base_csv_dir + 'base_{}.csv'.format(select_ratio)
            train_set = utils.dataset.TrainDataset(select_ratio,base_csv_path=base_csv_path,file_name_list=None, pool_list=None, file_path=file_Path, file_name=train_name,num_classes=num_classes, select_data=None, transform=transform['train'] )
        else:
            train_set = utils.dataset.TrainDataset(select_ratio,None, train_set.file_name_list, pool_set.file_name_list, file_Path, train_name, num_classes, select_data=select_data, transform=transform['train'])
        train_loader = data.DataLoader(
            dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # 测试集数据
        test_set = utils.dataset.TestDataset(file_Path, test_name, transform['test'])
        test_loader = data.DataLoader(
            dataset=test_set, batch_size=batch_size_tset, shuffle=True, num_workers=num_workers)
        # Pool中的数据
        if select_ratio != args.select_ratios[-1]:
            pool_set = utils.dataset.Pool(file_Path, train_set.pool_list, transform['test'])
            pool_loader = data.DataLoader(
                dataset=pool_set, batch_size=batch_size_tset, shuffle=True, num_workers=num_workers)
        else:
            pool_set = []

        classifier = utils.model_select(classifier_name, num_input, num_classes).to(DEVICE)
        # classifier = torch.nn.DataParallel(classifier.to(DEVICE))
        # 设置学习率与优化器
        optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9,weight_decay=5e-4)  # 优化器
        decay_epoch = [30, 60, 80]
        scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=decay_epoch, gamma=0.1)
        recoder = log_recorder(dataset_name, classifier_name, method_name, batch_size, EPOCH, lr,select_type,select_ratio)
        top_acc = 0
        max_epoch = 0
        currenttime = time.asctime(time.localtime(time.time()))


        print('基于{}模型的{}数据集，采用{}样本筛选方法，筛选种类{}，训练集{}张，测试集{}张，Pool {}张，训练开始！'.format(classifier_name, dataset_name, method_name,select_type,len(train_set),len(test_set),len(pool_set)))
        with open(txtfile, "a") as myfile:
            myfile.write(
                '基于{}模型的{}数据集，采用{}样本筛选方法，筛选种类{}，训练集{}张，测试集{}张，Pool {}张，训练开始！\n'.format(classifier_name, dataset_name, method_name,select_type,len(train_set),len(test_set),len(pool_set)))
        for i in range(EPOCH):
            # print('EPOCH:', i + 1)
            train_iter = iter(train_loader)
            train_iter2 = iter(train_loader)
            test_iter = iter(test_loader)

            train_loss, train_acc= utils.train(classifier, classifier_name,DEVICE, train_iter, optimizer, train_set, batch_size,
                                                recoder, dimension, num_classes,i,EPOCH)
            test_loss, test_acc,_,_,_,_,_,_,_,_,_ = utils.test(classifier,dimension,classifier_name,num_classes, DEVICE,test_iter, test_set,batch_size_tset, recoder,True, True)

            print('Epoch [{}/{}] Train Accuracy: {}% Test Accuracy: {}% '.format(i + 1, args.EPOCH,train_acc*100,test_acc*100))
            scheduler.step()

            if test_acc > top_acc:
                top_acc = test_acc
                max_epoch = i + 1
            if i+1 == EPOCH:
                model_path = './Model/{}/{}/{}/{}_{}_{}_{}/'.format(dataset_name, classifier_name, method_name, lr,
                                                                    currenttime, select_type, select_ratio)
                if os.path.exists(model_path):
                    pass
                else:
                    os.makedirs(model_path)
                torch.save(classifier.state_dict(), model_path + 'test_acc_{:.6f}_epoch_{}'.format(test_acc, i + 1),
                           _use_new_zipfile_serialization=False)
                test_loss, test_acc,TrainEmbeddinglist,TrainEmbedding,protolist,embed_maxlist, _, _,_, _, trainlabels\
                    = utils.test(classifier,dimension,classifier_name, num_classes,
                                                                         DEVICE, train_iter2, train_set, batch_size,
                                                                         recoder, True, True)
                if select_ratio != args.select_ratios[-1]:
                    pool_iter = iter(pool_loader)
                    pool_loss, pool_acc, _,pool_embeddinglist,_,_, outputlist, losslist,img_names, originlables, labels\
                        = utils.test(classifier,dimension,classifier_name, num_classes,DEVICE, pool_iter, pool_set, batch_size_tset, recoder, False, True)
                    if method_name == 'metric_select':
                        # 根据度量距离筛选
                        select_data = metric_select(method_name, originlables,num_classes, classifier_name, dataset_name, pool_embeddinglist,
                                      img_names,
                                      protolist, select_ratio, labels, select_ratio,select_type,add_ratio)
                    elif method_name == 'distance_entropy_select':
                        # 根据度量距离熵筛选
                        select_data = distance_entropy_select(method_name, originlables, num_classes, classifier_name,labels,
                                                              dataset_name, TrainEmbeddinglist, pool_embeddinglist,
                                                              img_names,
                                                              protolist, select_ratio, select_type, add_ratio)
                    elif method_name == 'distance_entropy_aver':
                        select_data = distance_entropy_aver(method_name, originlables, num_classes, classifier_name,labels,
                                                              dataset_name, TrainEmbeddinglist, pool_embeddinglist,
                                                              img_names,
                                                              protolist, select_ratio, select_type, add_ratio)

                    elif method_name == 'entropy_select':
                        # # 根据熵的样本筛选
                        select_data = entropy_select(method_name, originlables, classifier_name, dataset_name,
                                                     outputlist, img_names, select_ratio, labels, select_type,
                                                     add_ratio, num_classes)

                    else:
                        aa = "No Select"
                        print(aa)
                        select_data = []
                    recoder.log_close()
        # print(select_data)
        print('SelectRatio [{}/100]已完成 max_epoch: {} top_acc: {}%'.format(select_ratio*100,max_epoch,top_acc*100))
        with open(txtfile, "a") as myfile:
            myfile.write(
                'SelectRatio [{}/100]已完成 max_epoch: {} top_acc: {}% \n'.format(select_ratio*100,max_epoch,top_acc*100))

        print('********************************************************************************************************')
        j += 1