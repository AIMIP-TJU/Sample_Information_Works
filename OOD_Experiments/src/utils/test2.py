import math
from tqdm import tqdm
from utils.my_forward import my_forward
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import shutil
import os
import xml.etree.ElementTree as ET
#from PIL import ImageFile        
#ImageFile.LOAD_TRUNCATED_IMAGES = True

def copy_img(class_name, img_name):
    filepath = '/root/projects/gxl/fewshot/src/ResNet18/img_base/visual3/' + class_name
    if os.path.exists(filepath):
        pass
    else:
        os.mkdir(filepath)
    imgpath = '/root/projects/gxl/datasets/images/' + img_name
    shutil.copy(imgpath, filepath)

def crop_img(img, img_name):
    xmlfile = '/root/projects/gxl/datasets/17/xml/%s.xml' % img_name
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    

def test2(model, device, dataiter, test_set, batch_size, mission, recoder, tp=False): 
    model.eval()  
    img_names = []
    labels = []
    gray = {}
    
    
    with torch.no_grad():
        aloss = 0
        aacc = 0
        iter_times = math.ceil(len(test_set) / batch_size)
        #class_number = list(test_set)[0][1]
        
        data = np.zeros(shape = (iter_times,512))
        
        dists_1 = np.zeros(iter_times)
        dists_2 = np.zeros(iter_times)
        
        #proto = np.load(str('ResNet18/txt_0.613/protos/'+str(class_number)+'.npy'))
        print('!!!!!',iter_times)
        
        for iteration in tqdm(range(iter_times)):
            input_data, label, img_name = next(dataiter)
            #gray[img_name] = img_gray            
            labels.append(label.numpy()[0])
            input_data = input_data.clone().detach().float().to(device)
            label = label.clone().detach().to(device)
            
            if mission == 'classification':
                loss, acc = my_forward(mission, model, input_data, label, recoder, tp)
            if mission == 'extraction':
                loss, acc, embedding, pred_class = my_forward(mission, model, input_data, label, recoder, tp)
            
            img_names.append(img_name[0])            
            data[iteration] = embedding
            
            
            #######################
            aloss += loss.cpu().detach().numpy()
            aacc += acc.cpu().detach().numpy()
            recoder.log_test_loss(loss.cpu().detach().numpy())
            recoder.log_test_acc(acc.cpu().detach().numpy())
            
            #copy_img(pred_class, img_name[0])
               
    return aloss / iter_times, aacc / iter_times, data, img_names, labels

    