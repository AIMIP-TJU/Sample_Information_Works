import math
import time
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np

def entropy(c):
    result=-1;
    if(len(c)>0):
        result=0;
    for x in c:
        result+=(-x)*math.log(x,2)
    return result

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
def my_forward(mission, model, input_data, label, recoder, tp=False):
        
    #loss_function = vae_loss_function()    
    loss_function = nn.CrossEntropyLoss()

    if mission == 'classification':
        
        output, embedding = model(input_data)
        #print(model.children())
        corrent = torch.eq(torch.argmax(output, dim=1), label)
        time4 = time.time()
        time3 = time.time()
        loss = loss_function(output, label.long())
        acc = torch.mean(corrent.float())
        if tp:
            for i in range(len(torch.argmax(output, dim=1).cpu().numpy())):
                recoder.log_test_label(label.cpu().numpy()[i],
                                       torch.argmax(output, dim=1).cpu().numpy()[i])
        return loss, acc
    
    if mission == 'extraction':
        label_map = {'apple': 0, 'bottle': 1, 'car': 2, 'container': 3, 'cup': 4, 'doll': 5, 
                        'fleet': 6, 'headset': 7, 'milk': 8, 'pepper': 9, 'plant': 10}
        class_name = list(label_map.keys())
        
        output, embedding = model(input_data)
        #print(output)
        embedding = embedding.cpu().numpy()[0]
        #print(embedding)
        corrent = torch.eq(torch.argmax(output, dim=1), label)
        time4 = time.time()
        time3 = time.time()
        loss = loss_function(output, label.long())
        acc = torch.mean(corrent.float())
        if tp:
            
            for i in range(len(torch.argmax(output, dim=1).cpu().numpy())):
                recoder.log_test_label(label.cpu().numpy()[i],
                                       torch.argmax(output, dim=1).cpu().numpy()[i])
        
        pred_label = torch.argmax(output, dim=1).cpu().numpy()[0]
        pred_class = class_name[pred_label]
        #preds = F.softmax(output, dim=1).cpu().numpy()[0]  
        #e = entropy(preds)
        return loss, acc, embedding, pred_class
    else:
        pass
        return 0, 0
