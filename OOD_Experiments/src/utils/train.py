import math
from tqdm import tqdm
from utils.my_forward import my_forward
import torch.nn.functional as F
import torch
from torch import nn

def train(model, device, dataiter, optimizer, train_set, batch_size, mission, recoder, tp=False):  
   
    model.train()  
    aloss = 0
    aacc = 0
    iter_times = math.ceil(len(train_set) / batch_size)
    for iteration in tqdm(range(iter_times)):
        data, label, img_name = next(dataiter)
        data = data.clone().float().to(device)
        label = label.clone().detach().to(device)

        loss, acc = my_forward(mission, model, data, label, recoder, tp)

        #############
        aacc += acc.cpu().detach().numpy()
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()  
        aloss += loss.cpu().detach().numpy()
        recoder.log_train_loss(loss.cpu().detach().numpy())
        recoder.log_train_acc(acc.cpu().detach().numpy())


    return aloss / iter_times, aacc / iter_times
