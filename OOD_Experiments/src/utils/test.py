import math
from tqdm import tqdm
from utils.my_forward import my_forward
import torch.nn.functional as F
import torch
from torch import nn


def test(model, device, dataiter, test_set, batch_size, mission, recoder, tp=False):  
    model.eval()  
    with torch.no_grad():
        aloss = 0
        aacc = 0
        iter_times = math.ceil(len(test_set) / batch_size)
        for iteration in tqdm(range(iter_times)):
            input_data, label, img_name = next(dataiter)
            input_data = input_data.clone().detach().float().to(device)
            label = label.clone().detach().to(device)
            loss, acc = my_forward(mission, model, input_data, label, recoder, tp)

            #######################
            aloss += loss.cpu().detach().numpy()
            aacc += acc.cpu().detach().numpy()
            recoder.log_test_loss(loss.cpu().detach().numpy())
            recoder.log_test_acc(acc.cpu().detach().numpy())

    return aloss / iter_times, aacc / iter_times
