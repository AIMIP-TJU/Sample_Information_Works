
import torch.utils.data as Data
import pandas as pd
import numpy as np
from PIL import Image


def get_map(label):
    label = label.values[:, 1]
    label = np.unique(label)
    label_map = {}
    for l, i in enumerate(label):
        label_map[l] = i
    return label_map


class MyDataset(Data.Dataset):
    def __init__(self, file_path, file_name, label_map, select_method=None, transform=None):
        
        self.file_path = file_path
        if select_method:
            self.file_name_list = select_method.select(pd.read_csv(file_name, header=None))
        else:
            self.file_name_list = pd.read_csv(file_name, header=None)
        self.transform = transform
        self.target_map = label_map
        print(label_map)
        

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):        
        image_path = self.file_path + self.file_name_list.values[index, 0]
        img_name = self.file_name_list.values[index, 0]
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_map[self.file_name_list.values[index, 1]]
        #target = self.file_name_list.values[index, 1]

        return img, target, img_name