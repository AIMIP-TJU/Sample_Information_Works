
import torch.utils.data as Data
import pandas as pd
import numpy as np
from PIL import Image


def get_map(file_name):
    file_name_list = pd.read_csv(file_name[0], header=None)
    label = file_name_list.values[:, 1]
    label = np.unique(label)
    label_map = {}
    for i, l in enumerate(label):
        label_map[l] = i
    return label_map


class MyDataset(Data.Dataset):
    """
    数据集
    """

    def __init__(self, file_path, file_name, label_map, transform=None):
        """

        :param file_path:数据集位置
        :param file_name:样本索引目录
        :param target_map:数据集种类标签和实际标签间的映射
        :param transform:图像变换与增强

        """
        self.file_path = file_path  # '/root/disk/Zz/Dataset_rcs_structed/'
        if len(file_name) == 1:
            self.file_name_list = pd.read_csv(file_name[0], header=None)
        else:
            self.file_name_list = pd.concat([pd.read_csv(name, header=None) for name in file_name])
            # self.file_name_list = pd.concat([self.file_name_list, pd.read_csv(file_name[2], header=None)])
        self.transform = transform
        self.target_map = label_map


    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        image_path = self.file_path + self.file_name_list.values[index, 0]
        # img = Image.open('D:/Projects/Datasets/example/di.jpg')
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        # img = np.array(img).astype(np.float32)
        target = self.target_map[self.file_name_list.values[index, 1]]

        return img, target, self.file_name_list.values[index, 0], self.file_name_list.values[index, 1]