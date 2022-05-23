import torch.utils.data as Data
import pandas as pd
import numpy as np
from PIL import Image
import csv
import random

def get_map(label):
    label = label[:, 1]
    label = np.unique(label)
    label_map = {}
    for i, l in enumerate(label):
        label_map[l] = i
    return label_map


class TrainDataset(Data.Dataset):
    """
    数据集
    """

    def __init__(self, method_name, file_name_list, file_path, file_name, select_data, transform=None):
        """

        :param file_path:数据集位置
        :param file_name:样本索引目录
        :param target_map:数据集种类标签和实际标签间的映射
        :param transform:图像变换与增强

        """
        self.file_path = file_path  # '/root/disk/Zz/Dataset_rcs_structed/'
        # if select_method:
        #     self.file_name_list = select_method.select(pd.read_csv(file_name, header=None))
        # else:
        if select_data is None:
            with open(file_name, 'r') as f:
                files_reader = csv.reader(f)
                files = []
                for row in files_reader:
                    files.append(row)
            self.file_name_list = np.array(files)

        else:
            new_file_name_list = []
            for i in file_name_list:
                flag = True
                for j in select_data:
                    if i[0] == j[0] and i[1] == j[1]:
                        flag = False
                if flag:
                    new_file_name_list.append(i)
            self.file_name_list = np.array(new_file_name_list)
        self.transform = transform
        label_map = get_map(self.file_name_list)
        self.target_map = label_map

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        image_path = self.file_path + self.file_name_list[index, 0]
        img_name = self.file_name_list[index, 0]
        img_lable = self.file_name_list[index, 1]
        # img = Image.open('D:/Projects/Datasets/example/di.jp')g
        # from PIL import ImageFile
        # ImageFile.LOAD_TRUNCATED_IMAGES = True

        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        # img = np.array(img).astype(np.float32)
        target = self.target_map[self.file_name_list[index, 1]]

        return img, target, img_name, img_lable

class TestDataset(Data.Dataset):
    """
    数据集
    """

    def __init__(self, file_path, file_name, transform=None):
        """

        :param file_path:数据集位置
        :param file_name:样本索引目录
        :param target_map:数据集种类标签和实际标签间的映射
        :param transform:图像变换与增强

        """
        self.file_path = file_path  # '/root/disk/Zz/Dataset_rcs_structed/'
        # if select_method:
        #     self.file_name_list = select_method.select(pd.read_csv(file_name, header=None))
        # else:
        self.file_name_list = pd.read_csv(file_name, header=None)
        self.transform = transform
        label_map = get_map(self.file_name_list.values)
        self.target_map = label_map

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        image_path = self.file_path + self.file_name_list.values[index, 0]
        img_name = self.file_name_list.values[index, 0]
        img_lable = self.file_name_list.values[index, 1]
        # img = Image.open('D:/Projects/Datasets/example/di.jp')g
        # from PIL import ImageFile
        # ImageFile.LOAD_TRUNCATED_IMAGES = True

        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        # img = np.array(img).astype(np.float32)
        target = self.target_map[self.file_name_list.values[index, 1]]

        return img, target, img_name, img_lable
class Pool(Data.Dataset):
    """
    数据集
    """

    def __init__(self,file_path, pool_list, transform=None):
        self.file_path = file_path
        self.file_name_list = pool_list
        self.transform = transform
        label_map = get_map(self.file_name_list)
        self.target_map = label_map

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        image_path = self.file_path + self.file_name_list[index, 0]
        img_name = self.file_name_list[index, 0]
        img_lable = self.file_name_list[index, 1]
        # img = Image.open('D:/Projects/Datasets/example/di.jp')g
        # from PIL import ImageFile
        # ImageFile.LOAD_TRUNCATED_IMAGES = True

        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        # img = np.array(img).astype(np.float32)
        target = self.target_map[self.file_name_list[index, 1]]

        return img, target, img_name, img_lable